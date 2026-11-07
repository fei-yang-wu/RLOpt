"""Two real Gloo workers: pooled moments, PPO update parity, train and resume."""

from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rlopt.agent.ppo.distributed import DistributedPPO, global_moments
from rlopt.agent.ppo.ppo import PPO, PPORLOptConfig, RunningMeanStdCatInputs
from rlopt.base_class import BaseAlgorithm
from rlopt.env_utils import env_maker


def config(root, rank, global_advantage):
    cfg = PPORLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = cfg.device = "cpu"
    cfg.env.num_envs = 1
    cfg.seed = 17 + rank
    cfg.collector.frames_per_batch = 8
    cfg.collector.total_frames = 32
    cfg.loss.mini_batch_size = 8
    cfg.loss.epochs = 1
    cfg.replay_buffer.size = 8
    cfg.compile.compile = cfg.compile.cudagraphs = False
    cfg.ppo.entropy_coeff = (
        0.0  # Remove Monte Carlo entropy noise for exact update parity.
    )
    cfg.ppo.normalize_advantage_global = global_advantage
    cfg.policy.input_keys = ["observation"]
    cfg.value_function.input_keys = ["observation"]
    cfg.policy.num_cells = cfg.value_function.num_cells = [8]
    cfg.policy.normalize_input = cfg.value_function.normalize_input = True
    cfg.logger.backend = ""
    cfg.logger.log_to_console = False
    cfg.logger.log_dir = str(root / f"rank-{rank}")
    cfg.trainer.progress_bar = False
    cfg.optim.scheduler = "adaptive"
    cfg.save_interval = 16
    return cfg


def worker(rank, root, rendezvous, global_advantage):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    root = Path(root)
    try:
        all_values = torch.tensor([[1.0, 2.0], [2.0, 4.0], [9.0, -3.0], [20.0, 6.0]])
        local = all_values[rank * 2 : (rank + 1) * 2]
        mean, variance, count = global_moments(local)
        torch.testing.assert_close(mean, all_values.mean(0))
        torch.testing.assert_close(variance, all_values.var(0))
        assert count == 4
        local_norm = RunningMeanStdCatInputs(torch.nn.Identity(), 2)
        reference_norm = RunningMeanStdCatInputs(torch.nn.Identity(), 2)
        local_norm._update(local, moments=(mean, variance, count))
        reference_norm._update(all_values)
        torch.testing.assert_close(local_norm.running_var, reference_norm.running_var)

        cfg = config(root, rank, global_advantage)
        env = env_maker(cfg)
        agent = DistributedPPO(env, cfg)
        baseline = None
        if rank == 0:
            baseline_cfg = copy.deepcopy(cfg)
            baseline_cfg.logger.log_dir = str(root / "baseline")
            baseline = PPO(env_maker(baseline_cfg), baseline_cfg)
            baseline.actor_critic.load_state_dict(agent.actor_critic.state_dict())
        rollout = next(iter(agent.collector))
        rollout = agent.pre_iteration_compute(rollout).reshape(-1)
        batches = [None, None]
        dist.all_gather_object(batches, rollout)
        agent.update(rollout, 0)
        if rank == 0:
            baseline.update(torch.cat(batches, dim=0), 0)
            for left, right in zip(
                agent._grad_clip_params, baseline._grad_clip_params, strict=True
            ):
                torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
            baseline.collector.shutdown()
            baseline.env.close()
        agent.collector.shutdown()
        env.close()
        dist.barrier()

        cfg = config(root / "train", rank, global_advantage)
        env = env_maker(cfg)
        agent = DistributedPPO(env, cfg)
        agent.train()
        env.close()
        checkpoints = list((root / "train" / "rank-0").rglob("model_step_32.pt"))
        assert len(checkpoints) == 1
        cfg = config(root / "resume", rank, global_advantage)
        cfg.collector.total_frames = 64
        env = env_maker(cfg)
        resumed = DistributedPPO(env, cfg, max_training_seconds=1e-9)
        resumed.load_model(str(checkpoints[0]))
        resumed.train()
        env.close()
        health = json.loads((resumed.log_dir / "distributed_health.json").read_text())
        assert health["stop_reason"] == "time_budget"
        assert health["initial_frames"] == 32
        assert health["cumulative_env_frames"] == 48
        assert health["optimizer_updates_this_segment"] == 1
        assert health["max_model_difference"] == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("global_advantage", [True, False])
def test_two_worker_ppo_parity_and_resume(tmp_path, global_advantage):
    mp.spawn(
        worker,
        args=(str(tmp_path), str(tmp_path / "group"), global_advantage),
        nprocs=2,
        join=True,
    )
    assert not list((tmp_path / "train" / "rank-1").rglob("*.pt"))


def budget_worker(rank, root, rendezvous):
    """``trainer.max_train_seconds`` spent on ONE rank stops both together."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    root = Path(root)
    try:
        cfg = config(root, rank, False)
        cfg.collector.total_frames = 64
        # Only rank zero has a (tiny) budget; rank one would run to 64 frames.
        cfg.trainer.max_train_seconds = 1e-9 if rank == 0 else None
        env = env_maker(cfg)
        agent = DistributedPPO(env, cfg)
        assert agent.supports_train_time_budget
        agent.train()
        env.close()
        health = json.loads((agent.log_dir / "distributed_health.json").read_text())
        assert health["stop_reason"] == "time_budget"
        # One iteration of 8 local frames per rank: 16 global frames.
        assert health["cumulative_env_frames"] == 16
        assert health["max_model_difference"] == 0
    finally:
        dist.destroy_process_group()


def test_one_rank_time_budget_stops_every_rank(tmp_path):
    mp.spawn(
        budget_worker,
        args=(str(tmp_path), str(tmp_path / "group")),
        nprocs=2,
        join=True,
    )
    checkpoints = list((tmp_path / "rank-0").rglob("model_step_*.pt"))
    assert [p.name for p in checkpoints] == ["model_step_16.pt"]
    assert not list((tmp_path / "rank-1").rglob("*.pt"))


@pytest.fixture
def solo_group(tmp_path):
    """One rank is enough for what does not depend on the world size."""
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'solo'}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


@pytest.mark.usefixtures("solo_group")
def test_critic_only_steps_are_refused(tmp_path):
    """They run outside ``update``: unaveraged gradients would desync the critic."""
    cfg = config(tmp_path, 0, False)
    cfg.ppo.actor_learning_rate = cfg.ppo.critic_learning_rate = 1.0e-3
    cfg.ppo.critic_extra_epochs = 1
    with pytest.raises(ValueError, match="critic_extra_epochs"):
        DistributedPPO(None, cfg)


@pytest.mark.usefixtures("solo_group")
def test_rank_local_entries_are_labelled_on_the_one_write_path(tmp_path, monkeypatch):
    """The env payload joins a row after ``record`` pooled it, so it stays rank 0's."""
    cfg = config(tmp_path, 0, False)
    env = env_maker(cfg)
    agent = DistributedPPO(env, cfg)
    rows = []
    monkeypatch.setattr(
        BaseAlgorithm, "_env_log_metrics", lambda _self: {"Metrics/mpjpe": 1.0}
    )
    monkeypatch.setattr(
        BaseAlgorithm,
        "log_metrics",
        lambda _self, metrics, **_kwargs: rows.append(dict(metrics)),
    )
    agent.log_training_metrics(
        {
            "train/step_reward_mean": 0.5,
            "train/env_reward_mean": 0.5,
            "train/loss_critic": 2.0,
        },
        frames=8,
        iteration=1,
    )
    agent.collector.shutdown()
    env.close()
    (row,) = rows
    assert row["rank0_env/Metrics/mpjpe"] == 1.0
    assert "Metrics/mpjpe" not in row
    assert row["rank0/train/step_reward_mean"] == 0.5
    assert row["rank0/train/env_reward_mean"] == 0.5
    assert row["train/loss_critic"] == 2.0


@pytest.mark.usefixtures("solo_group")
def test_checkpoint_carries_scheduler_and_resume_counters(tmp_path):
    """The base save/load path owns the scheduler; this class adds the counters."""

    def build(name):
        cfg = config(tmp_path / name, 0, False)
        cfg.optim.scheduler = "linearlr"
        cfg.optim.scheduler_kwargs = {
            "start_factor": 1.0,
            "end_factor": 0.1,
            "total_iters": 100,
        }
        env = env_maker(cfg)
        return env, DistributedPPO(env, cfg)

    env, agent = build("saved")
    agent.optim.step()
    for _ in range(7):
        agent.lr_scheduler.step()
    agent._global_frames = 4096
    agent._updates_completed = 12
    agent.save_model(tmp_path / "ckpt", 4096)
    agent.collector.shutdown()
    env.close()
    checkpoint = tmp_path / "ckpt" / "model_step_4096.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["lr_scheduler_state_dict"]["last_epoch"] == 7
    assert saved["cumulative_env_frames"] == 4096
    assert saved["optimizer_updates"] == 12

    env, restored = build("restored")
    restored.load_model(str(checkpoint))
    assert restored.lr_scheduler.last_epoch == 7
    assert restored._resume_frame_offset == 4096
    assert restored._resume_updates == 12
    restored.collector.shutdown()
    env.close()
