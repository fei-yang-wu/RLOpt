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
