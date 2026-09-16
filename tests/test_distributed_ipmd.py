"""Four workers exercise IPMD's own update and checkpoint implementations."""
from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from test_distributed_ppo import config

from rlopt.agent.ipmd.distributed import DistributedIPMD
from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.env_utils import env_maker


def make_config(root, rank, global_advantage):
    cfg = IPMDRLOptConfig()
    base = config(root, rank, global_advantage)
    for name in (
        "env",
        "device",
        "seed",
        "collector",
        "loss",
        "replay_buffer",
        "compile",
        "ppo",
        "policy",
        "value_function",
        "logger",
        "trainer",
        "optim",
    ):
        setattr(cfg, name, copy.deepcopy(getattr(base, name)))
    cfg.collector.total_frames = 64
    cfg.save_interval = 32
    cfg.ppo.freeze_normalizers = True
    cfg.ppo.update_normalizers_after_rollout = False
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    for name in (
        "reward_loss_coeff",
        "reward_l2_coeff",
        "reward_grad_penalty_coeff",
        "reward_logit_reg_coeff",
        "reward_param_weight_decay_coeff",
        "bc_coef",
        "diversity_bonus_coeff",
        "rollout_bc_coef",
    ):
        setattr(cfg.ipmd, name, 0.0)
    return cfg


def worker(rank, root, rendezvous, global_advantage):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    root = Path(root)
    try:
        cfg = make_config(root / "parity", rank, global_advantage)
        env = env_maker(cfg)
        agent = DistributedIPMD(env, cfg)
        baseline = None
        if rank == 0:
            reference = copy.deepcopy(cfg)
            reference.logger.log_dir = str(root / "baseline")
            baseline = IPMD(env_maker(reference), reference)
            baseline.actor_critic.load_state_dict(agent.actor_critic.state_dict())
        batch = agent.pre_iteration_compute(next(iter(agent.collector))).reshape(-1)
        batches = [None] * 4
        dist.all_gather_object(batches, batch)
        empty = TensorDict({}, batch_size=batch.batch_size)
        agent.update(batch, 0, empty, torch.tensor(0.0))
        if rank == 0:
            full = torch.cat(batches)
            baseline.update(
                full, 0, TensorDict({}, batch_size=full.batch_size), torch.tensor(0.0)
            )
            for a, b in zip(
                agent._grad_clip_params, baseline._grad_clip_params, strict=True
            ):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
            baseline.collector.shutdown()
            baseline.env.close()
        agent.collector.shutdown()
        env.close()
        dist.barrier()
        cfg = make_config(root / "train", rank, global_advantage)
        env = env_maker(cfg)
        agent = DistributedIPMD(env, cfg)
        agent.train()
        env.close()
        checkpoint = next((root / "train" / "rank-0").rglob("model_step_64.pt"))
        cfg = make_config(root / "resume", rank, global_advantage)
        cfg.collector.total_frames = 128
        env = env_maker(cfg)
        agent = DistributedIPMD(env, cfg, max_training_seconds=1e-9)
        agent.load_model(str(checkpoint))
        agent.train()
        env.close()
        health = json.loads((agent.log_dir / "distributed_health.json").read_text())
        assert health["initial_frames"] == 64
        assert health["cumulative_env_frames"] == 96
        assert health["optimizer_updates_this_segment"] == 1
        assert health["max_model_difference"] == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("global_advantage", [True, False])
def test_four_worker_ipmd_parity_and_resume(tmp_path, global_advantage):
    mp.spawn(
        worker,
        args=(str(tmp_path), str(tmp_path / "group"), global_advantage),
        nprocs=4,
        join=True,
    )
    assert not list((tmp_path / "train" / "rank-1").rglob("*.pt"))
