from __future__ import annotations

import pytest

from rlopt.agent.rl import PPO, PPORLOptConfig
from rlopt.env_utils import make_parallel_env


def test_ppo_train_halfcheetah_v5_smoke():
    """Smoke test: construct PPO on HalfCheetah-v5 and run a tiny training loop.

    Uses small collector and batch sizes to keep the test fast. Skips if the
    environment cannot be created on the host (e.g., missing mujoco/gym).
    """
    cfg = PPORLOptConfig()
    cfg.env.env_name = "HalfCheetah-v5"
    cfg.env.device = "cpu"
    cfg.device = "cpu"

    # Tiny training setup
    cfg.collector.frames_per_batch = 8
    cfg.collector.total_frames = 8
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.loss.epochs = 1
    cfg.compile.compile = False

    try:
        env = make_parallel_env(cfg)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Cannot create environment: {exc}")

    agent = PPO(env, cfg, logger=None)
    agent.train()

    assert agent.__class__.__name__ == "PPO"
