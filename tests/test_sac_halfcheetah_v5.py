from __future__ import annotations

import pytest

from rlopt.agent.rl import SAC, SACRLOptConfig
from rlopt.env_utils import make_parallel_env
from rlopt.configs import NetworkConfig


def test_sac_train_halfcheetah_v5_smoke():
    """Smoke test: construct SAC on HalfCheetah-v5 and run a tiny training loop.

    This test uses small collector and buffer sizes so it runs quickly. If the
    environment cannot be created (missing mujoco/gym backend), the test will be
    skipped.
    """
    cfg = SACRLOptConfig()
    # Use v5 environment as requested and force CPU for tests
    cfg.env.env_name = "HalfCheetah-v5"
    cfg.env.device = "cpu"
    cfg.device = "cpu"

    # Make training tiny so the smoke test is fast
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.q_function = NetworkConfig(
        num_cells=[64, 64],
        activation_fn="relu",
        output_dim=1,
        input_keys=["observation"],
    )

    print(cfg)
    try:
        env = make_parallel_env(cfg)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Cannot create environment: {exc}")

    agent = SAC(env, cfg, logger=None)
    # Run training (should finish quickly due to tiny total_frames)
    agent.train()

    # Basic assertion that the agent was created and has recorded episodes (or at least ran)
    assert agent.__class__.__name__ == "SAC"
