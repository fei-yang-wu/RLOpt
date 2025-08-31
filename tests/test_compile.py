from __future__ import annotations

import sys

import pytest
import torch

from rlopt.agent.ppo import PPO


@pytest.mark.slow
@pytest.mark.compile
def test_ppo_compile_smoke(ppo_cfg_factory, make_env, compile_mode, compile_warmup, compile_cudagraphs):  # type: ignore
    # torch.compile is unstable/unsupported on macOS in many environments
    if sys.platform == "darwin":  # pragma: no cover - platform dependent
        pytest.skip("Skipping torch.compile tests on macOS")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    cfg = ppo_cfg_factory(env_name="Pendulum-v1", num_envs=4, frames_per_batch=32, total_frames=32)  # type: ignore
    cfg.compile.compile = True
    cfg.compile.compile_mode = compile_mode
    cfg.compile.warmup = int(compile_warmup)
    cfg.compile.cudagraphs = bool(compile_cudagraphs)

    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore
    agent = PPO(env=env, config=cfg)

    # Just ensure it runs a short train without error
    agent.train()
