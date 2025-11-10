#!/usr/bin/env python3
"""Test script to verify compilation works on CPU (without CUDAGraphs)."""

from __future__ import annotations

import torch
from rlopt.agent.rl import SAC, SACRLOptConfig
from rlopt.configs import NetworkConfig
from rlopt.env_utils import make_parallel_env


def test_sac_cpu_with_cudagraphs_enabled():
    """Test that CUDAGraphs is safely skipped on CPU even when enabled in config."""
    print("=" * 70)
    print("Testing SAC on CPU with CUDAGraphs Enabled in Config")
    print("=" * 70)

    # Create config with CUDAGraphs enabled (should be safely ignored on CPU)
    cfg = SACRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.num_envs = 2
    cfg.device = "cpu"  # Force CPU
    cfg.collector.total_frames = 1000
    cfg.collector.frames_per_batch = 200
    cfg.collector.init_random_frames = 200
    cfg.compile.compile = True
    cfg.compile.compile_mode = "reduce-overhead"
    cfg.compile.cudagraphs = True  # This should be safely ignored on CPU
    cfg.logger.backend = None  # Disable logging for this test

    # Network configurations
    cfg.q_function = NetworkConfig(
        num_cells=[64, 64],
        activation_fn="relu",
        output_dim=1,
        input_keys=["observation"],
    )
    cfg.policy = NetworkConfig(
        num_cells=[64, 64],
        activation_fn="relu",
        input_keys=["observation"],
    )

    print(f"\nConfiguration:")
    print(f"  Device: {cfg.device}")
    print(f"  Compile: {cfg.compile.compile}")
    print(f"  Compile mode: {cfg.compile.compile_mode}")
    print(f"  CUDAGraphs in config: {cfg.compile.cudagraphs}")
    print(f"  Expected behavior: CUDAGraphs should be skipped")

    # Create environment
    print("\nCreating environment...")
    env = make_parallel_env(cfg)

    # Create and train agent
    print("Creating agent...")
    agent = SAC(env=env, config=cfg)

    print("\n✅ Agent created successfully")
    print("   Starting training...")

    agent.train()

    print("\n✅ Training completed successfully on CPU!")
    print("   CUDAGraphs was safely skipped (as expected on CPU)")
    print(f"   Total episodes: {len(agent.episode_rewards)}")
    if agent.episode_rewards:
        print(f"   Final reward: {agent.episode_rewards[-1]:.2f}")


if __name__ == "__main__":
    print("\n🚀 Testing CPU Safety with CUDAGraphs Config")
    print("=" * 70)
    print("This test verifies that CUDAGraphs is safely skipped on CPU devices,")
    print("even when enabled in configuration.")
    print("=" * 70)

    test_sac_cpu_with_cudagraphs_enabled()

    print("\n" + "=" * 70)
    print("✅ CPU safety test passed!")
    print("=" * 70)
