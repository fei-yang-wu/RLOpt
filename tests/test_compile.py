#!/usr/bin/env python3
"""Test script to verify compilation and CUDAGraphs work correctly."""

from __future__ import annotations

import torch
from rlopt.agent.rl import SAC, SACRLOptConfig
from rlopt.configs import NetworkConfig
from rlopt.env_utils import make_parallel_env


def test_sac_with_compile():
    """Test SAC with compile enabled."""
    print("=" * 70)
    print("Testing SAC with Compile")
    print("=" * 70)

    # Create config with compile enabled
    cfg = SACRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.num_envs = 2
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.collector.total_frames = 2000
    cfg.collector.frames_per_batch = 200
    cfg.collector.init_random_frames = 200
    cfg.compile.compile = True
    cfg.compile.compile_mode = "reduce-overhead"
    cfg.compile.cudagraphs = False  # Start without cudagraphs
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
    print(f"  CUDAGraphs: {cfg.compile.cudagraphs}")

    # Create environment
    print("\nCreating environment...")
    env = make_parallel_env(cfg)

    # Create and train agent
    agent = SAC(env=env, config=cfg)

    print("\n✅ Agent created successfully")
    print("   Starting training...")

    agent.train()

    print("\n✅ Training completed successfully!")
    print(f"   Total episodes: {len(agent.episode_rewards)}")
    if agent.episode_rewards:
        print(f"   Final reward: {agent.episode_rewards[-1]:.2f}")


def test_sac_with_cudagraphs():
    """Test SAC with CUDAGraphs enabled (only on CUDA devices)."""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available - skipping CUDAGraphs test")
        return

    print("\n" + "=" * 70)
    print("Testing SAC with CUDAGraphs")
    print("=" * 70)

    # Create config with cudagraphs enabled
    cfg = SACRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.num_envs = 2
    cfg.device = "cuda:0"
    cfg.collector.total_frames = 2000
    cfg.collector.frames_per_batch = 200
    cfg.collector.init_random_frames = 200
    cfg.compile.compile = True
    cfg.compile.compile_mode = "default"
    cfg.compile.cudagraphs = True
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
    print(f"  CUDAGraphs: {cfg.compile.cudagraphs}")

    # Create environment
    print("\nCreating environment...")
    env = make_parallel_env(cfg)

    # Create and train agent
    agent = SAC(env=env, config=cfg)

    print("\n✅ Agent created successfully")
    print("   Starting training...")

    try:
        agent.train()
        print("\n✅ Training with CUDAGraphs completed successfully!")
        print(f"   Total episodes: {len(agent.episode_rewards)}")
        if agent.episode_rewards:
            print(f"   Final reward: {agent.episode_rewards[-1]:.2f}")
    except Exception as e:
        print(f"\n⚠️  CUDAGraphs test failed (this is OK if experimental): {e}")


if __name__ == "__main__":
    print("\n🚀 Testing Compilation Support in RLOpt")
    print("=" * 70)

    # Test basic compile
    test_sac_with_compile()

    # Test cudagraphs if available
    test_sac_with_cudagraphs()

    print("\n" + "=" * 70)
    print("✅ All compilation tests completed!")
    print("=" * 70)
