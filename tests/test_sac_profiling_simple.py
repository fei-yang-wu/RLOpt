#!/usr/bin/env python3
"""
Simple SAC profiling test for RLOpt implementation.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    InitTracker,
    RewardSum,
    StepCounter,
)

from rlopt.agent.sac.sac import SAC, SACRLOptConfig
from rlopt.configs import (
    EnvConfig,
    RLOptConfig,
    NetworkLayout,
    MLPBlockConfig,
    ModuleNetConfig,
    CriticConfig,
)
from rlopt.env_utils import make_parallel_env


def test_sac_profiling():
    """Simple SAC profiling test."""
    print("Starting RLOpt SAC profiling test...")

    # Configure a shorter training run for profiling
    cfg = SACRLOptConfig()

    # Environment configuration
    cfg.env.env_name = "HalfCheetah-v4"
    cfg.env.device = "cuda:0"
    cfg.env.num_envs = 8

    # Collector configuration
    cfg.collector.num_collectors = 8
    cfg.collector.frames_per_batch = 1000
    cfg.collector.total_frames = 10000  # Reduced for quick profiling
    cfg.collector.set_truncated = False
    cfg.collector.init_random_frames = 1000  # Reduced for quick start

    # Optimization configuration
    cfg.optim.lr = 3e-4
    cfg.optim.target_update_polyak = 0.995
    cfg.optim.weight_decay = 0.0

    # Loss configuration
    cfg.loss.mini_batch_size = 256

    # Create NetworkLayout to match TorchRL architecture exactly
    # No shared features, direct input to networks, ReLU activation
    from rlopt.configs import SharedFeatures

    # Empty shared features (no feature extractor)
    shared_features = SharedFeatures(features={})

    network_layout = NetworkLayout(
        # No shared features - direct input like TorchRL
        shared=shared_features,
        # Policy module - direct input, ReLU activation
        policy=ModuleNetConfig(
            feature_ref=None,  # No feature extractor
            head=MLPBlockConfig(
                num_cells=[256, 256],  # Match TorchRL hidden sizes
                activation="relu",  # Match TorchRL activation
                init="orthogonal",
                layer_norm=False,
                dropout=0.0,
            ),
            in_keys=["observation"],  # Direct input
            out_key="hidden",
        ),
        # Critic module - direct input, ReLU activation, twin Q-networks
        critic=CriticConfig(
            template=ModuleNetConfig(
                feature_ref=None,  # No feature extractor
                head=MLPBlockConfig(
                    num_cells=[256, 256],  # Match TorchRL hidden sizes
                    activation="relu",  # Match TorchRL activation
                    init="orthogonal",
                    layer_norm=False,
                    dropout=0.0,
                ),
                in_keys=["observation", "action"],  # Direct input
                out_key="hidden",
            ),
            num_nets=2,  # Twin Q-networks like TorchRL
            shared_feature_ref=None,  # No shared features
            use_target=True,
            target_update="polyak",
            polyak_eps=0.995,
        ),
    )

    # Apply the network layout
    cfg.network = network_layout

    # Feature extractor configuration - DISABLE to match TorchRL
    cfg.use_feature_extractor = False
    cfg.feature_extractor.output_dim = 256
    cfg.feature_extractor.num_cells = [64, 64]

    # Network architecture - match TorchRL exactly
    cfg.policy.num_cells = [256, 256]
    cfg.action_value_net.num_cells = [256, 256]

    # Input/output keys - direct input like TorchRL
    cfg.policy_in_keys = ["observation"]
    cfg.value_net_in_keys = ["observation"]
    cfg.total_input_keys = ["observation"]

    # Logger configuration (disable for profiling)
    cfg.logger.backend = ""

    # Device configuration
    cfg.device = "cuda:0"

    # Compile configuration
    cfg.compile.compile = False

    # SAC-specific configuration - match TorchRL exactly
    cfg.sac.utd_ratio = 1.0
    cfg.sac.alpha_init = 1.0
    cfg.sac.fixed_alpha = False
    cfg.sac.target_entropy = "auto"
    cfg.sac.min_alpha = None
    cfg.sac.max_alpha = None

    # Replay buffer settings
    cfg.replay_buffer.size = 100000  # Reduced for profiling

    # Save interval (disable for profiling)
    cfg.save_interval = 0

    # Create environment
    env = make_parallel_env(cfg)
    env = TransformedEnv(
        env,
        Compose(
            [
                InitTracker(),
                StepCounter(),
                DoubleToFloat(),
                RewardSum(),
            ]
        ),
    )

    # Create agent without logger for profiling
    agent = SAC(env=env, config=cfg, logger=None)

    # Run training with profiling
    print("Starting training with profiling...")
    agent.train()
    print("RLOpt SAC profiling test completed!")


if __name__ == "__main__":
    test_sac_profiling()
