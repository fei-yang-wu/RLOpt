from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import pytest
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    InitTracker,
    RewardSum,
    StepCounter,
)
from torchrl.record.loggers import generate_exp_name, get_logger

from rlopt.agent.sac.sac import SAC, SACRLOptConfig
from rlopt.configs import (
    CriticConfig,
    EnvConfig,
    MLPBlockConfig,
    ModuleNetConfig,
    NetworkLayout,
    RLOptConfig,
    SharedFeatures,
)
from rlopt.env_utils import make_parallel_env

import torch
import numpy as np


@pytest.mark.full_halfcheetah
@pytest.mark.mujoco("HalfCheetah-v4")
@pytest.mark.filterwarnings("ignore::Warning")
def test_sac_halfcheetah_v5_full_wandb(sac_cfg_factory) -> None:  # type: ignore
    """Optional long-run SAC training on HalfCheetah-v5 with wandb logging.

    Skipped by default; enable with --run-full-halfcheetah.
    """
    # Configure a longer training run with SOTA SAC hyperparameters
    cfg: SACRLOptConfig = sac_cfg_factory(  # type: ignore
        env_name="HalfCheetah-v4",
        num_envs=8,
        frames_per_batch=1000,
        total_frames=1_000_000,
        feature_dim=256,  # Increased for better representation
        lr=3e-4,
        mini_batch_size=256,
        utd_ratio=1.0,
        init_random_frames=25000,  # Increased for better exploration
        save_interval=100000,
    )

    cfg.compile.compile = False
    cfg.compile.compile_mode = "default"
    cfg.compile.cudagraphs = False
    cfg.compile.warmup = 1

    network_layout = NetworkLayout(
        # No shared features - direct input like TorchRL
        shared=SharedFeatures(features={}),
        # Policy module - direct input, ReLU activation
        policy=ModuleNetConfig(
            feature_ref=None,  # No feature extractor
            head=MLPBlockConfig(
                num_cells=[256, 256],  # Match TorchRL hidden sizes
                activation="relu",  # Match TorchRL activation
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
                    activation="relu",  # Match TorchRL activatio
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

    cfg.network = network_layout

    cfg.use_feature_extractor = False

    # SAC-specific configuration - match TorchRL exactly
    cfg.sac.utd_ratio = 1.0
    cfg.sac.alpha_init = 1.0
    cfg.sac.fixed_alpha = False
    cfg.sac.target_entropy = "auto"
    cfg.sac.min_alpha = None
    cfg.sac.max_alpha = None

    cfg.policy_in_keys = ["observation"]
    cfg.value_net_in_keys = ["observation"]
    cfg.total_input_keys = ["observation"]

    # Build a wandb logger with the requested entity
    run_dir = (
        Path("logs")
        / "sac"
        / "halfcheetah-v4"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    exp_name = generate_exp_name("SAC", "HalfCheetah-v4")
    wandb_logger = get_logger(
        "wandb",
        logger_name=str(run_dir),
        experiment_name=exp_name,
        wandb_kwargs={
            "project": cfg.logger.project_name,
            "group": "HalfCheetah-v4",
            "entity": "fywu",
            "config": asdict(cfg),
        },
    )

    # Replay buffer settings
    cfg.replay_buffer.size = 1000000  # Reduced for profiling

    # Save interval (disable for profiling)
    cfg.save_interval = 0

    cfg.seed = 42

    # Parallel env and agent
    env = make_parallel_env(cfg)
    env = TransformedEnv(
        env,
        Compose(
            [
                InitTracker(),
                StepCounter(1000),
                DoubleToFloat(),
                RewardSum(),
            ]
        ),
    )

    agent = SAC(env=env, config=cfg, logger=wandb_logger)

    # Full training run (no strict assertions; ensures it executes without error)
    agent.train()
