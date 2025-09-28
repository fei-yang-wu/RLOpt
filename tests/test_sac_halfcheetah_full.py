from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import pytest
from torchrl.envs.libs.gym import GymWrapper
from torchrl.record.loggers import generate_exp_name, get_logger

from rlopt.agent.sac.sac import SAC, SACRLOptConfig


@pytest.mark.full_halfcheetah
@pytest.mark.mujoco("HalfCheetah-v5")
@pytest.mark.filterwarnings("ignore::Warning")
def test_sac_halfcheetah_v5_full_wandb(sac_cfg_factory, make_env_parallel) -> None:  # type: ignore
    """Optional long-run SAC training on HalfCheetah-v5 with wandb logging.

    Skipped by default; enable with --run-full-halfcheetah.
    """
    # Ensure env is available
    try:
        _ = GymWrapper(gym.make("HalfCheetah-v5"), device="cpu")
    except Exception as e:  # pragma: no cover - env availability varies
        pytest.skip(f"HalfCheetah-v5 unavailable: {e}")

    # Configure a longer training run with SOTA SAC hyperparameters
    cfg: SACRLOptConfig = sac_cfg_factory(  # type: ignore
        env_name="HalfCheetah-v5",
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
    
    # Disable feature extractor for direct policy/Q-function input
    cfg.use_feature_extractor = True
    
    # Update SAC-specific parameters to match SOTA implementations
    cfg.sac.alpha_init = 1.0  # Typical value for continuous control
    cfg.sac.fixed_alpha = False  # Enable automatic entropy tuning
    cfg.sac.target_entropy = "auto"  # Automatically set to -action_dim
    cfg.sac.min_alpha = None  # Minimum alpha value
    cfg.sac.max_alpha = None  # Maximum alpha value

    cfg.feature_extractor.num_cells = [64, 64]
    
    # Network architecture improvements
    cfg.policy.num_cells = [256, 256]  # Larger networks for better performance
    cfg.action_value_net.num_cells = [256, 256]
    
    # Optimizer improvements
    cfg.optim.target_update_polyak = 0.995
    cfg.optim.weight_decay = 0.0  # Typically disabled for RL
    
    # Replay buffer improvements
    cfg.replay_buffer.size = 1_000_000  # Standard size for continuous control

    # Build a wandb logger with the requested entity
    run_dir = (
        Path("logs")
        / "sac"
        / "halfcheetah-v5"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    exp_name = generate_exp_name("SAC", "HalfCheetah-v5")
    wandb_logger = get_logger(
        "wandb",
        logger_name=str(run_dir),
        experiment_name=exp_name,
        wandb_kwargs={
            "project": cfg.logger.project_name,
            "group": "HalfCheetah-v5",
            "entity": "fywu",
            "config": asdict(cfg),
        },
    )

    # Parallel env and agent
    env = make_env_parallel(cfg.env.env_name, num_workers=cfg.env.num_envs, device="cpu")  # type: ignore
    agent = SAC(env=env, config=cfg, logger=wandb_logger)

    # Full training run (no strict assertions; ensures it executes without error)
    agent.train()


