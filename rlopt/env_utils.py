from __future__ import annotations

import functools

import torch
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    RewardSum,
    StepCounter,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend

from rlopt.configs import RLOptConfig


def env_maker(cfg: RLOptConfig, device="cpu", from_pixels=False):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            env = GymEnv(
                cfg.env.env_name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )
            # Add dtype conversion transform (float64 -> float32) only if needed
            # Check if observations are float64
            obs_spec = env.observation_spec["observation"]
            if hasattr(obs_spec, "dtype") and obs_spec.dtype == torch.float64:
                env = env.append_transform(DoubleToFloat(in_keys=["observation"]))

            # Add transforms to track episode rewards and lengths
            # RewardSum accumulates rewards and adds "episode_reward" when done
            # StepCounter tracks steps and adds "step_count"
            env = env.append_transform(StepCounter(max_steps=1000))
            env = env.append_transform(RewardSum())

            return env
    else:
        msg = f"Unknown lib {lib}."
        raise NotImplementedError(msg)


def make_parallel_env(cfg: RLOptConfig):
    """Make environments for training and evaluation."""
    partial = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.env.num_envs,
        EnvCreator(partial),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.seed)

    return parallel_env
