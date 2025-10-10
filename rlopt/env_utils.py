from __future__ import annotations

import functools

from torchrl.envs import (
    EnvCreator,
    ParallelEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend

from rlopt.configs import RLOptConfig


def env_maker(cfg: RLOptConfig, device="cpu", from_pixels=False):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.env_name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


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
