# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools

import torch
from torch import nn
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)

from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter

from torchrl.record import VideoRecorder

import torch


# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")  # type: ignore
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def apply_env_transforms(env, max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, logger=None):
    """Make environments for training and evaluation."""
    partial = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(partial),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    partial = functools.partial(env_maker, cfg=cfg, from_pixels=cfg.logger.video)
    trsf_clone = train_env.transform.clone()
    if cfg.logger.video:
        trsf_clone.insert(
            0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])  # type: ignore
        )
    eval_env = TransformedEnv(
        ParallelEnv(
            1,
            EnvCreator(partial),
            serial_for_single=True,
        ),
        trsf_clone,
    )
    return train_env, eval_env


# ====================================================================
# General utils
# ---------


### Saving/loading models


def save_model(model, path):
    torch.save(
        {
            "actor": model[0].state_dict(),
            "critic": model[1].state_dict(),
            "reward_estimate": model[2].state_dict(),
        },
        path,
    )


def load_model(model, path, device):
    states = torch.load(path, map_location=device)
    model[0].load_state_dict(states["actor"])
    model[1].load_state_dict(states["critic"])
    model[2].load_state_dict(states["reward_estimate"])


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
