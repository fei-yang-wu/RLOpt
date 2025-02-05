from __future__ import annotations

import torch.nn
import torch.optim
import os
import uuid
from torch import multiprocessing

try:
    multiprocessing.set_start_method("fork")
    mp_context = "fork"
except RuntimeError:
    # If we can't set the method globally we can still run the parallel env with "fork"
    # This will fail on windows! Use "spawn" and put the script within `if __name__ == "__main__"`
    mp_context = "fork"
    pass

import torch
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import DuelingCnnDQNet, EGreedyModule, QValueActor

from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers.csv import CSVLogger
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import EnvBase
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder

from omegaconf import DictConfig


def make_mujoco_env(
    env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False
) -> EnvBase:
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


def make_gym_env(
    env_name="HalfCheetah-v4",
    parallel=False,
    obs_norm_sd=None,
    num_workers=1,
    device="cpu",
    from_pixels: bool = False,
    pixels_only=True,
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:

        def maker():
            return GymEnv(
                env_name=env_name,
                from_pixels=from_pixels,
                pixels_only=pixels_only,
                device=device,
            )

        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            # Don't create a sub-process if we have only one worker
            serial_for_single=True,
            mp_start_method=mp_context,
        )
    else:
        base_env = GymEnv(
            env_name=env_name,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
            device=device,
        )

    env = TransformedEnv(
        base_env,
        Compose(
            VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2),
            ClipTransform(in_keys=["observation"], low=-10, high=10),
            RewardSum(),
            StepCounter(),  # to count the steps of each trajectory
            DoubleToFloat(in_keys=["observation"]),
        ),
    )

    return env
