import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Tuple, List
from collections import deque
import time
import statistics
import pathlib
import io


import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from copy import deepcopy

from stable_baselines3.common.buffers import RolloutBuffer, BaseBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)


from sb3_contrib.common.recurrent.type_aliases import RNNStates  # type: ignore
from stable_baselines3.common.save_util import (
    load_from_zip_file,
    recursive_getattr,
    recursive_setattr,
    save_to_zip_file,
)
from stable_baselines3.common.utils import get_system_info
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    TensorDict,
)
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate
from stable_baselines3.common import utils
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import (
    get_device,
)
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecNormalize,
    unwrap_vec_normalize,
)

from rlopt.common.torch.buffer import RLOptDictRecurrentReplayBuffer

from rlopt.utils.torch.utils import (
    obs_as_tensor,
    explained_variance,
    unpad_trajectories,
)
from rlopt.agent.torch.l2t.policies import (
    MlpLstmPolicy,
    CnnLstmPolicy,
    MultiInputLstmPolicy,
    RecurrentActorCriticPolicy,
)
from rlopt.agent.torch.l2t.recurrent_l2t import RecurrentL2T


SelfTeacherStudentLearning = TypeVar(
    "SelfTeacherStudentLearning", bound="TeacherStudentLearning"
)


class Teacher(RecurrentL2T):
    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train_teacher(self):
        # Implement training logic for the teacher policy
        self.policy.train()
        # Add training steps specific to the teacher
