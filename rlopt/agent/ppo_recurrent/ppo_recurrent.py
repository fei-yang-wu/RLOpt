"""
Only supports MuJoCo environments for now
"""

from os import sync
from tty import CFLAG
from typing import List, Tuple, Dict, Any, Optional

import tqdm
import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from torch.optim.optimizer import Optimizer as Optimizer

from tensordict.nn import (
    AddStateIndependentNormalScale,
    TensorDictModule,
    set_composite_lp_aggregate,
)

from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.data import ReplayBuffer
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import EnvBase, TransformedEnv, ExplorationType
from torchrl.record import CSVLogger, TensorboardLogger, WandbLogger
from torchrl.record.loggers.common import Logger
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl._utils import timeit, compile_with_warmup

from omegaconf import DictConfig
from rlopt.common.base_class import BaseAlgorithm
from rlopt.agent import PPO

set_composite_lp_aggregate(True).set()


class PPORECURRENT(PPO):

    def __init__(
        self,
        env: EnvBase,
        config: DictConfig,
        policy: Optional[nn.Module] = None,
        value_net: Optional[nn.Module] = None,
        q_net: Optional[nn.Module] = None,
        reward_estimator: Optional[nn.Module] = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: type[Logger] = TensorboardLogger,
        **kwargs,
    ):
        super().__init__(
            env,
            config,
            policy,
            value_net,
            q_net,
            reward_estimator,
            replay_buffer,
            logger,
            **kwargs,
        )

    def _construct_policy(self) -> nn.Module:
        raise NotImplementedError

    def predict(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict action and value given observation"""
        raise NotImplementedError
