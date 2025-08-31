from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from torch import Tensor
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    ReplayBuffer,
)
from torchrl.envs import TransformedEnv
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.record.loggers.common import Logger

from rlopt.configs import RLOptConfig


class BaseAlgorithm(ABC):
    """
    Base class for all RL algorithms.
    args:
        env: Environment instance
        collector: Data collector instance
        config: Algorithm configuration
        policy: Policy network
        value_net: Value network
        q_net: Q network
        reward_estimator: Reward estimator network
        replay_buffer: Replay buffer class
        logger: Logger class
        **

    """

    def __init__(
        self,
        env: TransformedEnv,
        config: RLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ):
        super().__init__()
        self.env = env
        self.config = config
        self.logger = logger
        self.kwargs = kwargs

        # Seed for reproducibility
        torch.manual_seed(config.seed)
        self.np_rng = np.random.default_rng(config.seed)
        self.mp_context = "fork"

        # Construct or attach networks based on existence in config
        self.feature_extractor = self._construct_feature_extractor(
            feature_extractor_net
        )
        self.policy = self._construct_policy(policy_net)

        # determine using value function or q function
        if self.config.use_value_function:
            self.value_function = (
                value_net if value_net is not None else self._construct_value_function()
            )
        else:
            self.q_function = (
                q_net if q_net is not None else self._construct_q_function()
            )

        self.actor_critic = self._construct_actor_critic()

        # Move them to device
        if self.policy:
            self.policy.to(self.device)
        if self.value_function:
            self.value_function.to(self.device)
        if self.q_function:
            self.q_function.to(self.device)

        # Create optimizers
        self.optimizers = self._configure_optimizers()

        # Replay buffer / experience
        self.replay_buffer = replay_buffer
        self.step_count = 0
        self.start_time = time.time()

        # build collector
        self.collector = self._construct_collector(self.env)

        # build loss module
        self.loss_module = self._construct_loss_module()

        # optimizers
        self.optim = self._configure_optimizers()

        # buffer
        self.data_buffer = self._construct_data_buffer()

        # logger_video attribute must be defined before use
        self.logger_video = False

        # logger
        self._configure_logger()

        # episode length
        self.episode_lengths = deque(maxlen=100)

        # episode rewards
        self.episode_rewards = deque(maxlen=100)

    @property
    def value_input_shape(self) -> int:
        if self.config.use_feature_extractor:
            return self.config.feature_extractor.output_dim
        return int(
            torch.tensor(
                [
                    self.env.observation_spec[key].shape[-1]
                    for key in self.config.value_net_in_keys
                ]
            )
            .sum()
            .item()  # type: ignore
        )

    @property
    def policy_input_shape(self) -> int:
        if self.config.use_feature_extractor:
            return self.config.feature_extractor.output_dim
        return int(
            torch.tensor(
                [
                    self.env.observation_spec[key].shape[-1]
                    for key in self.config.policy_in_keys
                    + self.config.value_net_in_keys
                ]
            )
            .sum()
            .item()  # type: ignore
        )

    @property
    def total_input_keys(self) -> list[str]:
        return self.config.total_input_keys

    @property
    def total_input_shape(self) -> int:
        return int(
            torch.tensor(
                [
                    self.env.observation_spec[key].shape[-1]
                    for key in self.total_input_keys
                ]
            )
            .sum()
            .item()  # type: ignore
        )

    @property
    def policy_output_shape(self) -> int:
        return int(self.env.action_spec_unbatched.shape[-1])  # type: ignore

    @property
    def value_output_shape(self) -> int:
        return 1

    @property
    def device(self) -> torch.device:
        """Return the device used for training."""
        return self._get_device(self.config.device)

    def _get_device(self, device_str: str) -> torch.device:
        """Decide on CPU or GPU device."""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _construct_collector(
        self,
        env: TransformedEnv,
    ) -> SyncDataCollector:
        # We can't use nested child processes with mp_start_method="fork"

        return SyncDataCollector(
            create_env_fn=env,
            policy=self.actor_critic.get_policy_operator(),
            frames_per_batch=self.config.collector.frames_per_batch,
            total_frames=self.config.collector.total_frames,
            # this is the default behavior: the collector runs in ``"random"`` (or explorative) mode
            # exploration_type=ExplorationType.RANDOM,
            # We set the all the devices to be identical. Below is an example of
            compile_policy=(
                {"mode": self.config.compile.compile_mode, "warmup": 1}
                if self.config.compile.compile
                else False
            ),
            # heterogeneous devices
            device=self.device,
            storing_device=self.device,
            reset_at_each_iter=False,
            set_truncated=self.config.collector.set_truncated,
        )

    @abstractmethod
    def _construct_policy(
        self, policy_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Override to build your policy network from config."""

    def _construct_value_function(
        self, value_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Override to build your V-network from config."""
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def _construct_q_function(
        self, q_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Override to build your Q-network from config."""
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @abstractmethod
    def _construct_feature_extractor(
        self, feature_extractor_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Override to build your feature extractor network from config."""

    @abstractmethod
    def _construct_actor_critic(self) -> TensorDictModule:
        """Override to build your actor-critic network from config."""

    @abstractmethod
    def _construct_loss_module(self) -> torch.nn.Module:
        """Override to build your loss module from config."""

    @abstractmethod
    def _construct_data_buffer(self) -> ReplayBuffer:
        """Override to build your data buffer from config."""

    def _compile_components(self) -> None:  # noqa: B027
        """compile performance-critical methods.
        Examples:
            >>> if self.config.compile:
            ...     self.compute_action = torch.compile(self._compute_action, dynamic=True)
            ...     self.compute_returns = torch.compile(self._compute_returns, dynamic=True)
            >>> else:
            ...     self.compute_action = self._compute_action
            ...     self.compute_returns = self._compute_returns
        """

    @abstractmethod
    def _configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for different components."""
        # Basic example: one optimizer for each component

    def _configure_logger(self) -> None:
        # Create logger
        self.logger = None
        cfg = self.config
        if cfg.logger.backend:
            exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}")
            self.logger = get_logger(
                cfg.logger.backend,
                logger_name="ppo",
                experiment_name=exp_name,
                log_dir=cfg.logger.log_dir,
                wandb_kwargs={
                    "config": dict(cfg),
                    "project": cfg.logger.project_name,
                    "group": cfg.logger.group_name,
                },
            )
            self.logger_video = cfg.logger.video

        else:
            self.logger_video = False

    @abstractmethod
    def train(self) -> None:
        """Main training loop."""

    @abstractmethod
    def predict(self, obs: Tensor) -> Tensor:
        """Predict action given observation."""

    def save_model(
        self, path: str | Path | None = None, step: int | None = None
    ) -> None:
        """Save the model and related parameters to a file."""
        assert self.logger is not None
        prefix = f"{self.config.logger.log_dir}" if path is None else path
        # Include step in filename if provided
        if step is not None:
            path = f"{prefix}/model_step_{step}.pt"
        else:
            path = f"{prefix}/model.pt"
        data_to_save = {
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value_function.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
        }
        if self.config.use_feature_extractor:
            data_to_save["feature_extractor_state_dict"] = (
                self.feature_extractor.state_dict()
            )
        # if we are using VecNorm, we need to save the running mean and std
        if (
            hasattr(self.env, "is_closed")
            and not self.env.is_closed
            and hasattr(self.env, "normalize_obs")
        ):
            data_to_save["vec_norm_msg"] = self.env.state_dict()

        torch.save(data_to_save, path)

    def load_model(self, path: str) -> None:
        """Load the model and related parameters from a file."""
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy_state_dict"])
        self.value_function.load_state_dict(data["value_state_dict"])
        self.optim.load_state_dict(data["optimizer_state_dict"])
        if self.config.use_feature_extractor and "feature_extractor_state_dict" in data:
            self.feature_extractor.load_state_dict(data["feature_extractor_state_dict"])
        if hasattr(self.env, "normalize_obs") and "vec_norm_msg" in data:
            self.env.load_state_dict(data["vec_norm_msg"])
