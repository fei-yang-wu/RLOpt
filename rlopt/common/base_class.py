from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Sequence

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,
)
from torch import Tensor
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data import (
    ReplayBuffer,
)
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.record.loggers.common import Logger
from torchrl.trainers import Trainer
from torchrl.envs.env_creator import EnvCreator

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
        config: DictConfig,
        policy: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        reward_estimator: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        **kwargs,
    ):
        super().__init__()
        self.env = env
        self.config = config
        self.logger = logger

        # Seed for reproducibility
        torch.manual_seed(config.seed)
        self.np_rng = np.random.default_rng(config.seed)
        self.mp_context = "fork"

        # Construct or attach networks based on existence in config
        self.policy = policy if policy else self._construct_policy()
        self.value_net = value_net if value_net else self._construct_value_function()
        self.q_net = q_net if q_net else self._construct_q_function()
        self.reward_estimator = reward_estimator

        # Move them to device
        if self.policy:
            self.policy.to(self.device)
        if self.value_net:
            self.value_net.to(self.device)
        if self.q_net:
            self.q_net.to(self.device)
        if self.reward_estimator:
            self.reward_estimator.to(self.device)

        # Construct (optional) target networks
        self.target_value_net: torch.nn.Module | None = None
        self.target_q_net: torch.nn.Module | None = None
        construct_target_value = self.config.get("construct_target_value", None)
        construct_target_q = self.config.get("construct_target_q", None)
        # If you want a separate target for the value function:
        if self.value_net is not None and construct_target_value:
            self.target_value_net = self._construct_target_network(self.value_net)
        # If you want a separate target for the Q function:
        if self.q_net is not None and construct_target_q:
            self.target_q_net = self._construct_target_network(self.q_net)

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

        # trainer
        self.trainer = self._construct_trainer()

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
        create_env_fn: Callable,
    ) -> SyncDataCollector | MultiSyncDataCollector:
        # try:
        #     # Set multiprocessing start method
        #     multiprocessing.set_start_method("spawn")
        #     self.mp_context = "spawn"
        # except RuntimeError:
        #     # If we can't set the method globally we can still run the parallel env with "fork"
        #     # This will fail on windows! Use "spawn" and put the script within `if __name__ == "__main__"`
        #     self.mp_context = "fork"
        #     pass
        # multiprocessing.set_start_method("fork")

        # We can't use nested child processes with mp_start_method="fork"
        if self.mp_context == "fork":
            cls = SyncDataCollector
            env_arg = EnvCreator(create_env_fn)
        else:
            cls = MultiSyncDataCollector
            env_arg = [EnvCreator(create_env_fn) for _ in range(self.config.collector.num_collectors)]
        
        return cls(
            create_env_fn=env_arg, # type: ignore
            policy=self.policy,
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
        )

    @abstractmethod
    def _construct_policy(self) -> torch.nn.Module:
        """Override to build your policy network from config."""
        # use torchrl modules for common policy components
        # e.g. CategoricalPolicy, GaussianPolicy, etc.
        out_features = self.env.action_spec_unbatched.shape[-1]  # type: ignore
        module = torch.nn.LazyLinear(out_features=out_features)
        return TensorDictModule(
            module,
            in_keys=["observation"],
            out_keys=["action"],
        )

    def _construct_value_function(self) -> torch.nn.Module | None:
        """Override to build your V-network from config."""
        if self.config.get("value_net", None) is None:
            return None
        value_net_config = self.config.value_net
        # Example: simple MLP for state-value
        in_features = value_net_config.get("in_features", 4)
        hidden_size = value_net_config.get("hidden_size", 64)
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),  # V(s)
        )

    def _construct_q_function(self) -> torch.nn.Module | None:
        """Override to build your Q-network from config."""
        if self.config.get("q_net", None) is None:
            return None

        q_net_config = self.config.q_net
        # Example: simple MLP for state-action value
        obs_dim = q_net_config.get("obs_dim", 4)
        act_dim = q_net_config.get("act_dim", 2)
        hidden_size = q_net_config.get("hidden_size", 64)
        return torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),  # Q(s,a)
        )

    def _construct_target_network(self, net: torch.nn.Module) -> torch.nn.Module:
        """Create a target network as a copy of the provided net."""
        target_net = type(net)()  # same class
        # This assumes your net is a torch.nn.Sequential or a custom class:
        # We copy the state dict so that it starts off identical.
        target_net.load_state_dict(net.state_dict())

        # Freeze parameters if you want them non-trainable:
        for param in target_net.parameters():
            param.requires_grad = False

        return target_net.to(self.device)

    def _construct_loss_module(self) -> torch.nn.Module:
        loss_config = self.config.loss
        """Override to build your loss module from config."""
        # Example: simple PPO loss
        return ClipPPOLoss(
            actor_network=None,
            critic_network=None,
            clip_epsilon=loss_config.clip_epsilon,
            loss_critic_type=loss_config.loss_critic_type,
            entropy_coef=loss_config.entropy_coef,
            critic_coef=loss_config.critic_coef,
            normalize_advantage=True,
        )

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
        optimizers: dict[str, torch.optim.Optimizer] = {}
        if self.policy:
            optimizers["policy"] = torch.optim.Adam(
                self.policy.parameters(), lr=self.config.learning_rate
            )
        if self.value_net:
            optimizers["value"] = torch.optim.Adam(
                self.value_net.parameters(), lr=self.config.learning_rate
            )
        if self.q_net:
            optimizers["q"] = torch.optim.Adam(
                self.q_net.parameters(), lr=self.config.learning_rate
            )
        if self.reward_estimator and self.config.reward_estimation:
            lr = self.config.reward_estimation.get("learning_rate", 3e-4)
            optimizers["reward"] = torch.optim.Adam(
                self.reward_estimator.parameters(),
                lr=lr,
            )
        return group_optimizers(*optimizers.values())

    def _configure_logger(self) -> None:
        # Create logger
        self.logger = None
        cfg = self.config
        if cfg.logger.backend:
            exp_name = generate_exp_name(
                "PPO", f"{cfg.logger.exp_name}_{cfg.env.env_name}"
            )
            self.logger = get_logger(
                cfg.logger.backend,
                logger_name="ppo",
                experiment_name=exp_name,
                wandb_kwargs={
                    "config": dict(cfg),
                    "project": cfg.logger.project_name,
                    "group": cfg.logger.group_name,
                },
            )
            self.logger_video = cfg.logger.video

        else:
            self.logger_video = False

    def soft_update(
        self,
        source_net: torch.nn.Module,
        target_net: torch.nn.Module,
        tau: float = 0.005,
    ):
        """Polyak/soft-update target network parameters."""
        with torch.no_grad():
            for param, target_param in zip(
                source_net.parameters(), target_net.parameters(), strict=False
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

    def hard_update(self, source_net: torch.nn.Module, target_net: torch.nn.Module):
        """Hard update for target network (full copy)."""
        target_net.load_state_dict(source_net.state_dict())

    @abstractmethod
    def _compute_action(self, tensordict: TensorDict) -> TensorDict:
        """Compute the next action given current policy (abstract)."""

    @abstractmethod
    def _compute_returns(self, rollout: Tensor) -> Tensor:
        """Compute returns and possibly advantages from a rollout."""

    @abstractmethod
    def _update_policy(self, batch: TensorDict) -> dict[str, float]:
        """Algorithm-specific policy (and/or value) update logic."""

    def collect_experience(self) -> TensorDict:
        """Generic experience collection (for on-policy).
        Override if you do something different (like off-policy sampling).
        """
        if self.config.offline:
            return self._load_offline_data()

        msg = "Override for online collection."
        raise NotImplementedError(msg)

    def _load_offline_data(self) -> TensorDict:
        """Load from replay buffer or offline dataset."""
        assert self.replay_buffer is not None, (
            "ReplayBuffer must be provided for offline."
        )
        return self.replay_buffer.sample(self.config.batch_size)        

    def update_parameters(self, batch: TensorDict) -> dict[str, float]:
        """Generic parameter update (handles policy, value, reward, etc.)."""
        metrics = {}
        policy_metrics = self._update_policy(batch)
        metrics.update(policy_metrics)

        return metrics

    def _construct_trainer(self) -> Trainer:
        # Use the policy optimizer if available, otherwise pick any optimizer from the dict
        main_optimizer = None
        if isinstance(self.optim, dict):
            main_optimizer = self.optim.get("policy", next(iter(self.optim.values())))
        else:
            main_optimizer = self.optim

        return Trainer(
            collector=self.collector,
            total_frames=self.config.collector.total_frames,
            loss_module=self.loss_module,
            optimizer=main_optimizer,
            logger=self.logger,
            **self.config.trainer,
        )

    def train(self) -> None:
        """Main training loop."""
        # get the torchrl trainer
        self.trainer.train()

    @abstractmethod
    def predict(self, obs: Tensor) -> Tensor:
        """Predict action given observation."""

    def save_checkpoint(self, path: str):
        """Save complete training state."""
        checkpoint = {
            "policy": self.policy.state_dict() if self.policy else None,
            "value_net": self.value_net.state_dict() if self.value_net else None,
            "q_net": self.q_net.state_dict() if self.q_net else None,
            "target_value_net": (
                self.target_value_net.state_dict() if self.target_value_net else None
            ),
            "target_q_net": (
                self.target_q_net.state_dict() if self.target_q_net else None
            ),
            "reward_estimator": (
                self.reward_estimator.state_dict() if self.reward_estimator else None
            ),
            "optimizers": self.optimizers.state_dict() if self.optimizers else None, # type: ignore
            "step_count": self.step_count,
            "config": OmegaConf.to_container(self.config),
        }
        torch.save(checkpoint, path)

    def export_onnx_policy_to(self, path: str):
        """Export policy network to ONNX format."""
        if self.policy:
            dummy_input: Tensor = torch.randn(1, *self.env.observation_spec.shape)  # type: ignore
            torch.onnx.export(
                self.policy,
                args=(dummy_input,),
                f=path,
                verbose=True,
                input_names=["observation"],
                output_names=["action"],
            )

    def load_from(self, path: str):
        """Load training state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        if self.value_net:
            self.value_net.load_state_dict(checkpoint["value_net"])
        if self.q_net:
            self.q_net.load_state_dict(checkpoint["q_net"])
        if self.target_value_net:
            self.target_value_net.load_state_dict(checkpoint["target_value_net"])
        if self.target_q_net:
            self.target_q_net.load_state_dict(checkpoint["target_q_net"])
        if self.reward_estimator:
            self.reward_estimator.load_state_dict(checkpoint["reward_estimator"])
        if self.optimizers:
            self.optimizers.load_state_dict(checkpoint["optimizers"])
