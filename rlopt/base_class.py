from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
        # Initialize component placeholders
        self.policy: TensorDictModule | None = None
        self.value_function: TensorDictModule | None = None
        self.q_function: TensorDictModule | None = None

        self.policy = self._construct_policy(policy_net)

        # determine using value function or q function
        if self.config.use_value_function:
            self.value_function = (
                self._construct_value_function()
                if value_net is None
                else self._construct_value_function(value_net)
            )
        else:
            self.q_function = (
                self._construct_q_function()
                if q_net is None
                else self._construct_q_function(q_net)
            )

        self.actor_critic = self._construct_actor_critic()

        # Move them to device
        if self.policy is not None:
            self.policy.to(self.device)
        if self.value_function is not None:
            self.value_function.to(self.device)
        if self.q_function is not None:
            self.q_function.to(self.device)

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

        # Print model overview once components are initialized
        try:  # noqa: SIM105
            self._print_model_overview()
        except Exception:
            # Avoid hard failures if summary printing hits an edge case
            pass

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
                {
                    "mode": self.config.compile.compile_mode,
                    "warmup": int(getattr(self.config.compile, "warmup", 1)),
                }
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

    # ---------------------------
    # Architecture introspection
    # ---------------------------
    def _print_model_overview(self) -> None:
        """Log an elegant, informative model summary with dims and details."""
        algo = self.__class__.__name__

        def _act_name(act) -> str:
            try:
                return act.__name__ if isinstance(act, type) else act.__class__.__name__
            except Exception:
                return str(act)

        # Inputs
        obs_keys = list(self.total_input_keys)
        dims_by_key: dict[str, int] = {}
        for k in obs_keys:
            try:
                dims_by_key[k] = int(self.env.observation_spec[k].shape[-1])  # type: ignore[index]
            except Exception:
                pass
        obs_dim = (
            sum(dims_by_key.values()) if dims_by_key else int(self.total_input_shape)
        )
        act_dim = int(self.policy_output_shape)

        # Feature extractor
        fe_td = self.feature_extractor
        fe_mod = getattr(fe_td, "module", fe_td)
        fe_cls = fe_mod.__class__.__name__
        uses_fe = bool(self.config.use_feature_extractor)
        fe_is_identity = fe_cls == "Identity"
        fe_shared = uses_fe and not fe_is_identity
        fe_out_key = "hidden" if uses_fe else ", ".join(obs_keys)
        fe_out_dim = (
            int(getattr(self.config.feature_extractor, "output_dim", -1))
            if uses_fe and not fe_is_identity
            else obs_dim
        )
        # FE layers (best-effort from config)
        fe_layers = []
        fe_act = ""
        try:
            fe_layers = list(getattr(self.config.feature_extractor, "num_cells", []))
            fe_act = "ELU"
        except Exception:
            pass

        # Actor
        policy_op = self.actor_critic.get_policy_operator()
        policy_in_keys = list(getattr(self.config, "policy_in_keys", []))
        actor_dist = getattr(policy_op, "distribution_class", None)
        actor_dist_name = actor_dist.__name__ if actor_dist is not None else "Unknown"
        actor_interaction = getattr(policy_op, "default_interaction_type", "?")
        actor_layers = []
        actor_act = ""
        try:
            if (
                getattr(self.config, "network", None)
                and getattr(self.config.network, "policy", None)
                and getattr(self.config.network.policy, "head", None)
            ):
                actor_layers = list(self.config.network.policy.head.num_cells)  # type: ignore[attr-defined]
                actor_act = _act_name(self._get_activation_class(self.config.network.policy.head.activation))  # type: ignore[attr-defined]
            else:
                actor_layers = list(getattr(self.config.policy, "num_cells", []))
                actor_act = "ELU"
        except Exception:
            pass

        # Critic / Q
        using_value = bool(self.config.use_value_function)
        if using_value:
            critic_type = "state-value"
            value_in_keys = list(getattr(self.config, "value_net_in_keys", []))
            critic_layers = []
            critic_act = ""
            try:
                if (
                    getattr(self.config, "network", None)
                    and getattr(self.config.network, "value", None)
                    and getattr(self.config.network.value, "head", None)
                ):
                    critic_layers = list(self.config.network.value.head.num_cells)  # type: ignore[attr-defined]
                    critic_act = _act_name(self._get_activation_class(self.config.network.value.head.activation))  # type: ignore[attr-defined]
                else:
                    critic_layers = list(
                        getattr(self.config.value_net, "num_cells", [])
                    )
                    critic_act = "ELU"
            except Exception:
                pass
            critic_in_dim = int(self.value_input_shape)
            critic_out_dim = 1
        else:
            # SAC Q(s)
            num_q = None
            try:
                if getattr(self.config, "network", None) and getattr(
                    self.config.network, "critic", None
                ):
                    num_q = int(getattr(self.config.network.critic, "num_nets", 1))
            except Exception:
                pass
            if num_q is None:
                try:
                    num_q = int(getattr(self.config.sac, "num_qvalue_nets", 1))
                except Exception:
                    num_q = 1
            critic_type = (
                f"Q-value x{num_q}"
                if (isinstance(num_q, int) and num_q > 1)
                else "Q-value"
            )
            value_in_keys = ["action"] + (
                policy_in_keys if self.config.use_feature_extractor else obs_keys
            )
            critic_layers = list(getattr(self.config.action_value_net, "num_cells", []))
            critic_act = "ELU"
            critic_in_dim = int(self.policy_input_shape + act_dim)
            critic_out_dim = 1

        # Build message
        title = f"RLOpt Model Summary [{algo}] — Device: {self.device}"
        bar = "=" * len(title)
        parts: list[str] = [bar, title, bar]

        parts.append("Inputs")
        parts.append(f"- Keys: {obs_keys}")
        if dims_by_key:
            parts.append(f"- Dims: {dims_by_key} (sum={obs_dim})")
        else:
            parts.append(f"- Total dim: {obs_dim}")
        parts.append(f"- Action dim: {act_dim}")
        parts.append("")

        parts.append("Feature Extractor")
        parts.append(f"- Shared: {bool(fe_shared)}")
        parts.append(f"- Type: {fe_cls}")
        parts.append(f"- Out: key='{fe_out_key}', dim={fe_out_dim}")
        if fe_layers:
            parts.append(f"- Layers: {fe_layers}; act: {fe_act}")
        parts.append("")

        parts.append("Actor")
        parts.append(f"- In keys: {policy_in_keys}")
        parts.append(
            f"- Head: {policy_op.__class__.__name__} (dist={actor_dist_name}, interaction={actor_interaction})"
        )
        if actor_layers:
            parts.append(
                f"- Net: MLP(in={int(self.policy_input_shape)} → out={act_dim}, layers={actor_layers}, act={actor_act})"
            )
        parts.append("")

        parts.append("Critic")
        parts.append(f"- Type: {critic_type}")
        parts.append(f"- In keys: {value_in_keys}")
        if critic_layers:
            parts.append(
                f"- Net: MLP(in={critic_in_dim} → out={critic_out_dim}, layers={critic_layers}, act={critic_act})"
            )

        logger.info("\n".join(parts))

    @abstractmethod
    def predict(self, obs: Tensor) -> Tensor:
        """Predict action given observation."""

    def save_model(
        self, path: str | Path | None = None, step: int | None = None
    ) -> None:
        """Save the model and related parameters to a file."""
        # Handle case where no logger is provided (e.g., in testing)
        if path is None and self.logger is None:
            # Default to current directory if no logger and no path provided
            prefix = "."
        elif path is None:
            prefix = f"{self.config.logger.log_dir}"
        else:
            prefix = path

        # Include step in filename if provided
        if step is not None:
            if Path(prefix).is_file():
                # If prefix is a file, add step to the filename
                path = str(prefix).rsplit(".", 1)[0] + f"_step_{step}.pt"
            else:
                # If prefix is a directory, create filename with step
                path = f"{prefix}/model_step_{step}.pt"
        elif Path(prefix).is_file():
            # If prefix is a file, use it directly
            path = prefix
        else:
            # If prefix is a directory, create default filename
            path = f"{prefix}/model.pt"

        # Ensure directory exists only if we're creating a new path
        if path != prefix or Path(prefix).is_dir():
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        data_to_save: dict[str, torch.Tensor | dict] = {
            "policy_state_dict": (
                self.policy.state_dict() if self.policy is not None else {}
            ),
            "optimizer_state_dict": self.optim.state_dict(),
        }
        if self.value_function is not None:
            data_to_save["value_state_dict"] = self.value_function.state_dict()
        if self.q_function is not None:
            data_to_save["q_state_dict"] = self.q_function.state_dict()
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
        if self.policy is not None and "policy_state_dict" in data:
            self.policy.load_state_dict(data["policy_state_dict"])  # type: ignore[arg-type]
        if self.value_function is not None and "value_state_dict" in data:
            self.value_function.load_state_dict(data["value_state_dict"])  # type: ignore[arg-type]
        if self.q_function is not None and "q_state_dict" in data:
            self.q_function.load_state_dict(data["q_state_dict"])  # type: ignore[arg-type]
        if "optimizer_state_dict" in data:
            self.optim.load_state_dict(data["optimizer_state_dict"])  # type: ignore[arg-type]
        if self.config.use_feature_extractor and "feature_extractor_state_dict" in data:
            self.feature_extractor.load_state_dict(data["feature_extractor_state_dict"])  # type: ignore[arg-type]
        if hasattr(self.env, "normalize_obs") and "vec_norm_msg" in data:
            self.env.load_state_dict(data["vec_norm_msg"])  # type: ignore[arg-type]
