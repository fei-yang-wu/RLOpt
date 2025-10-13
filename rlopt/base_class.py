from __future__ import annotations

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn
import torch.optim
from tensordict import TensorDict
from tensordict.base import TensorDictBase
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import Tensor
from torch.optim import lr_scheduler
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    ReplayBuffer,
)
from torchrl.envs import TransformedEnv
from torchrl.modules import MLP, ValueOperator
from torchrl.objectives import group_optimizers
from torchrl.record.loggers.common import Logger

from rlopt.configs import (
    FeatureBlockSpec,
    ModuleNetConfig,
    NetworkLayout,
    RLOptConfig,
)
from rlopt.logging_utils import ROOT_LOGGER_NAME, LoggingManager, MetricReporter
from rlopt.type_aliases import OptimizerClass, SchedulerClass
from rlopt.utils import log_agent_overview

torch.set_float32_matmul_precision("high")


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
        self.kwargs = kwargs

        # Keep Python logger and TorchRL logger state on the instance for reuse
        self.log: logging.Logger = logging.getLogger(
            f"{ROOT_LOGGER_NAME}.{self.__class__.__name__}"
        )
        self.logger: Logger | None = None
        self.metrics: MetricReporter | None = None
        self._logging_manager: LoggingManager | None = None
        self.logger_video = False

        self._configure_logger(logger_override=logger)
        self.log_dir: Path = self._logging_manager.run_dir  # type: ignore[union-attr]
        self.log.debug(
            "Initialized %s (metrics_backend=%s, level=%s)",
            self.__class__.__name__,
            self.config.logger.backend or "none",
            logging.getLevelName(self.log.level),
        )

        # Seed for reproducibility
        self.manual_seed(config.seed)
        self.mp_context = "fork"

        # Construct or attach networks based on existence in config
        self.lr_scheduler = None
        self.lr_scheduler_step = "update"

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

        # build collector, collector_policy can be customized
        self.collector = self._construct_collector(self.env, self.collector_policy)

        # build loss module
        self.loss_module = self._construct_loss_module()

        # optimizers
        self.optim = self._configure_optimizers()

        # buffer
        self.data_buffer = self._construct_data_buffer()

        # episode length
        self.episode_lengths = deque(maxlen=100)

        # episode rewards
        self.episode_rewards = deque(maxlen=100)

        # Cache parameters for fast NaN checks during training
        self._parameter_monitor: list[tuple[str, torch.nn.Parameter]] = []
        self._refresh_parameter_monitor()
        self._update_stage_context: str = ""

        # Print model overview once components are initialized
        try:  # noqa: SIM105
            self._print_model_overview()
        except Exception:
            # Avoid hard failures if summary printing hits an edge case
            pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.th_rng = torch.Generator()
        self.th_rng.manual_seed(seed)
        # save for non-cuda device
        torch.cuda.manual_seed_all(seed)

    @property
    def collector_policy(self) -> TensorDictModule:
        """By default, the collector_policy is self.policy or self.actor_critic.policy_operator()"""
        return self.actor_critic.get_policy_operator()

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

    # ---------------------
    # Common NN utilities
    # ---------------------
    def _get_activation_class(self, activation_name: str) -> type[torch.nn.Module]:
        """Get activation class from activation name across agents."""
        activation_map = {
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
            "tanh": torch.nn.Tanh,
            "gelu": torch.nn.GELU,
        }
        return activation_map.get(activation_name, torch.nn.ELU)

    def _initialize_weights(self, module: torch.nn.Module, init_type: str) -> None:
        """Initialize linear layer weights based on initialization type."""
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                if init_type == "orthogonal":
                    torch.nn.init.orthogonal_(layer.weight, 1.0)
                elif init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(layer.weight)
                elif init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight)
                # bias exists for Linear by default
                layer.bias.data.zero_()

    def _construct_collector(
        self, env: TransformedEnv, policy: TensorDictModule
    ) -> SyncDataCollector:
        # We can't use nested child processes with mp_start_method="fork"

        collector = SyncDataCollector(
            create_env_fn=env,
            policy=policy,
            init_random_frames=self.config.collector.init_random_frames,
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
            # reset_at_each_iter=False,
            # set_truncated=self.config.collector.set_truncated,
        )
        collector.set_seed(self.config.seed)
        return collector

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

    def _configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for different components.

        This method provides a general implementation that can be overridden by
        specific algorithms if they need custom optimizer configuration.
        """
        cfg = self.config.optim

        optimizer_map: dict[str, OptimizerClass] = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "adamax": torch.optim.Adamax,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }

        optimizer_name = cfg.optimizer.lower()
        if optimizer_name not in optimizer_map:
            available = ", ".join(sorted(optimizer_map))
            msg = f"Unknown optimizer '{cfg.optimizer}'. Choose one of: {available}."
            raise ValueError(msg)

        optimizer_cls = optimizer_map[optimizer_name]

        base_kwargs = {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        }
        optimizer_kwargs = {**base_kwargs, **cfg.optimizer_kwargs}

        # Create optimizers for different components
        optimizers: list[torch.optim.Optimizer] = []

        # Actor/policy optimizer
        if hasattr(self, "actor_critic"):
            policy_module = None
            if hasattr(self.actor_critic, "get_policy_head"):
                try:
                    policy_module = self.actor_critic.get_policy_head()
                except Exception:
                    policy_module = None
            if policy_module is None and hasattr(
                self.actor_critic, "get_policy_operator"
            ):
                try:
                    policy_module = self.actor_critic.get_policy_operator()
                except Exception:
                    policy_module = None
            if policy_module is None and getattr(self, "policy", None) is not None:
                policy_module = self.policy
            if policy_module is not None:
                actor_optim = optimizer_cls(
                    policy_module.parameters(), **optimizer_kwargs
                )
                optimizers.append(actor_optim)

        # Critic/value optimizer
        if hasattr(self, "actor_critic"):
            value_module = None
            if hasattr(self.actor_critic, "get_value_head"):
                try:
                    value_module = self.actor_critic.get_value_head()
                except Exception:
                    value_module = None
            if value_module is None and hasattr(
                self.actor_critic, "get_value_operator"
            ):
                try:
                    value_module = self.actor_critic.get_value_operator()
                except Exception:
                    value_module = None
            if value_module is None:
                # fall back to explicit networks when available
                value_module = getattr(self, "value_function", None) or getattr(
                    self, "q_function", None
                )
            if value_module is not None:
                critic_optim = optimizer_cls(
                    value_module.parameters(), **optimizer_kwargs
                )
                optimizers.append(critic_optim)

        # Feature extractor optimizer
        if self.config.use_feature_extractor and hasattr(self, "feature_extractor"):
            feature_optim = optimizer_cls(
                self.feature_extractor.parameters(),
                **optimizer_kwargs,
            )
            optimizers.append(feature_optim)

        # Additional optimizers for specific algorithms
        additional_optimizers = self._get_additional_optimizers(
            optimizer_cls, optimizer_kwargs
        )
        optimizers.extend(additional_optimizers)

        if not optimizers:
            msg = "No optimizers could be created. Check that the algorithm has the required components."
            raise ValueError(msg)

        optim = group_optimizers(*optimizers)

        # Configure learning rate scheduler
        scheduler_name = (cfg.scheduler or "").lower() or None
        scheduler_map: dict[str, SchedulerClass] = {
            "steplr": lr_scheduler.StepLR,
            "multiplicativelr": lr_scheduler.MultiplicativeLR,
            "exponentiallr": lr_scheduler.ExponentialLR,
            "cosineannealinglr": lr_scheduler.CosineAnnealingLR,
            "cosineannealingwarmrestarts": lr_scheduler.CosineAnnealingWarmRestarts,
            "linearlr": lr_scheduler.LinearLR,
            "polynomiallr": lr_scheduler.PolynomialLR,
        }

        self.lr_scheduler = None
        if scheduler_name is not None:
            if scheduler_name not in scheduler_map:
                available = ", ".join(sorted(scheduler_map))
                msg = (
                    f"Unknown scheduler '{cfg.scheduler}'. Choose one of: {available}."
                )
                raise ValueError(msg)

            scheduler_cls = scheduler_map[scheduler_name]
            scheduler_kwargs = dict(cfg.scheduler_kwargs)
            self.lr_scheduler = scheduler_cls(optim, **scheduler_kwargs)

        self.lr_scheduler_step = getattr(cfg, "scheduler_step", "update").lower()

        return optim

    def _get_additional_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Get additional optimizers for algorithm-specific components.

        Override this method in subclasses to add optimizers for components
        like alpha parameters in SAC, reward estimators in IPMD, etc.

        Returns:
            List of additional optimizers
        """
        return []

    def _configure_logger(self, logger_override: Logger | None = None) -> None:
        """Configure Python logging and the optional TorchRL metrics backend."""

        self._logging_manager = LoggingManager(
            config=self.config,
            component=self.__class__.__name__,
            metrics_logger=logger_override,
        )
        self.log = self._logging_manager.logger
        self.logger = self._logging_manager.metrics_logger
        self.metrics = self._logging_manager.metric_reporter
        self.logger_video = self._logging_manager.video_enabled

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: Any,
        log_python: bool | None = None,
        python_level: int = logging.INFO,
    ) -> None:
        """Log scalar metrics via the configured metric backend and optional Python logs."""

        if self.metrics is None:
            return

        should_log_python = (
            bool(self.config.logger.log_to_console)
            if log_python is None
            else log_python
        )
        self.metrics.log_scalars(
            metrics,
            step=step,
            log_python=should_log_python,
            python_level=python_level,
        )

    @abstractmethod
    def train(self) -> None:
        """Main training loop."""

    # ---------------------------
    # Architecture introspection
    # ---------------------------
    def _print_model_overview(self) -> None:
        """Use shared utils to print a concise, customizable overview."""
        obs_keys = list(self.total_input_keys)
        dims_by_key: dict[str, int] = {}
        for k in obs_keys:
            try:  # noqa: SIM105
                dims_by_key[k] = int(self.env.observation_spec[k].shape[-1])  # type: ignore[index]
            except Exception:
                pass
        obs_dim = (
            sum(dims_by_key.values()) if dims_by_key else int(self.total_input_shape)
        )
        act_dim = int(self.policy_output_shape)

        extra = {
            "Inputs": {
                "Keys": ", ".join(obs_keys),
                "Dims": dims_by_key if dims_by_key else obs_dim,
                "Action dim": act_dim,
            },
            "Device": str(self.device),
        }

        title = f"RLOpt Model Summary [{self.__class__.__name__}]"
        log_agent_overview(
            self,
            title=title,
            logger=self.log,
            extra=extra,
            max_depth=1,
            indent=0,
        )

    # ---------------------------
    # Reusable builders (MLP-based)
    # ---------------------------
    def _sum_input_dim(self, keys: list[str]) -> int:
        dims = []
        for k in keys:
            with contextlib.suppress(Exception):
                dims.append(int(self.env.observation_spec[k].shape[-1]))  # type: ignore[index]
        if dims:
            return int(torch.tensor(dims).sum().item())
        # Fallback to total_input_shape if nothing matched
        return int(self.total_input_shape)

    def _resolve_shared_feature_spec(
        self, layout: NetworkLayout | None
    ) -> FeatureBlockSpec | None:
        if not layout:
            return None
        shared_name: str | None = None
        if (
            getattr(layout, "policy", None)
            and layout.policy
            and layout.policy.feature_ref
        ):
            shared_name = layout.policy.feature_ref
        elif (
            getattr(layout, "value", None) and layout.value and layout.value.feature_ref
        ):
            shared_name = layout.value.feature_ref
        if shared_name and shared_name in layout.shared.features:
            return layout.shared.features[shared_name]
        return None

    def _resolve_head_config(
        self,
        module_cfg: ModuleNetConfig | None,
        fallback_cells: list[int],
        fallback_activation: type[torch.nn.Module],
        fallback_init: str = "orthogonal",
    ) -> tuple[list[int], type[torch.nn.Module], str]:
        if module_cfg and module_cfg.head:
            return (
                list(module_cfg.head.num_cells),
                self._get_activation_class(module_cfg.head.activation),
                module_cfg.head.init,
            )
        return (list(fallback_cells), fallback_activation, fallback_init)

    def _build_feature_extractor_module(
        self,
        *,
        feature_extractor_net: torch.nn.Module | None = None,
        in_keys: list[str] | None = None,
        out_key: str = "hidden",
        layout: NetworkLayout | None = None,
        use_feature_extractor: bool | None = None,
    ) -> TensorDictModule:
        in_keys = list(in_keys) if in_keys is not None else list(self.total_input_keys)
        if feature_extractor_net is not None:
            return TensorDictModule(
                module=feature_extractor_net, in_keys=in_keys, out_keys=[out_key]  # type: ignore[call-arg]
            )

        if use_feature_extractor is None:
            use_feature_extractor = bool(self.config.use_feature_extractor)

        if use_feature_extractor:
            # Prefer advanced layout spec, else legacy config
            spec = self._resolve_shared_feature_spec(layout)
            if spec is not None:
                if spec.type == "mlp" and spec.mlp is not None:
                    in_dim = self._sum_input_dim(in_keys)
                    fe = MLP(
                        in_features=in_dim,
                        out_features=int(spec.output_dim),
                        num_cells=list(spec.mlp.num_cells),
                        activation_class=self._get_activation_class(
                            spec.mlp.activation
                        ),
                        device=self.device,
                    )
                    self._initialize_weights(fe, spec.mlp.init)
                    return TensorDictModule(
                        module=fe, in_keys=in_keys, out_keys=[out_key]  # type: ignore[call-arg]
                    )
                msg = f"Feature type {spec.type} not supported (expected mlp)"
                raise NotImplementedError(msg)

            # Legacy feature-extractor config
            in_dim = self._sum_input_dim(in_keys)
            fe = MLP(
                in_features=in_dim,
                out_features=int(self.config.feature_extractor.output_dim),
                num_cells=list(self.config.feature_extractor.num_cells),
                activation_class=torch.nn.ELU,
                device=self.device,
            )
            self._initialize_weights(fe, "orthogonal")
            return TensorDictModule(module=fe, in_keys=in_keys, out_keys=[out_key])  # type: ignore[call-arg]

        # No feature extractor: identity mapping
        return TensorDictModule(
            module=torch.nn.Identity(), in_keys=in_keys, out_keys=in_keys  # type: ignore[call-arg]
        )

    def _build_policy_head_module(
        self,
        *,
        policy_net: torch.nn.Module | None = None,
        in_keys: list[str] | None = None,
        out_keys: list[str] | tuple[str, str] = ("loc", "scale"),
        layout: NetworkLayout | None = None,
    ) -> TensorDictModule:
        in_keys = (
            list(in_keys) if in_keys is not None else list(self.config.policy_in_keys)
        )

        if policy_net is None:
            # Read head config
            module_cfg = (
                layout.policy if layout and getattr(layout, "policy", None) else None
            )
            num_cells, activation_class, init = self._resolve_head_config(
                module_cfg,
                fallback_cells=list(self.config.policy.num_cells),
                fallback_activation=torch.nn.ELU,
                fallback_init="orthogonal",
            )
            # Infer input dim: prefer feature spec output dim if using FE
            if self.config.use_feature_extractor:
                spec = self._resolve_shared_feature_spec(layout)
                in_dim = (
                    int(spec.output_dim)
                    if spec is not None
                    else int(self.config.feature_extractor.output_dim)
                )
            else:
                # Sum over raw observation keys (policy+value in_keys when no FE)
                raw_keys = in_keys
                in_dim = self._sum_input_dim(raw_keys)

            net = MLP(
                in_features=in_dim,
                activation_class=activation_class,
                out_features=int(self.policy_output_shape),
                num_cells=list(num_cells),
                device=self.device,
            )
            # self._initialize_weights(net, init)
        else:
            net = policy_net

        # net = torch.nn.Sequential(
        #     net,
        #     AddStateIndependentNormalScale(
        #         int(self.policy_output_shape), scale_lb=1e-4
        #     ).to(self.device),
        # )
        extractor = NormalParamExtractor(
            scale_mapping="biased_softplus_1.0", scale_lb=0.1  # type: ignore
        ).to(self.device)
        net = torch.nn.Sequential(net, extractor)
        return TensorDictModule(module=net, in_keys=in_keys, out_keys=list(out_keys))  # type: ignore[call-arg]

    def _build_value_module(
        self,
        *,
        value_net: torch.nn.Module | None = None,
        in_keys: list[str] | None = None,
        out_key: str | None = None,
        layout: NetworkLayout | None = None,
    ) -> TensorDictModule:
        in_keys = (
            list(in_keys)
            if in_keys is not None
            else list(self.config.value_net_in_keys)
        )

        if value_net is None:
            module_cfg = (
                layout.value if layout and getattr(layout, "value", None) else None
            )
            num_cells, activation_class, init = self._resolve_head_config(
                module_cfg,
                fallback_cells=list(self.config.value_net.num_cells),
                fallback_activation=torch.nn.ELU,
                fallback_init="orthogonal",
            )

            if self.config.use_feature_extractor:
                spec = self._resolve_shared_feature_spec(layout)
                in_dim = (
                    int(spec.output_dim)
                    if spec is not None
                    else int(self.config.feature_extractor.output_dim)
                )
            else:
                in_dim = self._sum_input_dim(in_keys)

            net = MLP(
                in_features=in_dim,
                activation_class=activation_class,
                out_features=int(self.value_output_shape),
                num_cells=list(num_cells),
                device=self.device,
            )
            self._initialize_weights(net, init)
        else:
            net = value_net

        # ValueOperator allows optional out_keys override
        if out_key is not None:
            return ValueOperator(net, in_keys=in_keys, out_keys=[out_key])
        return ValueOperator(net, in_keys=in_keys)

    def _build_qvalue_module(
        self,
        *,
        q_net: torch.nn.Module | None = None,
        in_keys: list[str] | None = None,
        layout: NetworkLayout | None = None,
    ) -> TensorDictModule:
        # Default in_keys: ["action"] + (policy_in_keys if FE else total_input_keys)
        if in_keys is None:
            in_keys = ["action"] + (
                list(self.config.policy_in_keys)
                if self.config.use_feature_extractor
                else list(self.total_input_keys)
            )

        if q_net is None:
            # Read critic head config if present
            head_cfg = None
            if (
                layout
                and getattr(layout, "critic", None)
                and layout.critic
                and layout.critic.template
                and layout.critic.template.head
            ):
                head_cfg = layout.critic.template.head
                num_cells = list(head_cfg.num_cells)
                activation_class = self._get_activation_class(head_cfg.activation)
                init = head_cfg.init
            else:
                num_cells = list(getattr(self.config.action_value_net, "num_cells", []))
                activation_class = torch.nn.ELU
                init = "orthogonal"

            # Input dim: action + obs/hidden
            if self.config.use_feature_extractor:
                spec = self._resolve_shared_feature_spec(layout)
                obs_dim = (
                    int(spec.output_dim)
                    if spec is not None
                    else int(self.config.feature_extractor.output_dim)
                )
            else:
                # Remove action before summing
                obs_keys = [k for k in in_keys if k != "action"]
                obs_dim = self._sum_input_dim(obs_keys)
            in_dim = int(self.policy_output_shape) + obs_dim

            net = MLP(
                in_features=in_dim,
                out_features=1,
                num_cells=list(num_cells),
                activation_class=activation_class,
                device=self.device,
            )
            self._initialize_weights(net, init)
        else:
            net = q_net

        return ValueOperator(module=net, in_keys=in_keys)

    @abstractmethod
    def predict(self, obs: Tensor) -> Tensor:
        """Predict action given observation."""

    def save_model(
        self, path: str | Path | None = None, step: int | None = None
    ) -> None:
        """Save the model and related parameters to a file."""
        default_dir = self.log_dir
        target_base = Path(path).expanduser() if path is not None else default_dir
        base_exists = target_base.exists()
        base_is_file = base_exists and target_base.is_file()
        has_suffix = bool(target_base.suffix)

        if step is not None:
            if base_is_file or (has_suffix and not target_base.is_dir()):
                suffix = target_base.suffix
                stemmed = target_base.with_suffix("")
                target_path = stemmed.with_name(stemmed.name + f"_step_{step}{suffix}")
            else:
                target_path = target_base / f"model_step_{step}.pt"
        elif base_is_file or (has_suffix and not target_base.is_dir()):
            target_path = target_base
        else:
            target_path = target_base / "model.pt"
        target_path.parent.mkdir(parents=True, exist_ok=True)

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

        torch.save(data_to_save, target_path)

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

    def _refresh_parameter_monitor(self) -> None:
        """Collect parameter references for NaN checks while avoiding duplicates."""
        modules_to_check: list[tuple[str, torch.nn.Module | None]] = [
            ("actor_critic", self.actor_critic),
            ("loss_module", self.loss_module),
        ]
        if getattr(self.config, "use_feature_extractor", False):
            modules_to_check.append(("feature_extractor", self.feature_extractor))

        seen: set[int] = set()
        monitored: list[tuple[str, torch.nn.Parameter]] = []
        for module_label, module in modules_to_check:
            if module is None:
                continue
            for name, param in module.named_parameters(recurse=True):
                if param is None or not torch.is_floating_point(param):
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                monitored.append((f"{module_label}.{name}", param))

        self._parameter_monitor = monitored

    def _validate_parameters(self, stage: str) -> None:
        """Ensure all monitored parameters remain finite."""
        if not self._parameter_monitor:
            self._refresh_parameter_monitor()

        for name, param in self._parameter_monitor:
            if not torch.isfinite(param).all():
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                msg = (
                    "Detected non-finite parameter values "
                    f"during '{stage}' in {name}: nan={nan_count}, inf={inf_count}"
                )
                raise FloatingPointError(msg)

    def _validate_gradients(self, stage: str, raise_error: bool = False) -> bool:
        """Ensure gradients for monitored parameters are finite."""
        if not self._parameter_monitor:
            self._refresh_parameter_monitor()

        all_finite = True
        for name, param in self._parameter_monitor:
            grad = param.grad
            if grad is None or not torch.is_floating_point(grad):
                continue
            if not torch.isfinite(grad).all():
                nan_count = torch.isnan(grad).sum().item()
                inf_count = torch.isinf(grad).sum().item()
                msg = (
                    "Detected non-finite gradients "
                    f"during '{stage}' in {name}: nan={nan_count}, inf={inf_count}"
                )
                all_finite = False
                if raise_error:
                    raise FloatingPointError(msg)
                self.log.debug("%s", msg)
        return all_finite

    def _validate_tensordict(
        self,
        td: TensorDictBase,
        stage: str,
        prefix: tuple[str, ...] = (),
        raise_error: bool = False,
    ) -> bool:
        """Recursively ensure TensorDict floating tensors are finite."""

        all_finite = True
        for key, value in td.items():  # type: ignore
            key_str = str(key)
            next_prefix = (*prefix, key_str)
            if isinstance(value, TensorDict):
                all_finite &= self._validate_tensordict(
                    value, stage, next_prefix, raise_error
                )
            elif (
                torch.is_tensor(value)
                and torch.is_floating_point(value)  # type: ignore[call-arg]
                and not torch.isfinite(value).all()  # type: ignore
            ):
                nan_count = torch.isnan(value).sum().item()  # type: ignore[call-arg]
                inf_count = torch.isinf(value).sum().item()  # type: ignore[call-arg]
                joined_key = ".".join(next_prefix)
                msg = (
                    "Detected non-finite tensor values "
                    f"during '{stage}' at key '{joined_key}': "
                    f"nan={nan_count}, inf={inf_count}"
                )
                all_finite = False
                if raise_error:
                    raise FloatingPointError(msg)
                self.log.debug("%s", msg)

        return all_finite

    def _sanitize_loss_tensordict(self, loss_td: TensorDict, stage: str) -> TensorDict:
        """Replace non-finite loss terms with zeros and warn the user."""

        for key, value in list(loss_td.items()):
            if isinstance(value, TensorDict):
                self._sanitize_loss_tensordict(value, stage)
                continue

            if torch.is_tensor(value) and torch.is_floating_point(value):  # type: ignore[call-arg]
                if torch.isfinite(value).all():  # type: ignore[call-arg]
                    continue

                nan_count = torch.isnan(value).sum().item()  # type: ignore[call-arg]
                inf_count = torch.isinf(value).sum().item()  # type: ignore[call-arg]
                key_repr = key if isinstance(key, str) else str(key)
                self.log.debug(
                    "Non-finite loss detected; replacing with zeros | stage=%s key=%s nan=%d inf=%d",
                    stage,
                    key_repr,
                    nan_count,
                    inf_count,
                )
                loss_td.set(key, torch.zeros_like(value))  # type: ignore

        return loss_td
