from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn
import torch.optim
from tensordict import TensorDict
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torch import Tensor
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    ReplayBuffer,
)
from torchrl.envs import TransformedEnv
from torchrl.modules import MLP, ValueOperator
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.record.loggers.common import Logger

from rlopt.configs import (
    FeatureBlockSpec,
    ModuleNetConfig,
    NetworkLayout,
    RLOptConfig,
)
from rlopt.utils import log_agent_overview

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

        # logger_video attribute must be defined before use
        self.logger_video = False

        # logger
        self._configure_logger()

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

        return SyncDataCollector(
            create_env_fn=env,
            policy=policy,
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
                    "config": asdict(cfg),
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
            logger=logger,
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
            try:
                dims.append(int(self.env.observation_spec[k].shape[-1]))  # type: ignore[index]
            except Exception:
                # Unknown key (e.g., hidden); caller should override in_dim
                pass
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
                module=feature_extractor_net, in_keys=in_keys, out_keys=[out_key]
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
                        module=fe, in_keys=in_keys, out_keys=[out_key]
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
            return TensorDictModule(module=fe, in_keys=in_keys, out_keys=[out_key])

        # No feature extractor: identity mapping
        return TensorDictModule(
            module=torch.nn.Identity(), in_keys=in_keys, out_keys=in_keys
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
            self._initialize_weights(net, init)
        else:
            net = policy_net

        net = torch.nn.Sequential(
            net,
            AddStateIndependentNormalScale(
                int(self.policy_output_shape), scale_lb=1e-8
            ).to(self.device),
        )
        return TensorDictModule(module=net, in_keys=in_keys, out_keys=list(out_keys))

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
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return all_finite

    def _validate_tensordict(
        self,
        td: TensorDict,
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
            elif torch.is_tensor(value) and torch.is_floating_point(value):  # noqa: SIM102
                if not torch.isfinite(value).all():
                    nan_count = torch.isnan(value).sum().item()
                    inf_count = torch.isinf(value).sum().item()
                    joined_key = ".".join(next_prefix)
                    msg = (
                        "Detected non-finite tensor values "
                        f"during '{stage}' at key '{joined_key}': "
                        f"nan={nan_count}, inf={inf_count}"
                    )
                    all_finite = False
                    if raise_error:
                        raise FloatingPointError(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return all_finite

    def _sanitize_loss_tensordict(self, loss_td: TensorDict, stage: str) -> TensorDict:
        """Replace non-finite loss terms with zeros and warn the user."""

        for key, value in list(loss_td.items()):
            if isinstance(value, TensorDict):
                self._sanitize_loss_tensordict(value, stage)
                continue

            if torch.is_tensor(value) and torch.is_floating_point(value):
                if torch.isfinite(value).all():
                    continue

                nan_count = torch.isnan(value).sum().item()
                inf_count = torch.isinf(value).sum().item()
                key_repr = key if isinstance(key, str) else str(key)
                warnings.warn(
                    "Non-finite PPO loss detected; replacing with zeros. "
                    f"Stage='{stage}', key='{key_repr}', nan={nan_count}, inf={inf_count}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                loss_td.set(key, torch.zeros_like(value))  # type: ignore

        return loss_td
