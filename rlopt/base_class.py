from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn
import torch.optim
from tensordict import TensorDict
from tensordict.base import TensorDictBase
from tensordict.nn import CudaGraphModule, TensorDictModule
from torch import Tensor
from torch.optim import lr_scheduler
from torchrl._utils import compile_with_warmup
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    ReplayBuffer,
)
from torchrl.envs import TransformedEnv
from torchrl.objectives import group_optimizers
from torchrl.record.loggers.common import Logger

from rlopt.config_base import (
    RLOptConfig,
)
from rlopt.logging_utils import ROOT_LOGGER_NAME, LoggingManager, MetricReporter
from rlopt.type_aliases import OptimizerClass, SchedulerClass


class BaseAlgorithm(ABC):
    """Abstract base class for reinforcement learning algorithms.

    This class provides a unified framework for implementing various RL algorithms
    (PPO, SAC, DDPG, etc.) with standardized components including policy networks,
    value functions, data collection, logging, and training infrastructure.

    The class follows a template method pattern where concrete algorithms implement
    specific abstract methods while inheriting common functionality for optimization,
    logging, model persistence, and numerical stability checks.

    Args:
        env: A TorchRL TransformedEnv instance representing the RL environment.
            Should include appropriate transforms (e.g., normalization, action clipping).
        config: Algorithm-specific configuration dataclass (subclass of RLOptConfig)
            containing hyperparameters, network architectures, and training settings.
        logger: Optional TorchRL Logger instance for metrics tracking. If None,
            a logger will be created based on config.logger settings.
        **kwargs: Additional keyword arguments for algorithm-specific extensions.

    Attributes:
        policy: TensorDictModule implementing the policy network (actor).
        value_function: Optional TensorDictModule for value function estimation (critic).
        q_function: Optional TensorDictModule for Q-function estimation (used in off-policy methods).
        feature_extractor: Optional shared feature extraction network.
        actor_critic: Combined actor-critic module for unified policy and value computation.
        loss_module: TorchRL loss module for computing training objectives.
        data_buffer: ReplayBuffer for storing and sampling experience.
        collector: SyncDataCollector for environment interaction and data collection.
        optim: Grouped optimizer for all trainable parameters.
        lr_scheduler: Optional learning rate scheduler.

    Workflow for Implementing New Algorithms:
        1. Define configuration dataclass in `rlopt/configs.py` inheriting from RLOptConfig.
        2. Create algorithm class inheriting from BaseAlgorithm.
        3. Implement abstract methods:
           - _construct_policy(): Build policy network
           - _construct_actor_critic(): Combine policy and value networks
           - _construct_loss_module(): Define loss computation
           - _construct_data_buffer(): Configure replay buffer
           - _construct_feature_extractor(): Build shared feature network (if needed)
           - train(): Implement main training loop
           - predict(): Define inference behavior
        4. Override _set_optimizers() if custom optimizers are needed.
        5. Optionally override _compile_components() for torch.compile optimization.

    Note:
        Networks are lazily initialized. Call fake_tensordict() on the environment
        to generate a dummy batch for initialization before training begins.

    Example:
        >>> config = PPOConfig(...)
        >>> env = make_env(config.env)
        >>> algorithm = PPOAlgorithm(env, config)
        >>> algorithm.train()
        >>> algorithm.save_model("checkpoints/model.pt")
    """

    def __init__(
        self,
        env: TransformedEnv,
        config: RLOptConfig,
        logger: Logger | None = None,
        **kwargs,
    ):
        """Initialize the base algorithm with environment, configuration, and logging.

        Sets up all core components including networks, data collection, optimization,
        and monitoring infrastructure. Concrete algorithms should not override this
        unless absolutely necessary; use the abstract construction methods instead.

        Args:
            env: Configured TorchRL environment with appropriate transforms.
            config: Algorithm configuration with hyperparameters and settings.
            logger: Optional external logger for metrics tracking.
            **kwargs: Additional algorithm-specific parameters.
        """
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
        self.seed(config.seed)

        self.mp_context = "fork"

        # Construct or attach networks based on existence in config
        self.lr_scheduler = None
        self.lr_scheduler_step = "update"

        # Initialize component placeholders
        self.policy: TensorDictModule | None = None
        self.value_function: TensorDictModule | None = None
        self.q_function: TensorDictModule | None = None
        self.feature_extractor: TensorDictModule | None = None

        # construct networks based on config
        self.policy = self._construct_policy()
        if config.feature_extractor:
            self.feature_extractor = self._construct_feature_extractor()
        if config.value_function:
            self.value_function = self._construct_value_function()
        if config.q_function:
            self.q_function = self._construct_q_function()

        # Use actor_critic to encapsulate policy and value networks for easy management
        self.actor_critic = self._construct_actor_critic()

        # buffer
        self.data_buffer = self._construct_data_buffer()

        self.step_count = 0

        self.start_time = time.time()

        # build collector, collector_policy can be customized
        self.collector = self._construct_collector(self.env, self.collector_policy)

        # build loss module
        self.loss_module = self._construct_loss_module()

        # optimizers
        self.optim = self._configure_optimizers()

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

        self._compile_components()

    def seed(self, seed: int) -> None:
        """Set random seeds for reproducibility across all random number generators.

        Configures seeds for PyTorch (CPU and CUDA), NumPy, and creates dedicated
        RNG instances for controlled stochasticity in the algorithm.

        Args:
            seed: Integer seed value for all random number generators.
        """
        torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.th_rng = torch.Generator(device=self.device)
        self.th_rng.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @property
    def collector_policy(self) -> TensorDictModule:
        """Return the policy used for data collection during training.

        By default, returns the policy operator from the actor-critic module.
        Override in subclasses if a different exploration policy is needed
        (e.g., with added noise for off-policy algorithms).

        Returns:
            TensorDictModule that maps observations to actions for collection.
        """
        return self.actor_critic.get_policy_operator()

    @property
    def policy_output_shape(self) -> int:
        """Get the dimensionality of the action space.

        Returns:
            Integer representing the action space dimension.
        """
        return int(self.env.action_spec_unbatched.shape[-1])  # type: ignore

    @property
    def value_output_shape(self) -> int:
        """Get the output dimension for value function networks.

        Returns:
            1 (scalar value function output).
        """
        return 1

    @property
    def device(self) -> torch.device:
        """Return the PyTorch device used for training computations.

        Returns:
            torch.device instance (CPU or CUDA based on config).
        """
        return self._get_device(self.config.device)

    def _get_device(self, device_str: str) -> torch.device:
        """Resolve device string to a concrete PyTorch device.

        Args:
            device_str: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.).
                "auto" selects CUDA if available, otherwise CPU.

        Returns:
            torch.device instance corresponding to the specification.
        """
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _initialize_weights(
        self, module: torch.nn.Module, init_type: str | None
    ) -> None:
        """Initialize neural network weights using specified initialization scheme.

        Applies weight initialization to all Linear layers in the module recursively.
        Biases are always initialized to zero.

        Args:
            module: PyTorch module whose weights should be initialized.
            init_type: Initialization method. Supported values:
                - "orthogonal": Orthogonal initialization with gain=1.0
                - "xavier_uniform": Xavier/Glorot uniform initialization
                - "kaiming_uniform": He uniform initialization
                - None: Skip initialization (use PyTorch defaults)
        """
        if init_type is None:
            return  # No initialization specified, skip
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
        """Create a synchronized data collector for environment interaction.

        Configures a TorchRL SyncDataCollector with appropriate settings for
        parallel data collection, initialization, and device placement.

        Args:
            env: Environment instance to collect data from.
            policy: Policy module for action selection during collection.

        Returns:
            Configured SyncDataCollector instance ready for use.

        Note:
            Uses fork context by default. Compilation is enabled if config.compile
            is set, with warmup period for JIT optimization.
        """
        # We can't use nested child processes with mp_start_method="fork"

        collector = SyncDataCollector(
            env,
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
            cudagraph_policy=(
                {"warmup": 10}
                if self.config.compile.cudagraphs and self.device.type == "cuda"
                else False
            ),
            # heterogeneous devices
            device=self.device,
            # storing_device=self.device,
            # reset_at_each_iter=False,
            # set_truncated=self.config.collector.set_truncated,
        )
        collector.set_seed(self.config.seed)
        return collector

    @abstractmethod
    def _construct_policy(self) -> TensorDictModule:
        """Construct the policy network (actor) from configuration.

        Implement this method in subclasses to build the policy network architecture.
        The network should map observations to action distributions or deterministic actions.

        Args:
            policy_net: Optional pre-constructed network. If None, build from config.

        Returns:
            TensorDictModule wrapping the policy network with appropriate input/output keys.

        Note:
            For stochastic policies, typically wraps the network with a distribution
            (e.g., TanhNormal, Normal) via ProbabilisticActor.
        """

    def _construct_value_function(self) -> TensorDictModule:
        """Construct the value function network (critic) from configuration.

        Implement this method in subclasses that use value-based methods (e.g., PPO, A2C).

        Args:
            value_net: Optional pre-constructed network. If None, build from config.

        Returns:
            TensorDictModule wrapping the value network.

        Raises:
            NotImplementedError: If the algorithm doesn't use a value function.
        """
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def _construct_q_function(self) -> TensorDictModule:
        """Construct the Q-function network from configuration.

        Implement this method in subclasses that use Q-learning (e.g., SAC, DDPG, TD3).

        Args:
            q_net: Optional pre-constructed network. If None, build from config.

        Returns:
            TensorDictModule wrapping the Q-network.

        Raises:
            NotImplementedError: If the algorithm doesn't use Q-functions.
        """
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @abstractmethod
    def _construct_feature_extractor(self) -> TensorDictModule:
        """Construct a shared feature extraction network from configuration.

        Implement this method when using shared representations between policy and value networks.

        Args:
            feature_extractor_net: Optional pre-constructed network. If None, build from config.

        Returns:
            TensorDictModule for shared feature extraction.

        Note:
            Common for vision-based tasks where convolutional layers are shared.
        """

    @abstractmethod
    def _construct_actor_critic(self) -> TensorDictModule:
        """Construct combined actor-critic module for unified forward passes.

        Implement this method to combine policy and value networks into a single module.
        This enables efficient computation when both are needed simultaneously.

        Returns:
            TensorDictModule combining actor and critic components.

        Example:
            For PPO, this typically returns an ActorValueOperator wrapping
            the policy and value function.
        """

    @abstractmethod
    def _construct_loss_module(self) -> torch.nn.Module:
        """Construct the loss module for computing training objectives.

        Implement this method to create the algorithm-specific loss module
        (e.g., ClipPPOLoss, SACLoss, DDPGLoss).

        Returns:
            TorchRL loss module configured for the algorithm.

        Note:
            The loss module should accept TensorDict inputs and return
            a TensorDict with loss components.
        """

    @abstractmethod
    def _construct_data_buffer(self) -> ReplayBuffer:
        """Construct the replay buffer for storing and sampling experience.

        Implement this method to configure the buffer based on algorithm needs
        (on-policy vs off-policy, prioritization, etc.).

        Returns:
            Configured ReplayBuffer instance.

        Example:
            On-policy algorithms typically use simple TensorDictReplayBuffer,
            while off-policy methods may use larger buffers with prioritization.
        """

    def _compile_components(self) -> None:  # noqa: B027
        """Compile performance-critical methods using torch.compile for acceleration.

        Override this method in subclasses to apply torch.compile to hot paths
        (e.g., action computation, advantage estimation, return calculation).

        Note:
            Only beneficial for algorithms with heavy computational bottlenecks.
            May increase compilation time and memory usage.

        Example:
            >>> if self.config.compile:
            ...     self.compute_action = torch.compile(self._compute_action, dynamic=True)
            ...     self.compute_returns = torch.compile(self._compute_returns, dynamic=True)
            >>> else:
            ...     self.compute_action = self._compute_action
            ...     self.compute_returns = self._compute_returns
        """
        compile_mode = None

        if self.config.compile.compile:
            compile_mode = self.config.compile.compile_mode
            if compile_mode in ("", None):
                if self.config.compile.cudagraphs:
                    compile_mode = "default"
                else:
                    compile_mode = "reduce-overhead"
        self.compile_mode = compile_mode

    def _configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers and learning rate schedulers for training.

        Creates optimizers for all trainable components (policy, value, Q-networks, etc.)
        based on configuration settings. Supports common optimizers (Adam, SGD, RMSprop)
        and optional learning rate scheduling.

        Returns:
            Grouped optimizer containing all component optimizers.

        Raises:
            ValueError: If unknown optimizer or scheduler is specified in config.

        Note:
            Subclasses can override _set_optimizers() to add algorithm-specific
            optimizers (e.g., for temperature parameters, reward models).
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
        optimizers: list[torch.optim.Optimizer] = self._set_optimizers(
            optimizer_cls, optimizer_kwargs
        )

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
        if scheduler_name and scheduler_name != "adaptive":
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

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizers for algorithm-specific trainable components.

        Override this method in subclasses to instantiate optimizers for all
        trainable modules (policy, value, Q-networks, temperature parameters, etc.).

        Args:
            optimizer_cls: Optimizer class to instantiate (e.g., torch.optim.Adam).
            optimizer_kwargs: Keyword arguments for optimizer construction (lr, weight_decay, etc.).

        Returns:
            List of optimizer instances, one per trainable component.

        Example:
            >>> def _set_optimizers(self, optimizer_cls, optimizer_kwargs):
            ...     return [
            ...         optimizer_cls(self.policy.parameters(), **optimizer_kwargs),
            ...         optimizer_cls(self.value_function.parameters(), **optimizer_kwargs)
            ...     ]
        """
        return []

    def _configure_logger(self, logger_override: Logger | None = None) -> None:
        """Configure Python logging and metrics tracking backend.

        Initializes the logging infrastructure including Python logger, optional
        TorchRL metrics backend (TensorBoard, WandB, etc.), and video recording.

        Args:
            logger_override: Optional external TorchRL Logger to use instead of
                creating one from config. Useful for sharing loggers across components.
        """

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
        """Log scalar metrics to configured backends (TensorBoard, WandB, console).

        Sends metrics to the configured TorchRL logger and optionally prints
        formatted metrics to the Python logger for console monitoring.

        Args:
            metrics: Dictionary mapping metric names to scalar values.
            step: Training step/iteration number for x-axis alignment.
            log_python: Whether to also log to Python logger. If None, uses
                config.logger.log_to_console setting.
            python_level: Logging level for Python logger output (default: INFO).

        Example:
            >>> self.log_metrics(
            ...     {"loss": 0.5, "reward": 100},
            ...     step=1000,
            ...     log_python=True
            ... )
        """

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
        """Execute the main training loop for the algorithm.

        Implement this method in subclasses to define the complete training procedure
        including data collection, loss computation, optimization steps, and logging.

        The training loop should:
        1. Collect batches of experience using self.collector
        2. Process data and compute losses via self.loss_module
        3. Perform gradient updates with self.optim
        4. Log metrics periodically
        5. Save checkpoints as needed
        6. Handle episode termination and reset logic

        Raises:
            NotImplementedError: If not implemented by subclass.
        """

    def _print_model_overview(self) -> None:
        """Print a comprehensive overview of model architecture and components.

        Logs network architectures for all initialized components (policy, value,
        Q-networks) to help with debugging and verification.
        """
        self.log.info("Model Overview:\n%s", str(self))
        self.log.info("Policy Network:\n%s", str(self.policy))

        if self.config.value_function:
            self.log.info("Value Network:\n%s", str(self.value_function))
        if self.config.q_function:
            self.log.info("Q Network:\n%s", str(self.q_function))

    @abstractmethod
    def predict(self, obs: Tensor) -> Tensor:
        """Predict action(s) for given observation(s) using the trained policy.

        Implement this method for inference/deployment. Should handle both single
        observations and batched inputs appropriately.

        Args:
            obs: Observation tensor(s) from the environment. Shape depends on
                observation space (e.g., [obs_dim] or [batch, obs_dim]).

        Returns:
            Action tensor(s). Shape depends on action space (e.g., [action_dim]
            or [batch, action_dim]).

        Note:
            For stochastic policies, this typically returns the mode or mean
            of the action distribution rather than sampling.
        """

    def save_model(
        self, path: str | Path | None = None, step: int | None = None
    ) -> None:
        """Save model state and training components to disk.

        Persists state dictionaries for all networks, optimizer, and environment
        statistics (if using normalization). Supports both single-file checkpoints
        and versioned checkpoints with step numbers.

        Args:
            path: Target file path or directory. If None, saves to log directory.
                If a directory, creates "model.pt" or "model_step_{step}.pt" inside.
                If a file path with suffix, uses that name (with optional step suffix).
            step: Optional training step number to append to filename for versioning.
                Creates separate checkpoint files for different training stages.

        Note:
            Saves policy, value/Q networks, feature extractor (if used), optimizer state,
            and VecNorm statistics (if environment uses observation normalization).

        Example:
            >>> algorithm.save_model("checkpoints/ppo_model.pt")
            >>> algorithm.save_model("checkpoints", step=10000)  # -> model_step_10000.pt
        """
        default_dir = self.log_dir
        target_base = Path(path).expanduser() if path else default_dir
        base_exists = target_base.exists()
        base_is_file = base_exists and target_base.is_file()
        has_suffix = bool(target_base.suffix)

        if step:
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
            "policy_state_dict": (self.policy.state_dict() if self.policy else {}),
            "optimizer_state_dict": self.optim.state_dict(),
        }
        if self.value_function:
            data_to_save["value_state_dict"] = self.value_function.state_dict()
        if self.q_function:
            data_to_save["q_state_dict"] = self.q_function.state_dict()
        if self.feature_extractor:
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
        """Load model state and training components from a saved checkpoint.

        Restores network parameters, optimizer state, and environment statistics
        from a previously saved checkpoint file. Allows resuming training or
        deploying trained models.

        Args:
            path: Path to the checkpoint file (.pt) to load.

        Note:
            Only loads components that exist in both the checkpoint and current model.
            Missing keys are safely skipped. Loads to the device specified in config.

        Example:
            >>> algorithm = PPOAlgorithm(env, config)
            >>> algorithm.load_model("checkpoints/model_step_10000.pt")
            >>> algorithm.train()  # Resume training
        """
        data = torch.load(path, map_location=self.device)
        if self.policy and "policy_state_dict" in data:
            self.policy.load_state_dict(data["policy_state_dict"])  # type: ignore[arg-type]
        if self.value_function and "value_state_dict" in data:
            self.value_function.load_state_dict(data["value_state_dict"])  # type: ignore[arg-type]
        if self.q_function and "q_state_dict" in data:
            self.q_function.load_state_dict(data["q_state_dict"])  # type: ignore[arg-type]
        if "optimizer_state_dict" in data:
            self.optim.load_state_dict(data["optimizer_state_dict"])  # type: ignore[arg-type]
        if self.config.feature_extractor and "feature_extractor_state_dict" in data:
            self.feature_extractor.load_state_dict(data["feature_extractor_state_dict"])  # type: ignore[arg-type]
        if hasattr(self.env, "normalize_obs") and "vec_norm_msg" in data:
            self.env.load_state_dict(data["vec_norm_msg"])  # type: ignore[arg-type]

    def _refresh_parameter_monitor(self) -> None:
        """Build a registry of all trainable parameters for numerical stability monitoring.

        Collects references to all parameters from core modules (actor_critic, loss_module,
        feature_extractor) while avoiding duplicates. Used by validation methods to
        efficiently check for NaN/Inf values during training.

        Note:
            Should be called after network initialization and when architecture changes.
            Automatically filters out non-floating-point parameters.
        """
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
        """Verify that all monitored parameters contain finite values.

        Checks all registered parameters for NaN or Inf values, which typically
        indicate numerical instability, exploding gradients, or improper initialization.

        Args:
            stage: Descriptive label for the current training stage (e.g., "loss_computation",
                "after_optimizer_step") for informative error messages.

        Raises:
            FloatingPointError: If any parameter contains NaN or Inf values, with
                detailed information about the problematic parameter.

        Note:
            Call this at critical points in training (after loss computation, after
            optimizer steps) to catch numerical issues early.
        """
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
        """Verify that parameter gradients are finite.

        Checks gradients of all monitored parameters for NaN or Inf values,
        which can occur due to vanishing/exploding gradients, numerical issues
        in loss computation, or improper backpropagation.

        Args:
            stage: Descriptive label for the current training stage for error context.
            raise_error: If True, raises exception on detection. If False, logs
                warnings and returns False.

        Returns:
            True if all gradients are finite, False otherwise.

        Raises:
            FloatingPointError: If raise_error=True and non-finite gradients detected.

        Note:
            Typically called after loss.backward() but before optimizer.step() to
            prevent corrupted parameter updates.
        """
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
        """Recursively validate that all floating-point tensors in a TensorDict are finite.

        Performs deep inspection of TensorDict structures to detect NaN or Inf values
        in observations, actions, rewards, or other intermediate computations.

        Args:
            td: TensorDict to validate (may contain nested TensorDicts).
            stage: Training stage label for error context.
            prefix: Tuple of parent keys for nested path construction (internal use).
            raise_error: If True, raises exception on detection. If False, logs and returns False.

        Returns:
            True if all tensors are finite, False otherwise.

        Raises:
            FloatingPointError: If raise_error=True and non-finite values detected.

        Note:
            Useful for validating batches from replay buffer, rollout data, or loss outputs.
        """

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
        """Replace non-finite loss values with zeros to prevent gradient corruption.

        Defensive mechanism to handle numerical instabilities in loss computation.
        Replaces NaN/Inf loss terms with zero while logging warnings for debugging.

        Args:
            loss_td: TensorDict containing loss components from loss_module.
            stage: Training stage label for logging context.

        Returns:
            Sanitized TensorDict with non-finite values replaced by zeros.

        Warning:
            This is a fallback for numerical issues. Frequent sanitization indicates
            underlying problems (e.g., unstable hyperparameters, gradient explosion)
            that should be addressed at the source.

        Note:
            Operates in-place on the TensorDict structure.
        """

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

    @staticmethod
    def _reduce_log_prob(log_prob: Tensor, action: Tensor) -> Tensor:
        if log_prob.ndim == action.ndim:
            return log_prob.sum(dim=-1)
        return log_prob

    def _prepare_kl_context(
        self, sampled_tensordict: TensorDict, policy_op: TensorDictModule
    ) -> dict[str, Tensor]:
        obs_td = sampled_tensordict.select(*policy_op.in_keys).detach()
        if "loc" not in sampled_tensordict.keys(True) or "scale" not in sampled_tensordict.keys(True):
            msg = "Expected 'loc' and 'scale' in sampled_tensordict for KL computation."
            raise KeyError(msg)
        return {
            "obs_td": obs_td,
            "old_loc": sampled_tensordict.get("loc").detach(),
            "old_scale": sampled_tensordict.get("scale").detach(),
        }

    def _compute_kl_after_update(
        self, kl_context: dict[str, Tensor], policy_op: TensorDictModule
    ) -> Tensor | None:
        with torch.no_grad():
            obs_td = kl_context["obs_td"].clone()
            obs_td = policy_op(obs_td)
            new_loc = obs_td.get("loc")
            new_scale = obs_td.get("scale")
            old_loc = kl_context["old_loc"]
            old_scale = kl_context["old_scale"]

            var_old = old_scale.pow(2)
            var_new = new_scale.pow(2)
            kl = torch.log(new_scale / old_scale) + (
                var_old + (old_loc - new_loc).pow(2)
            ) / (2.0 * var_new) - 0.5
            if kl.ndim > 0:
                kl = kl.sum(dim=-1)
            kl_approx = kl.mean()
        if not torch.isfinite(kl_approx):
            return None
        return kl_approx

    def _maybe_adjust_lr(self, kl_approx: Tensor, schedule_cfg: Any) -> None:
        schedule = (getattr(schedule_cfg, "scheduler", "") or "").lower()
        if schedule != "adaptive":
            return
        kl_value = float(kl_approx.detach().mean().cpu().item())
        if not np.isfinite(kl_value) or kl_value <= 0.0:
            return

        desired_kl = float(getattr(schedule_cfg, "desired_kl", 0.01))
        factor = float(getattr(schedule_cfg, "lr_adaptation_factor", 1.5))
        min_lr = getattr(schedule_cfg, "min_lr", None)
        max_lr = getattr(schedule_cfg, "max_lr", None)

        lr = float(self.optim.param_groups[0]["lr"])
        if kl_value > desired_kl * 2.0:
            lr /= factor
        elif kl_value < desired_kl / 2.0:
            lr *= factor
        else:
            return

        if min_lr is not None:
            lr = max(lr, float(min_lr))
        if max_lr is not None:
            lr = min(lr, float(max_lr))

        for group in self.optim.param_groups:
            group["lr"] = lr
