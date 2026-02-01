from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class EnvConfig:
    """Environment configuration for RLOpt  ."""

    env_name: Any = "HalfCheetah-v4"
    """Name of the environment."""

    num_envs: int = 1
    """Number of environments to simulate."""

    device: str = "auto"
    """Device to run the environment on."""

    library: str = "gymnasium"
    """Library to use for the environment."""


@dataclass
class CollectorConfig:
    """Data collector configuration for RLOpt  ."""

    num_collectors: int = 1
    """Number of data collectors."""

    frames_per_batch: int = 12
    """Number of frames per batch."""

    total_frames: int = 100_000_000
    """Total number of frames to collect."""

    set_truncated: bool = False
    """Whether to set truncated to True when the episode is done."""

    init_random_frames: int = 1000
    """Number of random frames to collect."""

    scratch_dir: str | None = None
    """Directory to save scratch data."""

    shared: bool = False
    """Whether the buffer will be shared using multiprocessing or not.."""

    prefetch: int | None = None
    """Number of prefetch batches."""


@dataclass
class ReplayBufferConfig:
    """Replay buffer configuration for RLOpt  ."""

    size: int = 1_000_000
    """Size of the replay buffer."""

    prb: bool = False
    """Whether to use a prioritized replay buffer."""

    scratch_dir: str | None = None
    """Directory to save scratch data."""

    prefetch: int = 3
    """Number of prefetch batches."""


@dataclass
class LoggerConfig:
    """Logger configuration for RLOpt  ."""

    backend: str = "wandb"
    """Logger backend to use."""

    project_name: str = "RLOpt"
    """Project name for logging."""

    entity: str | None = None
    """W&B entity (username or team name) for logging."""

    group_name: str | None = None
    """Group name for logging."""

    exp_name: Any = "RLOpt"
    """Experiment name for logging."""

    test_interval: int = 1_000_000
    """Interval between test evaluations."""

    num_test_episodes: int = 5
    """Number of test episodes to run."""

    video: bool = False
    """Whether to record videos."""

    log_dir: str = "logs"
    """Base directory for logging. Structure: {log_dir}/{algorithm}/{env_name}/{timestamp}/

    Default creates: ./logs/SAC/Pendulum-v1/2025-10-27_19-49-59/
    """

    save_path: str = "models"
    """Path to save model checkpoints (relative to run directory)."""

    python_level: str | None = None
    """Overrides :attr:`RLOptConfig.log_level` for standard Python logging when provided."""

    log_to_console: bool = True
    """Whether to emit Python logs to the console."""

    console_use_rich: bool = True
    """Attempt to use ``rich``'s console handler when available for better readability."""

    console_format: str = "%(message)s"
    """Logging format string for console output (ignored when ``rich`` handler is used)."""

    log_to_file: bool = True
    """Whether to persist Python logs to a file."""

    file_name: str = "rlopt.log"
    """Filename (relative to ``log_dir``) for file logging."""

    file_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    """Logging format string for the file handler."""

    file_rotation_bytes: int = 10_000_000
    """Rotate the log file once it reaches this many bytes (<=0 disables rotation)."""

    file_backup_count: int = 5
    """Number of rotated log files to keep when rotation is enabled."""


@dataclass
class OptimizerConfig:
    """Optimizer configuration for RLOpt."""

    optimizer: str = "adamw"
    """Name of the optimizer to use (e.g. ``"adamw"``, ``"adam"``, ``"sgd"``)."""

    lr: float = 3e-4
    """Base learning rate."""

    weight_decay: float = 0.0
    """Weight decay applied to all parameter groups."""

    optimizer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"betas": (0.9, 0.999), "eps": 1e-8}
    )
    """Extra keyword arguments forwarded to the optimizer (defaults tailored for Adam-family optimizers)."""

    scheduler: str | None = "steplr"
    """Learning-rate schedule. Use ``"adaptive"`` for KL-based adjustment or a
    PyTorch scheduler name (e.g. ``"steplr"``, ``"cosineannealinglr"``)."""

    scheduler_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"step_size": 1_000, "gamma": 0.9}
    )
    """Keyword arguments passed to the scheduler constructor."""

    scheduler_step: Literal["update", "epoch"] = "update"
    """Whether to step the scheduler after each optimizer update or once per epoch."""

    device: str = "auto"
    """Device for optimizer state when applicable."""

    target_update_polyak: float = 0.995
    """Polyak averaging coefficient for target network updates."""

    max_grad_norm: float | None = 0.5
    """Maximum gradient norm for clipping; set to ``None`` to disable clipping."""

    desired_kl: float = 0.01
    """Target KL divergence for adaptive learning rate."""

    lr_adaptation_factor: float = 1.5
    """Factor to scale LR up/down for adaptive schedule."""

    min_lr: float | None = 1e-6
    """Lower bound for adaptive learning rate. None disables clamping."""

    max_lr: float | None = 1e-2
    """Upper bound for adaptive learning rate. None disables clamping."""


@dataclass
class LossConfig:
    """Loss function configuration for RLOpt  ."""

    gamma: float = 0.99
    """Discount factor."""

    mini_batch_size: int = 256
    """Mini-batch size for training."""

    epochs: int = 4
    """Number of training epochs."""

    loss_critic_type: str = "smooth_l1"
    """Type of critic loss."""


@dataclass
class CompileConfig:
    """Compilation configuration for RLOpt  ."""

    compile: bool = False
    """Whether to compile the model."""

    compile_mode: str = "default"
    """Compilation mode."""

    cudagraphs: bool = False
    """Whether to use CUDA graphs."""

    warmup: int = 1
    """Number of warmup iterations when compiling policies.
    Used by collectors that accept a warmup parameter.
    """


@dataclass
class PolicyConfig:
    """Policy network configuration for RLOpt  ."""

    num_cells: list[int] = field(
        default_factory=lambda: [256, 256]
    )  # Match TorchRL stable architecture
    """Number of cells in each layer."""

    default_policy_scale: float = 1.0  # Match TorchRL default
    """Default policy scale."""


@dataclass
class ValueNetConfig:
    """Value network configuration for RLOpt  ."""

    num_cells: list[int] = field(
        default_factory=lambda: [256, 256]
    )  # Match TorchRL stable architecture
    """Number of cells in each layer."""


@dataclass
class ActionValueNetConfig:
    """Action-value (Q) network configuration for RLOpt."""

    num_cells: list[int] = field(
        default_factory=lambda: [256, 256]
    )  # Match TorchRL stable architecture
    """Number of cells in each layer."""


@dataclass
class FeatureExtractorConfig:
    """Feature extractor configuration for RLOpt  ."""

    num_cells: list[int] = field(
        default_factory=lambda: [256, 256]
    )  # Match TorchRL stable architecture
    """Number of cells in each layer."""

    output_dim: int = 256  # Match TorchRL stable architecture
    """Output dimension of the feature extractor."""


@dataclass
class NetworkConfig:
    """Network configuration for RLOpt  ."""

    num_cells: list[int] = field(
        default_factory=lambda: [256, 128, 128]
    )  # Match TorchRL stable architecture
    """Number of cells in each layer."""

    input_dim: int | None = None
    """Input dimension of the network. Defaults to lazy initialization if None."""

    output_dim: int = 128
    """Output dimension of the feature extractor."""

    input_keys: list[str] = field(default_factory=lambda: ["observation"])
    """Input keys for the network."""

    output_keys: list[str] = field(default_factory=list)
    """Output keys for the network."""

    activation_fn: str = "elu"
    """Activation function."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments for the network."""


@dataclass
class FeatureExtractorNetworkConfig(NetworkConfig):
    """Feature network config with type-discriminated config.

    One of mlp, lstm, or cnn may be set depending on `type`.
    """

    type: Literal["mlp", "lstm", "cnn"] = "mlp"
    # Feature extractor configuration for RLOpt.

    mlp: NetworkConfig | None = None

    lstm: NetworkConfig | None = None

    cnn: NetworkConfig | None = None

    output_dim: int = 256  # Match TorchRL stable architecture


@dataclass
class TrainerConfig:
    """Trainer configuration for RLOpt  ."""

    optim_steps_per_batch: int = 10
    """Number of optimization steps per batch."""

    clip_grad_norm: bool = True
    """Whether to clip gradient norm."""

    clip_norm: float = 0.5
    """Gradient clipping norm."""

    progress_bar: bool = True
    """Whether to show progress bar."""

    save_trainer_interval: int = 10_000
    """Interval for saving trainer."""

    log_interval: int = 1000
    """Interval for logging."""

    save_trainer_file: str | None = None
    """File to save trainer to."""

    frame_skip: int = 1
    """Frame skip for training."""


@dataclass
class RLOptConfig:
    """Main configuration class for RLOpt  ."""

    env: EnvConfig = field(default_factory=EnvConfig)
    """Environment configuration."""

    collector: CollectorConfig = field(default_factory=CollectorConfig)
    """Data collector configuration."""

    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    """Replay buffer configuration."""

    logger: LoggerConfig = field(default_factory=LoggerConfig)
    """Logger configuration."""

    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer configuration."""

    loss: LossConfig = field(default_factory=LossConfig)
    """Loss function configuration."""

    compile: CompileConfig = field(default_factory=CompileConfig)
    """Compilation configuration."""

    policy: NetworkConfig = field(default_factory=NetworkConfig)
    """Policy network configuration."""

    value_function: NetworkConfig | None = None
    """Value network configuration."""

    q_function: NetworkConfig | None = None
    """Action-value network configuration (used by off-policy agents such as SAC)."""

    feature_extractor: FeatureExtractorNetworkConfig | None = None
    """Feature extractor configuration."""

    trainer: TrainerConfig | None = None
    """Trainer configuration."""

    device: str = "cuda:0"
    """Device for training."""

    seed: int = 42
    """Random seed."""

    log_level: str = "warning"
    """Verbosity for internal debug logging (e.g. ``"debug"``, ``"info"``)."""

    save_interval: int = 10
    """Interval for saving the model."""
