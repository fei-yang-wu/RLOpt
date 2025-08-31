from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Any, ClassVar


@dataclass
class EnvConfig:
    """Environment configuration for RLOpt  ."""

    env_name: Any = MISSING
    """Name of the environment."""

    device: str = "cuda:0"
    """Device to run the environment on."""

    num_envs: Any = MISSING
    """Number of environments to simulate."""


@dataclass
class CollectorConfig:
    """Data collector configuration for RLOpt  ."""

    num_collectors: int = 1
    """Number of data collectors."""

    frames_per_batch: int = 4096 * 12
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

    group_name: str | None = None
    """Group name for logging."""

    exp_name: Any = MISSING
    """Experiment name for logging."""

    test_interval: int = 1_000_000
    """Interval between test evaluations."""

    num_test_episodes: int = 5
    """Number of test episodes to run."""

    video: bool = False
    """Whether to record videos."""

    log_dir: str = "logs"
    """Directory to save logs."""

    save_path: str = "models"
    """Path to save the model."""


@dataclass
class OptimizerConfig:
    """Optimizer configuration for RLOpt  ."""

    lr: float = 3e-4
    """Learning rate."""

    weight_decay: float = 0.0
    """Weight decay for optimizer."""

    anneal_lr: bool = True
    """Whether to anneal learning rate."""

    device: str = "cuda:0"
    """Device for optimizer."""

    target_update_polyak: float = 0.995
    """Polyak averaging coefficient for target network updates."""


@dataclass
class LossConfig:
    """Loss function configuration for RLOpt  ."""

    gamma: float = 0.99
    """Discount factor."""

    mini_batch_size: Any = MISSING
    """Mini-batch size for training."""

    epochs: int = 4
    """Number of training epochs."""

    loss_critic_type: str = "l2"
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


@dataclass
class PolicyConfig:
    """Policy network configuration for RLOpt  ."""

    num_cells: ClassVar[list[int]] = [512, 256, 128]
    """Number of cells in each layer."""


@dataclass
class ValueNetConfig:
    """Value network configuration for RLOpt  ."""

    num_cells: ClassVar[list[int]] = [512, 256, 128]
    """Number of cells in each layer."""


@dataclass
class FeatureExtractorConfig:
    """Feature extractor configuration for RLOpt  ."""

    num_cells: ClassVar[list[int]] = [512, 256, 128]
    """Number of cells in each layer."""

    output_dim: int = 128
    """Output dimension of the feature extractor."""


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

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    """Policy network configuration."""

    value_net: ValueNetConfig = field(default_factory=ValueNetConfig)
    """Value network configuration."""

    feature_extractor: FeatureExtractorConfig = field(
        default_factory=FeatureExtractorConfig
    )
    """Feature extractor configuration."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """Trainer configuration."""

    use_feature_extractor: bool = True
    """Whether to use a feature extractor."""

    use_value_function: bool = True
    """Whether to use a value function. 
    If use action_value function, then q network is used."""

    device: str = "cuda:0"
    """Device for training."""

    seed: int = 0
    """Random seed."""

    save_interval: int = 500
    """Interval for saving the model."""

    policy_in_keys: ClassVar[list[str]] = ["hidden"]
    """Keys to use for the policy."""

    value_net_in_keys: ClassVar[list[str]] = ["hidden"]
    """Keys to use for the value network."""

    total_input_keys: ClassVar[list[str]] = ["policy"]
    """Keys to use for the total input."""
