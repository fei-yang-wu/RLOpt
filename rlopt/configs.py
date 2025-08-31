from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal


@dataclass
class EnvConfig:
    """Environment configuration for RLOpt  ."""

    env_name: Any = "Pendulum-v1"
    """Name of the environment."""

    num_envs: int = 1
    """Number of environments to simulate."""

    device: str = "cpu"
    """Device to run the environment on."""


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

    exp_name: Any = "RLOpt"
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

    mini_batch_size: int = 256
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

    warmup: int = 1
    """Number of warmup iterations when compiling policies.
    Used by collectors that accept a warmup parameter.
    """


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
class ActionValueNetConfig:
    """Action-value (Q) network configuration for RLOpt."""

    num_cells: ClassVar[list[int]] = [512, 256, 128]
    """Number of cells in each layer."""


@dataclass
class FeatureExtractorConfig:
    """Feature extractor configuration for RLOpt  ."""

    num_cells: ClassVar[list[int]] = [512, 256, 128]
    """Number of cells in each layer."""

    output_dim: int = 128
    """Output dimension of the feature extractor."""


# ------------------------------
# Advanced network configuration
# ------------------------------


@dataclass
class MLPBlockConfig:
    """Config for an MLP block (torso or head)."""

    num_cells: list[int]
    activation: Literal["relu", "elu", "tanh", "gelu"] = "elu"
    init: Literal["orthogonal", "xavier_uniform", "kaiming_uniform"] = "orthogonal"
    layer_norm: bool = False
    dropout: float = 0.0


@dataclass
class LSTMBlockConfig:
    """Config for an LSTM block (torso)."""

    hidden_size: int
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0


@dataclass
class CNNBlockConfig:
    """Config for a CNN block (torso for pixel inputs)."""

    channels: list[int]
    kernels: list[int]
    strides: list[int]
    paddings: list[int]
    activation: Literal["relu", "elu", "tanh", "gelu"] = "relu"


@dataclass
class FeatureBlockSpec:
    """Feature block spec with type-discriminated config.

    One of mlp, lstm, or cnn may be set depending on `type`.
    """

    type: Literal["mlp", "lstm", "cnn"] = "mlp"
    mlp: MLPBlockConfig | None = None
    lstm: LSTMBlockConfig | None = None
    cnn: CNNBlockConfig | None = None
    output_dim: int = 128


@dataclass
class ModuleNetConfig:
    """Full module network config with optional feature sharing.

    If `feature_ref` is provided, it references a named feature in `SharedFeatures.features`.
    If `feature_ref` is None and `feature` is provided, a private feature is built.
    The `head` typically refers to an MLP used after features (policy/value/Q head).
    """

    feature_ref: str | None = None
    feature: FeatureBlockSpec | None = None
    head: MLPBlockConfig | None = None
    in_keys: list[str] = field(default_factory=lambda: ["hidden"])  # after FE
    out_key: str = "hidden"


@dataclass
class SharedFeatures:
    """Registry of shared feature extractors that modules can reference by name."""

    features: dict[str, FeatureBlockSpec] = field(default_factory=dict)


@dataclass
class CriticConfig:
    """Generic critic configuration (state-value or action-value).

    - Supports multiple critics (e.g., twin Q for SAC) via `num_nets`.
    - Supports shared or private torso across critics.
    - Supports optional target networks and update strategies.
    """

    template: ModuleNetConfig = field(default_factory=ModuleNetConfig)
    num_nets: int = 1
    shared_feature_ref: str | None = None
    # Target network options
    use_target: bool = True
    target_update: Literal["polyak", "hard"] = "polyak"
    polyak_eps: float = 0.995
    hard_update_interval: int = 1_000


@dataclass
class NetworkLayout:
    """High-level network layout for an agent.

    - PPO typically uses `policy` + `value` modules.
    - SAC uses `policy` + `q_ensemble` modules.
    """

    shared: SharedFeatures = field(default_factory=SharedFeatures)
    policy: ModuleNetConfig = field(default_factory=ModuleNetConfig)
    value: ModuleNetConfig | None = None
    critic: CriticConfig | None = None


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

    action_value_net: ActionValueNetConfig = field(default_factory=ActionValueNetConfig)
    """Action-value network configuration (used by off-policy agents such as SAC)."""

    feature_extractor: FeatureExtractorConfig = field(
        default_factory=FeatureExtractorConfig
    )
    """Feature extractor configuration."""

    # Optional advanced network layout. If provided, agents may use this
    # to build shared/private feature extractors and heads.
    network: NetworkLayout | None = None

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
