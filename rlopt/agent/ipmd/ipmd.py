from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import InteractionType
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.data import ReplayBuffer
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import MLP
from torchrl.record.loggers import Logger

from rlopt.agent.imitation.latent_commands import LatentCommandController
from rlopt.agent.imitation.latent_learning import (
    BaseLatentLearner,
    build_latent_learner,
)
from rlopt.agent.ipmd.utils import (
    IPMD_COMMAND_SOURCES,
    IPMD_REWARD_INPUT_TYPES,
    IPMD_REWARD_MODEL_TYPES,
    IPMD_REWARD_OUTPUT_ACTIVATIONS,
    RewardInputBlock,
    build_reward_input_blocks,
    normalize_ipmd_command_source,
    normalize_ipmd_reward_input_type,
    normalize_ipmd_reward_model_type,
    normalize_ipmd_reward_output_activation,
    require_non_negative,
    require_positive_int,
    required_batch_keys_from_reward_blocks,
    reward_alignment_metrics,
    reward_grad_penalty_from_input,
)
from rlopt.agent.ppo.ppo import (
    PPO,
    PPOConfig,
    PPOIterationData,
    PPORLOptConfig,
    PPOTrainingMetadata,
)
from rlopt.config_utils import (
    BatchKey,
    ObsKey,
    dedupe_keys,
    flatten_action_features_from_td,
    flatten_obs_features_from_td,
    infer_batch_shape,
    mapping_get_obs_value,
    normalize_batch_key,
    obs_key_feature_dim,
    obs_key_feature_ndim,
)
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info


@dataclass
class IPMDLatentLearningConfig:
    """Nested latent-learning configuration used by IsaacLab imitation tasks."""

    method: str = "patch_autoencoder"
    """Latent learning method to use."""

    posterior_input_keys: list[ObsKey] = field(default_factory=list)
    """Keys for posterior."""

    prior_input_keys: list[ObsKey] = field(default_factory=list)
    """Keys for prior. """

    encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    """Encoder hidden dimension if use NN. """

    encoder_activation: str = "elu"
    """Encoder activation if use NN. """

    decoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    """Decoder hidden dimension if use NN with a decoder attached, either for computing reconstructions or generating samples. """

    decoder_activation: str = "elu"
    """Decoder activation if use NN with a decoder attached, either for computing reconstructions or generating samples. """

    prior_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    """Prior network hidden dimension if use NN for the prior. E.g. for state-dependent Gaussian prior. """

    prior_activation: str = "elu"
    """Prior network activation if use NN for the prior. """

    patch_past_steps: int = 0
    patch_future_steps: int = 0
    """A patch is a sliding window over the expert trajectory segment. The past and future steps are defined relative to the current time step. For example, with patch_past_steps=2 and patch_future_steps=3, the patch at time t will include observations from time steps [t-2, t-1, t, t+1, t+2, t+3]. Will pad if not enough past or future steps are available. """

    lr: float = 3e-4
    """Learning rate for the latent learner. """

    grad_clip_norm: float = 1.0
    """Gradient clipping norm for the latent learner. """

    recon_coeff: float = 0.0
    """Reconstruction loss coefficient. Only used if the latent learning method involves a decoder and reconstruction loss. If recon_coeff = 0, we will not update encoder and decoder with the reconstruction loss, but let the encoder update through policy improvements. """

    weight_decay_coeff: float = 0.0
    """Weight decay coefficient for the latent learner parameters. """

    train_posterior_through_policy: bool = False
    """If True, recompute the latent via the encoder during the PPO policy loss so that
    policy-improvement gradients update the posterior encoder. Encoder parameters are added
    to the main PPO optimizer. Combining this with recon_coeff > 0 updates the encoder from
    both PPO and reconstruction simultaneously."""

    freeze_encoder: bool = False
    """If True, the reconstruction update steps only the decoder. The encoder forward in the
    reconstruction loss is detached and its parameters are excluded from the reconstruction
    optimizer. Use for decoder-only diagnostics (probe whether a fixed encoder preserves the
    information in posterior_input_keys)."""

    kl_coeff: float = 0.0
    """KL regularization penalty. """

    probe_enabled: bool = False
    probe_condition_on_state: bool = False
    probe_target_keys: list[ObsKey] = field(default_factory=list)
    probe_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    probe_activation: str = "elu"
    probe_lr: float = 3e-4
    probe_grad_clip_norm: float = 1.0
    probe_batch_size: int = 8192
    """Probe for analyzing the latent learner. """

    log_std_min: float = -5.0
    log_std_max: float = 2.0
    """Clamps applied to learned log_std heads (Gaussian latent models)."""

    uniformity_coeff: float = 0.0
    """Weight on the latent uniformity penalty for ``policy_mlp_embedding``."""

    # ----- VQ-VAE / discrete-skill latent settings (used by patch_vqvae) -----
    quantizer: str = "fsq"
    """Quantizer choice for ``patch_vqvae``.

    One of ``"fsq" | "vq_ema" | "gumbel" | "identity"``. ``"identity"``
    keeps the same temporal hold path but removes discrete quantization.
    """

    code_latent_dim: int | None = None
    """Latent code width before optional command-phase features.

    ``None`` uses ``ipmd.latent_dim`` when ``command_phase_mode="none"`` and
    ``ipmd.latent_dim - 2`` when ``command_phase_mode="sin_cos"``.
    """

    command_phase_mode: str = "none"
    """Optional phase features appended to held rollout commands.

    Supported values:
    - ``"none"``: publish only the held code latent.
    - ``"sin_cos"``: append ``sin(phase), cos(phase)`` within the current hold.
    """

    fsq_levels: list[int] = field(default_factory=lambda: [8, 8, 8, 5, 5])
    """Per-dimension level counts for FSQ. Effective codebook size = prod(levels)."""

    codebook_size: int = 512
    """Codebook size for ``vq_ema`` and ``gumbel`` quantizers."""

    codebook_embed_dim: int | None = None
    """Embedding dim for codebook entries. ``None`` defaults to ``IPMDConfig.latent_dim``."""

    commitment_coeff: float = 0.25
    """Commitment-loss weight for ``vq_ema`` (``beta`` in van den Oord 2017)."""

    ema_decay: float = 0.99
    """EMA decay for codebook updates in ``vq_ema``."""

    dead_code_reset_iters: int = 0
    """Cadence (in update calls) for reviving unused codes via reinit. 0 disables."""

    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.3
    gumbel_tau_anneal_iters: int = 200_000
    """Gumbel-Softmax temperature anneal schedule (linear)."""

    gumbel_hard: bool = True
    """Straight-through hard Gumbel forward, soft backward."""

    gumbel_kl_to_uniform_coeff: float = 0.0
    """``KL(q(c|x) || Uniform(K))`` regularizer (gumbel only)."""

    code_usage_entropy_coeff: float = 0.0
    """Marginal-code-usage entropy bonus to discourage collapse (gumbel only)."""

    action_recon_coeff: float = 0.0
    """If > 0, an extra decoder reconstructs ``expert_action`` from the quantized latent."""

    code_period: int = 30
    """Env steps a sampled code is held during rollout collection.

    Used by ``patch_vqvae`` when ``ipmd.command_source == "posterior"``.
    The generic latent command controller's ``latent_steps_min/max`` settings
    are bypassed in that posterior path.
    """

    latent_dropout_to_random_code_prob: float = 0.0
    """Reserved for random-code substitution during PPO updates."""

    def validate(self, *, command_dim: int) -> None:
        self.quantizer = str(self.quantizer).strip().lower()
        if self.quantizer not in {"fsq", "vq_ema", "gumbel", "identity"}:
            msg = (
                "ipmd.latent_learning.quantizer must be one of "
                "'fsq', 'vq_ema', 'gumbel', or 'identity', got "
                f"{self.quantizer!r}."
            )
            raise ValueError(msg)

        self.command_phase_mode = str(self.command_phase_mode).strip().lower()
        if self.command_phase_mode not in {"none", "sin_cos"}:
            msg = (
                "ipmd.latent_learning.command_phase_mode must be 'none' or "
                f"'sin_cos', got {self.command_phase_mode!r}."
            )
            raise ValueError(msg)

        self.code_period = require_positive_int(
            "ipmd.latent_learning.code_period", self.code_period
        )
        if self.code_latent_dim is not None:
            self.code_latent_dim = require_positive_int(
                "ipmd.latent_learning.code_latent_dim", self.code_latent_dim
            )

        phase_dim = 2 if self.command_phase_mode == "sin_cos" else 0
        code_dim = (
            int(self.code_latent_dim)
            if self.code_latent_dim is not None
            else int(command_dim) - phase_dim
        )
        if code_dim <= 0:
            msg = (
                "ipmd.latent_learning.code_latent_dim must leave a positive code "
                f"width after phase features, got command_dim={command_dim}, "
                f"phase_dim={phase_dim}."
            )
            raise ValueError(msg)
        if code_dim + phase_dim != int(command_dim):
            msg = (
                "ipmd.latent_dim must equal code_latent_dim plus phase feature "
                f"width, got latent_dim={command_dim}, code_latent_dim={code_dim}, "
                f"phase_dim={phase_dim}."
            )
            raise ValueError(msg)


@dataclass
class IPMDConfig(PPOConfig):
    """IPMD-specific configuration (PPO-based)."""

    use_latent_command: bool = True
    """Whether to use agent-managed latent commands."""

    latent_dim: int = 16
    """Dimension of the latent skill space."""

    latent_key: ObsKey = "latent_command"
    """Key for the latent skill."""

    latent_steps_min: int = 30
    """Minimum steps before resampling latent."""

    latent_steps_max: int = 120
    """Maximum steps before resampling latent."""

    command_source: str = "random"
    """Source used to generate rollout latent commands.

    Supported values:
    - ``"random"`` - sample unit latents uniformly at random
    - ``"posterior"`` - encode the current rollout posterior inputs and
      publish the latent state as the latent command
    """

    latent_learning: IPMDLatentLearningConfig = field(
        default_factory=IPMDLatentLearningConfig
    )
    """Nested latent-learning configuration used by IsaacLab imitation tasks."""

    diversity_bonus_coeff: float = 0.05
    """Coefficient for the diversity bonus."""

    diversity_target: float = 1.0
    """Target diversity level."""

    latent_uniformity_temperature: float = 2.0
    """Temperature for the latent uniformity loss."""

    # Reward estimator network and loss settings
    reward_model_type: str = "mlp"
    """Reward estimator parameterization.

    Supported values:
    - ``"mlp"``     - one monolithic MLP over ``reward_input_type`` features
    - ``"grouped"`` - one small MLP per reward-state head, sharing goal context
    """

    reward_input_type: str = "sas"
    """Input type for the reward estimator.

    Supported values:
    - ``"s"``   - current state only  r(s)
    - ``"s'"``  - next state only     r(s')
    - ``"sa"``  - state-action         r(s, a)
    - ``"sas"`` - state-action-state   r(s, a, s')  (default)
    """

    reward_input_keys: list[ObsKey] | None = None
    """Observation keys used for reward state inputs.

    If ``None`` (default), falls back to ``value_function.get_input_keys()``
    (then ``policy.get_input_keys()``). Set this to let the reward model
    consume a different subset of observation keys than policy / value.
    """

    reward_num_cells: tuple[int, ...] = (256, 256)
    """Hidden layer sizes for the reward estimator MLP."""

    reward_group_num_cells: tuple[int, ...] = (128, 128)
    """Hidden layer sizes for each grouped reward head."""

    reward_group_context_keys: list[ObsKey] | None = None
    """Shared observation keys concatenated into every grouped reward head.

    If ``None``, grouped mode looks for a reward input key whose final component
    is ``"reference_command"``.
    """

    reward_group_head_keys: list[ObsKey] | None = None
    """Observation keys that receive separate grouped reward heads.

    If ``None``, grouped mode uses all reward input keys except the context keys.
    """

    reward_group_head_weights: list[float] | None = None
    """Optional scalar weights used when summing grouped reward heads."""

    reward_activation: str = "elu"
    """Activation for reward estimator MLP."""

    reward_init: str = "orthogonal"
    """Weight init for reward estimator."""

    reward_output_activation: str = "tanh"
    """Output activation applied to the reward estimator.

    Supported values:
    - ``"none"``    - no activation (unbounded, default)
    - ``"tanh"``    - tanh scaled by ``reward_output_scale``
    - ``"sigmoid"`` - sigmoid scaled by ``reward_output_scale``
    """

    reward_output_scale: float = 1.0
    """Scale factor for bounded output activations (tanh / sigmoid)."""

    reward_loss_coeff: float = 1.0
    """Scale for (sum r_pi - sum r_expert)."""

    reward_l2_coeff: float = 0.05
    """L2 regularization weight for reward parameters."""

    reward_grad_penalty_coeff: float = 0.5
    """Gradient penalty weight for reward parameters."""

    reward_detach_features: bool = True
    """Detach features when computing reward loss (avoid leaking grads)."""

    reward_lr: float | None = None
    """Learning rate for the reward estimator optimizer.

    If ``None``, reuses ``optim.lr``.
    """

    reward_updates_per_policy_update: int = 1
    """Number of reward-estimator updates to run when a reward step is due."""

    reward_update_interval: int = 1
    """Run reward-estimator updates every N policy updates."""

    reward_update_warmup_updates: int = 0
    """Number of policy updates to complete before reward updates begin."""

    reward_logit_reg_coeff: float = 0.0
    """L2 penalty coefficient on reward logits before output activation."""

    reward_param_weight_decay_coeff: float = 0.0
    """Explicit L2 penalty coefficient on reward-estimator parameters."""

    use_estimated_rewards_for_ppo: bool = False
    """Whether to use estimated rewards instead of environment rewards for PPO (GAE + value target).

    Default is False. Set to True to train PPO on estimated rewards.
    """

    estimated_reward_clamp_min: float | None = -np.inf
    """Optional lower bound applied to estimated rewards before PPO reward mixing.

    Set to ``None`` to disable lower clipping.
    """

    estimated_reward_clamp_max: float | None = np.inf
    """Optional upper bound applied to estimated rewards before PPO reward mixing.

    Set to ``None`` to disable upper clipping.
    """

    est_reward_weight: float = 0.3
    """Linear mixing coefficient for estimated rewards in PPO.
    """

    env_reward_weight: float = 1.0
    """Linear mixing coefficient for environment rewards in PPO.
    """

    estimated_reward_done_penalty: float = 0.0
    """Penalty subtracted from estimated reward on terminal (non-truncated) steps.

    This is applied only when ``next.done`` is true and ``next.truncated`` is false.
    Set to 0 to disable.
    """

    expert_batch_size: int | None = None
    """Batch size for expert data sampling. If None, uses the same as mini_batch_size."""

    bc_coef: float = 0.0
    """Behavior cloning (MLE) loss coefficient on expert actions.

    When > 0, adds ``-bc_coef * mean(log_prob(expert_action | policy))``
    to each update step.  This regularises the policy toward expert actions
    during early training.  Set to 0 to disable (default).
    """

    def validate(self) -> None:
        self.command_source = normalize_ipmd_command_source(self.command_source)
        self.reward_model_type = normalize_ipmd_reward_model_type(
            self.reward_model_type
        )
        self.reward_input_type = normalize_ipmd_reward_input_type(
            self.reward_input_type
        )
        self.reward_output_activation = normalize_ipmd_reward_output_activation(
            self.reward_output_activation
        )
        self.latent_dim = require_positive_int("ipmd.latent_dim", self.latent_dim)
        self.latent_steps_min = require_positive_int(
            "ipmd.latent_steps_min", self.latent_steps_min
        )
        self.latent_steps_max = require_positive_int(
            "ipmd.latent_steps_max", self.latent_steps_max
        )
        if self.latent_steps_min > self.latent_steps_max:
            msg = (
                "ipmd.latent_steps_min must be <= ipmd.latent_steps_max, got "
                f"{self.latent_steps_min} > {self.latent_steps_max}."
            )
            raise ValueError(msg)
        self.latent_learning.validate(command_dim=self.latent_dim)
        require_non_negative("ipmd.diversity_bonus_coeff", self.diversity_bonus_coeff)
        require_non_negative("ipmd.diversity_target", self.diversity_target)
        if float(self.latent_uniformity_temperature) <= 0.0:
            msg = (
                "ipmd.latent_uniformity_temperature must be > 0, got "
                f"{self.latent_uniformity_temperature!r}."
            )
            raise ValueError(msg)
        if any(int(width) <= 0 for width in self.reward_num_cells):
            msg = (
                "ipmd.reward_num_cells must contain only positive widths, got "
                f"{self.reward_num_cells!r}."
            )
            raise ValueError(msg)
        if any(int(width) <= 0 for width in self.reward_group_num_cells):
            msg = (
                "ipmd.reward_group_num_cells must contain only positive widths, got "
                f"{self.reward_group_num_cells!r}."
            )
            raise ValueError(msg)
        if self.reward_model_type == "grouped" and self.reward_input_type != "s":
            msg = (
                "ipmd.reward_model_type='grouped' requires "
                "ipmd.reward_input_type='s' because grouped heads consume current "
                "reward-state observation slices."
            )
            raise ValueError(msg)
        if (
            self.reward_group_context_keys is not None
            and len(self.reward_group_context_keys) == 0
        ):
            msg = "ipmd.reward_group_context_keys cannot be empty when configured."
            raise ValueError(msg)
        if (
            self.reward_group_head_keys is not None
            and len(self.reward_group_head_keys) == 0
        ):
            msg = "ipmd.reward_group_head_keys cannot be empty when configured."
            raise ValueError(msg)
        if self.reward_group_head_weights is not None:
            normalized_weights: list[float] = []
            for weight in self.reward_group_head_weights:
                normalized_weight = require_non_negative(
                    "ipmd.reward_group_head_weights", weight
                )
                if not math.isfinite(normalized_weight):
                    msg = (
                        "ipmd.reward_group_head_weights must be finite, got "
                        f"{weight!r}."
                    )
                    raise ValueError(msg)
                normalized_weights.append(normalized_weight)
            self.reward_group_head_weights = normalized_weights
        if (
            self.reward_output_activation != "none"
            and float(self.reward_output_scale) <= 0.0
        ):
            msg = (
                "ipmd.reward_output_scale must be > 0 when reward_output_activation "
                f"is {self.reward_output_activation!r}, got {self.reward_output_scale!r}."
            )
            raise ValueError(msg)
        require_non_negative("ipmd.reward_loss_coeff", self.reward_loss_coeff)
        require_non_negative("ipmd.reward_l2_coeff", self.reward_l2_coeff)
        require_non_negative(
            "ipmd.reward_grad_penalty_coeff", self.reward_grad_penalty_coeff
        )
        if self.reward_lr is not None and float(self.reward_lr) <= 0.0:
            msg = f"ipmd.reward_lr must be > 0, got {self.reward_lr!r}."
            raise ValueError(msg)
        self.reward_updates_per_policy_update = require_positive_int(
            "ipmd.reward_updates_per_policy_update",
            self.reward_updates_per_policy_update,
        )
        self.reward_update_interval = require_positive_int(
            "ipmd.reward_update_interval",
            self.reward_update_interval,
        )
        if int(self.reward_update_warmup_updates) < 0:
            msg = (
                "ipmd.reward_update_warmup_updates must be >= 0, got "
                f"{self.reward_update_warmup_updates!r}."
            )
            raise ValueError(msg)
        require_non_negative("ipmd.reward_logit_reg_coeff", self.reward_logit_reg_coeff)
        require_non_negative(
            "ipmd.reward_param_weight_decay_coeff",
            self.reward_param_weight_decay_coeff,
        )
        require_non_negative("ipmd.est_reward_weight", self.est_reward_weight)
        require_non_negative("ipmd.env_reward_weight", self.env_reward_weight)
        require_non_negative("ipmd.bc_coef", self.bc_coef)
        if self.expert_batch_size is not None:
            self.expert_batch_size = require_positive_int(
                "ipmd.expert_batch_size", self.expert_batch_size
            )
        clamp_min = self.estimated_reward_clamp_min
        clamp_max = self.estimated_reward_clamp_max
        if clamp_min is not None and clamp_max is not None and clamp_min > clamp_max:
            msg = (
                "ipmd.estimated_reward_clamp_min must be <= "
                f"ipmd.estimated_reward_clamp_max, got {clamp_min!r} > {clamp_max!r}."
            )
            raise ValueError(msg)

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class IPMDRLOptConfig(PPORLOptConfig):
    """IPMD configuration extending PPORLOptConfig (PPO-based)."""

    ipmd: IPMDConfig = field(default_factory=IPMDConfig)
    """IPMD configuration."""


@dataclass(frozen=True)
class GroupedRewardHeadSpec:
    """One construction-time grouped reward-head specification."""

    name: str
    obs_key: ObsKey
    input_dim: int
    weight: float


@dataclass(frozen=True)
class GroupedRewardOutput:
    """Grouped reward forward outputs used by reward losses and diagnostics."""

    reward: Tensor
    head_rewards: dict[str, Tensor]
    head_logits: dict[str, Tensor]
    head_inputs: dict[str, Tensor]


class IPMD(PPO):
    """IPMD algorithm with PPO as the base RL algorithm.

    Uses the same on-policy rollout + GAE + multiple epochs over mini-batches as PPO,
    and adds a reward estimator r(s, a, s') trained with the IPMD objective
    (policy estimated return - expert estimated return). Optionally uses estimated
    rewards for PPO updates.
    """

    _REWARD_INPUT_TYPES = IPMD_REWARD_INPUT_TYPES
    _REWARD_MODEL_TYPES = IPMD_REWARD_MODEL_TYPES
    _REWARD_OUTPUT_ACTIVATIONS = IPMD_REWARD_OUTPUT_ACTIVATIONS
    _COMMAND_SOURCES = IPMD_COMMAND_SOURCES

    @staticmethod
    def _normalize_command_source(command_source: str) -> str:
        return normalize_ipmd_command_source(command_source)

    def __init__(
        self,
        env,
        config: IPMDRLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ):
        self.config: IPMDRLOptConfig = config
        self.config.ipmd.validate()
        self.env = env
        self._reward_model_type = self.config.ipmd.reward_model_type
        self._use_latent_command = bool(self.config.ipmd.use_latent_command)
        self._latent_key = normalize_batch_key(self.config.ipmd.latent_key)
        self._latent_dim = int(self.config.ipmd.latent_dim)
        self._latent_command_controller: LatentCommandController | None = None
        self._command_source = self._normalize_command_source(
            self.config.ipmd.command_source
        )

        if self._use_latent_command:
            self._latent_command_controller = LatentCommandController(
                env=env,
                latent_key=self._latent_key,
                latent_dim=self._latent_dim,
                latent_steps_min=int(self.config.ipmd.latent_steps_min),
                latent_steps_max=int(self.config.ipmd.latent_steps_max),
                discover_env_method=self._discover_env_method,
            )

        # Observation key groups can differ across policy, value, and reward model.
        self._policy_obs_keys: list[ObsKey] = [
            normalize_batch_key(key) for key in self.config.policy.get_input_keys()
        ]
        value_cfg = self.config.value_function
        self._value_obs_keys: list[ObsKey] = (
            [normalize_batch_key(key) for key in value_cfg.get_input_keys()]
            if value_cfg is not None
            else list(self._policy_obs_keys)
        )
        self._policy_obs_keys_without_latent: list[ObsKey] = (
            [key for key in self._policy_obs_keys if key != self._latent_key]
            if self._use_latent_command
            else list(self._policy_obs_keys)
        )
        reward_keys = self.config.ipmd.reward_input_keys
        if not reward_keys:
            reward_keys = (
                value_cfg.get_input_keys()
                if value_cfg is not None
                else self._policy_obs_keys_without_latent
            )
        self._reward_obs_keys: list[ObsKey] = dedupe_keys(
            [normalize_batch_key(key) for key in reward_keys]
        )
        latent_cfg = self.config.ipmd.latent_learning
        posterior_keys = latent_cfg.posterior_input_keys or self._reward_obs_keys
        self._posterior_obs_keys: list[ObsKey] = dedupe_keys(
            [normalize_batch_key(key) for key in posterior_keys]
        )
        self._prior_obs_keys: list[ObsKey] = dedupe_keys(
            [normalize_batch_key(key) for key in latent_cfg.prior_input_keys]
        )
        self._current_reference_obs_getter = None

        available_keys = set(env.observation_spec.keys(True))
        if self._use_latent_command:
            latent_present = self._latent_key in available_keys
            if not latent_present:
                msg = (
                    "IPMD use_latent_command=True requires the environment to expose "
                    f"the latent observation key {self._latent_key!r}."
                )
                msg += self._latent_mode_hint()
                raise ValueError(msg)
            if self._latent_key not in self._policy_obs_keys:
                msg = (
                    "IPMD use_latent_command=True requires the policy input keys to "
                    f"contain {self._latent_key!r}."
                )
                msg += self._latent_mode_hint()
                raise ValueError(msg)

        all_obs_keys = dedupe_keys(
            self._policy_obs_keys
            + self._value_obs_keys
            + self._reward_obs_keys
            + self._posterior_obs_keys
            + self._prior_obs_keys
        )
        for key in all_obs_keys:
            env.observation_spec[key]
        self._obs_feature_ndims: dict[ObsKey, int] = {
            key: obs_key_feature_ndim(self.env, key) for key in all_obs_keys
        }
        self._obs_feature_dims: dict[ObsKey, int] = {
            key: obs_key_feature_dim(self.env, key) for key in all_obs_keys
        }
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        action_shape = tuple(int(dim) for dim in action_spec.shape)
        self._action_feature_ndim = len(action_shape)
        self._action_feature_dim = int(math.prod(action_shape)) if action_shape else 1
        self._reward_input_blocks: tuple[RewardInputBlock, ...] = (
            build_reward_input_blocks(
                reward_input_type=self.config.ipmd.reward_input_type,
                reward_obs_keys=self._reward_obs_keys,
                obs_feature_dims=self._obs_feature_dims,
                action_feature_dim=self._action_feature_dim,
            )
        )
        self._reward_input_dim: int = sum(
            block.dim for block in self._reward_input_blocks
        )
        self._reward_group_context_keys: tuple[ObsKey, ...] = ()
        self._reward_group_head_specs: tuple[GroupedRewardHeadSpec, ...] = ()
        if self._reward_model_type == "grouped":
            (
                self._reward_group_context_keys,
                self._reward_group_head_specs,
            ) = self._resolve_grouped_reward_head_specs()
        self._reward_infers_latent_condition: bool = (
            self._use_latent_command
            and self._latent_key in self._reward_required_batch_keys()
        )

        # Reward estimator: r(s, a, s') -> scalar
        self.reward_estimator: torch.nn.Module = self._construct_reward_estimator()
        self.reward_estimator.to(self.device)
        self.reward_optim: torch.optim.Optimizer | None = None

        self._expert_batch_sampler: (
            Callable[[int, list[BatchKey]], TensorDict | None] | None
        ) = None

        self._latent_learner: BaseLatentLearner | None = None

        cfg = self.config.ipmd
        # Scalar caches
        self._reward_loss_coeff: float = float(cfg.reward_loss_coeff)
        self._reward_l2_coeff: float = float(cfg.reward_l2_coeff)
        self._reward_grad_penalty_coeff: float = float(cfg.reward_grad_penalty_coeff)
        self._reward_logit_reg_coeff: float = float(cfg.reward_logit_reg_coeff)
        self._reward_param_weight_decay_coeff: float = float(
            cfg.reward_param_weight_decay_coeff
        )
        self._reward_updates_per_policy_update: int = int(
            cfg.reward_updates_per_policy_update
        )
        self._reward_update_interval: int = int(cfg.reward_update_interval)
        self._reward_update_warmup_updates: int = int(cfg.reward_update_warmup_updates)
        self._reward_lr: float | None = (
            float(cfg.reward_lr) if cfg.reward_lr is not None else None
        )
        self._use_estimated_rewards_for_ppo: bool = bool(
            cfg.use_estimated_rewards_for_ppo
        )
        self._bc_coeff: float = float(cfg.bc_coef)
        self._reward_update_enabled: bool = any(
            coeff > 0.0
            for coeff in (
                self._reward_loss_coeff,
                self._reward_l2_coeff,
                self._reward_grad_penalty_coeff,
                self._reward_logit_reg_coeff,
                self._reward_param_weight_decay_coeff,
            )
        )
        self._expert_minibatch_update_enabled: bool = bool(
            self._bc_coeff > 0.0 or self._reward_update_enabled
        )
        self._reward_detach_features: bool = bool(cfg.reward_detach_features)
        max_grad = getattr(self.config.optim, "max_grad_norm", None)
        self._max_grad_norm: float = float(max_grad) if max_grad else 1e10
        self._reward_max_grad_norm: float = self._max_grad_norm
        self._reward_grad_clip_params: list[torch.nn.Parameter] = list(
            self.reward_estimator.parameters()
        )
        # Output activation as a callable (eliminates string dispatch at call time)
        out_act = cfg.reward_output_activation
        scale = float(cfg.reward_output_scale)
        if out_act == "tanh":
            self._reward_out_fn: Callable[[Tensor], Tensor] = lambda r: (
                torch.tanh(r) * scale
            )
        elif out_act == "sigmoid":
            self._reward_out_fn = lambda r: torch.sigmoid(r) * scale
        else:
            self._reward_out_fn = lambda r: r

        if self._use_latent_command:
            method = str(self.config.ipmd.latent_learning.method)
            self._latent_learner = build_latent_learner(method)
            self._latent_learner.initialize(self)
        self._validate_latent_reward_conditioning()

        super().__init__(
            env=env,
            config=config,
            policy_net=policy_net,
            value_net=value_net,
            q_net=q_net,
            replay_buffer=replay_buffer,
            logger=logger,
            feature_extractor_net=feature_extractor_net,
            **kwargs,
        )
        self._auto_attach_env_expert_sampler()
        self._validate_expert_transition_alignment()
        self._refresh_grad_clip_params()
        self._reward_grad_clip_params = list(self.reward_estimator.parameters())
        self._policy_operator = self.actor_critic.get_policy_operator()
        self._bc_debug_anomaly_prints = 0

    def _require_latent_command_controller(self) -> LatentCommandController:
        controller = self._latent_command_controller
        if controller is None:
            msg = (
                "IPMD latent-command controller is unavailable. "
                "Construct the agent with ipmd.use_latent_command=True."
            )
            raise RuntimeError(msg)
        return controller

    @property
    def collector_policy(self):
        """Return the collector policy, optionally stamping latent commands."""
        policy_operator = self.actor_critic.get_policy_operator()
        if not self._use_latent_command:
            return policy_operator
        controller = self._require_latent_command_controller()
        return controller.collector_policy(
            inject_fn=self._inject_latent_command,
            policy_module=policy_operator,
        )

    def _inject_latent_command(
        self,
        td: TensorDict,
    ) -> None:
        controller = self._require_latent_command_controller()
        if self._command_source == "random":
            controller.inject_random_latent_command(
                td,
                device=self.device,
                dtype=torch.float32,
            )
            return

        assert self._latent_learner is not None
        collector_hook = getattr(self._latent_learner, "infer_collector_latents", None)
        if callable(collector_hook):
            latents = collector_hook(td)
        else:
            latents = self._latent_learner.infer_batch_latents(
                td,
                detach=True,
                context="collector posterior latent",
            )
        if latents is None:
            msg = (
                "Active latent learner does not support collector-time posterior "
                "latent inference."
            )
            raise RuntimeError(msg)
        latents = latents.to(
            device=self.device,
            dtype=torch.float32,
        )
        td.set(
            self._latent_key,
            latents.reshape(*td.batch_size, self._latent_dim),
        )
        published_latents = latents.reshape(-1, self._latent_dim)
        controller.publish_latents_to_env(published_latents)

    def _expert_latents_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        assert self._latent_learner is not None
        return self._latent_learner.infer_expert_latents(
            td,
            detach=detach,
        ).to(self.device)

    def _validate_latent_reward_conditioning(self) -> None:
        """Validate reward-side latent conditioning at construction time."""
        if not self._reward_infers_latent_condition:
            return
        if self._latent_learner is None:
            msg = (
                "IPMD reward inputs include the latent key "
                f"{self._latent_key!r}, but no latent learner is active."
            )
            raise ValueError(msg)
        if (
            self._reward_model_type == "grouped"
            and self._latent_key not in self._reward_group_context_keys
        ):
            msg = (
                "Grouped IPMD reward inputs include the latent key "
                f"{self._latent_key!r}; configure it as "
                "ipmd.reward_group_context_keys so it is shared conditioning "
                "instead of a grouped reward head."
            )
            raise ValueError(msg)
        latent_head_keys = [
            spec.obs_key
            for spec in self._reward_group_head_specs
            if spec.obs_key == self._latent_key
        ]
        if latent_head_keys:
            msg = (
                "Grouped IPMD reward heads must not include the latent key "
                f"{self._latent_key!r}; use it only as reward group context."
            )
            raise ValueError(msg)

        expert_keys = self._latent_learner.required_expert_batch_keys()
        missing_expert_keys = [
            key for key in self._posterior_obs_keys if key not in expert_keys
        ]
        if missing_expert_keys:
            msg = (
                "Latent-conditioned reward updates require expert posterior input "
                f"keys {missing_expert_keys!r}, but the active latent learner does "
                "not declare them in required_expert_batch_keys()."
            )
            raise ValueError(msg)

    def _reward_sampler_required_keys(self) -> list[BatchKey]:
        """Return reward keys that must come directly from the expert sampler."""
        required = self._reward_required_batch_keys()
        if self._reward_infers_latent_condition:
            required = [key for key in required if key != self._latent_key]
        return required

    def _expert_required_keys(self) -> list[BatchKey]:
        """Return expert-batch keys required by current IPMD settings."""
        bc_enabled = float(self.config.ipmd.bc_coef) > 0.0

        required = self._reward_sampler_required_keys()
        if self._reward_infers_latent_condition:
            assert self._latent_learner is not None
            required.extend(self._latent_learner.required_expert_batch_keys())
        if bc_enabled:
            required.append("expert_action")
            required.extend(
                key
                for key in self._policy_obs_keys
                if not self._use_latent_command or key != self._latent_key
            )
            if self._use_latent_command:
                assert self._latent_learner is not None
                required.extend(self._latent_learner.required_expert_batch_keys())
        return dedupe_keys(required)

    def _validate_expert_transition_alignment(self) -> None:
        """Fail fast when expert next-state requests cannot be satisfied exactly."""
        requires_next_obs = any(
            isinstance(key, tuple) and len(key) > 0 and key[0] == "next"
            for key in self._expert_required_keys()
        )
        if not requires_next_obs or self._expert_batch_sampler is None:
            return

        aligned_next = getattr(self.env, "_reference_has_aligned_next", None)
        if aligned_next is None or bool(aligned_next):
            return

        msg = (
            "IPMD expert inputs require transition-aligned next observations, but "
            "the attached env.sample_expert_batch(...) does not expose aligned next "
            "reference transitions. Disable next_obs reward inputs or provide "
            "transition-aligned expert next state data."
        )
        raise ValueError(msg)

    def _reward_required_batch_keys(self) -> list[BatchKey]:
        """Return the batch keys needed to materialize reward-model inputs."""
        return required_batch_keys_from_reward_blocks(self._reward_input_blocks)

    @staticmethod
    def _obs_key_leaf_name(key: ObsKey) -> str:
        if isinstance(key, tuple):
            return str(key[-1])
        return str(key)

    @staticmethod
    def _metric_safe_name(value: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in value)
        return safe.strip("_") or "head"

    @classmethod
    def _reward_head_names_for_keys(cls, keys: list[ObsKey]) -> list[str]:
        leaf_names = [
            cls._metric_safe_name(cls._obs_key_leaf_name(key)) for key in keys
        ]
        if len(set(leaf_names)) == len(leaf_names):
            return leaf_names
        names: list[str] = []
        for key in keys:
            parts = key if isinstance(key, tuple) else (key,)
            names.append(cls._metric_safe_name("__".join(str(part) for part in parts)))
        if len(set(names)) != len(names):
            msg = (
                "Grouped IPMD reward head metric names are not unique. Configure "
                "reward_group_head_keys without duplicate names."
            )
            raise ValueError(msg)
        return names

    def _resolve_grouped_reward_head_specs(
        self,
    ) -> tuple[tuple[ObsKey, ...], tuple[GroupedRewardHeadSpec, ...]]:
        """Resolve grouped reward context and head specs at construction time."""
        cfg = self.config.ipmd
        reward_keys = list(self._reward_obs_keys)

        if cfg.reward_group_context_keys is None:
            context_keys = [
                key
                for key in reward_keys
                if self._obs_key_leaf_name(key) == "reference_command"
            ]
        else:
            context_keys = [
                normalize_batch_key(key) for key in cfg.reward_group_context_keys
            ]
        missing_context = [key for key in context_keys if key not in reward_keys]
        if missing_context:
            msg = (
                "ipmd.reward_group_context_keys must be included in "
                f"ipmd.reward_input_keys; missing {missing_context!r}."
            )
            raise ValueError(msg)
        if not context_keys:
            msg = (
                "ipmd.reward_model_type='grouped' requires reward group context. "
                "Include a reward_input key named 'reference_command' or configure "
                "ipmd.reward_group_context_keys explicitly."
            )
            raise ValueError(msg)

        if cfg.reward_group_head_keys is None:
            head_keys = [key for key in reward_keys if key not in context_keys]
        else:
            head_keys = [normalize_batch_key(key) for key in cfg.reward_group_head_keys]
        missing_heads = [key for key in head_keys if key not in reward_keys]
        if missing_heads:
            msg = (
                "ipmd.reward_group_head_keys must be included in "
                f"ipmd.reward_input_keys; missing {missing_heads!r}."
            )
            raise ValueError(msg)
        overlap = [key for key in head_keys if key in context_keys]
        if overlap:
            msg = (
                "ipmd.reward_group_head_keys must not overlap "
                f"ipmd.reward_group_context_keys; got {overlap!r}."
            )
            raise ValueError(msg)
        if not head_keys:
            msg = "ipmd.reward_model_type='grouped' requires at least one reward head."
            raise ValueError(msg)

        if cfg.reward_group_head_weights is None:
            weights = [1.0 for _ in head_keys]
        else:
            weights = [float(weight) for weight in cfg.reward_group_head_weights]
            if len(weights) != len(head_keys):
                msg = (
                    "ipmd.reward_group_head_weights must have one weight per grouped "
                    f"reward head, got {len(weights)} weights for {len(head_keys)} "
                    "heads."
                )
                raise ValueError(msg)

        context_dim = sum(int(self._obs_feature_dims[key]) for key in context_keys)
        names = self._reward_head_names_for_keys(head_keys)
        specs = tuple(
            GroupedRewardHeadSpec(
                name=name,
                obs_key=head_key,
                input_dim=context_dim + int(self._obs_feature_dims[head_key]),
                weight=weight,
            )
            for name, head_key, weight in zip(names, head_keys, weights, strict=True)
        )
        return tuple(context_keys), specs

    def _construct_reward_estimator(self) -> torch.nn.Module:
        """Create reward network whose input depends on ``reward_input_type``."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)

        if self._reward_model_type == "grouped":
            heads = torch.nn.ModuleDict()
            for spec in self._reward_group_head_specs:
                head = MLP(
                    in_features=spec.input_dim,
                    out_features=1,
                    num_cells=list(cfg.ipmd.reward_group_num_cells),
                    activation_class=get_activation_class(cfg.ipmd.reward_activation),
                    device=self.device,
                )
                self._initialize_weights(head, cfg.ipmd.reward_init)
                heads[spec.name] = head
            return heads

        net = MLP(
            in_features=self._reward_input_dim,
            out_features=1,
            num_cells=list(cfg.ipmd.reward_num_cells),
            activation_class=get_activation_class(cfg.ipmd.reward_activation),
            device=self.device,
        )
        self._initialize_weights(net, cfg.ipmd.reward_init)
        return net

    def _compile_components(self) -> None:
        """Compile reward estimator and update method with torch.compile (if enabled)."""
        if not self.config.compile.compile:
            return
        super()._compile_components()
        self.reward_estimator = torch.compile(self.reward_estimator)

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create the PPO optimizer and a dedicated reward-estimator optimizer."""
        joint_params: list[torch.nn.Parameter] = []
        if (
            self._use_latent_command
            and self._latent_learner is not None
            and bool(self.config.ipmd.latent_learning.train_posterior_through_policy)
        ):
            joint_params = list(self._latent_learner.joint_parameters())

        if not hasattr(self, "reward_estimator"):
            if not joint_params:
                return super()._set_optimizers(optimizer_cls, optimizer_kwargs)
            all_params = list(self.actor_critic.parameters()) + joint_params
            return [optimizer_cls(all_params, **optimizer_kwargs)]

        policy_params = list(self.actor_critic.parameters()) + joint_params
        reward_optimizer_kwargs = dict(optimizer_kwargs)
        reward_optimizer_kwargs["lr"] = (
            self._reward_lr
            if self._reward_lr is not None
            else float(reward_optimizer_kwargs["lr"])
        )
        reward_optimizer_kwargs["weight_decay"] = 0.0
        self.reward_optim = optimizer_cls(
            self.reward_estimator.parameters(),
            **reward_optimizer_kwargs,
        )
        return [optimizer_cls(policy_params, **optimizer_kwargs)]

    def _next_expert_batch(
        self,
        batch_size: int | None = None,
        required_keys: list[BatchKey] | None = None,
    ) -> TensorDict:
        assert isinstance(self.config, IPMDRLOptConfig)
        effective_batch_size = int(
            batch_size
            or self.config.ipmd.expert_batch_size
            or self.config.loss.mini_batch_size
        )

        if self._expert_batch_sampler is None:
            msg = (
                "IPMD training requires env.sample_expert_batch(...). "
                "Tests may install a private expert sampler override."
            )
            raise RuntimeError(msg)
        required_keys = (
            self._expert_required_keys() if required_keys is None else required_keys
        )
        expert_batch = self._expert_batch_sampler(
            effective_batch_size,
            required_keys,
        )
        if expert_batch is None:
            msg = "Expert sampler returned None."
            raise RuntimeError(msg)
        available_keys = set(expert_batch.keys(True))
        missing_keys = [key for key in required_keys if key not in available_keys]
        if missing_keys:
            msg = f"Expert sampler batch is missing required keys: {missing_keys!r}."
            raise RuntimeError(msg)
        if expert_batch.numel() != effective_batch_size:
            msg = (
                "Expert sampler must return exactly the requested batch size, got "
                f"{expert_batch.numel()} for request {effective_batch_size}."
            )
            raise RuntimeError(msg)
        return expert_batch.to(self.device)

    def _apply_estimated_reward_done_penalty(
        self,
        rollout: TensorDict,
        est_reward: Tensor,
    ) -> Tensor:
        """Subtract the configured penalty on terminal non-truncated transitions."""
        penalty = float(self.config.ipmd.estimated_reward_done_penalty)
        if penalty == 0.0:
            return est_reward
        done_key: BatchKey = ("next", "done")
        if done_key not in rollout.keys(True):
            return est_reward
        done = rollout[done_key]
        truncated_key: BatchKey = ("next", "truncated")
        truncated = (
            rollout[truncated_key] if truncated_key in rollout.keys(True) else None
        )
        done_mask = done.to(device=est_reward.device, dtype=torch.bool)
        if truncated is None:
            truncated_mask = torch.zeros_like(done_mask)
        else:
            truncated_mask = truncated.to(device=est_reward.device, dtype=torch.bool)
        terminal_mask = done_mask & ~truncated_mask
        while terminal_mask.ndim < est_reward.ndim:
            terminal_mask = terminal_mask.unsqueeze(-1)
        return torch.where(terminal_mask, est_reward - penalty, est_reward)

    def _reward_input_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
        requires_grad: bool,
    ) -> Tensor:
        """Materialize the canonical reward-model input tensor for one batch."""
        parts: list[Tensor] = []
        for block in self._reward_input_blocks:
            if block.kind == "obs":
                parts.append(
                    flatten_obs_features_from_td(
                        td,
                        list(block.obs_keys),
                        self._obs_feature_ndims,
                        next_obs=False,
                        detach=detach,
                    )
                )
            elif block.kind == "next_obs":
                parts.append(
                    flatten_obs_features_from_td(
                        td,
                        list(block.obs_keys),
                        self._obs_feature_ndims,
                        next_obs=True,
                        detach=detach,
                    )
                )
            elif block.kind == "action":
                parts.append(
                    flatten_action_features_from_td(
                        td,
                        self._action_feature_ndim,
                        detach=detach,
                    )
                )
            else:
                msg = f"Unhandled reward input block kind {block.kind!r}."
                raise RuntimeError(msg)

        if not parts:
            msg = "Reward input specification must contain at least one block."
            raise RuntimeError(msg)
        reward_input = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if requires_grad:
            reward_input = reward_input.detach().requires_grad_(True)
        return reward_input

    def _grouped_reward_outputs_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
        requires_grad: bool,
    ) -> GroupedRewardOutput:
        """Evaluate grouped reward heads on one TensorDict batch."""
        context = flatten_obs_features_from_td(
            td,
            list(self._reward_group_context_keys),
            self._obs_feature_ndims,
            next_obs=False,
            detach=detach,
        )
        assert isinstance(self.reward_estimator, torch.nn.ModuleDict)
        head_rewards: dict[str, Tensor] = {}
        head_logits: dict[str, Tensor] = {}
        head_inputs: dict[str, Tensor] = {}
        weighted_rewards: list[Tensor] = []
        for spec in self._reward_group_head_specs:
            head_features = flatten_obs_features_from_td(
                td,
                [spec.obs_key],
                self._obs_feature_ndims,
                next_obs=False,
                detach=detach,
            )
            head_input = torch.cat((context, head_features), dim=-1)
            if requires_grad:
                head_input = head_input.detach().requires_grad_(True)
            logits = self.reward_estimator[spec.name](head_input)
            reward = self._reward_out_fn(logits)
            head_inputs[spec.name] = head_input
            head_logits[spec.name] = logits
            head_rewards[spec.name] = reward
            weighted_rewards.append(reward * spec.weight)

        total_reward = weighted_rewards[0]
        for reward in weighted_rewards[1:]:
            total_reward = total_reward + reward
        return GroupedRewardOutput(
            reward=total_reward,
            head_rewards=head_rewards,
            head_logits=head_logits,
            head_inputs=head_inputs,
        )

    def _reward_from_td(
        self,
        td: TensorDict,
        *,
        batch_role: str,
        detach: bool | None = None,
        requires_grad: bool = False,
        return_input: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute estimated reward for a batch of transitions."""
        del batch_role
        if detach is None:
            detach = self._reward_detach_features
        if self._reward_model_type == "grouped":
            grouped = self._grouped_reward_outputs_from_td(
                td,
                detach=detach,
                requires_grad=requires_grad,
            )
            if return_input:
                grouped_input = torch.cat(
                    [
                        grouped.head_inputs[spec.name]
                        for spec in self._reward_group_head_specs
                    ],
                    dim=-1,
                )
                return grouped.reward, grouped_input
            return grouped.reward
        x = self._reward_input_from_td(
            td,
            detach=detach,
            requires_grad=requires_grad,
        )
        reward = self._reward_out_fn(self.reward_estimator(x))
        if return_input:
            return reward, x
        return reward

    def _reward_from_batch(
        self,
        td: TensorDict,
        *,
        batch_role: str,
        detach: bool | None = None,
        requires_grad: bool = False,
        return_input: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Backward-compatible alias for reward evaluation on one batch."""
        return self._reward_from_td(
            td,
            batch_role=batch_role,
            detach=detach,
            requires_grad=requires_grad,
            return_input=return_input,
        )

    def _reward_parameter_weight_decay(self) -> Tensor:
        """Return the explicit L2 penalty term over reward-estimator parameters."""
        if self._reward_param_weight_decay_coeff <= 0.0:
            return torch.zeros((), device=self.device)
        terms = [param.pow(2).sum() for param in self.reward_estimator.parameters()]
        if len(terms) == 0:
            return torch.zeros((), device=self.device)
        return torch.stack(terms).sum()

    def _reward_update_due(self, policy_update_idx: int) -> bool:
        """Return whether the reward estimator should update on this policy step."""
        if not self._reward_update_enabled:
            return False
        if policy_update_idx < self._reward_update_warmup_updates:
            return False
        relative_idx = policy_update_idx - self._reward_update_warmup_updates
        return relative_idx % self._reward_update_interval == 0

    def _latent_uniformity(self, latent_pred: Tensor) -> Tensor:
        """Not used. Placeholder."""
        return torch.zeros((), device=latent_pred.device, dtype=latent_pred.dtype)

    def _extra_actor_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        del batch
        return torch.zeros((), device=self.device), {}

    def _refresh_grad_clip_params(self) -> None:
        """Refresh the cached optimizer parameter list used for grad clipping."""
        self._grad_clip_params = [  # type: ignore[attr-defined]
            param for group in self.optim.param_groups for param in group["params"]
        ]

    def _empty_reward_metrics(self) -> dict[str, Tensor]:
        metrics = {
            "loss_reward_diff": torch.zeros((), device=self.device),
            "loss_reward_l2": torch.zeros((), device=self.device),
            "loss_reward_grad_penalty": torch.zeros((), device=self.device),
            "loss_reward_grad_penalty_batch": torch.zeros((), device=self.device),
            "loss_reward_grad_penalty_expert": torch.zeros((), device=self.device),
            "loss_reward_logit_reg": torch.zeros((), device=self.device),
            "loss_reward_param_weight_decay": torch.zeros((), device=self.device),
            "estimated_reward_mean": torch.zeros((), device=self.device),
            "estimated_reward_std": torch.zeros((), device=self.device),
            "expert_reward_mean": torch.zeros((), device=self.device),
            "expert_reward_std": torch.zeros((), device=self.device),
        }
        for key in self._grouped_reward_loss_metric_keys():
            metrics[key] = torch.zeros((), device=self.device)
        for key in self._reward_condition_metric_keys():
            metrics[key] = torch.zeros((), device=self.device)
        return metrics

    def _grouped_reward_loss_metric_keys(self) -> list[str]:
        keys: list[str] = []
        for spec in self._reward_group_head_specs:
            prefix = f"reward_head_{spec.name}"
            keys.extend(
                [
                    f"{prefix}_policy_mean",
                    f"{prefix}_expert_mean",
                    f"{prefix}_diff",
                    f"{prefix}_grad_penalty",
                    f"{prefix}_l2",
                    f"{prefix}_logit_reg",
                ]
            )
        return keys

    def _reward_condition_metric_keys(self) -> list[str]:
        if not self._reward_infers_latent_condition:
            return []
        return [
            "reward_condition_policy_norm",
            "reward_condition_expert_norm",
            "reward_condition_mean_distance",
            "reward_condition_mean_cosine",
        ]

    @property
    def _required_loss_metrics(self) -> list[str]:
        return [
            *super()._required_loss_metrics,
            "loss_reward_diff",
            "loss_reward_l2",
            "loss_reward_grad_penalty",
            "loss_reward_grad_penalty_batch",
            "loss_reward_grad_penalty_expert",
            "loss_reward_logit_reg",
            "loss_reward_param_weight_decay",
            "reward_updates_applied",
        ]

    @property
    def _optional_loss_metrics(self) -> list[str]:
        return [
            *super()._optional_loss_metrics,
            "loss_diversity",
            "loss_bc",
            "bc_nll",
            "bc_has_expert",
            "bc_log_prob_mean",
            "bc_log_prob_nan_frac",
            "bc_expert_action_abs_mean",
            "bc_expert_action_zero_frac",
            "bc_expert_action_nan_frac",
            "bc_policy_action_abs_mean",
            "bc_policy_action_mae",
            "bc_policy_action_rmse",
            "bc_actor_grad_norm",
            "bc_policy_scale_mean",
            "estimated_reward_mean",
            "estimated_reward_std",
            "expert_reward_mean",
            "expert_reward_std",
            "reward_grad_norm",
            *self._grouped_reward_loss_metric_keys(),
            *self._reward_condition_metric_keys(),
        ]

    def _select_reported_loss_metrics(self, loss: TensorDict) -> TensorDict:
        """Filter IPMD update outputs down to the metrics recorded per minibatch."""
        loss_keys = [
            key
            for key in [*self._required_loss_metrics, *self._optional_loss_metrics]
            if key in loss
        ]
        return loss.select(*loss_keys)

    def _prepare_rollout_rewards(self, rollout: TensorDict) -> dict[str, float]:
        """Attach reward-model diagnostics and PPO reward mixing to one rollout."""
        metrics: dict[str, float] = {}
        reward_key = ("next", "reward")

        with torch.no_grad():
            env_reward = rollout[reward_key]
            est_reward: Tensor | None = None
            if self.config.ipmd.use_estimated_rewards_for_ppo:
                est_reward = (
                    self._reward_from_td(
                        rollout,
                        batch_role="rollout",
                    )
                    .detach()  # type: ignore[attr-defined]
                    .clamp(
                        min=self.config.ipmd.estimated_reward_clamp_min,
                        max=self.config.ipmd.estimated_reward_clamp_max,
                    )
                )  # type: ignore[attr-defined]
                est_reward = self._apply_estimated_reward_done_penalty(
                    rollout,
                    est_reward,
                )

            if self._use_estimated_rewards_for_ppo:
                assert est_reward is not None
                mixed_reward = (
                    self.config.ipmd.env_reward_weight * env_reward
                    + self.config.ipmd.est_reward_weight * est_reward
                )
            else:
                mixed_reward = env_reward

        rollout.set(("next", "env_reward"), env_reward)
        rollout.set(reward_key, mixed_reward)

        metrics.update(
            {
                "train/env_reward_mean": env_reward.mean().item(),
            }
        )
        if est_reward is not None:
            rollout.set(("next", "est_reward"), est_reward)
            metrics["train/est_reward_mean"] = est_reward.mean().item()
            metrics.update(
                reward_alignment_metrics(
                    "reward/env_vs_est",
                    env_reward,
                    est_reward,
                )
            )

        return metrics

    def _expert_batch_for_update(self, batch: TensorDict) -> tuple[TensorDict, Tensor]:
        """Return an expert batch aligned with one PPO minibatch update."""
        del batch
        expert_batch = self._next_expert_batch()
        return (
            expert_batch,
            torch.ones((), device=self.device, dtype=torch.float32),
        )

    def _backward_ppo_terms(self, batch: TensorDict) -> TensorDict:
        """Backpropagate the PPO objective and return detached update metrics."""
        if (
            self._use_latent_command
            and self._latent_learner is not None
            and bool(self.config.ipmd.latent_learning.train_posterior_through_policy)
        ):
            latents = self._latent_learner.infer_batch_latents(
                batch,
                detach=False,
                context="ppo posterior latent",
            )
            if latents is not None:
                batch.set(
                    self._latent_key,
                    latents.reshape(*batch.batch_size, self._latent_dim),
                )
        loss: TensorDict = self.loss_module(batch)
        extra_actor_loss, extra_actor_metrics = self._extra_actor_loss(batch)
        (
            loss["loss_critic"]
            + loss["loss_objective"]
            + loss["loss_entropy"]
            + extra_actor_loss
        ).backward()
        output_loss = loss.clone().detach_()
        for key, value in extra_actor_metrics.items():
            output_loss.set(key, value.detach())
        return output_loss

    def _backward_bc_terms(
        self,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> dict[str, Tensor]:
        """Backpropagate behavior-cloning terms when enabled."""
        metrics: dict[str, Tensor] = {}
        if self._bc_coeff <= 0.0:
            return metrics

        expert_action = expert_batch["expert_action"]
        if not isinstance(expert_action, Tensor):
            msg = "BC update requires expert_batch['expert_action'] to be a Tensor."
            raise RuntimeError(msg)
        expert_obs_td = expert_batch.clone(False)
        if self._latent_key not in expert_obs_td.keys(True):
            expert_latents = self._expert_latents_from_td(
                expert_batch,
                detach=True,
            ).reshape(*expert_batch.batch_size, self._latent_dim)
            expert_obs_td.set(self._latent_key, expert_latents)
        expert_obs_td = expert_obs_td.select(*self._policy_obs_keys)
        dist = self._policy_operator.get_dist(expert_obs_td)
        log_prob = dist.log_prob(expert_action)
        log_prob = self._reduce_log_prob(log_prob, expert_action)
        has_expert_float = has_expert.to(dtype=log_prob.dtype)
        bc_nll = -log_prob.mean() * has_expert_float
        bc_loss = bc_nll * self._bc_coeff
        bc_loss.backward()
        metrics["loss_bc"] = bc_loss.detach()
        metrics["bc_nll"] = bc_nll.detach()
        return metrics

    def _backward_reward_terms(
        self,
        batch: TensorDict,
        expert_batch: TensorDict,
    ) -> dict[str, Tensor]:
        """Backpropagate the IPMD reward objective and return detached metrics."""
        if self._reward_model_type == "grouped":
            return self._backward_grouped_reward_terms(batch, expert_batch)

        metrics = self._empty_reward_metrics()
        if not self._reward_update_enabled:
            return metrics

        requires_grad = self._reward_grad_penalty_coeff > 0.0
        r_pi_input = self._reward_input_from_td(
            batch,
            detach=self._reward_detach_features,
            requires_grad=requires_grad,
        )
        r_exp_input = self._reward_input_from_td(
            expert_batch,
            detach=self._reward_detach_features,
            requires_grad=requires_grad,
        )
        r_pi_logits = self.reward_estimator(r_pi_input)
        r_exp_logits = self.reward_estimator(r_exp_input)
        r_pi = self._reward_out_fn(r_pi_logits)
        r_exp = self._reward_out_fn(r_exp_logits)

        if requires_grad:
            reward_grad_penalty_batch = reward_grad_penalty_from_input(r_pi, r_pi_input)
            reward_grad_penalty_expert = reward_grad_penalty_from_input(
                r_exp, r_exp_input
            )
        else:
            reward_grad_penalty_batch = torch.zeros((), device=self.device)
            reward_grad_penalty_expert = torch.zeros((), device=self.device)

        diff = r_pi.mean() - r_exp.mean()
        l2 = r_pi.pow(2).mean() + r_exp.pow(2).mean()
        reward_grad_penalty = reward_grad_penalty_batch + reward_grad_penalty_expert
        logit_reg = r_pi_logits.pow(2).mean() + r_exp_logits.pow(2).mean()
        param_weight_decay = self._reward_parameter_weight_decay()
        (
            self._reward_loss_coeff * diff
            + self._reward_l2_coeff * l2.pow(0.5)
            + self._reward_grad_penalty_coeff * reward_grad_penalty
            + self._reward_logit_reg_coeff * logit_reg
            + self._reward_param_weight_decay_coeff * param_weight_decay
        ).backward()

        metrics["loss_reward_diff"] = diff.detach()
        metrics["loss_reward_l2"] = l2.detach()
        metrics["loss_reward_grad_penalty"] = reward_grad_penalty.detach()
        metrics["loss_reward_grad_penalty_batch"] = reward_grad_penalty_batch.detach()
        metrics["loss_reward_grad_penalty_expert"] = reward_grad_penalty_expert.detach()
        metrics["loss_reward_logit_reg"] = logit_reg.detach()
        metrics["loss_reward_param_weight_decay"] = param_weight_decay.detach()
        metrics["estimated_reward_mean"] = r_pi.mean().detach()
        metrics["estimated_reward_std"] = r_pi.std().detach()
        metrics["expert_reward_mean"] = r_exp.mean().detach()
        metrics["expert_reward_std"] = r_exp.std().detach()
        return metrics

    def _backward_grouped_reward_terms(
        self,
        batch: TensorDict,
        expert_batch: TensorDict,
    ) -> dict[str, Tensor]:
        """Backpropagate grouped reward heads with per-head diagnostics."""
        metrics = self._empty_reward_metrics()
        if not self._reward_update_enabled:
            return metrics

        requires_grad = self._reward_grad_penalty_coeff > 0.0
        pi = self._grouped_reward_outputs_from_td(
            batch,
            detach=self._reward_detach_features,
            requires_grad=requires_grad,
        )
        exp = self._grouped_reward_outputs_from_td(
            expert_batch,
            detach=self._reward_detach_features,
            requires_grad=requires_grad,
        )

        diff = pi.reward.mean() - exp.reward.mean()
        head_l2_terms: list[Tensor] = []
        head_logit_terms: list[Tensor] = []
        head_gp_terms: list[Tensor] = []
        head_gp_batch_terms: list[Tensor] = []
        head_gp_expert_terms: list[Tensor] = []
        for spec in self._reward_group_head_specs:
            name = spec.name
            pi_head = pi.head_rewards[name]
            exp_head = exp.head_rewards[name]
            pi_logits = pi.head_logits[name]
            exp_logits = exp.head_logits[name]

            head_l2 = pi_head.pow(2).mean() + exp_head.pow(2).mean()
            head_logit_reg = pi_logits.pow(2).mean() + exp_logits.pow(2).mean()
            if requires_grad:
                head_gp_batch = reward_grad_penalty_from_input(
                    pi_head,
                    pi.head_inputs[name],
                )
                head_gp_expert = reward_grad_penalty_from_input(
                    exp_head,
                    exp.head_inputs[name],
                )
            else:
                head_gp_batch = torch.zeros((), device=self.device)
                head_gp_expert = torch.zeros((), device=self.device)
            head_gp = head_gp_batch + head_gp_expert

            head_l2_terms.append(head_l2)
            head_logit_terms.append(head_logit_reg)
            head_gp_terms.append(head_gp)
            head_gp_batch_terms.append(head_gp_batch)
            head_gp_expert_terms.append(head_gp_expert)

            prefix = f"reward_head_{name}"
            metrics[f"{prefix}_policy_mean"] = pi_head.mean().detach()
            metrics[f"{prefix}_expert_mean"] = exp_head.mean().detach()
            metrics[f"{prefix}_diff"] = (pi_head.mean() - exp_head.mean()).detach()
            metrics[f"{prefix}_grad_penalty"] = head_gp.detach()
            metrics[f"{prefix}_l2"] = head_l2.detach()
            metrics[f"{prefix}_logit_reg"] = head_logit_reg.detach()

        l2 = torch.stack(head_l2_terms).sum()
        l2_loss = torch.stack([term.pow(0.5) for term in head_l2_terms]).sum()
        reward_grad_penalty_batch = torch.stack(head_gp_batch_terms).sum()
        reward_grad_penalty_expert = torch.stack(head_gp_expert_terms).sum()
        reward_grad_penalty = torch.stack(head_gp_terms).sum()
        logit_reg = torch.stack(head_logit_terms).sum()
        param_weight_decay = self._reward_parameter_weight_decay()
        (
            self._reward_loss_coeff * diff
            + self._reward_l2_coeff * l2_loss
            + self._reward_grad_penalty_coeff * reward_grad_penalty
            + self._reward_logit_reg_coeff * logit_reg
            + self._reward_param_weight_decay_coeff * param_weight_decay
        ).backward()

        metrics["loss_reward_diff"] = diff.detach()
        metrics["loss_reward_l2"] = l2.detach()
        metrics["loss_reward_grad_penalty"] = reward_grad_penalty.detach()
        metrics["loss_reward_grad_penalty_batch"] = reward_grad_penalty_batch.detach()
        metrics["loss_reward_grad_penalty_expert"] = reward_grad_penalty_expert.detach()
        metrics["loss_reward_logit_reg"] = logit_reg.detach()
        metrics["loss_reward_param_weight_decay"] = param_weight_decay.detach()
        metrics["estimated_reward_mean"] = pi.reward.mean().detach()
        metrics["estimated_reward_std"] = pi.reward.std().detach()
        metrics["expert_reward_mean"] = exp.reward.mean().detach()
        metrics["expert_reward_std"] = exp.reward.std().detach()
        return metrics

    def _run_reward_updates(
        self,
        batch: TensorDict,
        expert_batch: TensorDict,
        *,
        policy_update_idx: int,
    ) -> dict[str, Tensor]:
        """Run scheduled reward-estimator updates and return averaged metrics."""
        metrics: dict[str, Tensor] = self._empty_reward_metrics()
        metrics["reward_updates_applied"] = torch.zeros((), device=self.device)
        metrics["reward_grad_norm"] = torch.zeros((), device=self.device)
        if not self._reward_update_due(policy_update_idx):
            return metrics
        if self.reward_optim is None:
            msg = "Reward update is enabled, but reward_optim was not initialized."
            raise RuntimeError(msg)

        policy_latents = self._set_reward_policy_condition_from_batch(batch)
        expert_latents = self._set_reward_expert_condition_from_batch(expert_batch)
        condition_metrics = self._reward_condition_metrics(
            policy_latents,
            expert_latents,
        )

        for _ in range(self._reward_updates_per_policy_update):
            self.reward_optim.zero_grad(set_to_none=True)
            step_metrics = self._backward_reward_terms(batch, expert_batch)
            reward_grad_norm = clip_grad_norm_(
                self._reward_grad_clip_params,
                self._reward_max_grad_norm,
            )
            self.reward_optim.step()
            for key, value in step_metrics.items():
                metrics[key] = metrics[key] + value
            metrics["reward_grad_norm"] = (
                metrics["reward_grad_norm"] + reward_grad_norm.detach()
            )

        steps = float(self._reward_updates_per_policy_update)
        for key in tuple(metrics.keys()):
            if key == "reward_updates_applied":
                continue
            metrics[key] = metrics[key] / steps
        metrics["reward_updates_applied"] = torch.tensor(steps, device=self.device)
        for key, value in condition_metrics.items():
            metrics[key] = value
        return metrics

    def _record_env_metrics(self, iteration: PPOIterationData) -> None:
        """Record IsaacLab env metrics while draining the env log queue."""
        if (
            "Isaac" in self.config.env.env_name
            and hasattr(self.env, "log_infos")
            and len(self.env.log_infos) > 0
        ):
            log_info_dict: dict[str, Tensor] = self.env.log_infos.pop()
            self.env.log_infos.clear()
            log_info(log_info_dict, iteration.metrics)

    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        return (
            ("train/step_reward_mean", "r_step"),
            ("episode/length", "ep_len"),
            ("episode/return", "r_ep"),
            ("train/loss_objective", "pi_loss"),
            ("train/loss_reward_diff", "reward_diff"),
            ("train/expert_reward_mean", "exp_r"),
            ("time/speed", "fps"),
        )

    def _file_summary_fields(self) -> tuple[tuple[str, str], ...]:
        fields = (
            *super()._file_summary_fields(),
            ("train/env_reward_mean", "env_r"),
            ("train/estimated_reward_mean", "est_r"),
            ("train/expert_reward_mean", "exp_r"),
            ("train/expert_reward_std", "exp_r_std"),
            ("train/loss_reward_l2", "reward_l2"),
            ("train/loss_reward_grad_penalty", "reward_gp"),
            ("train/loss_bc", "bc_loss"),
            ("train/bc_policy_action_mae", "bc_act_mae"),
            ("train/latent_posterior_recon_mae", "latent_recon_mae"),
            ("train/latent_posterior_recon_max_abs", "latent_recon_max"),
        )
        grouped_fields: list[tuple[str, str]] = []
        for spec in self._reward_group_head_specs:
            grouped_fields.append(
                (f"train/reward_head_{spec.name}_diff", f"{spec.name}_diff")
            )
            grouped_fields.append(
                (f"train/reward_head_{spec.name}_grad_penalty", f"{spec.name}_gp")
            )
        return (*fields, *grouped_fields)

    def record(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,
    ) -> None:
        """Flush IPMD diagnostics, reward metrics, and checkpoints for one rollout."""
        rollout = iteration.rollout

        if "train/step_reward_mean" not in iteration.metrics and (
            "next",
            "reward",
        ) in rollout.keys(True):
            step_rewards = rollout["next", "reward"]
            iteration.metrics.update(
                {
                    "train/step_reward_mean": step_rewards.mean().item(),
                    "train/step_reward_std": step_rewards.std().item(),
                    "train/step_reward_max": step_rewards.max().item(),
                    "train/step_reward_min": step_rewards.min().item(),
                }
            )

        episode_rewards = rollout["next", "episode_reward"][rollout["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = rollout["next", "step_count"][rollout["next", "done"]]
            episode_lengths = episode_length.cpu().tolist()
            episode_reward_values = episode_rewards.cpu().tolist()
            self.episode_lengths.extend(episode_lengths)
            self.episode_rewards.extend(episode_reward_values)
            iteration.metrics.update(
                {
                    "episode/length": float(np.mean(self.episode_lengths)),
                    "episode/return": float(np.mean(self.episode_rewards)),
                    "train/reward": float(np.mean(episode_reward_values)),
                }
            )

        iteration.metrics.update(self._build_control_metrics(metadata))
        iteration.metrics.update(self._build_timing_metrics(iteration, metadata))
        self._record_env_metrics(iteration)
        iteration.metrics.update(timeit.todict(prefix="time"))  # type: ignore[arg-type]
        # Always stream scalar metrics to the configured backend (e.g. WandB).
        # Sparse terminal/file summaries remain controlled by
        # ``trainer.log_interval`` via ``_refresh_progress_display`` below.
        self.log_metrics(
            iteration.metrics,
            step=metadata.frames_processed,
            log_python=False,
        )
        self._log_iteration_file_summary(metadata, iteration)
        self.collector.update_policy_weights_()
        self._refresh_progress_display(metadata, iteration)

        if self._should_save_checkpoint(
            frames_processed=metadata.frames_processed,
            frames_in_iteration=iteration.frames,
        ):
            self.save_model(
                path=self.log_dir / self.config.logger.save_path,
                step=metadata.frames_processed,
            )

    def save_model(
        self, path: str | Path | None = None, step: int | None = None
    ) -> None:
        """Save PPO and reward-estimator state into one checkpoint."""
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
            "reward_estimator_state_dict": self.reward_estimator.state_dict(),
        }
        if self.value_function:
            data_to_save["value_state_dict"] = self.value_function.state_dict()
        if self.q_function:
            data_to_save["q_state_dict"] = self.q_function.state_dict()
        if self.feature_extractor:
            data_to_save["feature_extractor_state_dict"] = (
                self.feature_extractor.state_dict()
            )
        if self.reward_optim is not None:
            data_to_save["reward_optimizer_state_dict"] = self.reward_optim.state_dict()
        if self.lr_scheduler is not None:
            data_to_save["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if (
            hasattr(self.env, "is_closed")
            and not self.env.is_closed
            and hasattr(self.env, "normalize_obs")
        ):
            data_to_save["vec_norm_msg"] = self.env.state_dict()

        torch.save(data_to_save, target_path)

    def load_model(self, path: str) -> None:
        """Load PPO and reward-estimator state from a checkpoint."""
        data = torch.load(path, map_location=self.device)
        if self.policy and "policy_state_dict" in data:
            self.policy.load_state_dict(data["policy_state_dict"])  # type: ignore[arg-type]
        if self.value_function and "value_state_dict" in data:
            self.value_function.load_state_dict(data["value_state_dict"])  # type: ignore[arg-type]
        if self.q_function and "q_state_dict" in data:
            self.q_function.load_state_dict(data["q_state_dict"])  # type: ignore[arg-type]
        if "optimizer_state_dict" in data:
            self.optim.load_state_dict(data["optimizer_state_dict"])  # type: ignore[arg-type]
        if "reward_estimator_state_dict" in data:
            self.reward_estimator.load_state_dict(
                data["reward_estimator_state_dict"]  # type: ignore[arg-type]
            )
        if self.reward_optim is not None and "reward_optimizer_state_dict" in data:
            self.reward_optim.load_state_dict(
                data["reward_optimizer_state_dict"]  # type: ignore[arg-type]
            )
        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in data:
            self.lr_scheduler.load_state_dict(data["lr_scheduler_state_dict"])
        if self.config.feature_extractor and "feature_extractor_state_dict" in data:
            self.feature_extractor.load_state_dict(
                data["feature_extractor_state_dict"]  # type: ignore[arg-type]
            )
        if hasattr(self.env, "normalize_obs") and "vec_norm_msg" in data:
            self.env.load_state_dict(data["vec_norm_msg"])  # type: ignore[arg-type]

    def predict(self, obs: Tensor | np.ndarray | Mapping[Any, Any]) -> Tensor:  # type: ignore[override]
        """Predict action given observation (deterministic)."""
        policy_op = self.actor_critic.get_policy_operator()
        policy_op.eval()
        with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
            input_keys = list(self._policy_obs_keys)
            td_data: dict[BatchKey, Tensor] = {}
            batch_shape: tuple[int, ...] | None = None

            if isinstance(obs, Mapping):
                for key in input_keys:
                    if key == self._latent_key:
                        try:
                            latent_value = mapping_get_obs_value(obs, key)
                        except Exception:
                            continue
                        value = torch.as_tensor(latent_value, device=self.device)
                    else:
                        value = torch.as_tensor(
                            mapping_get_obs_value(obs, key),
                            device=self.device,
                        )
                    feature_ndim = self._obs_feature_ndims[key]
                    if (feature_ndim == 0 and value.ndim == 0) or (
                        feature_ndim > 0 and value.ndim == feature_ndim
                    ):
                        value = value.unsqueeze(0)
                    current_batch_shape = infer_batch_shape(value, feature_ndim)
                    if batch_shape is None:
                        batch_shape = current_batch_shape
                    elif current_batch_shape != batch_shape:
                        msg = (
                            "All observation tensors passed to predict() must have the "
                            f"same batch shape, got {batch_shape} and {current_batch_shape}."
                        )
                        raise ValueError(msg)
                    td_data[key] = value
            else:
                if len(input_keys) != 1:
                    msg = (
                        "predict() received a single tensor observation, but policy "
                        f"expects multiple keys {input_keys}. Pass a dict instead."
                    )
                    raise ValueError(msg)
                key = input_keys[0]
                value = torch.as_tensor(obs, device=self.device)
                feature_ndim = self._obs_feature_ndims[key]
                if (feature_ndim == 0 and value.ndim == 0) or (
                    feature_ndim > 0 and value.ndim == feature_ndim
                ):
                    value = value.unsqueeze(0)
                td_data[key] = value
                batch_shape = infer_batch_shape(value, feature_ndim)

            td = TensorDict(
                td_data, batch_size=list(batch_shape or [1]), device=self.device
            )
            if self._use_latent_command:
                self._inject_latent_command(td)
            td = policy_op(td)
            return td["action"]

    def _latent_mode_hint(self) -> str:
        task_name = str(getattr(self.config.env, "env_name", "") or "")
        if task_name in {"Isaac-Imitation-G1-v0", "Isaac-Imitation-G1-LafanTrack-v0"}:
            return (
                " For the vanilla G1 task, pass ipmd.use_latent_command=False, "
                "or switch to Isaac-Imitation-G1-Latent-v0."
            )
        if task_name == "Isaac-Imitation-G1-Latent-v0":
            return (
                " For the latent G1 task, keep ipmd.use_latent_command=True, "
                "or switch to Isaac-Imitation-G1-v0."
            )
        return ""

    def pre_iteration_compute(self, rollout: TensorDict) -> TensorDict:
        """Compute advantages."""
        with torch.no_grad():
            rollout = self.adv_module(rollout)

            if getattr(self.config.compile, "compile", False):
                rollout = rollout.clone()

        self.data_buffer.extend(rollout.reshape(-1))
        return rollout

    def _set_reward_policy_condition_from_batch(
        self,
        batch: TensorDict,
    ) -> Tensor | None:
        """Infer and stamp reward-conditioning latents for a policy minibatch."""
        if not self._reward_infers_latent_condition:
            return None
        assert self._latent_learner is not None
        latents = self._latent_learner.infer_batch_latents(
            batch,
            detach=True,
            context="reward policy latent",
        )
        if latents is None:
            msg = (
                "Active latent learner does not support policy-batch latent "
                "inference for reward conditioning."
            )
            raise RuntimeError(msg)
        latents = latents.to(device=self.device, dtype=torch.float32)
        batch.set(
            self._latent_key,
            latents.reshape(*batch.batch_size, self._latent_dim),
        )
        return latents.reshape(-1, self._latent_dim)

    def _set_reward_expert_condition_from_batch(
        self,
        expert_batch: TensorDict,
    ) -> Tensor | None:
        """Infer and stamp reward-conditioning latents for an expert minibatch."""
        if not self._reward_infers_latent_condition:
            return None
        latents = self._expert_latents_from_td(
            expert_batch,
            detach=True,
        ).to(device=self.device, dtype=torch.float32)
        expert_batch.set(
            self._latent_key,
            latents.reshape(*expert_batch.batch_size, self._latent_dim),
        )
        return latents.reshape(-1, self._latent_dim)

    def _reward_condition_metrics(
        self,
        policy_latents: Tensor | None,
        expert_latents: Tensor | None,
    ) -> dict[str, Tensor]:
        """Build diagnostics for the latent context used by reward updates."""
        metrics: dict[str, Tensor] = {}
        if policy_latents is None or expert_latents is None:
            return metrics

        policy_flat = policy_latents.reshape(-1, self._latent_dim)
        expert_flat = expert_latents.reshape(-1, self._latent_dim)
        policy_mean = policy_flat.mean(dim=0)
        expert_mean = expert_flat.mean(dim=0)
        metrics["reward_condition_policy_norm"] = (
            policy_flat.norm(dim=-1).mean().detach()
        )
        metrics["reward_condition_expert_norm"] = (
            expert_flat.norm(dim=-1).mean().detach()
        )
        metrics["reward_condition_mean_distance"] = (
            (policy_mean - expert_mean).norm().detach()
        )
        metrics["reward_condition_mean_cosine"] = (
            torch.nn.functional.cosine_similarity(
                policy_mean.unsqueeze(0),
                expert_mean.unsqueeze(0),
                dim=-1,
                eps=1.0e-6,
            )
            .squeeze(0)
            .detach()
        )
        return metrics

    def _empty_reward_update_metrics(self) -> dict[str, Tensor]:
        metrics = self._empty_reward_metrics()
        metrics["reward_updates_applied"] = torch.zeros((), device=self.device)
        metrics["reward_grad_norm"] = torch.zeros((), device=self.device)
        return metrics

    def _run_pre_policy_reward_updates(
        self,
        rollout_flat: TensorDict,
        metadata: PPOTrainingMetadata,
    ) -> dict[str, Tensor]:
        """Update the reward estimator before PPO reward and advantage computation."""
        if not self._reward_update_enabled:
            return {}

        metrics = self._empty_reward_update_metrics()
        minibatch_slots = 0
        updates_completed = metadata.updates_completed
        start_update_idx = (
            int(updates_completed.item())
            if isinstance(updates_completed, Tensor)
            else int(updates_completed)
        )
        self.data_buffer.empty()
        self.data_buffer.extend(rollout_flat)
        try:
            for _epoch_idx in range(metadata.epochs_per_rollout):
                for batch in self.data_buffer:
                    policy_update_idx = start_update_idx + minibatch_slots
                    if self._reward_update_due(policy_update_idx):
                        reward_batch = batch.clone()
                        expert_batch, _has_expert = self._expert_batch_for_update(
                            reward_batch
                        )
                        step_metrics = self._run_reward_updates(
                            reward_batch,
                            expert_batch,
                            policy_update_idx=policy_update_idx,
                        )
                    else:
                        step_metrics = self._empty_reward_update_metrics()
                    for key, value in step_metrics.items():
                        metrics[key] = metrics[key] + value
                    minibatch_slots += 1
        finally:
            self.data_buffer.empty()

        if minibatch_slots == 0:
            return metrics
        scale = float(minibatch_slots)
        for key in tuple(metrics.keys()):
            metrics[key] = metrics[key] / scale
        return metrics

    def update(
        self,
        batch: TensorDict,
        num_network_updates: int,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> tuple[TensorDict, int]:
        """PPO update plus optional BC loss."""
        self.optim.zero_grad(set_to_none=True)
        output_loss = self._backward_ppo_terms(batch)
        bc_metrics = self._backward_bc_terms(expert_batch, has_expert)

        # Gradient clipping — always call for a fixed graph
        grad_norm_tensor = clip_grad_norm_(self._grad_clip_params, self._max_grad_norm)

        self.optim.step()

        output_loss.set("alpha", torch.ones((), device=self.device))
        for key, value in bc_metrics.items():
            output_loss.set(key, value)
        output_loss.set("grad_norm", grad_norm_tensor.detach())

        return output_loss, num_network_updates + 1

    def prepare(
        self,
        iteration: PPOIterationData,  # noqa: ARG002
        metadata: PPOTrainingMetadata,  # noqa: ARG002
    ) -> None:
        """Reward preparation runs in iterate() after pre-PPO reward updates."""
        return

    def iterate(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,
    ) -> None:
        """Run PPO-style epochs over the prepared rollout with IPMD expert updates."""
        losses = TensorDict(
            batch_size=[metadata.epochs_per_rollout, metadata.minibatches_per_epoch]
        )
        learn_start = time.perf_counter()

        self.data_buffer.empty()
        self.actor_critic.train()
        self.adv_module.train()
        self.reward_estimator.train()

        with timeit("training"):
            rollout_flat = iteration.rollout.reshape(-1)
            if self._latent_learner is not None:
                learner_metrics = self._latent_learner.update(rollout_flat)
                if learner_metrics:
                    iteration.metrics.update(
                        {f"train/{k}": v for k, v in learner_metrics.items()}
                    )

            reward_metrics = self._run_pre_policy_reward_updates(
                rollout_flat,
                metadata,
            )
            iteration.metrics.update(
                {f"train/{key}": value.item() for key, value in reward_metrics.items()}
            )
            iteration.metrics.update(self._prepare_rollout_rewards(iteration.rollout))
            iteration.rollout = self.pre_iteration_compute(iteration.rollout)

            for epoch_idx in range(metadata.epochs_per_rollout):
                for batch_idx, batch in enumerate(self.data_buffer):
                    kl_context = None
                    if (self.config.optim.scheduler or "").lower() == "adaptive":
                        kl_context = self._prepare_kl_context(
                            batch, metadata.policy_operator
                        )

                    needs_expert_batch = bool(self._bc_coeff > 0.0)
                    if needs_expert_batch:
                        expert_batch, has_expert = self._expert_batch_for_update(batch)
                    else:
                        expert_batch = TensorDict(
                            {},
                            batch_size=batch.batch_size,
                            device=self.device,
                        )
                        has_expert = torch.zeros(
                            (),
                            device=self.device,
                            dtype=torch.float32,
                        )
                    with timeit("training/update"):
                        loss, metadata.updates_completed = self.update(
                            batch,
                            metadata.updates_completed,
                            expert_batch,
                            has_expert,
                        )

                    if self.lr_scheduler and self.lr_scheduler_step == "update":
                        self.lr_scheduler.step()
                    if kl_context is not None:
                        kl_approx = self._compute_kl_after_update(
                            kl_context, metadata.policy_operator
                        )
                        if kl_approx is not None:
                            loss.set("kl_approx", kl_approx.detach())
                            self._maybe_adjust_lr(kl_approx, self.config.optim)

                    losses[epoch_idx, batch_idx] = self._select_reported_loss_metrics(
                        loss
                    )

                if self.lr_scheduler and self.lr_scheduler_step == "epoch":
                    self.lr_scheduler.step()

        iteration.learn_time = time.perf_counter() - learn_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():  # type: ignore[attr-defined]
            iteration.metrics[f"train/{key}"] = value.item()  # type: ignore[attr-defined]
