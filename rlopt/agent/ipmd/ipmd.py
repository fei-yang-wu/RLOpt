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
    held_command_policy_surrogate,
)
from rlopt.agent.ipmd.dagger_config import DAggerConfig
from rlopt.agent.ipmd.reward_averaging import (
    PrioritizedFifoFeatureReplay,
    ReservoirFeatureReplay,
    polyak_update,
)
from rlopt.agent.ipmd.utils import (
    IPMD_COMMAND_SOURCES,
    IPMD_REWARD_INPUT_TYPES,
    IPMD_REWARD_MODEL_TYPES,
    IPMD_REWARD_OUTPUT_ACTIVATIONS,
    LinearDualRewardHead,
    RewardInputBlock,
    build_reward_input_blocks,
    normalize_ipmd_command_source,
    normalize_ipmd_reward_input_type,
    normalize_ipmd_reward_model_type,
    normalize_ipmd_reward_objective,
    normalize_ipmd_reward_output_activation,
    normalize_ipmd_reward_ppo_squash,
    require_non_negative,
    require_positive_int,
    required_batch_keys_from_reward_blocks,
    resolve_ipmd_reward_ppo_squash,
    reward_alignment_metrics,
    reward_grad_penalty_from_input,
    reward_saturation_fraction,
    reward_separation_auc,
    rollout_done_mask,
    squash_reward_for_ppo,
)
from rlopt.agent.ppo.ppo import (
    PPO,
    PPOConfig,
    PPOIterationData,
    PPORLOptConfig,
    PPOTrainingMetadata,
    _normalize_advantage_over_rollout,
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
from rlopt.expert import build_offline_expert_sampler
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
    """Keys for the learned prior."""

    reconstruction_target_keys: list[ObsKey] = field(default_factory=list)
    """Expert keys reconstructed by decoder-based latent learners.

    ``future_cvae`` should set this to the future reference window. When left
    empty, it falls back to ``posterior_input_keys`` for backward-compatible
    autoencoder behavior.
    """

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
    """If True, route PPO policy gradients through a recomputed posterior latent.

    The PPO forward value remains the exact command stored during rollout, including
    multi-step holding and phase. The recomputed latent is used only as a straight-through
    gradient path into the encoder. Encoder parameters are added to the main PPO optimizer.
    Combining this with recon_coeff > 0 updates the encoder from both PPO and reconstruction.
    """

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

    code_latent_dim: int | None = 0
    """Latent code width before optional command-phase features.

    ``0`` or ``None`` uses ``ipmd.latent_dim`` when ``command_phase_mode="none"`` and
    ``ipmd.latent_dim - 2`` when ``command_phase_mode="sin_cos"``.
    """

    command_phase_mode: str = "none"
    """Optional phase features appended to held rollout commands.

    Supported values:
    - ``"none"``: publish only the held code latent.
    - ``"sin_cos"``: append ``sin(phase), cos(phase)`` within the current hold.
    """

    command_phase_source: str = "hold"
    """What drives the ``sin_cos`` phase clock.

    - ``"hold"``: position within the current code hold. Informationally dead
      at ``code_period=1``, where it pins to a constant.
    - ``"episode"``: env steps since reset modulo ``command_phase_period`` — a
      live clock for hold-1 interfaces. Requires ``latent_steps_max == 1``.
    """

    command_phase_period: int = 0
    """Phase clock period in env steps. ``0`` uses ``code_period``."""

    command_quantizer: str = "none"
    """Quantize the published command AT THE COMMAND BOUNDARY.

    The encoder stays continuous; only what the tracker observes is snapped to
    a lattice. This is NOT ``latent_mode="sonic_fsq"``, which makes the encoder
    itself an FSQ bottleneck at pretrain time.

    - ``"none"``: publish the continuous code (the default).
    - ``"sonic_fsq"``: SONIC's token space -- per-dimension FSQ, then divide by
      ``levels // 2`` so values land on the lattice in ``[-1, 1]``. At the
      default 32 levels that is exact multiples of ``1/16``, which is what
      every entry of ``gear_sonic``'s ``LATENT_INITIAL_MOTION_TOKEN`` is.

    The phase features, when present, are appended AFTER quantization and are
    never quantized.
    """

    command_fsq_levels: list[int] = field(default_factory=lambda: [32] * 64)
    """Per-dimension levels for ``command_quantizer="sonic_fsq"``.

    SONIC publishes 64 dimensions at 32 levels. The length must equal the code
    width (``code_latent_dim``), since the quantizer acts per dimension.
    """

    fsq_levels: list[int] = field(default_factory=lambda: [8, 8, 8, 5, 5])
    """Per-dimension level counts for FSQ. Effective codebook size = prod(levels)."""

    fsq_normalize_codes: bool = False
    """Divide FSQ lattice values by ``levels // 2`` before publication.

    This preserves legacy reconstruction arms by default. Enable it for SONIC,
    whose public token values live on a normalized lattice in [-1, 1].
    """

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

    posterior_command_period: int = 1
    """Env steps a continuous posterior command is held during rollout.

    Used by continuous posterior learners such as ``patch_autoencoder``.
    A value of 1 preserves the original behavior and refreshes the posterior
    command every env step.
    """

    token_sequence_horizon: int = 10
    """Number of per-step tokens in ``per_step_vq_sequence`` packets."""

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

        self.command_phase_source = str(self.command_phase_source).strip().lower()
        if self.command_phase_source not in {"hold", "episode"}:
            msg = (
                "ipmd.latent_learning.command_phase_source must be 'hold' or "
                f"'episode', got {self.command_phase_source!r}."
            )
            raise ValueError(msg)
        self.command_phase_period = int(self.command_phase_period or 0)
        if self.command_phase_period < 0:
            msg = (
                "ipmd.latent_learning.command_phase_period must be >= 0, got "
                f"{self.command_phase_period}."
            )
            raise ValueError(msg)

        self.code_period = require_positive_int(
            "ipmd.latent_learning.code_period", self.code_period
        )
        self.posterior_command_period = require_positive_int(
            "ipmd.latent_learning.posterior_command_period",
            self.posterior_command_period,
        )
        self.token_sequence_horizon = require_positive_int(
            "ipmd.latent_learning.token_sequence_horizon",
            self.token_sequence_horizon,
        )
        if self.code_latent_dim is not None and int(self.code_latent_dim) > 0:
            self.code_latent_dim = require_positive_int(
                "ipmd.latent_learning.code_latent_dim", self.code_latent_dim
            )
        else:
            self.code_latent_dim = 0

        phase_dim = 2 if self.command_phase_mode == "sin_cos" else 0
        code_dim = (
            int(self.code_latent_dim)
            if self.code_latent_dim is not None and int(self.code_latent_dim) > 0
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
    - ``"hl_skill"`` - load a frozen high-level DiffSR skill encoder and
      publish its online expert-window latents as commands
    """

    hl_skill_checkpoint_path: str = ""
    """Checkpoint path used when ``command_source="hl_skill"``."""

    hl_skill_horizon_steps: int = 0
    """Optional horizon assertion for the frozen high-level skill checkpoint; 0 infers from checkpoint."""

    hl_skill_command_mode: str = "z"
    """HL skill command representation: z, phi, or z_phi."""

    hl_skill_finetune_enabled: bool = False
    """Enable online policy-gradient finetuning for command_source='hl_skill'."""

    hl_skill_restore_from_checkpoint: bool = True
    """Restore the command sampler (skill encoder weights, DiffSR, optimizer)
    from a tracker checkpoint's ``hl_skill_command_sampler_state_dict``.

    ``False`` keeps the encoder loaded from ``hl_skill_checkpoint_path`` when
    a tracker is fine-tuned onto a DIFFERENT encoder (an encoder swap): the
    tracker checkpoint's embedded encoder is the old one, and restoring it
    would silently undo the swap (same width) or fail (new width). The
    encoder is frozen in that setting, so no sampler state is lost.
    """

    hl_skill_pg_coeff: float = 0.05
    # Weight of the DiffSR endpoint loss on ACHIEVED windows sampled from the
    # environment's raw-pose ring (env.achieved_ring_capacity > 0). 0 disables.
    # This is the no-policy-gradient online-dynamics path: the encoder learns
    # the transition structure of what the robot actually does, through the
    # same objective and window convention as its expert pretraining.
    hl_skill_achieved_coeff: float = 0.0
    """Weight on the second-pass PPO actor objective for the high-level skill encoder."""

    hl_skill_offline_diffsr_coeff: float = 1.0
    """Weight on the original offline DiffSR loss during online skill finetuning."""

    hl_skill_anchor_coeff: float = 0.01
    """Weight anchoring the online encoder output to the checkpoint encoder output."""

    hl_skill_z_norm_coeff: float | None = None
    """Optional z-norm penalty during online finetuning; None reuses checkpoint config."""

    hl_skill_lr: float = 3.0e-5
    """Learning rate for the online high-level skill optimizer."""

    hl_skill_grad_clip_norm: float | None = 1.0
    """Gradient clip for online high-level skill updates. None disables clipping."""

    hl_skill_offline_batch_size: int = 8192
    """Expert macro batch size for DiffSR/anchor regularization."""

    hl_skill_update_interval: int = 1
    """Run one high-level skill update every N PPO minibatch updates."""

    hl_skill_train_diffsr: bool = False
    """Also update the loaded DiffSR model during online high-level skill finetuning."""

    hl_skill_online_objective: str = "legacy"
    """Online finetune objective. ``legacy`` = PG + offline/achieved DiffSR +
    anchor (mlp-head JEPA checkpoints, no source history). ``ntp_live_offset``
    = the checkpoint's own pretraining next-chunk DiffSR loss (diff_chunk head,
    source history supported) on offline windows moved by the policy's live
    slot-0 offsets, plus the anchor term; requires hl_skill_pg_coeff=0."""

    hl_skill_live_offset_ring: int = 65536
    """Capacity of the live slot-0 offset ring read by ``ntp_live_offset``."""

    skill_commander_checkpoint_path: str = ""
    """Skill-commander checkpoint used when command_source='skill_commander'."""

    token_planner_checkpoint_path: str = ""
    """Categorical planner checkpoint used when command_source='token_planner'."""

    continuous_planner_checkpoint_path: str = ""
    """Continuous planner checkpoint used when command_source='continuous_planner'."""

    skill_commander_embeddings_path: str = ""
    """Language goal embedding table (.pt) used when command_source='skill_commander'."""

    skill_commander_use_achieved_state: bool = False
    """Condition the commander on the robot's achieved macro state (full-M3 closed loop)."""

    skill_commander_goal_name: str = ""
    """Optional fixed language-goal name for command_source='skill_commander'."""

    skill_commander_goal_rank: int = -1
    """Optional fixed language-goal rank from the embedding table; -1 disables."""

    skill_commander_flow_num_inference_steps: int = 0
    """Optional override for flow-matching commander inference steps; 0 uses checkpoint config."""

    skill_commander_flow_inference_noise_std: float = -1.0
    """Optional override for flow-matching commander inference noise std; <0 uses checkpoint config."""

    skill_commander_diffusion_num_inference_steps: int = 0
    """Optional override for diffusion commander inference steps; 0 uses checkpoint config."""

    skill_commander_diffusion_inference_scheduler: str = ""
    """Optional diffusion commander inference scheduler override: ddpm or ddim."""

    skill_commander_diffusion_ddim_eta: float = -1.0
    """Optional DDIM eta override for diffusion commander inference; <0 uses checkpoint config."""

    skill_commander_diffusion_inference_noise_std: float = -1.0
    """Optional diffusion commander initial-noise std override; <0 uses checkpoint config."""

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

    # Reward-estimation revival (plan 2026-10-07, section 5.4 B2/B3). Every
    # default below reproduces the historical behaviour bitwise.
    reward_objective: str = "ipmd_diff"
    """Reward-estimator objective in ``_backward_reward_terms``.

    - ``"ipmd_diff"``      - the parked objective (default; historical path).
    - ``"logistic_ratio"`` - label-smoothed BCE, reward = logit.
    - ``"chi2_lsgan"``     - AMP least-squares discriminator with R1 on expert
      samples (``reward_grad_penalty_coeff`` is the R1 coefficient, applied
      as ``coeff / 2``); reward = D(x).
    - ``"linear_dual"``    - parameter-free IPMD linear dual over whitened
      features; no optimizer; ``w* = Delta / reward_linear_weight_decay``.

    The three new objectives require ``reward_output_activation == "none"``
    and ``reward_model_type == "mlp"``. They scale their main term by
    ``reward_loss_coeff`` and add ``reward_param_weight_decay_coeff``; the
    ``reward_l2_coeff`` output penalty belongs to ``ipmd_diff`` only.
    """

    reward_ppo_squash: str = "auto"
    """How the raw estimate ``r`` becomes ``r_hat`` before PPO mixing.

    ``"auto"`` resolves per objective: ipmd_diff -> ``"none"``,
    logistic_ratio -> ``"sigmoid"``, chi2_lsgan -> ``"amp"`` (``max(0, 1 -
    0.25 (r - 1)^2)``), linear_dual -> ``"affine_frozen"`` (``clip(0.5 + (r -
    m0) / (6 s0), 0, 1)`` with ``m0, s0`` captured on the first rollout after
    PPO driving turns on and frozen into the checkpoint).
    """

    estimated_reward_scale: float = 1.0
    """Multiplier on ``r_hat`` before mixing: ``est = scale * r_hat``.

    Magnitude-matches a [0, 1] squash to the env reward (ceiling 0.30 per step
    on the campaign recipe). 1.0 keeps the historical path.
    """

    reward_ramp_updates: int = 0
    """Linear ramp of ``est_reward_weight`` over this many PPO iterations.

    Counted in rollouts (``_prepare_rollout_rewards`` calls) since PPO driving
    turned on: ``effective = est_reward_weight * min(1, seen / ramp)``. 0
    disables the ramp (full weight from the first rollout).
    """

    reward_mask_done_rows: bool = False
    """Zero the estimated reward on rows whose ``("next", "done")`` is true.

    Both terminated and truncated rows. With >= 75% of episode ends being
    terminations, any non-zero estimate on done rows pays for dying.
    """

    reward_constant_bonus: bool = False
    """Survival-effect control: replace ``r_hat`` by its frozen batch mean.

    The mean is captured on the first rollout after PPO driving turns on and
    frozen (checkpointed); every alive row gets ``scale * mean``, done rows 0.
    """

    reward_whiten_ridge: float = 1e-3
    """Ridge added to the pooled covariance before the inverse square root
    (``linear_dual`` whitening)."""

    reward_linear_weight_decay: float = 1.0
    """``lambda`` of the ``linear_dual`` step ``w <- w + alpha (Delta - lambda w)``;
    fixed point ``w* = Delta / lambda``."""

    reward_linear_closed_form: bool = True
    """``linear_dual``: jump to ``w* = Delta / lambda`` each step instead of
    taking the ``alpha``-sized dual step (``alpha = reward_lr or optim.lr``)."""

    reward_moment_momentum: float = 0.05
    """EMA momentum of the ``linear_dual`` running moments over pooled rows;
    ``<= 0`` switches to the exact cumulative mean."""

    reward_label_smoothing: float = 0.05
    """``logistic_ratio`` targets: expert -> ``1 - eps``, policy -> ``eps``."""

    reward_prior_checkpoint: str | None = None
    """``hl_skill_diffsr`` checkpoint whose next-chunk head gives the fixed
    prior reward ``g`` (``rlopt.agent.ipmd.reward_prior``). ``None`` disables
    the prior. The PPO reward becomes ``env_reward_weight * r_env (+ est) + g``
    and the estimator never sees ``g`` (MaxEnt IRL with the reward
    regularizer centred at ``g``)."""

    reward_prior_obs_key: list[str] = field(
        default_factory=lambda: ["reward_prior_state", "macro_pair"]
    )
    """Rollout key of the per-step raw ``[robot | reference]`` macro components."""

    reward_prior_weight: float = 0.3
    """Per-step scale of ``g = weight * stride * exp(r_B / kappa)``."""

    reward_prior_kappa: float = 150.0
    """Temperature of ``exp(r_B / kappa)``; ``r_B`` is a negative denoising error."""

    reward_prior_stride: int = 10
    """Evaluate one chunk every ``stride`` steps of each episode."""

    reward_prior_draws: int = 1
    """Noise draws per diffusion level in the ``r_B`` estimate."""

    reward_target_ema_tau: float = 0.0
    """Polyak rate of an EMA copy of the reward estimator that rewards the
    policy (``0`` = off: the policy sees the latest estimator). Averaging the
    reward iterates of the saddle (``rlopt.agent.ipmd.reward_averaging``)."""

    reward_policy_replay_size: int = 0
    """Rows in a reservoir replay of past policy reward inputs (``0`` = off).
    The estimator's policy rows then mix the current batch with a uniform
    sample of all past rows: the classifier fits ``log p_E / p_bar`` against
    the mixture of past policies (averaging the policy iterates)."""

    reward_policy_replay_fraction: float = 0.5
    """Fraction of each estimator policy batch drawn from the replay."""

    reward_policy_replay_rows_per_rollout: int = 65536
    """Non-done rollout rows offered to the replay per rollout (subsampled)."""

    reward_policy_replay_mode: str = "reservoir"
    """``"reservoir"``: uniform sample of the whole history. ``"fifo"``: old
    rows leave first-in-first-out (a SAC-style window of the last
    ``size / rows_per_rollout`` rollouts) and rows are drawn by prioritized
    replay on the per-row classifier loss
    (``rlopt.agent.ipmd.reward_averaging.PrioritizedFifoFeatureReplay``)."""

    reward_policy_replay_priority_alpha: float = 0.0
    """FIFO replay: ``P(i) ∝ (loss_i + eps)^alpha``; ``0`` = uniform."""

    reward_policy_replay_priority_beta: float = 1.0
    """FIFO replay: importance weight ``(N P(i))^-beta``; ``1`` keeps the
    weighted classifier loss unbiased for the uniform-over-window loss."""

    reward_absorbing_terminal: bool = False
    """Terminated (non-truncated) rows enter an absorbing state (Kostrikov et
    al. 2019, DAC). The absorbing state is never visited by the reference, so
    its reward is the floor ``r_floor`` of the estimate's range (0 for the
    sigmoid/amp/affine squashes, ``estimated_reward_clamp_min`` for
    ``"none"``). The estimate on a terminated row becomes
    ``scale * r_floor / (1 - gamma)``: the floor for the terminal transition
    plus ``gamma * r_floor / (1 - gamma)`` for the absorbing tail. Applied after
    the done mask; reward shifts become policy-invariant under termination."""

    reward_prior_transform: str = "log"
    """Fixed map ``h(r_B)`` to [0, 1]: ``"log"`` (graded over the whole range)
    or ``"exp"`` (``exp(r_B / kappa)``)."""

    reward_prior_log_range: list[float] = field(default_factory=lambda: [5.0, 1.0e5])
    """``(lo, hi)`` of the ``"log"`` map: ``-r_B <= lo`` maps to 1, ``>= hi`` to 0."""

    expert_batch_size: int | None = None
    """Batch size for expert data sampling. If None, uses the same as mini_batch_size."""

    bc_coef: float = 0.0
    """Behavior cloning (MLE) loss coefficient on expert actions.

    When > 0, adds ``-bc_coef * mean(log_prob(expert_action | policy))``
    to each update step.  This regularises the policy toward expert actions
    during early training.  Set to 0 to disable (default).
    """

    bc_pretrain_updates: int = 0
    """Number of initial optimizer updates that use BC without PPO gradients.

    This provides an offline policy bootstrap when early on-policy rollouts are
    too short to learn from. After this many updates, the usual PPO plus BC
    update resumes. Set to 0 to mix BC and PPO from the first update.
    """

    rollout_bc_coef: float = 0.0
    """Behavior-cloning coefficient on states visited by the current policy.

    The target is read from ``rollout_bc_action_key`` in the rollout. This key
    is a training-only label and must not be an actor, critic, reward-model, or
    latent-model input. Unlike ``bc_coef``, this loss keeps the policy state
    distribution from the live rollout instead of replacing it with clean
    demonstration observations.
    """

    rollout_bc_loss_type: str = "nll"
    """Loss used for live-state action supervision: ``"nll"`` or ``"mse"``.

    ``"nll"`` preserves the probabilistic behavior-cloning objective. ``"mse"``
    supervises the deterministic policy mean and is useful when matching an
    action-reconstruction auxiliary objective independently of exploration std.
    """

    rollout_bc_action_key: ObsKey = ("policy_supervision", "expert_action")
    """Rollout observation key containing the aligned expert action label."""

    actor_learning_rate: float | None = None
    """Optional actor-specific learning rate for PPO/IPMD.

    When set together with ``critic_learning_rate``, the main optimizer uses
    separate actor and critic parameter groups. The adaptive KL schedule updates
    only the actor group, matching asymmetric actor/critic PPO recipes.
    """

    critic_learning_rate: float | None = None
    """Optional critic-specific learning rate for PPO/IPMD."""

    critic_lr_schedule: str = "constant"
    """Decay shape for the critic group: ``"constant"``, ``"linear"`` or ``"cosine"``.

    The adaptive KL rule owns the ACTOR group only (``adaptive_lr=True``), so the
    critic learning rate is otherwise fixed for the whole run. A non-constant
    schedule interpolates ``critic_learning_rate`` -> ``critic_lr_final`` on
    cumulative environment frames, not on the in-segment iteration counter, so a
    walltime-chained resume continues the decay instead of restarting it.
    """

    critic_lr_final: float | None = None
    """End-of-budget critic learning rate. Required when the schedule is not constant."""

    first_layer_bias_lr_scale: float = 1.0
    """Learning-rate multiplier for the FIRST linear layer's bias, actor and critic.

    ``1.0`` (default) changes nothing. Any other value moves each network's
    first-layer bias into its own parameter group at ``scale x`` the parent
    group's learning rate: ``actor_first_bias`` (adaptive with the actor, so the
    KL rule keeps the ratio; ``lr_min``/``lr_max`` scaled to match) and
    ``critic_first_bias`` (follows ``critic_lr_schedule``). Weight decay is the
    parent group's. Requires split actor/critic learning rates.

    Why this knob exists (2026-09-01): a constant input channel (the hold-1
    sin/cos phase pair, which is ``(0, 1)`` on every step) has a weight column
    whose gradient equals the bias gradient, so under Adam the effective
    first-layer bias moves at exactly twice the rate. This scale reproduces
    that effect without the channel, to test whether it is what separates the
    64-D hold-1 arms that train from those that stall.
    """

    def validate(self) -> None:
        self.command_source = normalize_ipmd_command_source(self.command_source)
        self.first_layer_bias_lr_scale = float(self.first_layer_bias_lr_scale)
        if self.first_layer_bias_lr_scale <= 0.0:
            msg = (
                "ipmd.first_layer_bias_lr_scale must be > 0, got "
                f"{self.first_layer_bias_lr_scale}."
            )
            raise ValueError(msg)
        if self.first_layer_bias_lr_scale != 1.0 and self.actor_learning_rate is None:
            msg = (
                "ipmd.first_layer_bias_lr_scale != 1 requires split actor/critic "
                "learning rates (ipmd.actor_learning_rate and "
                "ipmd.critic_learning_rate)."
            )
            raise ValueError(msg)
        self.reward_model_type = normalize_ipmd_reward_model_type(
            self.reward_model_type
        )
        self.reward_input_type = normalize_ipmd_reward_input_type(
            self.reward_input_type
        )
        self.reward_output_activation = normalize_ipmd_reward_output_activation(
            self.reward_output_activation
        )
        self.reward_objective = normalize_ipmd_reward_objective(self.reward_objective)
        self.reward_ppo_squash = normalize_ipmd_reward_ppo_squash(
            self.reward_ppo_squash
        )
        self.latent_dim = require_positive_int("ipmd.latent_dim", self.latent_dim)
        self.latent_steps_min = require_positive_int(
            "ipmd.latent_steps_min", self.latent_steps_min
        )
        self.latent_steps_max = require_positive_int(
            "ipmd.latent_steps_max", self.latent_steps_max
        )
        if (
            self.command_source == "token_planner"
            and not str(self.token_planner_checkpoint_path).strip()
        ):
            msg = (
                "ipmd.token_planner_checkpoint_path is required when "
                "command_source='token_planner'."
            )
            raise ValueError(msg)
        if (
            self.command_source == "continuous_planner"
            and not str(self.continuous_planner_checkpoint_path).strip()
        ):
            msg = (
                "ipmd.continuous_planner_checkpoint_path is required when "
                "command_source='continuous_planner'."
            )
            raise ValueError(msg)
        if self.command_source in ("hl_skill", "skill_commander"):
            self.hl_skill_horizon_steps = int(self.hl_skill_horizon_steps)
            if self.hl_skill_horizon_steps < 0:
                msg = "ipmd.hl_skill_horizon_steps must be >= 0."
                raise ValueError(msg)
            if self.hl_skill_horizon_steps > 0:
                self.hl_skill_horizon_steps = require_positive_int(
                    "ipmd.hl_skill_horizon_steps",
                    self.hl_skill_horizon_steps,
                )
            self.hl_skill_command_mode = str(self.hl_skill_command_mode).strip().lower()
            hl_skill_command_aliases = {"fz": "phi", "z_fz": "z_phi"}
            self.hl_skill_command_mode = hl_skill_command_aliases.get(
                self.hl_skill_command_mode,
                self.hl_skill_command_mode,
            )
            if self.hl_skill_command_mode not in {"z", "phi", "z_phi"}:
                msg = (
                    "ipmd.hl_skill_command_mode must be one of 'z', 'phi', "
                    f"or 'z_phi', got {self.hl_skill_command_mode!r}."
                )
                raise ValueError(msg)
        if self.command_source == "hl_skill":
            self.hl_skill_checkpoint_path = str(self.hl_skill_checkpoint_path).strip()
            if not self.hl_skill_checkpoint_path:
                msg = (
                    "ipmd.hl_skill_checkpoint_path is required when "
                    "ipmd.command_source='hl_skill'."
                )
                raise ValueError(msg)
        if self.command_source == "skill_commander":
            self.skill_commander_checkpoint_path = str(
                self.skill_commander_checkpoint_path
            ).strip()
            self.skill_commander_embeddings_path = str(
                self.skill_commander_embeddings_path
            ).strip()
            if not self.skill_commander_checkpoint_path:
                msg = (
                    "ipmd.skill_commander_checkpoint_path is required when "
                    "ipmd.command_source='skill_commander'."
                )
                raise ValueError(msg)
            self.skill_commander_goal_name = str(self.skill_commander_goal_name).strip()
            self.skill_commander_goal_rank = int(self.skill_commander_goal_rank)
            if self.skill_commander_goal_rank < -1:
                msg = "ipmd.skill_commander_goal_rank must be >= -1."
                raise ValueError(msg)
            if self.skill_commander_goal_name and self.skill_commander_goal_rank >= 0:
                msg = (
                    "Set only one of ipmd.skill_commander_goal_name or "
                    "ipmd.skill_commander_goal_rank."
                )
                raise ValueError(msg)
            self.skill_commander_flow_num_inference_steps = int(
                self.skill_commander_flow_num_inference_steps
            )
            if self.skill_commander_flow_num_inference_steps < 0:
                msg = "ipmd.skill_commander_flow_num_inference_steps must be >= 0."
                raise ValueError(msg)
            self.skill_commander_diffusion_num_inference_steps = int(
                self.skill_commander_diffusion_num_inference_steps
            )
            if self.skill_commander_diffusion_num_inference_steps < 0:
                msg = "ipmd.skill_commander_diffusion_num_inference_steps must be >= 0."
                raise ValueError(msg)
            scheduler = str(self.skill_commander_diffusion_inference_scheduler).strip()
            self.skill_commander_diffusion_inference_scheduler = scheduler.lower()
            if self.skill_commander_diffusion_inference_scheduler not in {
                "",
                "ddpm",
                "ddim",
            }:
                msg = (
                    "ipmd.skill_commander_diffusion_inference_scheduler must be "
                    "'ddpm', 'ddim', or empty."
                )
                raise ValueError(msg)
            self.skill_commander_flow_inference_noise_std = float(
                self.skill_commander_flow_inference_noise_std
            )
            if self.skill_commander_flow_inference_noise_std >= 0.0:
                require_non_negative(
                    "ipmd.skill_commander_flow_inference_noise_std",
                    self.skill_commander_flow_inference_noise_std,
                )
            self.skill_commander_diffusion_ddim_eta = float(
                self.skill_commander_diffusion_ddim_eta
            )
            if self.skill_commander_diffusion_ddim_eta >= 0.0:
                require_non_negative(
                    "ipmd.skill_commander_diffusion_ddim_eta",
                    self.skill_commander_diffusion_ddim_eta,
                )
            self.skill_commander_diffusion_inference_noise_std = float(
                self.skill_commander_diffusion_inference_noise_std
            )
            if self.skill_commander_diffusion_inference_noise_std >= 0.0:
                require_non_negative(
                    "ipmd.skill_commander_diffusion_inference_noise_std",
                    self.skill_commander_diffusion_inference_noise_std,
                )
        if self.hl_skill_finetune_enabled and self.command_source != "hl_skill":
            msg = (
                "ipmd.hl_skill_finetune_enabled=True requires "
                "ipmd.command_source='hl_skill'."
            )
            raise ValueError(msg)
        require_non_negative("ipmd.hl_skill_pg_coeff", self.hl_skill_pg_coeff)
        require_non_negative(
            "ipmd.hl_skill_offline_diffsr_coeff",
            self.hl_skill_offline_diffsr_coeff,
        )
        require_non_negative("ipmd.hl_skill_anchor_coeff", self.hl_skill_anchor_coeff)
        if self.hl_skill_z_norm_coeff is not None:
            require_non_negative(
                "ipmd.hl_skill_z_norm_coeff",
                self.hl_skill_z_norm_coeff,
            )
        if float(self.hl_skill_lr) <= 0.0:
            msg = f"ipmd.hl_skill_lr must be > 0, got {self.hl_skill_lr!r}."
            raise ValueError(msg)
        if (
            self.hl_skill_grad_clip_norm is not None
            and float(self.hl_skill_grad_clip_norm) <= 0.0
        ):
            msg = (
                "ipmd.hl_skill_grad_clip_norm must be > 0 or None, got "
                f"{self.hl_skill_grad_clip_norm!r}."
            )
            raise ValueError(msg)
        self.hl_skill_offline_batch_size = require_positive_int(
            "ipmd.hl_skill_offline_batch_size",
            self.hl_skill_offline_batch_size,
        )
        self.hl_skill_update_interval = require_positive_int(
            "ipmd.hl_skill_update_interval",
            self.hl_skill_update_interval,
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
        self._validate_reward_objective_fields()
        require_non_negative("ipmd.bc_coef", self.bc_coef)
        require_non_negative("ipmd.rollout_bc_coef", self.rollout_bc_coef)
        if self.actor_learning_rate is not None and self.actor_learning_rate <= 0.0:
            msg = (
                "ipmd.actor_learning_rate must be > 0 when configured, got "
                f"{self.actor_learning_rate!r}."
            )
            raise ValueError(msg)
        if self.critic_learning_rate is not None and self.critic_learning_rate <= 0.0:
            msg = (
                "ipmd.critic_learning_rate must be > 0 when configured, got "
                f"{self.critic_learning_rate!r}."
            )
            raise ValueError(msg)
        if (self.actor_learning_rate is None) != (self.critic_learning_rate is None):
            msg = (
                "ipmd.actor_learning_rate and ipmd.critic_learning_rate must be "
                "configured together."
            )
            raise ValueError(msg)
        self.critic_lr_schedule = str(self.critic_lr_schedule).strip().lower()
        if self.critic_lr_schedule not in {"constant", "linear", "cosine"}:
            msg = (
                "ipmd.critic_lr_schedule must be 'constant', 'linear' or "
                f"'cosine', got {self.critic_lr_schedule!r}."
            )
            raise ValueError(msg)
        if self.critic_lr_schedule != "constant":
            if self.critic_learning_rate is None:
                msg = (
                    "ipmd.critic_lr_schedule="
                    f"{self.critic_lr_schedule!r} requires ipmd.critic_learning_rate "
                    "(the schedule drives the named critic parameter group, which "
                    "only exists under split actor/critic learning rates)."
                )
                raise ValueError(msg)
            if self.critic_lr_final is None or self.critic_lr_final <= 0.0:
                msg = (
                    "ipmd.critic_lr_final must be > 0 when "
                    f"ipmd.critic_lr_schedule={self.critic_lr_schedule!r}, got "
                    f"{self.critic_lr_final!r}."
                )
                raise ValueError(msg)
        self.rollout_bc_loss_type = str(self.rollout_bc_loss_type).strip().lower()
        if self.rollout_bc_loss_type not in {"nll", "mse"}:
            msg = (
                "ipmd.rollout_bc_loss_type must be 'nll' or 'mse', got "
                f"{self.rollout_bc_loss_type!r}."
            )
            raise ValueError(msg)
        self.rollout_bc_action_key = normalize_batch_key(self.rollout_bc_action_key)
        if int(self.bc_pretrain_updates) < 0:
            msg = (
                "ipmd.bc_pretrain_updates must be >= 0, got "
                f"{self.bc_pretrain_updates!r}."
            )
            raise ValueError(msg)
        if (
            int(self.bc_pretrain_updates) > 0
            and float(self.bc_coef) <= 0.0
            and float(self.rollout_bc_coef) <= 0.0
        ):
            msg = (
                "ipmd.bc_pretrain_updates requires ipmd.bc_coef > 0 or "
                "ipmd.rollout_bc_coef > 0."
            )
            raise ValueError(msg)
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

    def _validate_reward_objective_fields(self) -> None:
        """Checks for the reward-estimation revival fields (plan 5.4 B2/B3)."""
        objective = self.reward_objective
        if objective != "ipmd_diff":
            if self.reward_output_activation != "none":
                msg = (
                    f"ipmd.reward_objective={objective!r} requires "
                    "ipmd.reward_output_activation='none' (the objective's reward "
                    "is the raw head output), got "
                    f"{self.reward_output_activation!r}."
                )
                raise ValueError(msg)
            if self.reward_model_type != "mlp":
                msg = (
                    f"ipmd.reward_objective={objective!r} requires "
                    "ipmd.reward_model_type='mlp'; grouped heads implement the "
                    "ipmd_diff objective only."
                )
                raise ValueError(msg)
        require_non_negative("ipmd.estimated_reward_scale", self.estimated_reward_scale)
        if int(self.reward_ramp_updates) < 0:
            msg = f"ipmd.reward_ramp_updates must be >= 0, got {self.reward_ramp_updates!r}."
            raise ValueError(msg)
        self.reward_ramp_updates = int(self.reward_ramp_updates)
        if float(self.reward_whiten_ridge) <= 0.0:
            msg = f"ipmd.reward_whiten_ridge must be > 0, got {self.reward_whiten_ridge!r}."
            raise ValueError(msg)
        if float(self.reward_linear_weight_decay) <= 0.0:
            msg = (
                "ipmd.reward_linear_weight_decay must be > 0 (the dual fixed point "
                f"is Delta / lambda), got {self.reward_linear_weight_decay!r}."
            )
            raise ValueError(msg)
        momentum = float(self.reward_moment_momentum)
        if momentum > 1.0:
            msg = (
                "ipmd.reward_moment_momentum must be <= 1 (<= 0 selects the exact "
                f"cumulative mean), got {self.reward_moment_momentum!r}."
            )
            raise ValueError(msg)
        if not 0.0 <= float(self.reward_target_ema_tau) <= 1.0:
            raise ValueError(
                "ipmd.reward_target_ema_tau must lie in [0, 1], got "
                f"{self.reward_target_ema_tau!r}."
            )
        if int(self.reward_policy_replay_size) < 0:
            raise ValueError("ipmd.reward_policy_replay_size must be >= 0.")
        if not 0.0 <= float(self.reward_policy_replay_fraction) <= 1.0:
            raise ValueError("ipmd.reward_policy_replay_fraction must lie in [0, 1].")
        if self.reward_policy_replay_mode not in {"reservoir", "fifo"}:
            raise ValueError(
                "ipmd.reward_policy_replay_mode must be 'reservoir' or 'fifo', got "
                f"{self.reward_policy_replay_mode!r}."
            )
        alpha = float(self.reward_policy_replay_priority_alpha)
        if alpha < 0.0 or not 0.0 <= float(self.reward_policy_replay_priority_beta) <= 1.0:
            raise ValueError(
                "ipmd.reward_policy_replay_priority_alpha must be >= 0 and "
                "ipmd.reward_policy_replay_priority_beta in [0, 1]."
            )
        if alpha > 0.0 and (
            self.reward_policy_replay_mode != "fifo"
            or normalize_ipmd_reward_objective(self.reward_objective) != "logistic_ratio"
        ):
            raise ValueError(
                "Prioritized replay (reward_policy_replay_priority_alpha > 0) needs "
                "reward_policy_replay_mode='fifo' and reward_objective='logistic_ratio'."
            )
        if self.reward_absorbing_terminal and float(self.estimated_reward_done_penalty) != 0.0:
            raise ValueError(
                "ipmd.reward_absorbing_terminal replaces estimated_reward_done_penalty; "
                "set the penalty to 0."
            )
        if self.reward_prior_checkpoint is not None:
            if float(self.reward_prior_kappa) <= 0.0 or int(self.reward_prior_stride) < 1:
                raise ValueError(
                    "ipmd.reward_prior_kappa must be > 0 and ipmd.reward_prior_stride >= 1."
                )
            if int(self.reward_prior_draws) < 1:
                raise ValueError("ipmd.reward_prior_draws must be >= 1.")
            if self.reward_prior_transform not in {"log", "exp"}:
                raise ValueError(
                    "ipmd.reward_prior_transform must be 'log' or 'exp', got "
                    f"{self.reward_prior_transform!r}."
                )
        smoothing = float(self.reward_label_smoothing)
        if not 0.0 <= smoothing < 0.5:
            msg = (
                "ipmd.reward_label_smoothing must lie in [0, 0.5), got "
                f"{self.reward_label_smoothing!r}."
            )
            raise ValueError(msg)

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class IPMDRLOptConfig(PPORLOptConfig):
    """IPMD configuration extending PPORLOptConfig (PPO-based)."""

    dagger: DAggerConfig = field(default_factory=DAggerConfig)
    """Optional frozen-teacher warm start; ordinary IPMD ignores disabled settings."""

    ipmd: IPMDConfig = field(default_factory=IPMDConfig)
    """IPMD configuration."""

    abort_on_nonfinite: bool = True
    """Stop the run the first time a rollout metric or policy weight is not
    finite.

    A NaN policy keeps training at full speed and silently poisons every later
    checkpoint; two chains lost about a day each that way (2026-08-23 fsq64,
    2026-08-24 emastack) before eval caught it. Set false only to reproduce
    such a failure on purpose."""


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


def _first_linear_bias(module: torch.nn.Module) -> torch.nn.Parameter:
    """The bias of the first ``nn.Linear`` in ``module.modules()`` order.

    Wrappers (TensorDictModule, the running-mean-std input normalizer) register
    no linear layers, so the first one met is the first hidden layer of the
    MLP.
    """
    for sub in module.modules():
        if isinstance(sub, torch.nn.Linear) and sub.bias is not None:
            return sub.bias
    msg = "first_layer_bias_lr_scale: no nn.Linear with a bias found in the module."
    raise ValueError(msg)


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

    def _latent_command_consumer_obs_keys(self) -> list[ObsKey]:
        """Return the observation keys of the policy that consumes the latent.

        Ordinary IPMD publishes the latent to its main policy. Asymmetric
        variants can override this hook when another policy consumes the same
        command while the main rollout policy uses a different command view.
        """
        return self._policy_obs_keys

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
        if bool(self.config.ipmd.hl_skill_finetune_enabled) and bool(
            self.config.compile.compile
        ):
            msg = (
                "ipmd.hl_skill_finetune_enabled=True is not compatible with "
                "compile.compile=True in the v1 rollout-cache implementation."
            )
            raise ValueError(msg)
        self.env = env
        self._reward_model_type = self.config.ipmd.reward_model_type
        self._use_latent_command = bool(self.config.ipmd.use_latent_command)
        self._latent_key = normalize_batch_key(self.config.ipmd.latent_key)
        self._latent_dim = int(self.config.ipmd.latent_dim)
        self._latent_command_controller: LatentCommandController | None = None
        self._hl_skill_command_sampler: Any | None = None
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
        self._rollout_bc_action_key = normalize_batch_key(
            self.config.ipmd.rollout_bc_action_key
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
        target_keys = latent_cfg.reconstruction_target_keys or self._posterior_obs_keys
        self._reconstruction_target_obs_keys: list[ObsKey] = dedupe_keys(
            [normalize_batch_key(key) for key in target_keys]
        )
        self._uses_internal_latent_learner = bool(
            self._use_latent_command
            and self._command_source not in ("hl_skill", "skill_commander")
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
            latent_consumer_keys = self._latent_command_consumer_obs_keys()
            if self._latent_key not in latent_consumer_keys:
                msg = (
                    "IPMD use_latent_command=True requires the latent-command "
                    "consumer input keys to contain "
                    f"{self._latent_key!r}, got {latent_consumer_keys!r}."
                )
                msg += self._latent_mode_hint()
                raise ValueError(msg)

        all_obs_keys = list(
            self._policy_obs_keys + self._value_obs_keys + self._reward_obs_keys
        )
        if self._uses_internal_latent_learner:
            all_obs_keys.extend(
                self._posterior_obs_keys
                + self._prior_obs_keys
                + self._reconstruction_target_obs_keys
            )
        all_obs_keys = dedupe_keys(all_obs_keys)
        if (
            float(self.config.ipmd.rollout_bc_coef) > 0.0
            and self._rollout_bc_action_key in all_obs_keys
        ):
            msg = (
                "ipmd.rollout_bc_action_key is a training-only label and must not "
                "be used by the policy, value function, reward model, posterior, "
                f"prior, or reconstruction model: {self._rollout_bc_action_key!r}."
            )
            raise ValueError(msg)
        for key in all_obs_keys:
            env.observation_spec[key]
        if float(self.config.ipmd.rollout_bc_coef) > 0.0:
            try:
                env.observation_spec[self._rollout_bc_action_key]
            except KeyError as err:
                msg = (
                    "ipmd.rollout_bc_coef > 0 requires the environment observation "
                    f"{self._rollout_bc_action_key!r}."
                )
                raise ValueError(msg) from err
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

        # Reward-estimation revival (plan 5.4): objective and PPO squash are
        # structural, so they are resolved once; the mixing scalars are read
        # live from the config like est/env weights.
        self._reward_objective: str = self.config.ipmd.reward_objective
        self._reward_ppo_squash: str = resolve_ipmd_reward_ppo_squash(
            self._reward_objective, self.config.ipmd.reward_ppo_squash
        )
        self._reward_mix_state: dict[str, Tensor] = self._initial_reward_mix_state()

        # Fixed prior reward g (2026-10-11): a frozen DiffSR next-chunk head
        # scored on rollouts only. None when ipmd.reward_prior_checkpoint is unset.
        self._reward_prior = None
        if self.config.ipmd.reward_prior_checkpoint:
            from rlopt.agent.ipmd.reward_prior import DiffSRChunkPrior

            cfg_ipmd = self.config.ipmd
            self._reward_prior = DiffSRChunkPrior(
                cfg_ipmd.reward_prior_checkpoint,
                num_envs=int(self.env.batch_size[0]),
                device=self.device,
                weight=float(cfg_ipmd.reward_prior_weight),
                kappa=float(cfg_ipmd.reward_prior_kappa),
                stride=int(cfg_ipmd.reward_prior_stride),
                draws=int(cfg_ipmd.reward_prior_draws),
                transform=str(cfg_ipmd.reward_prior_transform),
                log_range=tuple(float(v) for v in cfg_ipmd.reward_prior_log_range),
            )

        # Reward estimator: r(s, a, s') -> scalar
        self.reward_estimator: torch.nn.Module = self._construct_reward_estimator()
        # Averaging of the reward saddle (2026-10-12): an EMA target that
        # rewards the policy, and a reservoir replay of past policy inputs.
        self._reward_target: torch.nn.Module | None = None
        tau = float(self.config.ipmd.reward_target_ema_tau)
        if tau > 0.0 and any(True for _ in self.reward_estimator.parameters()):
            import copy

            self._reward_target = copy.deepcopy(self.reward_estimator).requires_grad_(False)
        self._reward_target_tau = tau
        self._reward_replay: ReservoirFeatureReplay | PrioritizedFifoFeatureReplay | None = None
        # Set by _reward_pair_inputs when the replay is on: the current batch's
        # policy rows (for the current-rows AUC) and, for the FIFO replay, the
        # drawn slots and importance weights of the replay rows.
        self._reward_current_policy_input: Tensor | None = None
        self._reward_replay_slots: Tensor | None = None
        self._reward_replay_weights: Tensor | None = None
        cfg_replay = self.config.ipmd
        if int(cfg_replay.reward_policy_replay_size) > 0:
            if cfg_replay.reward_policy_replay_mode == "fifo":
                self._reward_replay = PrioritizedFifoFeatureReplay(
                    int(cfg_replay.reward_policy_replay_size),
                    int(self._reward_input_dim),
                    self.device,
                    alpha=float(cfg_replay.reward_policy_replay_priority_alpha),
                    beta=float(cfg_replay.reward_policy_replay_priority_beta),
                )
            else:
                self._reward_replay = ReservoirFeatureReplay(
                    int(cfg_replay.reward_policy_replay_size), int(self._reward_input_dim), self.device
                )
        self.reward_estimator.to(self.device)
        self.reward_optim: torch.optim.Optimizer | None = None

        self._expert_batch_sampler: (
            Callable[[int, list[BatchKey]], TensorDict | None] | None
        ) = None
        self._offline_expert_batch_sampler: (
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
        self._rollout_bc_coeff: float = float(cfg.rollout_bc_coef)
        self._bc_pretrain_updates: int = int(cfg.bc_pretrain_updates)
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

        if self._uses_internal_latent_learner:
            method = str(self.config.ipmd.latent_learning.method)
            self._latent_learner = build_latent_learner(method)
            self._latent_learner.initialize(self)
        if self._use_latent_command and self._command_source == "hl_skill":
            from rlopt.agent.hl_skill_diffsr import (  # noqa: PLC0415
                FrozenHighLevelSkillCommandSampler,
            )

            self._hl_skill_command_sampler = FrozenHighLevelSkillCommandSampler(
                env=self.env,
                checkpoint_path=str(self.config.ipmd.hl_skill_checkpoint_path),
                latent_dim=self._latent_dim,
                latent_steps_min=int(self.config.ipmd.latent_steps_min),
                latent_steps_max=int(self.config.ipmd.latent_steps_max),
                horizon_steps=(
                    self.config.ipmd.hl_skill_horizon_steps
                    if self.config.ipmd.hl_skill_horizon_steps > 0
                    else None
                ),
                command_phase_mode=str(
                    self.config.ipmd.latent_learning.command_phase_mode
                ),
                code_latent_dim=self.config.ipmd.latent_learning.code_latent_dim,
                phase_period=int(
                    self.config.ipmd.latent_learning.command_phase_period
                    or self.config.ipmd.latent_learning.code_period
                ),
                phase_source=str(self.config.ipmd.latent_learning.command_phase_source),
                command_mode=str(self.config.ipmd.hl_skill_command_mode),
                command_quantizer=str(
                    self.config.ipmd.latent_learning.command_quantizer
                ),
                command_fsq_levels=list(
                    self.config.ipmd.latent_learning.command_fsq_levels
                ),
                discover_env_method=self._discover_env_method,
                device=self._get_device(self.config.device),
                finetune_enabled=bool(self.config.ipmd.hl_skill_finetune_enabled),
                pg_coeff=float(self.config.ipmd.hl_skill_pg_coeff),
                offline_diffsr_coeff=float(
                    self.config.ipmd.hl_skill_offline_diffsr_coeff
                ),
                anchor_coeff=float(self.config.ipmd.hl_skill_anchor_coeff),
                achieved_coeff=float(self.config.ipmd.hl_skill_achieved_coeff),
                z_norm_coeff=self.config.ipmd.hl_skill_z_norm_coeff,
                lr=float(self.config.ipmd.hl_skill_lr),
                grad_clip_norm=self.config.ipmd.hl_skill_grad_clip_norm,
                offline_batch_size=int(self.config.ipmd.hl_skill_offline_batch_size),
                update_interval=int(self.config.ipmd.hl_skill_update_interval),
                train_diffsr=bool(self.config.ipmd.hl_skill_train_diffsr),
                online_objective=str(self.config.ipmd.hl_skill_online_objective),
                live_offset_ring=int(self.config.ipmd.hl_skill_live_offset_ring),
            )
        if self._use_latent_command and self._command_source == "skill_commander":
            from rlopt.agent.skill_commander import (  # noqa: PLC0415
                FrozenSkillCommanderSampler,
            )

            commander_overrides: dict[str, object] = {}
            if self.config.ipmd.skill_commander_flow_num_inference_steps > 0:
                commander_overrides["flow_num_inference_steps"] = int(
                    self.config.ipmd.skill_commander_flow_num_inference_steps
                )
            if self.config.ipmd.skill_commander_flow_inference_noise_std >= 0.0:
                commander_overrides["flow_inference_noise_std"] = float(
                    self.config.ipmd.skill_commander_flow_inference_noise_std
                )
            if self.config.ipmd.skill_commander_diffusion_num_inference_steps > 0:
                commander_overrides["diffusion_num_inference_steps"] = int(
                    self.config.ipmd.skill_commander_diffusion_num_inference_steps
                )
            if self.config.ipmd.skill_commander_diffusion_inference_scheduler:
                commander_overrides["diffusion_inference_scheduler"] = str(
                    self.config.ipmd.skill_commander_diffusion_inference_scheduler
                )
            if self.config.ipmd.skill_commander_diffusion_ddim_eta >= 0.0:
                commander_overrides["diffusion_ddim_eta"] = float(
                    self.config.ipmd.skill_commander_diffusion_ddim_eta
                )
            if self.config.ipmd.skill_commander_diffusion_inference_noise_std >= 0.0:
                commander_overrides["diffusion_inference_noise_std"] = float(
                    self.config.ipmd.skill_commander_diffusion_inference_noise_std
                )

            if str(self.config.ipmd.latent_learning.command_phase_source) != "hold":
                msg = (
                    "command_phase_source='episode' is only wired into the "
                    "hl_skill command sampler; the skill-commander path still "
                    "uses the hold clock."
                )
                raise ValueError(msg)
            self._hl_skill_command_sampler = FrozenSkillCommanderSampler(
                env=self.env,
                checkpoint_path=str(self.config.ipmd.skill_commander_checkpoint_path),
                language_embeddings_path=str(
                    self.config.ipmd.skill_commander_embeddings_path
                ),
                latent_dim=self._latent_dim,
                latent_steps_min=int(self.config.ipmd.latent_steps_min),
                latent_steps_max=int(self.config.ipmd.latent_steps_max),
                generator_config_overrides=commander_overrides,
                horizon_steps=(
                    self.config.ipmd.hl_skill_horizon_steps
                    if self.config.ipmd.hl_skill_horizon_steps > 0
                    else None
                ),
                command_phase_mode=str(
                    self.config.ipmd.latent_learning.command_phase_mode
                ),
                code_latent_dim=self.config.ipmd.latent_learning.code_latent_dim,
                phase_period=int(self.config.ipmd.latent_learning.code_period),
                command_mode=str(self.config.ipmd.hl_skill_command_mode),
                use_achieved_state=bool(
                    self.config.ipmd.skill_commander_use_achieved_state
                ),
                goal_name=str(self.config.ipmd.skill_commander_goal_name),
                goal_rank=int(self.config.ipmd.skill_commander_goal_rank),
                discover_env_method=self._discover_env_method,
                device=self._get_device(self.config.device),
            )
        if self._use_latent_command and self._command_source == "token_planner":
            from rlopt.agent.causal_interface_planner import (  # noqa: PLC0415
                FrozenCategoricalTokenPlannerSampler,
            )

            if self._latent_learner is None:
                msg = "token_planner requires an initialized per-step latent learner."
                raise RuntimeError(msg)
            decode_token_ids = getattr(self._latent_learner, "decode_token_ids", None)
            if not callable(decode_token_ids):
                msg = (
                    "token_planner requires latent_learning.method="
                    "'per_step_vq_sequence'."
                )
                raise ValueError(msg)
            self._hl_skill_command_sampler = FrozenCategoricalTokenPlannerSampler(
                env=self.env,
                checkpoint_path=str(self.config.ipmd.token_planner_checkpoint_path),
                decode_token_ids=decode_token_ids,
                latent_dim=self._latent_dim,
                discover_env_method=self._discover_env_method,
                device=self._get_device(self.config.device),
            )
        if self._use_latent_command and self._command_source == "continuous_planner":
            from rlopt.agent.causal_interface_planner import (  # noqa: PLC0415
                FrozenContinuousInterfacePlannerSampler,
            )

            self._hl_skill_command_sampler = FrozenContinuousInterfacePlannerSampler(
                env=self.env,
                checkpoint_path=str(
                    self.config.ipmd.continuous_planner_checkpoint_path
                ),
                latent_dim=self._latent_dim,
                discover_env_method=self._discover_env_method,
                device=self._get_device(self.config.device),
            )
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
        self._auto_attach_offline_expert_sampler()
        self._validate_expert_transition_alignment()
        self._refresh_grad_clip_params()
        self._reward_grad_clip_params = list(self.reward_estimator.parameters())
        self._policy_operator = self.actor_critic.get_policy_operator()
        self._bc_debug_anomaly_prints = 0

    def _auto_attach_offline_expert_sampler(self) -> None:
        """Attach a configured offline dataset sampler after env construction."""
        sampler = build_offline_expert_sampler(self.config, self.env)
        if sampler is None:
            return
        self._offline_expert_batch_sampler = sampler
        offline_cfg = self.config.offline_dataset
        self.log.info(
            "Using offline dataset sampler: source=%s repo_id=%s mapper=%s",
            offline_cfg.source,
            offline_cfg.repo_id,
            offline_cfg.mapper,
        )
        if not bool(offline_cfg.replace_expert_sampler):
            return

        self._expert_batch_sampler = sampler
        self._expert_sampler_source_name = "offline_dataset"
        required_keys = (
            self._expert_required_keys()
            if self._expert_minibatch_update_enabled
            else []
        )
        if required_keys:
            preflight_batch_size = min(
                int(
                    self.config.ipmd.expert_batch_size
                    or self.config.loss.mini_batch_size
                ),
                8,
            )
            _ = sampler(preflight_batch_size, required_keys)

    def _next_offline_expert_batch(
        self,
        batch_size: int | None = None,
        required_keys: list[BatchKey] | None = None,
    ) -> TensorDict:
        if self._offline_expert_batch_sampler is None:
            return self._next_expert_batch(
                batch_size=batch_size,
                required_keys=required_keys,
            )
        assert isinstance(self.config, IPMDRLOptConfig)
        effective_batch_size = int(
            batch_size
            or self.config.ipmd.expert_batch_size
            or self.config.loss.mini_batch_size
        )
        required_keys = (
            self._expert_required_keys() if required_keys is None else required_keys
        )
        expert_batch = self._offline_expert_batch_sampler(
            effective_batch_size,
            required_keys,
        )
        if expert_batch is None:
            msg = "Offline expert sampler returned None."
            raise RuntimeError(msg)
        available_keys = set(expert_batch.keys(True))
        missing_keys = [key for key in required_keys if key not in available_keys]
        if missing_keys:
            msg = (
                "Offline expert sampler batch is missing required keys: "
                f"{missing_keys!r}."
            )
            raise RuntimeError(msg)
        if expert_batch.numel() != effective_batch_size:
            msg = (
                "Offline expert sampler must return exactly the requested batch size, got "
                f"{expert_batch.numel()} for request {effective_batch_size}."
            )
            raise RuntimeError(msg)
        return expert_batch.to(self.device)

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

    def collect(self, run: PPOTrainingMetadata, iteration_idx: int) -> PPOIterationData:
        """Collect one rollout, resetting per-rollout HL skill caches when needed."""
        if self._hl_skill_command_sampler is not None:
            start_cache = getattr(
                self._hl_skill_command_sampler,
                "start_rollout_cache",
                None,
            )
            if callable(start_cache):
                start_cache()
        return super().collect(run, iteration_idx)

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
        if self._command_source in (
            "hl_skill",
            "skill_commander",
            "token_planner",
            "continuous_planner",
        ):
            if self._hl_skill_command_sampler is None:
                msg = "Frozen high-level skill command sampler is unavailable."
                raise RuntimeError(msg)
            latents = self._hl_skill_command_sampler.sample_for_step(
                td,
                device=self.device,
                dtype=torch.float32,
            )
            td.set(
                self._latent_key,
                latents.reshape(*td.batch_size, self._latent_dim),
            )
            controller.publish_latents_to_env(latents.reshape(-1, self._latent_dim))
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
        if getattr(self, "_expert_sampler_source_name", None) == "offline_dataset":
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

    @property
    def _reward_estimator_is_linear(self) -> bool:
        return self._reward_objective == "linear_dual"

    def _initial_reward_mix_state(self) -> dict[str, Tensor]:
        """Frozen PPO-mixing state (plan 5.4 B3), checkpointed by the hooks.

        ``driving_rollouts``: rollouts seen with PPO driving on (the ramp
        clock). ``affine_center`` / ``affine_scale`` with ``affine_frozen``:
        the ``m0, s0`` of the affine_frozen squash. ``constant_bonus_mean``
        with ``constant_bonus_frozen``: the batch mean of ``r_hat`` that the
        constant-bonus control pays on alive rows.
        """
        device = self.device
        return {
            "driving_rollouts": torch.zeros((), dtype=torch.int64, device=device),
            "affine_center": torch.zeros((), device=device),
            "affine_scale": torch.ones((), device=device),
            "affine_frozen": torch.zeros((), dtype=torch.bool, device=device),
            "constant_bonus_mean": torch.zeros((), device=device),
            "constant_bonus_frozen": torch.zeros((), dtype=torch.bool, device=device),
        }

    def _construct_reward_estimator(self) -> torch.nn.Module:
        """Create reward network whose input depends on ``reward_input_type``."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)

        if self._reward_objective == "linear_dual":
            return LinearDualRewardHead(
                self._reward_input_dim,
                ridge=float(cfg.ipmd.reward_whiten_ridge),
                weight_decay=float(cfg.ipmd.reward_linear_weight_decay),
                momentum=float(cfg.ipmd.reward_moment_momentum),
                closed_form=bool(cfg.ipmd.reward_linear_closed_form),
                device=self.device,
            )

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
        """Compile the loss forward and GAE with torch.compile (if enabled).

        Deliberately narrower than PPO's blanket ``torch.compile(self.update)``:
        IPMD's ``update()`` interleaves optimizer stepping and host-side
        bookkeeping (``.item()``), which force Dynamo graph breaks. Compiling
        ``loss_module`` captures the actor+critic forward that autograd then
        differentiates; ``adv_module`` covers the GAE scan. The reward
        estimator is compiled only when reward updates are actually enabled.

        Measured 2026-08-18 on the tuned G1 recipe (8,192 envs, RTX PRO
        6000): the loss step is GEMM-bound, so the win is small and comes
        from GAE (80 -> 57 ms); ``update/ppo_terms`` moves ~1%. ``max-autotune``
        ties ``default`` and its CUDA-graphs variant crashes on latent-sampler
        tensor reuse -- use ``max-autotune-no-cudagraphs`` if autotuning.
        """
        if not self.config.compile.compile:
            return
        if not hasattr(self, "adv_module"):
            # Base-class __init__ calls this hook before PPO.__init__ has
            # built the advantage module; the second call compiles.
            return
        if getattr(self, "_components_compiled", False):
            return
        compile_mode = self.config.compile.compile_mode or "default"
        self.loss_module = torch.compile(self.loss_module, mode=compile_mode)
        self.adv_module = torch.compile(self.adv_module, mode=compile_mode)
        if self._reward_update_enabled and not self._reward_estimator_is_linear:
            self.reward_estimator = torch.compile(self.reward_estimator)
        self._components_compiled = True

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

        split_actor_critic_lr = self.config.ipmd.actor_learning_rate is not None
        if split_actor_critic_lr:
            assert self.config.ipmd.critic_learning_rate is not None
            # The exploration std is a bare nn.Parameter, so AdamW's decoupled
            # weight decay would pull it toward log_std=0, i.e. sigma toward
            # 1.0 rad -- against the anneal that carries this recipe. Give it
            # its own group at weight_decay=0 so optim.weight_decay can be
            # tuned on the networks alone. The group stays adaptive, so the KL
            # rule scales it with the rest of the actor.
            log_std_params = [
                param
                for name, param in self.policy.named_parameters()
                if "log_std" in name
            ]
            log_std_ids = {id(param) for param in log_std_params}
            policy_params = [
                param
                for param in list(self.policy.parameters()) + joint_params
                if id(param) not in log_std_ids
            ]
            value_params = list(self.value_function.parameters())
            weight_decay = float(self.config.optim.weight_decay)
            optimizer_params: list[dict[str, Any]] = [
                {
                    "params": policy_params,
                    "lr": float(self.config.ipmd.actor_learning_rate),
                    "adaptive_lr": True,
                    "lr_min": self.config.optim.min_lr,
                    "lr_max": self.config.optim.max_lr,
                    "weight_decay": weight_decay,
                    "name": "actor",
                },
                {
                    "params": value_params,
                    "lr": float(self.config.ipmd.critic_learning_rate),
                    "adaptive_lr": False,
                    "weight_decay": weight_decay,
                    "name": "critic",
                },
            ]
            if log_std_params:
                optimizer_params.append(
                    {
                        "params": log_std_params,
                        "lr": float(self.config.ipmd.actor_learning_rate),
                        "adaptive_lr": True,
                        "lr_min": self.config.optim.min_lr,
                        "lr_max": self.config.optim.max_lr,
                        "weight_decay": 0.0,
                        "name": "actor_log_std",
                    }
                )
            bias_scale = float(self.config.ipmd.first_layer_bias_lr_scale)
            if bias_scale != 1.0:
                actor_bias = _first_linear_bias(self.policy)
                critic_bias = _first_linear_bias(self.value_function)
                actor_bias_id = id(actor_bias)
                critic_bias_id = id(critic_bias)
                optimizer_params[0]["params"] = [
                    p for p in policy_params if id(p) != actor_bias_id
                ]
                optimizer_params[1]["params"] = [
                    p for p in value_params if id(p) != critic_bias_id
                ]

                def _scaled(value: float | None) -> float | None:
                    return None if value is None else float(value) * bias_scale

                optimizer_params.append(
                    {
                        "params": [actor_bias],
                        "lr": float(self.config.ipmd.actor_learning_rate) * bias_scale,
                        "adaptive_lr": True,
                        "lr_min": _scaled(self.config.optim.min_lr),
                        "lr_max": _scaled(self.config.optim.max_lr),
                        "weight_decay": weight_decay,
                        "name": "actor_first_bias",
                    }
                )
                optimizer_params.append(
                    {
                        "params": [critic_bias],
                        "lr": float(self.config.ipmd.critic_learning_rate) * bias_scale,
                        "adaptive_lr": False,
                        "weight_decay": weight_decay,
                        "name": "critic_first_bias",
                    }
                )
        else:
            optimizer_params = []

        if not hasattr(self, "reward_estimator"):
            if split_actor_critic_lr:
                return [optimizer_cls(optimizer_params, **optimizer_kwargs)]
            if not joint_params:
                return super()._set_optimizers(optimizer_cls, optimizer_kwargs)
            all_params = list(self.actor_critic.parameters()) + joint_params
            return [optimizer_cls(all_params, **optimizer_kwargs)]

        policy_params = list(self.actor_critic.parameters()) + joint_params
        if self._reward_estimator_is_linear:
            # The linear dual has buffers only; its step is closed-form or the
            # explicit dual update, so there is nothing for Adam to own.
            self.reward_optim = None
        else:
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
        if split_actor_critic_lr:
            return [optimizer_cls(optimizer_params, **optimizer_kwargs)]
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
        use_target: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute estimated reward for a batch of transitions.

        ``use_target`` evaluates the EMA target estimator when one exists
        (``ipmd.reward_target_ema_tau > 0``); the rollout reward uses it.
        """
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
        estimator = (
            self._reward_target
            if use_target and self._reward_target is not None
            else self.reward_estimator
        )
        reward = self._reward_out_fn(estimator(x))
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
        # The PPO base carries the LCP gradient penalty (ppo.lcp_coeff);
        # inert at the default 0.0.
        return super()._extra_actor_loss(batch)

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
        for key in self._reward_objective_metric_keys():
            metrics[key] = torch.zeros((), device=self.device)
        for key in self._grouped_reward_loss_metric_keys():
            metrics[key] = torch.zeros((), device=self.device)
        for key in self._reward_condition_metric_keys():
            metrics[key] = torch.zeros((), device=self.device)
        return metrics

    @staticmethod
    def _reward_objective_metric_keys() -> list[str]:
        """Metrics every objective reports on its update batch (plan 5.4 B2)."""
        return [
            "reward_separation_auc",
            "reward_separation_auc_current",
            "reward_saturation_frac",
            "loss_reward_objective",
            "reward_linear_delta_norm",
            "reward_linear_w_norm",
            "reward_linear_dual_value",
            "reward_linear_whitening_condition",
        ]

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
            "loss_rollout_bc",
            "rollout_bc_nll",
            "rollout_bc_mse",
            "rollout_bc_log_prob_mean",
            "rollout_bc_target_abs_mean",
            "rollout_bc_policy_action_abs_mean",
            "rollout_bc_policy_action_mae",
            "rollout_bc_policy_action_rmse",
            "bc_pretrain_active",
            "hl_skill_total_loss",
            "hl_skill_pg_loss",
            "hl_skill_diffsr_loss",
            "hl_skill_achieved_loss",
            "hl_skill_anchor_loss",
            "hl_skill_z_norm_loss",
            "hl_skill_grad_norm",
            "hl_skill_updates",
            # ntp_live_offset online objective.
            "hl_skill_ntp_loss",
            "hl_skill_live_offset_xy_median",
            "hl_skill_live_offset_ring_count",
            "hl_skill_z_abs_mean",
            "hl_skill_z_rms",
            "hl_skill_z_dim_std_mean",
            "hl_skill_z_effective_rank",
            "estimated_reward_mean",
            "estimated_reward_std",
            "expert_reward_mean",
            "expert_reward_std",
            "reward_grad_norm",
            *self._reward_objective_metric_keys(),
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
        """Attach reward-model diagnostics and PPO reward mixing to one rollout.

        Plan 5.4 B3. The raw estimate ``r`` is computed over the whole rollout
        whenever the estimator trains or drives PPO, and the alignment metrics
        against the env reward are logged whenever it is computed. The PPO
        reward is ``env_w * env + est_w_eff * est`` only when
        ``use_estimated_rewards_for_ppo`` is on, with

        ``r -> clamp -> r_hat (reward_ppo_squash) -> scale -> done penalty ->
        done mask / constant bonus -> est``;

        otherwise the PPO reward is the env reward, untouched. With the
        historical defaults (squash none, scale 1, no mask, no bonus, no ramp)
        ``est`` is the historical clamped-and-penalized estimate.
        """
        metrics: dict[str, float] = {}
        reward_key = ("next", "reward")
        cfg = self.config.ipmd
        driving = bool(cfg.use_estimated_rewards_for_ppo)
        estimate = driving or self._reward_update_enabled

        with torch.no_grad():
            env_reward = rollout[reward_key]
            est_reward: Tensor | None = None
            raw_reward: Tensor | None = None
            r_hat: Tensor | None = None
            effective_est_weight = float(cfg.est_reward_weight)
            if estimate:
                raw_reward = (
                    self._reward_from_td(
                        rollout,
                        batch_role="rollout",
                        use_target=True,
                    )
                    .detach()  # type: ignore[attr-defined]
                    .clamp(
                        min=cfg.estimated_reward_clamp_min,
                        max=cfg.estimated_reward_clamp_max,
                    )
                )  # type: ignore[attr-defined]
                r_hat = self._squash_rollout_reward(raw_reward, driving=driving)
                scale = float(cfg.estimated_reward_scale)
                est_reward = r_hat if scale == 1.0 else scale * r_hat
                est_reward = self._apply_estimated_reward_done_penalty(
                    rollout,
                    est_reward,
                )
                est_reward = self._apply_reward_done_controls(
                    rollout, est_reward, r_hat, scale=scale, driving=driving
                )
                if bool(cfg.reward_absorbing_terminal):
                    est_reward = self._apply_absorbing_terminal(rollout, est_reward, scale=scale)

            if self._use_estimated_rewards_for_ppo:
                assert est_reward is not None
                effective_est_weight = self._effective_est_reward_weight()
                mixed_reward = (
                    cfg.env_reward_weight * env_reward
                    + effective_est_weight * est_reward
                )
                self._reward_mix_state["driving_rollouts"] += 1
            elif self._reward_prior is not None:
                mixed_reward = cfg.env_reward_weight * env_reward
            else:
                mixed_reward = env_reward
            prior_reward: Tensor | None = None
            if self._reward_prior is not None:
                prior_key = tuple(cfg.reward_prior_obs_key)
                done = rollout.get(("next", "done"))
                prior_reward, prior_metrics = self._reward_prior.compute(
                    rollout.get(prior_key), done
                )
                prior_reward = prior_reward.reshape(mixed_reward.shape).to(
                    mixed_reward.dtype
                )
                mixed_reward = mixed_reward + prior_reward

        rollout.set(("next", "env_reward"), env_reward)
        rollout.set(reward_key, mixed_reward)

        metrics.update(
            {
                "train/env_reward_mean": env_reward.mean().item(),
            }
        )
        if prior_reward is not None:
            rollout.set(("next", "prior_reward"), prior_reward)
            metrics.update({f"train/{k}": v for k, v in prior_metrics.items()})
            metrics.update(
                reward_alignment_metrics("reward/env_vs_prior", env_reward, prior_reward)
            )
        if est_reward is not None:
            assert raw_reward is not None
            assert r_hat is not None
            rollout.set(("next", "est_reward"), est_reward)
            metrics["train/est_reward_mean"] = est_reward.mean().item()
            metrics["train/est_reward_raw_mean"] = raw_reward.mean().item()
            metrics["train/est_reward_raw_std"] = raw_reward.std().item()
            metrics["train/est_reward_hat_mean"] = r_hat.mean().item()
            metrics["train/est_reward_saturation_frac"] = reward_saturation_fraction(
                raw_reward
            ).item()
            metrics["train/est_reward_weight_effective"] = (
                effective_est_weight if self._use_estimated_rewards_for_ppo else 0.0
            )
            metrics.update(
                reward_alignment_metrics(
                    "reward/env_vs_est",
                    env_reward,
                    raw_reward,
                )
            )

        return metrics

    def _squash_rollout_reward(self, raw_reward: Tensor, *, driving: bool) -> Tensor:
        """``r -> r_hat`` per the resolved squash; freezes ``m0, s0`` once."""
        squash = self._reward_ppo_squash
        if squash != "affine_frozen":
            return squash_reward_for_ppo(raw_reward, squash)
        state = self._reward_mix_state
        if driving and not bool(state["affine_frozen"].item()):
            flat = raw_reward.reshape(-1).float()
            state["affine_center"].copy_(flat.mean())
            state["affine_scale"].copy_(flat.std(unbiased=False).clamp_min(1e-6))
            state["affine_frozen"].fill_(True)
        if not bool(state["affine_frozen"].item()):
            # Diagnostics only (driving off): centre on the current rollout.
            flat = raw_reward.reshape(-1).float()
            return squash_reward_for_ppo(
                raw_reward,
                squash,
                affine_center=flat.mean(),
                affine_scale=flat.std(unbiased=False).clamp_min(1e-6),
            )
        return squash_reward_for_ppo(
            raw_reward,
            squash,
            affine_center=state["affine_center"],
            affine_scale=state["affine_scale"],
        )

    def _apply_reward_done_controls(
        self,
        rollout: TensorDict,
        est_reward: Tensor,
        r_hat: Tensor,
        *,
        scale: float,
        driving: bool,
    ) -> Tensor:
        """Done-row mask and the constant-bonus control (plan 5.4 B3)."""
        cfg = self.config.ipmd
        constant_bonus = bool(cfg.reward_constant_bonus)
        mask_done = bool(cfg.reward_mask_done_rows)
        if not constant_bonus and not mask_done:
            return est_reward
        if constant_bonus:
            state = self._reward_mix_state
            if driving and not bool(state["constant_bonus_frozen"].item()):
                state["constant_bonus_mean"].copy_(r_hat.float().mean())
                state["constant_bonus_frozen"].fill_(True)
            bonus = (
                state["constant_bonus_mean"]
                if bool(state["constant_bonus_frozen"].item())
                else r_hat.float().mean()
            )
            est_reward = torch.full_like(est_reward, 0.0) + (scale * bonus).to(
                est_reward.dtype
            )
        done = rollout_done_mask(rollout, est_reward)
        if done is None:
            return est_reward
        return torch.where(done, torch.zeros_like(est_reward), est_reward)

    def _reward_hat_floor(self) -> float:
        """Lower bound of ``r_hat``: the absorbing state's reward (DAC)."""
        if self._reward_ppo_squash in {"sigmoid", "amp", "affine_frozen"}:
            return 0.0
        floor = self.config.ipmd.estimated_reward_clamp_min
        if floor is None or not math.isfinite(float(floor)):
            raise ValueError(
                "reward_absorbing_terminal with reward_ppo_squash='none' needs a finite "
                "ipmd.estimated_reward_clamp_min (the floor of the reward class)."
            )
        return float(floor)

    def _apply_absorbing_terminal(
        self, rollout: TensorDict, est_reward: Tensor, *, scale: float
    ) -> Tensor:
        """Terminated rows pay ``scale * r_floor / (1 - gamma)``; truncated rows untouched."""
        done = rollout_done_mask(rollout, est_reward)
        if done is None:
            return est_reward
        truncated_key: BatchKey = ("next", "truncated")
        if truncated_key in rollout.keys(True):
            truncated = rollout[truncated_key].to(device=est_reward.device, dtype=torch.bool)
            while truncated.ndim < est_reward.ndim:
                truncated = truncated.unsqueeze(-1)
            done = done & ~truncated
        gamma = float(self.config.loss.gamma)
        value = scale * self._reward_hat_floor() / (1.0 - gamma)
        return torch.where(done, torch.full_like(est_reward, value), est_reward)

    def _effective_est_reward_weight(self) -> float:
        """``est_reward_weight`` under the linear ramp (unit: driving rollouts)."""
        cfg = self.config.ipmd
        weight = float(cfg.est_reward_weight)
        ramp = int(cfg.reward_ramp_updates)
        if ramp <= 0:
            return weight
        seen = int(self._reward_mix_state["driving_rollouts"].item())
        return weight * min(1.0, seen / float(ramp))

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
        posterior_pg_renew_frac: Tensor | None = None
        if (
            self._use_latent_command
            and self._latent_learner is not None
            and bool(self.config.ipmd.latent_learning.train_posterior_through_policy)
        ):
            stored_latents = batch.get(self._latent_key).reshape(
                *batch.batch_size, self._latent_dim
            )
            renewal_mask: Tensor | None = None
            posterior_batch = batch
            latent_cfg = self.config.ipmd.latent_learning
            if (
                str(latent_cfg.command_phase_mode) == "sin_cos"
                and int(latent_cfg.code_period) > 1
            ):
                phase = stored_latents[..., -2:]
                renewal_mask = (phase[..., 0].abs() <= 1.0e-5) & (
                    phase[..., 1] >= 1.0 - 1.0e-5
                )
                posterior_pg_renew_frac = renewal_mask.float().mean()
                posterior_batch = batch[renewal_mask]
            latents = None
            # TensorDict.numel() is lower-bounded to 1 by design (its own
            # docstring: "a stack of two tensordict with empty shape will
            # have two elements"), so it can never observe an empty-mask
            # selection here -- check the mask directly instead.
            if renewal_mask is None or bool(renewal_mask.any()):
                latents = self._latent_learner.infer_batch_latents(
                    posterior_batch,
                    detach=False,
                    context="ppo posterior latent",
                )
            if latents is not None:
                recomputed_latents = latents.reshape(-1, self._latent_dim)
                if renewal_mask is None:
                    recomputed_latents = recomputed_latents.reshape(
                        *batch.batch_size, self._latent_dim
                    )
                batch.set(
                    self._latent_key,
                    held_command_policy_surrogate(
                        stored_latents,
                        recomputed_latents,
                        renewal_mask,
                    ),
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
        if posterior_pg_renew_frac is not None:
            output_loss.set(
                "posterior_pg_renew_frac",
                posterior_pg_renew_frac.detach(),
            )
        for key, value in extra_actor_metrics.items():
            output_loss.set(key, value.detach())
        return output_loss

    def _ppo_metrics_without_backward(self, batch: TensorDict) -> TensorDict:
        """Return PPO diagnostics without adding PPO gradients during BC pretraining."""
        with torch.no_grad():
            loss: TensorDict = self.loss_module(batch)
        return loss.clone().detach_()

    def _hl_skill_actor_objective(self, batch: TensorDict) -> Tensor:
        """Return the PPO actor surrogate used to shape the HL skill encoder."""
        loss: TensorDict = self.loss_module(batch)
        actor_loss = loss["loss_objective"]
        if actor_loss.ndim > 0:
            actor_loss = actor_loss.mean()
        return actor_loss

    def _run_hl_skill_online_update(
        self,
        batch: TensorDict,
        *,
        update_idx: int,
    ) -> dict[str, Tensor]:
        """Update the online high-level skill encoder from PG + anchor losses."""
        sampler = self._hl_skill_command_sampler
        if sampler is None:
            return {}
        should_update = getattr(sampler, "should_update_online", None)
        if not callable(should_update) or not bool(should_update(update_idx)):
            return {}
        optimizer = getattr(sampler, "optimizer", None)
        if optimizer is None:
            msg = (
                "HL skill finetuning is enabled, but the sampler optimizer was "
                "not initialized."
            )
            raise RuntimeError(msg)

        actor_params = list(self.actor_critic.parameters())
        actor_requires_grad = [param.requires_grad for param in actor_params]
        for param in actor_params:
            param.requires_grad_(False)

        try:
            optimizer.zero_grad(set_to_none=True)
            total_loss, metrics = sampler.compute_online_finetune_loss(
                batch,
                latent_key=self._latent_key,
                actor_loss_fn=self._hl_skill_actor_objective,
            )
            total_loss.backward()
            trainable_params = sampler.trainable_parameters()
            grad_clip = getattr(sampler, "grad_clip_norm", None)
            if grad_clip is not None:
                grad_norm = clip_grad_norm_(trainable_params, float(grad_clip))
            else:
                grad_norm = torch.zeros((), device=self.device)
            optimizer.step()
            # After the step, never before: a JEPA target encoder is an EMA of
            # the ONLINE encoder, so it must see the post-step weights.
            after_step = getattr(sampler, "on_after_finetune_step", None)
            if callable(after_step):
                after_step()
            sampler.finetune_updates = int(getattr(sampler, "finetune_updates", 0)) + 1
        finally:
            for param, requires_grad in zip(
                actor_params,
                actor_requires_grad,
                strict=True,
            ):
                param.requires_grad_(requires_grad)

        metrics = dict(metrics)
        metrics["hl_skill_grad_norm"] = grad_norm.detach()
        metrics["hl_skill_updates"] = torch.tensor(
            float(getattr(sampler, "finetune_updates", 0)),
            device=self.device,
        )
        return metrics

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
        if self._use_latent_command and self._latent_key not in expert_obs_td.keys(
            True
        ):
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

    def _backward_rollout_bc_terms(self, batch: TensorDict) -> dict[str, Tensor]:
        """Backpropagate BC from live rollout states to aligned expert actions."""
        if self._rollout_bc_coeff <= 0.0:
            return {}

        target = batch.get(self._rollout_bc_action_key)
        if not isinstance(target, Tensor):
            msg = f"Rollout BC requires a Tensor at {self._rollout_bc_action_key!r}."
            raise RuntimeError(msg)
        policy_obs_td = batch.select(*self._policy_obs_keys)
        dist = self._policy_operator.get_dist(policy_obs_td)
        log_prob = dist.log_prob(target)
        log_prob = self._reduce_log_prob(log_prob, target)
        rollout_bc_nll = -log_prob.mean()
        policy_action = dist.deterministic_sample
        error = policy_action - target
        rollout_bc_mse = error.square().mean()
        rollout_bc_objective = (
            rollout_bc_mse
            if self.config.ipmd.rollout_bc_loss_type == "mse"
            else rollout_bc_nll
        )
        rollout_bc_loss = rollout_bc_objective * self._rollout_bc_coeff
        rollout_bc_loss.backward()

        return {
            "loss_rollout_bc": rollout_bc_loss.detach(),
            "rollout_bc_nll": rollout_bc_nll.detach(),
            "rollout_bc_mse": rollout_bc_mse.detach(),
            "rollout_bc_log_prob_mean": log_prob.detach().mean(),
            "rollout_bc_target_abs_mean": target.detach().abs().mean(),
            "rollout_bc_policy_action_abs_mean": policy_action.detach().abs().mean(),
            "rollout_bc_policy_action_mae": error.detach().abs().mean(),
            "rollout_bc_policy_action_rmse": error.detach().square().mean().sqrt(),
        }

    def _backward_reward_terms(
        self,
        batch: TensorDict,
        expert_batch: TensorDict,
    ) -> dict[str, Tensor]:
        """Backpropagate the IPMD reward objective and return detached metrics.

        ``reward_objective`` selects the branch; ``"ipmd_diff"`` (the parked
        campaign objective) is the historical arithmetic, kept bitwise.
        """
        if self._reward_model_type == "grouped":
            return self._backward_grouped_reward_terms(batch, expert_batch)
        if self._reward_objective == "logistic_ratio":
            return self._backward_logistic_ratio_reward_terms(batch, expert_batch)
        if self._reward_objective == "chi2_lsgan":
            return self._backward_chi2_lsgan_reward_terms(batch, expert_batch)
        if self._reward_objective == "linear_dual":
            msg = (
                "linear_dual has no backward pass; _run_reward_updates routes it "
                "through _linear_dual_reward_step."
            )
            raise RuntimeError(msg)

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

    def _reward_side_metrics(
        self,
        metrics: dict[str, Tensor],
        r_pi: Tensor,
        r_exp: Tensor,
    ) -> None:
        """Per-side moments, rank AUC and saturation shared by every objective."""
        r_pi = r_pi.detach()
        r_exp = r_exp.detach()
        metrics["estimated_reward_mean"] = r_pi.mean()
        metrics["estimated_reward_std"] = r_pi.std()
        metrics["expert_reward_mean"] = r_exp.mean()
        metrics["expert_reward_std"] = r_exp.std()
        metrics["reward_separation_auc"] = reward_separation_auc(r_exp, r_pi)
        # The policy side mixes in replay rows when the replay is on; this AUC
        # scores the current batch's own policy rows (one extra forward pass),
        # so arms with and without the replay stay comparable.
        current_input = self._reward_current_policy_input
        if current_input is None:
            r_pi_current = r_pi
        else:
            with torch.no_grad():
                r_pi_current = self.reward_estimator(current_input).reshape(-1)
        metrics["reward_separation_auc_current"] = reward_separation_auc(r_exp, r_pi_current)
        metrics["reward_saturation_frac"] = reward_saturation_fraction(
            torch.cat([r_pi.reshape(-1), r_exp.reshape(-1)], dim=0)
        )

    def _reward_pair_inputs(
        self, batch: TensorDict, expert_batch: TensorDict, *, expert_requires_grad: bool
    ) -> tuple[Tensor, Tensor]:
        r_pi_input = self._reward_input_from_td(
            batch,
            detach=self._reward_detach_features,
            requires_grad=False,
        )
        r_exp_input = self._reward_input_from_td(
            expert_batch,
            detach=self._reward_detach_features,
            requires_grad=expert_requires_grad,
        )
        replay = self._reward_replay
        self._reward_current_policy_input = None
        self._reward_replay_slots = None
        self._reward_replay_weights = None
        if replay is not None and replay.size > 0:
            rows = int(r_pi_input.shape[0])
            from_replay = int(round(float(self.config.ipmd.reward_policy_replay_fraction) * rows))
            if from_replay > 0:
                self._reward_current_policy_input = r_pi_input.detach()
                if isinstance(replay, PrioritizedFifoFeatureReplay):
                    replay_rows, slots, weights = replay.sample(from_replay)
                    self._reward_replay_slots = slots
                    self._reward_replay_weights = weights
                else:
                    replay_rows = replay.sample(from_replay)
                # Current rows first, replay rows last: the loss and the
                # priority update index the replay rows as the tail.
                r_pi_input = torch.cat(
                    [r_pi_input[from_replay:], replay_rows.to(r_pi_input.dtype)],
                    dim=0,
                )
        return r_pi_input, r_exp_input

    def _backward_logistic_ratio_reward_terms(
        self,
        batch: TensorDict,
        expert_batch: TensorDict,
    ) -> dict[str, Tensor]:
        """Label-smoothed BCE classifier; the reward is the logit.

        ``loss = BCE(logits; expert -> 1 - eps, policy -> eps)
        + reward_grad_penalty_coeff * R1(expert)
        + reward_logit_reg_coeff * mean logit^2`` (+ parameter weight decay).
        The cross-entropy optimum is the finite log density ratio where the
        supports overlap; smoothing bounds it where they do not.
        """
        metrics = self._empty_reward_metrics()
        if not self._reward_update_enabled:
            return metrics
        penalized = self._reward_grad_penalty_coeff > 0.0
        r_pi_input, r_exp_input = self._reward_pair_inputs(
            batch, expert_batch, expert_requires_grad=penalized
        )
        logits_pi = self.reward_estimator(r_pi_input)
        logits_exp = self.reward_estimator(r_exp_input)
        eps = float(self.config.ipmd.reward_label_smoothing)
        target_exp = torch.full_like(logits_exp, 1.0 - eps)
        target_pi = torch.full_like(logits_pi, eps)
        replay_weights = self._reward_replay_weights
        if replay_weights is None:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                torch.cat([logits_exp.reshape(-1), logits_pi.reshape(-1)], dim=0),
                torch.cat([target_exp.reshape(-1), target_pi.reshape(-1)], dim=0),
            )
        else:
            # Prioritized FIFO replay: the tail of the policy rows came from
            # the replay; weight them by their importance weights (mean 1) and
            # refresh their priorities with their per-row loss.
            per_row_pi = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_pi.reshape(-1), target_pi.reshape(-1), reduction="none"
            )
            per_row_exp = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_exp.reshape(-1), target_exp.reshape(-1), reduction="none"
            )
            tail = int(replay_weights.shape[0])
            pi_weights = torch.ones_like(per_row_pi)
            pi_weights[-tail:] = replay_weights.to(pi_weights.dtype)
            bce = torch.cat([per_row_exp, pi_weights * per_row_pi]).mean()
            assert self._reward_replay_slots is not None
            assert isinstance(self._reward_replay, PrioritizedFifoFeatureReplay)
            self._reward_replay.update_priorities(
                self._reward_replay_slots, per_row_pi[-tail:].detach()
            )
        if penalized:
            r1_expert = reward_grad_penalty_from_input(logits_exp, r_exp_input)
        else:
            r1_expert = torch.zeros((), device=self.device)
        logit_reg = logits_pi.pow(2).mean() + logits_exp.pow(2).mean()
        param_weight_decay = self._reward_parameter_weight_decay()
        (
            self._reward_loss_coeff * bce
            + self._reward_grad_penalty_coeff * r1_expert
            + self._reward_logit_reg_coeff * logit_reg
            + self._reward_param_weight_decay_coeff * param_weight_decay
        ).backward()

        metrics["loss_reward_objective"] = bce.detach()
        metrics["loss_reward_diff"] = (logits_pi.mean() - logits_exp.mean()).detach()
        metrics["loss_reward_grad_penalty"] = r1_expert.detach()
        metrics["loss_reward_grad_penalty_expert"] = r1_expert.detach()
        metrics["loss_reward_logit_reg"] = logit_reg.detach()
        metrics["loss_reward_param_weight_decay"] = param_weight_decay.detach()
        self._reward_side_metrics(metrics, logits_pi, logits_exp)
        return metrics

    def _backward_chi2_lsgan_reward_terms(
        self,
        batch: TensorDict,
        expert_batch: TensorDict,
    ) -> dict[str, Tensor]:
        """AMP least-squares discriminator (Peng et al. 2021); reward = D(x).

        ``loss = mean (D_exp - 1)^2 + mean (D_pi + 1)^2
        + (reward_grad_penalty_coeff / 2) * mean ||grad_x D(x_exp)||^2``
        (+ ``reward_logit_reg_coeff`` on D^2 and parameter weight decay).
        Population optimum without the penalty: ``(p_E - p_pi) / (p_E + p_pi)``
        in [-1, 1], so the reward is bounded with no output activation.
        """
        metrics = self._empty_reward_metrics()
        if not self._reward_update_enabled:
            return metrics
        penalized = self._reward_grad_penalty_coeff > 0.0
        r_pi_input, r_exp_input = self._reward_pair_inputs(
            batch, expert_batch, expert_requires_grad=penalized
        )
        d_pi = self.reward_estimator(r_pi_input)
        d_exp = self.reward_estimator(r_exp_input)
        ls_expert = (d_exp - 1.0).pow(2).mean()
        ls_policy = (d_pi + 1.0).pow(2).mean()
        objective = ls_expert + ls_policy
        if penalized:
            r1_expert = reward_grad_penalty_from_input(d_exp, r_exp_input)
        else:
            r1_expert = torch.zeros((), device=self.device)
        logit_reg = d_pi.pow(2).mean() + d_exp.pow(2).mean()
        param_weight_decay = self._reward_parameter_weight_decay()
        (
            self._reward_loss_coeff * objective
            + 0.5 * self._reward_grad_penalty_coeff * r1_expert
            + self._reward_logit_reg_coeff * logit_reg
            + self._reward_param_weight_decay_coeff * param_weight_decay
        ).backward()

        metrics["loss_reward_objective"] = objective.detach()
        metrics["loss_reward_diff"] = (d_pi.mean() - d_exp.mean()).detach()
        metrics["loss_reward_l2"] = (ls_expert.detach() + ls_policy.detach())
        metrics["loss_reward_grad_penalty"] = r1_expert.detach()
        metrics["loss_reward_grad_penalty_expert"] = r1_expert.detach()
        metrics["loss_reward_logit_reg"] = logit_reg.detach()
        metrics["loss_reward_param_weight_decay"] = param_weight_decay.detach()
        self._reward_side_metrics(metrics, d_pi, d_exp)
        return metrics

    def _linear_dual_reward_step(
        self,
        batch: TensorDict,
        expert_batch: TensorDict,
    ) -> dict[str, Tensor]:
        """One ``linear_dual`` reward step: moments, whitening, ``w`` update."""
        metrics = self._empty_reward_metrics()
        if not self._reward_update_enabled:
            return metrics
        head = self.reward_estimator
        assert isinstance(head, LinearDualRewardHead)
        r_pi_input, r_exp_input = self._reward_pair_inputs(
            batch, expert_batch, expert_requires_grad=False
        )
        alpha = (
            self._reward_lr
            if self._reward_lr is not None
            else float(self.config.optim.lr)
        )
        out = head.dual_step(r_pi_input, r_exp_input, alpha=alpha)
        metrics["reward_linear_delta_norm"] = out["delta_norm"]
        metrics["reward_linear_w_norm"] = out["w_norm"]
        metrics["reward_linear_dual_value"] = out["dual_value"]
        metrics["reward_linear_whitening_condition"] = head.whitening_condition.clone()
        metrics["loss_reward_objective"] = -out["dual_value"]
        metrics["loss_reward_diff"] = (
            out["reward_policy"].mean() - out["reward_expert"].mean()
        )
        self._reward_side_metrics(metrics, out["reward_policy"], out["reward_expert"])
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
        if self.reward_optim is None and not self._reward_estimator_is_linear:
            msg = "Reward update is enabled, but reward_optim was not initialized."
            raise RuntimeError(msg)

        policy_latents = self._set_reward_policy_condition_from_batch(batch)
        expert_latents = self._set_reward_expert_condition_from_batch(expert_batch)
        condition_metrics = self._reward_condition_metrics(
            policy_latents,
            expert_latents,
        )

        for _ in range(self._reward_updates_per_policy_update):
            if self._reward_estimator_is_linear:
                step_metrics = self._linear_dual_reward_step(batch, expert_batch)
                reward_grad_norm = torch.zeros((), device=self.device)
            else:
                assert self.reward_optim is not None
                self.reward_optim.zero_grad(set_to_none=True)
                step_metrics = self._backward_reward_terms(batch, expert_batch)
                reward_grad_norm = clip_grad_norm_(
                    self._reward_grad_clip_params,
                    self._reward_max_grad_norm,
                )
                self.reward_optim.step()
                if self._reward_target is not None:
                    polyak_update(
                        self._reward_target, self.reward_estimator, self._reward_target_tau
                    )
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
            ("train/loss_rollout_bc", "rollout_bc"),
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
            ("train/loss_rollout_bc", "rollout_bc_loss"),
            ("train/rollout_bc_policy_action_mae", "rollout_bc_act_mae"),
            ("train/rollout_bc_target_abs_mean", "rollout_bc_target_abs"),
            ("train/bc_pretrain_active", "bc_pretrain"),
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

        self._abort_on_nonfinite(iteration, metadata)

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

    # Watchdog keys: the first quantities to turn non-finite when a rollout
    # produces NaN. Checked every iteration because a NaN policy keeps training
    # at full speed and every later checkpoint is unusable -- two chains
    # (2026-08-23 fsq64, 2026-08-24 emastack) each burned about a day before
    # anyone noticed. Losses are excluded on purpose: an optimizer can recover
    # a single bad update, but a non-finite reward or policy parameter cannot.
    _NONFINITE_WATCH_KEYS: tuple[str, ...] = (
        "train/step_reward_mean",
        "episode/return",
    )

    def _abort_on_nonfinite(
        self, iteration: PPOIterationData, metadata: PPOTrainingMetadata
    ) -> None:
        """Raise as soon as training goes non-finite, before writing a checkpoint.

        Set ``agent.abort_on_nonfinite=false`` to downgrade this to a warning.
        """
        offenders = [
            key
            for key in self._NONFINITE_WATCH_KEYS
            if key in iteration.metrics
            and not math.isfinite(float(iteration.metrics[key]))
        ]
        if not offenders:
            bad_params = [
                name
                for name, tensor in (
                    self.policy.named_parameters() if self.policy else ()
                )
                if not torch.isfinite(tensor).all()
            ]
            if bad_params:
                offenders = [f"policy:{bad_params[0]}"]
        if not offenders:
            return
        message = (
            "Training went non-finite at "
            f"{metadata.frames_processed} cumulative frames: {offenders}. "
            "Every later checkpoint would be unusable; aborting so the chain "
            "can resume from the last good checkpoint."
        )
        if not bool(getattr(self.config, "abort_on_nonfinite", True)):
            self.log.error("%s (abort_on_nonfinite=false, continuing)", message)
            return
        raise RuntimeError(message)

    def _checkpoint_policy_state_dict(self) -> Mapping[str, Any]:
        """Return the policy payload stored under ``policy_state_dict``."""
        return self.policy.state_dict() if self.policy else {}

    def _extra_checkpoint_state_dict(self) -> dict[str, Any]:
        """Return algorithm-specific additions to an IPMD checkpoint.

        Carries the frozen PPO-mixing state (plan 5.4 B3/B4) so a walltime
        segment resumes with the same ``m0, s0``, constant-bonus mean and ramp
        clock. The ``linear_dual`` head's buffers travel in
        ``reward_estimator_state_dict``. Subclasses that override this hook
        must merge ``super()``'s dict.
        """
        return {
            "reward_mix_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self._reward_mix_state.items()
            }
        }

    def _load_checkpoint_policy_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore the primary IPMD policy from one checkpoint."""
        if self.policy and "policy_state_dict" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy_state_dict"])

    def _restore_training_state_from_checkpoint(
        self, checkpoint: Mapping[str, Any]
    ) -> bool:
        """Return whether auxiliary training state should be restored.

        An ordinary IPMD agent may consume the student-primary policy from an
        IPMD-L2T checkpoint for deployment. Its optimizer and privileged value
        network do not have the same layout, so those states are intentionally
        skipped. IPMD-L2T overrides this hook for exact training resume.
        """
        metadata = checkpoint.get("checkpoint_metadata")
        return not (
            isinstance(metadata, Mapping)
            and metadata.get("algorithm") == "IPMD_L2T"
            and metadata.get("primary_policy_role") == "student"
        )

    def _load_extra_checkpoint_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore algorithm-specific additions from an IPMD checkpoint.

        Pre-revival checkpoints carry no ``reward_mix_state_dict``; they resume
        with a fresh mixing state, which is the historical behaviour.
        """
        saved = checkpoint.get("reward_mix_state_dict")
        if not isinstance(saved, Mapping):
            return
        for key, target in self._reward_mix_state.items():
            if key in saved:
                target.copy_(torch.as_tensor(saved[key]).to(target.device, target.dtype))

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

        data_to_save: dict[str, Any] = {
            "policy_state_dict": self._checkpoint_policy_state_dict(),
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
        if self._reward_target is not None:
            data_to_save["reward_target_state_dict"] = self._reward_target.state_dict()
        if self.lr_scheduler is not None:
            data_to_save["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if self._latent_learner is not None:
            data_to_save["latent_learner_state_dict"] = (
                self._latent_learner.checkpoint_state_dict()
            )
        if self._hl_skill_command_sampler is not None:
            checkpoint_state = getattr(
                self._hl_skill_command_sampler,
                "checkpoint_state_dict",
                None,
            )
            if callable(checkpoint_state):
                data_to_save["hl_skill_command_sampler_state_dict"] = checkpoint_state()
        if (
            hasattr(self.env, "is_closed")
            and not self.env.is_closed
            and hasattr(self.env, "normalize_obs")
        ):
            data_to_save["vec_norm_msg"] = self.env.state_dict()
        data_to_save.update(self._extra_checkpoint_state_dict())
        # Global frame count for walltime-segmented resume. `step` is
        # metadata.frames_processed, which init_metadata already seeds with the
        # restored offset, so this is cumulative across segments -- the next
        # segment reads it and continues the SAME budget instead of restarting.
        if step is not None:
            data_to_save["cumulative_env_frames"] = int(step)
        # Cumulative training wall-clock (trainer.max_train_seconds budget).
        data_to_save["cumulative_train_seconds"] = self.train_elapsed_seconds()

        torch.save(data_to_save, target_path)

    def _load_optimizer_state_dict(self, state_dict: Any) -> None:
        """Restore optimizer state, tolerating a param-group-count change.

        `torch.optim.Optimizer.load_state_dict` requires the saved and live
        param groups to match one for one. Any change to how the actor's
        parameters are split into groups therefore makes every earlier
        checkpoint unloadable -- for evaluation as well as for resume, even
        though evaluation needs no optimizer state at all and the policy is
        already restored by the time this runs.

        A mismatch is reported and skipped rather than raised. The optimizer
        then starts from its fresh state: for evaluation that is exactly
        right, and for a resume it means losing the moment estimates, which
        the log says out loud because it changes the first updates after the
        restore.
        """
        groups = (
            state_dict.get("param_groups") if isinstance(state_dict, dict) else None
        )
        saved_groups = len(groups) if groups is not None else None
        live_groups = len(self.optim.param_groups)
        if saved_groups is not None and saved_groups != live_groups:
            self.log.warning(
                "Checkpoint optimizer has %d param group(s) but this optimizer "
                "has %d; skipping the optimizer restore. Evaluation is "
                "unaffected (the policy is already loaded); a training resume "
                "continues with fresh optimizer moments.",
                saved_groups,
                live_groups,
            )
            return
        self.optim.load_state_dict(state_dict)

    def load_model(self, path: str) -> None:
        """Load PPO and reward-estimator state from a checkpoint."""
        data = torch.load(path, map_location=self.device)
        # Continue the global frame budget on resume. Pre-2026-08-16
        # checkpoints lack the key; their filename step is segment-local, so
        # it is deliberately NOT used as a fallback -- treating a local step
        # as global would silently shrink the remaining budget.
        resume_frames = data.get("cumulative_env_frames")
        if resume_frames is not None:
            self._resume_frame_offset = int(resume_frames)
        self._restore_train_clock(data)
        self._load_checkpoint_policy_state_dict(data)
        if not self._restore_training_state_from_checkpoint(data):
            self._load_extra_checkpoint_state_dict(data)
            if hasattr(self.env, "normalize_obs") and "vec_norm_msg" in data:
                self.env.load_state_dict(data["vec_norm_msg"])  # type: ignore[arg-type]
            self.log.info(
                "Loaded student-primary IPMD-L2T policy for deployment; "
                "skipped incompatible training state."
            )
            return
        if self.value_function and "value_state_dict" in data:
            self.value_function.load_state_dict(data["value_state_dict"])  # type: ignore[arg-type]
        if self.q_function and "q_state_dict" in data:
            self.q_function.load_state_dict(data["q_state_dict"])  # type: ignore[arg-type]
        if "optimizer_state_dict" in data:
            self._load_optimizer_state_dict(data["optimizer_state_dict"])
        if "reward_estimator_state_dict" in data:
            self.reward_estimator.load_state_dict(
                data["reward_estimator_state_dict"]  # type: ignore[arg-type]
            )
        if self.reward_optim is not None and "reward_optimizer_state_dict" in data:
            self.reward_optim.load_state_dict(
                data["reward_optimizer_state_dict"]  # type: ignore[arg-type]
            )
        if self._reward_target is not None:
            if "reward_target_state_dict" in data:
                self._reward_target.load_state_dict(
                    data["reward_target_state_dict"]  # type: ignore[arg-type]
                )
            elif "reward_estimator_state_dict" in data:
                self._reward_target.load_state_dict(
                    data["reward_estimator_state_dict"]  # type: ignore[arg-type]
                )
        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in data:
            self.lr_scheduler.load_state_dict(data["lr_scheduler_state_dict"])
        if self.config.feature_extractor and "feature_extractor_state_dict" in data:
            self.feature_extractor.load_state_dict(
                data["feature_extractor_state_dict"]  # type: ignore[arg-type]
            )
        if self._latent_learner is not None and "latent_learner_state_dict" in data:
            self._latent_learner.load_checkpoint_state_dict(
                data["latent_learner_state_dict"]
            )
        if self._hl_skill_command_sampler is not None:
            reset_sampler = getattr(self._hl_skill_command_sampler, "reset", None)
            if callable(reset_sampler):
                reset_sampler()
        if (
            self._hl_skill_command_sampler is not None
            and "hl_skill_command_sampler_state_dict" in data
            and not bool(self.config.ipmd.hl_skill_restore_from_checkpoint)
        ):
            self.log.info(
                "hl_skill_restore_from_checkpoint=false: keeping the encoder "
                "from hl_skill_checkpoint_path, not the tracker checkpoint's."
            )
        elif (
            self._hl_skill_command_sampler is not None
            and "hl_skill_command_sampler_state_dict" in data
        ):
            load_checkpoint_state = getattr(
                self._hl_skill_command_sampler,
                "load_checkpoint_state_dict",
                None,
            )
            if callable(load_checkpoint_state):
                load_checkpoint_state(data["hl_skill_command_sampler_state_dict"])
        self._load_extra_checkpoint_state_dict(data)
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
        with torch.no_grad(), self._freeze_value_normalizer_updates():
            rollout = self.adv_module(rollout)

            if (
                self.config.ppo.normalize_advantage
                and self.config.ppo.normalize_advantage_global
            ):
                advantage = rollout.get("advantage")
                rollout.set(
                    "advantage",
                    _normalize_advantage_over_rollout(advantage),
                )

            if getattr(self.config.compile, "compile", False):
                rollout = rollout.clone()

        self.data_buffer.extend(self._rollout_for_buffer(rollout))
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

    @torch.no_grad()
    def _offer_rollout_to_reward_replay(self, rollout_flat: TensorDict) -> None:
        """Offer up to ``reward_policy_replay_rows_per_rollout`` non-done rows."""
        assert self._reward_replay is not None
        rows = self._reward_input_from_td(rollout_flat, detach=True, requires_grad=False)
        rows = rows.reshape(-1, rows.shape[-1])
        done = rollout_done_mask(rollout_flat, rows[..., :1])
        if done is not None:
            rows = rows[~done.reshape(-1)]
        limit = int(self.config.ipmd.reward_policy_replay_rows_per_rollout)
        if rows.shape[0] > limit:
            rows = rows[torch.randperm(rows.shape[0], device=rows.device)[:limit]]
        self._reward_replay.add(rows)

    def _run_pre_policy_reward_updates(
        self,
        rollout_flat: TensorDict,
        metadata: PPOTrainingMetadata,
    ) -> dict[str, Tensor]:
        """Update the reward estimator before PPO reward and advantage computation."""
        if not self._reward_update_enabled:
            return {}
        if self._reward_replay is not None:
            self._offer_rollout_to_reward_replay(rollout_flat)

        metrics = self._empty_reward_update_metrics()
        minibatch_slots = 0
        updated_slots = 0
        updates_completed = metadata.updates_completed
        start_update_idx = (
            int(updates_completed.item())
            if isinstance(updates_completed, Tensor)
            else int(updates_completed)
        )
        self.data_buffer.empty()
        # Route through the shaping helper: with a recurrent actor the buffer
        # stores [env, T] sequences and a flat extend silently corrupts it.
        self.data_buffer.extend(self._rollout_for_buffer(rollout_flat))
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
                        updated_slots += 1
                    else:
                        step_metrics = self._empty_reward_update_metrics()
                    for key, value in step_metrics.items():
                        metrics[key] = metrics[key] + value
                    minibatch_slots += 1
        finally:
            self.data_buffer.empty()

        if minibatch_slots == 0:
            return metrics
        # Estimator metrics are means over the slots that ran a reward step;
        # averaging over every slot would scale them by the fraction of due
        # slots (reward_update_interval > 1). `reward_updates_applied` stays a
        # per-slot rate.
        due = float(max(updated_slots, 1))
        for key in tuple(metrics.keys()):
            scale = float(minibatch_slots) if key == "reward_updates_applied" else due
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
        update_idx = (
            int(num_network_updates.item())
            if isinstance(num_network_updates, Tensor)
            else int(num_network_updates)
        )
        self.optim.zero_grad(set_to_none=True)
        bc_pretrain_active = update_idx < self._bc_pretrain_updates
        output_loss = (
            self._ppo_metrics_without_backward(batch)
            if bc_pretrain_active
            else self._backward_ppo_terms(batch)
        )
        bc_metrics = self._backward_bc_terms(expert_batch, has_expert)
        rollout_bc_metrics = self._backward_rollout_bc_terms(batch)

        # Gradient clipping — always call for a fixed graph
        grad_norm_tensor = clip_grad_norm_(self._grad_clip_params, self._max_grad_norm)

        self.optim.step()
        hl_skill_metrics = self._run_hl_skill_online_update(
            batch,
            update_idx=update_idx,
        )

        output_loss.set("alpha", torch.ones((), device=self.device))
        for key, value in bc_metrics.items():
            output_loss.set(key, value)
        for key, value in rollout_bc_metrics.items():
            output_loss.set(key, value)
        output_loss.set(
            "bc_pretrain_active",
            torch.tensor(
                float(bc_pretrain_active),
                device=self.device,
                dtype=torch.float32,
            ),
        )
        for key, value in hl_skill_metrics.items():
            output_loss.set(key, value.detach())
        output_loss.set("grad_norm", grad_norm_tensor.detach())

        return output_loss, num_network_updates + 1

    def prepare(
        self,
        iteration: PPOIterationData,  # noqa: ARG002
        metadata: PPOTrainingMetadata,  # noqa: ARG002
    ) -> None:
        """Reward preparation runs in iterate() after pre-PPO reward updates."""
        return

    def _apply_critic_lr_schedule(self, metadata: PPOTrainingMetadata) -> None:
        """Set the critic group's learning rate for this iteration.

        A no-op under ``ipmd.critic_lr_schedule="constant"``. Progress is
        cumulative environment frames over the configured frame budget --
        ``metadata.frames_processed`` is seeded with the resume offset, so a
        walltime-chained segment continues the decay where the previous one
        stopped instead of restarting at ``critic_learning_rate``.
        """
        schedule = str(self.config.ipmd.critic_lr_schedule)
        if schedule == "constant":
            return
        start = self.config.ipmd.critic_learning_rate
        end = self.config.ipmd.critic_lr_final
        if start is None or end is None:
            return
        total_frames = float(self.config.collector.total_frames)
        if total_frames <= 0.0:
            return
        progress = float(metadata.frames_processed) / total_frames
        progress = min(max(progress, 0.0), 1.0)
        start = float(start)
        end = float(end)
        if schedule == "linear":
            lr = start + (end - start) * progress
        else:  # cosine
            lr = end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
        bias_scale = float(self.config.ipmd.first_layer_bias_lr_scale)
        for group in self.optim.param_groups:
            if group.get("name") == "critic":
                group["lr"] = lr
            elif group.get("name") == "critic_first_bias":
                group["lr"] = lr * bias_scale

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

        self._apply_critic_lr_schedule(metadata)
        self.data_buffer.empty()
        self.actor_critic.train()
        self.adv_module.train()
        self.reward_estimator.train()

        with timeit("training"):
            rollout_flat = iteration.rollout.reshape(-1)
            with self._profile_iteration_phase(
                iteration.phase_times, "learn/latent_update"
            ):
                if self._latent_learner is not None:
                    learner_metrics = self._latent_learner.update(rollout_flat)
                    if learner_metrics:
                        iteration.metrics.update(
                            {f"train/{k}": v for k, v in learner_metrics.items()}
                        )

            with self._profile_iteration_phase(
                iteration.phase_times, "learn/reward_updates"
            ):
                reward_metrics = self._run_pre_policy_reward_updates(
                    rollout_flat,
                    metadata,
                )
            iteration.metrics.update(
                {f"train/{key}": value.item() for key, value in reward_metrics.items()}
            )
            with self._profile_iteration_phase(
                iteration.phase_times, "learn/reward_assignment"
            ):
                iteration.metrics.update(
                    self._prepare_rollout_rewards(iteration.rollout)
                )
            with self._profile_iteration_phase(
                iteration.phase_times, "learn/advantage_and_buffer"
            ):
                iteration.rollout = self.pre_iteration_compute(iteration.rollout)

            with self._profile_iteration_phase(
                iteration.phase_times, "learn/policy_updates"
            ):
                for epoch_idx in range(metadata.epochs_per_rollout):
                    for batch_idx, batch in enumerate(self.data_buffer):
                        kl_context = None
                        if (self.config.optim.scheduler or "").lower() == "adaptive":
                            kl_context = self._prepare_kl_context(
                                batch, metadata.policy_operator
                            )

                        needs_expert_batch = bool(self._bc_coeff > 0.0)
                        if needs_expert_batch:
                            expert_batch, has_expert = self._expert_batch_for_update(
                                batch
                            )
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
                                self._record_kl_for_lr_adaptation(
                                    kl_approx, self.config.optim
                                )

                        losses[epoch_idx, batch_idx] = (
                            self._select_reported_loss_metrics(loss)
                        )

                    if self.lr_scheduler and self.lr_scheduler_step == "epoch":
                        self.lr_scheduler.step()

                # One adaptive-LR step per iteration under
                # optim.kl_adapt_step="iteration"; a no-op otherwise.
                self._flush_kl_lr_adaptation(self.config.optim)

        iteration.learn_time = time.perf_counter() - learn_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():  # type: ignore[attr-defined]
            iteration.metrics[f"train/{key}"] = value.item()  # type: ignore[attr-defined]
