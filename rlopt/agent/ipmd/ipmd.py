from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import InteractionType
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.data import ReplayBuffer
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import MLP
from torchrl.record.loggers import Logger

from rlopt.agent.imitation.latent_skill import (
    LatentEncoder,
    LatentSkillMixin,
    generalized_advantage_estimate,
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
    epic_distance,
    flatten_feature_tensor,
    infer_batch_shape,
    mapping_get_obs_value,
    next_obs_key,
)
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info


@dataclass
class IPMDConfig(PPOConfig):
    """IPMD-specific configuration (PPO-based)."""

    latent_dim: int = 16
    """Dimension of the latent skill space."""

    latent_key: ObsKey = "latent_command"
    """Key for the latent skill."""

    latent_steps_min: int = 30
    """Minimum steps before resampling latent."""

    latent_steps_max: int = 120
    """Maximum steps before resampling latent."""

    latent_vmf_kappa: float = 1.0
    """VMF parameter for the latent skill distribution."""

    mi_reward_weight: float = 0.25
    """Weight for the mutual information reward."""

    mi_loss_coeff: float = 1.0
    """Coefficient for the mutual information loss."""

    mi_encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    """Hidden dimensions for the mutual information encoder."""

    mi_encoder_activation: str = "elu"
    """Activation function for the mutual information encoder."""

    mi_encoder_lr: float = 3e-4
    """Learning rate for the mutual information encoder."""

    mi_grad_clip_norm: float = 1.0
    """Gradient clipping norm for the mutual information encoder."""

    mi_weight_decay_coeff: float = 0.0
    """Weight decay coefficient for the mutual information encoder."""

    mi_grad_penalty_coeff: float = 0.0
    """Gradient penalty coefficient for the mutual information encoder."""

    # When True, shifts MI reward to [0, 1] via (dot + 1) / 2 (ProtoMotions / ASE style).
    # When False, clamps negative dot products to 0.
    mi_hypersphere_reward_shift: bool = True
    """Shift MI reward to [0, 1] via (dot + 1) / 2, matching ASE / ProtoMotions."""

    mi_critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    """Hidden dimensions for the MI critic value network."""

    mi_critic_activation: str = "elu"
    """Activation function for the MI critic."""

    mi_critic_lr: float = 3e-4
    """Learning rate for the MI critic optimizer."""

    mi_critic_grad_clip_norm: float = 1.0
    """Gradient clip norm for the MI critic."""

    latent_input_type: str = "s"
    """Input type for the latent encoder / MI posterior.

    Supported values:
    - ``"s"``   - current state only
    - ``"s'"``  - next state only
    - ``"sa"``  - state-action
    - ``"ss'"`` - state-next-state
    """

    diversity_bonus_coeff: float = 0.05
    """Coefficient for the diversity bonus."""

    diversity_target: float = 1.0
    """Target diversity level."""

    latent_uniformity_coeff: float = 0.0
    """Coefficient for the latent uniformity loss."""

    latent_uniformity_temperature: float = 2.0
    """Temperature for the latent uniformity loss."""

    # Reward estimator network and loss settings
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
    (then ``policy.get_input_keys()``).  Set this to let the reward model
    consume a different subset of observation keys than policy / value.
    """

    reward_num_cells: tuple[int, ...] = (256, 256)
    """Hidden layer sizes for the reward estimator MLP."""

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

    bc_coef: float = 1.0
    """Behavior cloning (MLE) loss coefficient on expert actions.

    When > 0, adds ``-bc_coef * mean(log_prob(expert_action | policy))``
    to each update step.  This regularises the policy toward expert actions
    during early training.  Set to 0 to disable (default).
    """


@dataclass
class IPMDRLOptConfig(PPORLOptConfig):
    """IPMD configuration extending PPORLOptConfig (PPO-based)."""

    ipmd: IPMDConfig = field(default_factory=IPMDConfig)
    """IPMD configuration."""


class IPMD(LatentSkillMixin, PPO):
    """IPMD algorithm with PPO as the base RL algorithm.

    Uses the same on-policy rollout + GAE + multiple epochs over mini-batches as PPO,
    and adds a reward estimator r(s, a, s') trained with the IPMD objective
    (policy estimated return - expert estimated return). Optionally uses estimated
    rewards for PPO updates.
    """

    # ------------------------------------------------------------------
    # Scalar and key-list caches populated by _cache_ipmd_scalars.
    # Declared here so Pyright treats them as proper instance attributes.
    # ------------------------------------------------------------------

    # Reward estimator
    _env_reward_weight: float
    _est_reward_weight: float
    _est_reward_clamp_min: float | None
    _est_reward_clamp_max: float | None

    # MI encoder
    _mi_loss_coeff: float
    _mi_vmf_kappa: float
    _mi_grad_clip_norm: float
    _mi_weight_decay_coeff: float
    _mi_encoder_grad_penalty_coeff: float
    _latent_uniformity_coeff: float
    _latent_uniformity_temperature: float

    # MI critic / reward
    _mi_critic_grad_clip_norm: float
    _mi_reward_weight: float
    _mi_hypersphere_reward_shift: bool

    # Diversity
    _diversity_bonus_coeff: float
    _diversity_target: float

    # Key-list caches (frozen after init)
    _encoder_keys_cache: list[BatchKey]
    _rollout_keys_cache: list[BatchKey]
    _expert_keys_cache: list[BatchKey]

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
        self.env = env

        self._init_latent_skills(
            env,
            latent_key=self.config.ipmd.latent_key,
            latent_dim=int(self.config.ipmd.latent_dim),
            latent_steps_min=int(self.config.ipmd.latent_steps_min),
            latent_steps_max=int(self.config.ipmd.latent_steps_max),
        )

        # Observation key groups can differ across policy, value, and reward model.
        self._policy_obs_keys: list[ObsKey] = self.config.policy.get_input_keys()
        value_cfg = self.config.value_function
        self._value_obs_keys: list[ObsKey] = (
            value_cfg.get_input_keys()
            if value_cfg is not None
            else list(self._policy_obs_keys)
        )
        self._reward_obs_keys: list[ObsKey] = self._resolve_reward_obs_keys()

        all_obs_keys = dedupe_keys(
            self._policy_obs_keys + self._value_obs_keys + self._reward_obs_keys
        )
        self._obs_feature_ndims: dict[ObsKey, int] = {
            k: self._obs_key_feature_ndim(k) for k in all_obs_keys
        }
        self._obs_feature_dims: dict[ObsKey, int] = {
            k: self._obs_key_feature_dim(k) for k in all_obs_keys
        }

        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        action_shape = tuple(int(d) for d in action_spec.shape)  # type: ignore[union-attr]
        self._action_feature_ndim = len(action_shape)
        self._action_feature_dim = int(math.prod(action_shape)) if action_shape else 1

        # Reward estimator: r(s, a, s') -> scalar
        self.reward_estimator: torch.nn.Module = self._construct_reward_estimator()
        self.reward_estimator.to(self.device)

        self._expert_batch_sampler: (
            Callable[[int, list[BatchKey]], TensorDict | None] | None
        ) = None

        self.mi_encoder: LatentEncoder | None = None
        self.mi_encoder_optim: torch.optim.Optimizer | None = None
        self.mi_critic: torch.nn.Module | None = None
        self.mi_critic_optim: torch.optim.Optimizer | None = None

        # Pre-cache compile-friendly scalars from config
        self._cache_ipmd_scalars()

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
        self._refresh_grad_clip_params()
        self._policy_operator = self.actor_critic.get_policy_operator()
        self._bc_debug_anomaly_prints = 0

    def _cache_ipmd_scalars(self) -> None:
        """Pre-cache config scalars and input-type flags for compile-friendly hot paths."""
        cfg = self.config.ipmd
        rit = cfg.reward_input_type
        lit = cfg.latent_input_type
        if lit not in ("s", "s'", "sa", "ss'"):
            msg = f"latent_input_type must be one of 's', \"s'\", 'sa', or \"ss'\". Got {lit!r}."
            raise ValueError(msg)
        # Boolean flags for feature assembly (avoids string dispatch in hot paths)
        self._rit_use_s: bool = rit in ("s", "sa", "sas")
        self._rit_use_a: bool = rit in ("sa", "sas")
        self._rit_use_sn: bool = rit in ("s'", "sas")
        self._lit_use_s: bool = lit in ("s", "sa", "ss'")
        self._lit_use_a: bool = lit == "sa"
        self._lit_use_sn: bool = lit in ("s'", "ss'")

        obs_dim = sum(self._obs_feature_dims[k] for k in self._reward_obs_keys)
        self._mi_obs_dim = (
            (obs_dim if self._lit_use_s else 0)
            + (self._action_feature_dim if self._lit_use_a else 0)
            + (obs_dim if self._lit_use_sn else 0)
        )

        # Reward estimator scalars
        self._reward_loss_coeff: float = float(cfg.reward_loss_coeff)
        self._reward_l2_coeff: float = float(cfg.reward_l2_coeff)
        self._reward_grad_penalty_coeff: float = float(cfg.reward_grad_penalty_coeff)
        self._reward_detach_features: bool = bool(cfg.reward_detach_features)
        self._env_reward_weight: float = float(cfg.env_reward_weight)
        self._est_reward_weight: float = float(cfg.est_reward_weight)
        self._est_reward_clamp_min: float | None = cfg.estimated_reward_clamp_min
        self._est_reward_clamp_max: float | None = cfg.estimated_reward_clamp_max

        # BC scalar
        self._bc_coeff: float = float(cfg.bc_coef)

        # MI encoder scalars
        self._mi_loss_coeff: float = float(cfg.mi_loss_coeff)
        self._mi_vmf_kappa: float = float(cfg.latent_vmf_kappa)
        self._mi_grad_clip_norm: float = float(cfg.mi_grad_clip_norm)
        self._mi_weight_decay_coeff: float = float(cfg.mi_weight_decay_coeff)
        self._mi_encoder_grad_penalty_coeff: float = float(cfg.mi_grad_penalty_coeff)
        self._latent_uniformity_coeff: float = float(cfg.latent_uniformity_coeff)
        self._latent_uniformity_temperature: float = float(
            max(cfg.latent_uniformity_temperature, 1e-6)
        )

        # MI critic scalars
        self._mi_critic_grad_clip_norm: float = float(cfg.mi_critic_grad_clip_norm)
        self._mi_reward_weight: float = float(cfg.mi_reward_weight)
        self._mi_hypersphere_reward_shift: bool = bool(cfg.mi_hypersphere_reward_shift)

        # Diversity scalars
        self._diversity_bonus_coeff: float = float(cfg.diversity_bonus_coeff)
        self._diversity_target: float = float(cfg.diversity_target)

        # Gradient clip
        max_grad = getattr(self.config.optim, "max_grad_norm", None)
        self._max_grad_norm: float = float(max_grad) if max_grad else 1e10

        # Compile the output activation into a callable (avoids string dispatch)
        out_act = cfg.reward_output_activation
        scale = float(cfg.reward_output_scale)
        if out_act == "tanh":
            self._reward_out_fn: Callable[[Tensor], Tensor] = (
                lambda r: torch.tanh(r) * scale
            )
        elif out_act == "sigmoid":
            self._reward_out_fn = lambda r: torch.sigmoid(r) * scale
        else:
            self._reward_out_fn = lambda r: r

        # Key lists: computed once from init-time flags; frozen for the lifetime of the agent.
        # _latent_encoder_keys must come before rollout/expert keys (both extend it).
        enc_keys: list[BatchKey] = []
        if self._lit_use_s:
            enc_keys.extend(self._reward_obs_keys)
        if self._lit_use_a:
            enc_keys.append("action")  # type: ignore[arg-type]
        if self._lit_use_sn:
            enc_keys.extend(next_obs_key(k) for k in self._reward_obs_keys)
        self._encoder_keys_cache: list[BatchKey] = dedupe_keys(enc_keys)

        rollout_keys: list[BatchKey] = [self._latent_key]  # type: ignore[list-item]
        if self._rit_use_s:
            rollout_keys.extend(self._reward_obs_keys)
        if self._rit_use_a:
            rollout_keys.append("action")  # type: ignore[arg-type]
        if self._rit_use_sn:
            rollout_keys.extend(next_obs_key(k) for k in self._reward_obs_keys)
        rollout_keys.extend(self._encoder_keys_cache)
        self._rollout_keys_cache: list[BatchKey] = dedupe_keys(rollout_keys)

        expert_keys: list[BatchKey] = []
        if self._rit_use_s:
            expert_keys.extend(self._reward_obs_keys)
        if self._rit_use_sn:
            expert_keys.extend(next_obs_key(k) for k in self._reward_obs_keys)
        expert_keys.extend(self._encoder_keys_cache)
        if self._rit_use_a or self._bc_coeff > 0.0:
            expert_keys.append("action")  # type: ignore[arg-type]
        if self._bc_coeff > 0.0:
            expert_keys.extend(
                k for k in self._policy_obs_keys if k != self._latent_key
            )
        self._expert_keys_cache: list[BatchKey] = dedupe_keys(expert_keys)

    _REWARD_INPUT_TYPES = frozenset({"s", "s'", "sa", "sas"})
    _REWARD_OUTPUT_ACTIVATIONS = frozenset({"none", "tanh", "sigmoid"})

    # -------------------------------------------------------------------------
    # Key / dimension helpers
    # -------------------------------------------------------------------------

    def _resolve_reward_obs_keys(self) -> list[ObsKey]:
        """Resolve observation keys used by reward estimator state inputs."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        reward_keys = cfg.ipmd.reward_input_keys
        if not reward_keys:
            reward_keys = (
                cfg.value_function.get_input_keys()
                if cfg.value_function is not None
                else cfg.policy.get_input_keys()
            )
        return dedupe_keys(list(reward_keys))

    def _obs_key_feature_shape(self, key: ObsKey) -> tuple[int, ...]:
        shape = tuple(int(d) for d in self.env.observation_spec[key].shape)
        batch_prefix = tuple(int(d) for d in getattr(self.env, "batch_size", ()))
        if len(batch_prefix) > 0 and shape[: len(batch_prefix)] == batch_prefix:
            return shape[len(batch_prefix) :]
        return shape

    def _obs_key_feature_ndim(self, key: ObsKey) -> int:
        return len(self._obs_key_feature_shape(key))

    def _obs_key_feature_dim(self, key: ObsKey) -> int:
        shape = self._obs_key_feature_shape(key)
        return int(math.prod(shape)) if shape else 1

    # -------------------------------------------------------------------------
    # Feature extraction from TensorDict
    # -------------------------------------------------------------------------

    def _obs_features_from_td(
        self,
        td: TensorDict,
        keys: list[ObsKey],
        *,
        next_obs: bool,
        detach: bool,
    ) -> Tensor:
        """Concatenate flattened observation tensors for the given keys."""
        parts = [
            flatten_feature_tensor(
                td[next_obs_key(k) if next_obs else k], self._obs_feature_ndims[k]
            )
            for k in keys
        ]
        if detach:
            parts = [t.detach() for t in parts]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _action_features_from_td(self, td: TensorDict, *, detach: bool) -> Tensor:
        action = flatten_feature_tensor(td["action"], self._action_feature_ndim)
        return action.detach() if detach else action

    def _latent_encoder_required_keys(self) -> list[BatchKey]:
        return self._encoder_keys_cache

    def _rollout_required_keys(self) -> list[BatchKey]:
        return self._rollout_keys_cache

    def _require_batch_keys(
        self, td: TensorDict, required_keys: list[BatchKey], *, context: str
    ) -> None:
        available = set(td.keys(True))
        missing = [k for k in required_keys if k not in available]
        if missing:
            msg = f"{context} is missing required keys: {missing}. Available: {list(td.keys(True))}."
            raise KeyError(msg)

    def _latent_encoder_features_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor:
        """Assemble encoder input (s, a, s') according to latent_input_type flags."""
        self._require_batch_keys(
            td, self._latent_encoder_required_keys(), context=context
        )
        parts: list[Tensor] = []
        if self._lit_use_s:
            parts.append(
                self._obs_features_from_td(
                    td, self._reward_obs_keys, next_obs=False, detach=detach
                )
            )
        if self._lit_use_a:
            parts.append(self._action_features_from_td(td, detach=detach))
        if self._lit_use_sn:
            parts.append(
                self._obs_features_from_td(
                    td, self._reward_obs_keys, next_obs=True, detach=detach
                )
            )
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _rollout_latents_from_td(self, td: TensorDict, *, detach: bool) -> Tensor:
        latent = self._latent_condition_from_td(td, detach=detach)
        if latent is None:
            raise RuntimeError("Rollout batch is missing stamped latent commands.")
        return latent.to(self.device)

    def _expert_latents_from_td(self, td: TensorDict, *, detach: bool) -> Tensor:
        latent = self._latent_condition_from_td(td, detach=detach)
        if latent is not None:
            return latent.to(self.device)
        # No stamped latent → synthesize via MI encoder
        if self.mi_encoder is None:
            raise RuntimeError(
                "MI encoder must exist before synthesizing expert latents."
            )
        obs_features = self._latent_encoder_features_from_td(
            td, detach=False, context="expert latent batch"
        )
        latent = self.mi_encoder(obs_features.to(self.device))
        return latent.detach() if detach else latent

    def _expert_required_keys(self) -> list[BatchKey]:
        return self._expert_keys_cache

    # -------------------------------------------------------------------------
    # Network construction
    # -------------------------------------------------------------------------

    def _construct_reward_estimator(self) -> torch.nn.Module:
        """Build reward MLP whose input dimension depends on ``reward_input_type``."""
        cfg = self.config.ipmd
        rit = cfg.reward_input_type
        if rit not in self._REWARD_INPUT_TYPES:
            msg = f"Unknown reward_input_type {rit!r}; expected one of {sorted(self._REWARD_INPUT_TYPES)}"
            raise ValueError(msg)
        out_act = cfg.reward_output_activation
        if out_act not in self._REWARD_OUTPUT_ACTIVATIONS:
            msg = f"Unknown reward_output_activation {out_act!r}; expected one of {sorted(self._REWARD_OUTPUT_ACTIVATIONS)}"
            raise ValueError(msg)

        obs_dim = sum(self._obs_feature_dims[k] for k in self._reward_obs_keys)
        act_dim = self._action_feature_dim
        in_dim = {
            "s": obs_dim,
            "s'": obs_dim,
            "sa": obs_dim + act_dim,
            "sas": obs_dim * 2 + act_dim,
        }[rit] + self._latent_dim

        net = MLP(
            in_features=in_dim,
            out_features=1,
            num_cells=list(cfg.reward_num_cells),
            activation_class=get_activation_class(cfg.reward_activation),
            device=self.device,
        )
        self._initialize_weights(net, cfg.reward_init)
        return net

    def _compile_components(self) -> None:
        """Compile reward estimator and PPO update if torch.compile is enabled."""
        if not self.config.compile.compile:
            return
        super()._compile_components()
        self.reward_estimator = torch.compile(self.reward_estimator)

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizers for actor-critic, reward estimator, MI encoder, and MI critic."""
        if not hasattr(self, "reward_estimator"):
            return super()._set_optimizers(optimizer_cls, optimizer_kwargs)

        # Build MI encoder if not yet created
        if self.mi_encoder is None:
            self.mi_encoder = LatentEncoder(
                input_dim=self._mi_obs_dim,
                latent_dim=self._latent_dim,
                hidden_dims=list(self.config.ipmd.mi_encoder_hidden_dims),
                activation=self.config.ipmd.mi_encoder_activation,
            ).to(self.device)
        self.mi_encoder_optim = torch.optim.Adam(
            self.mi_encoder.parameters(),
            lr=float(self.config.ipmd.mi_encoder_lr),
        )

        # Build MI critic if not yet created
        if self.mi_critic is None:
            obs_dim = sum(self._obs_feature_dims[k] for k in self._reward_obs_keys)
            self.mi_critic = MLP(
                in_features=obs_dim + self._latent_dim,
                out_features=1,
                num_cells=list(self.config.ipmd.mi_critic_hidden_dims),
                activation_class=get_activation_class(
                    self.config.ipmd.mi_critic_activation
                ),
                device=self.device,
            )
        self.mi_critic_optim = torch.optim.Adam(
            self.mi_critic.parameters(),
            lr=float(self.config.ipmd.mi_critic_lr),
        )

        all_params = [
            *self.actor_critic.parameters(),
            *self.reward_estimator.parameters(),
        ]
        return [optimizer_cls(all_params, **optimizer_kwargs)]

    # -------------------------------------------------------------------------
    # Expert batch utilities
    # -------------------------------------------------------------------------

    def _require_expert_batch_keys(
        self, expert_batch: TensorDict, required_keys: list[BatchKey]
    ) -> None:
        available = set(expert_batch.keys(True))
        missing = [
            k
            for k in required_keys
            if k not in available
            and not (k == "action" and "expert_action" in available)
        ]
        if missing:
            msg = f"Expert sampler contract violated. Missing keys: {missing}. Required: {required_keys}. Available: {list(available)}."
            raise KeyError(msg)

    @staticmethod
    def _expert_action_from_td(td: TensorDict) -> Tensor | None:
        """Return expert action, preferring 'expert_action' key over 'action'."""
        action = td.get("expert_action")
        return action if action is not None else td.get("action")

    def _next_expert_batch(self, batch_size: int | None = None) -> TensorDict:
        assert isinstance(self.config, IPMDRLOptConfig)
        n = int(
            batch_size
            or self.config.ipmd.expert_batch_size
            or self.config.loss.mini_batch_size
        )

        if self._expert_batch_sampler is None:
            raise RuntimeError(
                "IPMD training requires env.sample_expert_batch(...). "
                "Tests may install a private expert sampler override."
            )
        required_keys = self._expert_required_keys()
        try:
            expert_batch = self._expert_batch_sampler(n, required_keys)
        except Exception as err:
            raise RuntimeError("Failed to sample expert batch from sampler.") from err
        if expert_batch is None:
            raise RuntimeError("Expert sampler returned None.")
        if expert_batch.numel() > n:
            expert_batch = expert_batch[:n]
        expert_batch = expert_batch.to(self.device)
        self._log_batch_contract_once(
            flag_attr="_expert_batch_contract_logged",
            context="expert",
            batch=expert_batch,
            required_keys=required_keys,
        )
        self._require_expert_batch_keys(expert_batch, required_keys)
        return expert_batch

    # -------------------------------------------------------------------------
    # Reward estimator input assembly
    # -------------------------------------------------------------------------

    def _latents_for_role(
        self, td: TensorDict, *, batch_role: Literal["rollout", "expert"], detach: bool
    ) -> Tensor:
        """Resolve latent conditioning vector depending on batch origin."""
        if batch_role == "rollout":
            return self._rollout_latents_from_td(td, detach=detach)
        if batch_role == "expert":
            return self._expert_latents_from_td(td, detach=detach)
        raise ValueError(f"Unknown batch_role {batch_role!r}.")

    def _reward_input_from_batch(
        self,
        td: TensorDict,
        *,
        batch_role: Literal["rollout", "expert"],
        detach: bool | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Assemble reward-estimator input using pre-cached _rit_* flags (compile-friendly)."""
        if detach is None:
            detach = self._reward_detach_features
        parts: list[Tensor] = []
        if self._rit_use_s:
            parts.append(
                self._obs_features_from_td(
                    td, self._reward_obs_keys, next_obs=False, detach=detach
                )
            )
        if self._rit_use_a:
            parts.append(self._action_features_from_td(td, detach=detach))
        if self._rit_use_sn:
            parts.append(
                self._obs_features_from_td(
                    td, self._reward_obs_keys, next_obs=True, detach=detach
                )
            )
        # Latent is always detached — it's a conditioning variable, not a learned input here
        parts.append(self._latents_for_role(td, batch_role=batch_role, detach=True))
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return x.detach().requires_grad_(True) if requires_grad else x

    def _reward_from_batch(
        self,
        td: TensorDict,
        *,
        batch_role: Literal["rollout", "expert"],
        detach: bool | None = None,
        requires_grad: bool = False,
        return_input: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Evaluate reward estimator on a transition batch."""
        x = self._reward_input_from_batch(
            td, batch_role=batch_role, detach=detach, requires_grad=requires_grad
        )
        reward = self._reward_out_fn(self.reward_estimator(x))
        return (reward, x) if return_input else reward

    # -------------------------------------------------------------------------
    # MI encoder
    # -------------------------------------------------------------------------

    def _mi_features_and_latents(self, batch: TensorDict) -> tuple[Tensor, Tensor]:
        """Return (encoder input features, ground-truth latents) for MI training."""
        obs_features = self._latent_encoder_features_from_td(
            batch, detach=False, context="rollout latent batch"
        ).to(self.device)
        latents = self._rollout_latents_from_td(batch, detach=True)
        return obs_features, latents

    def _mi_grad_penalty(self, obs_features: Tensor, latents: Tensor) -> Tensor:
        if self.mi_encoder is None or self._mi_encoder_grad_penalty_coeff <= 0.0:
            return torch.zeros((), device=self.device)
        obs_req = obs_features.detach().requires_grad_(True)
        latent_pred = self.mi_encoder(obs_req)
        score = (latent_pred * latents.detach()).sum(dim=-1)
        grads = torch.autograd.grad(
            score.sum(), obs_req, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return grads.pow(2).sum(dim=-1).mean()

    def _latent_uniformity(self, latent_pred: Tensor) -> Tensor:
        if latent_pred.shape[0] <= 1:
            return torch.zeros((), device=latent_pred.device, dtype=latent_pred.dtype)
        # Subsample for large rollouts (pairwise distance is O(N²) memory)
        if latent_pred.shape[0] > 1024:
            idx = torch.randperm(latent_pred.shape[0], device=latent_pred.device)[:1024]
            latent_pred = latent_pred[idx]
        pairwise_dist = torch.cdist(latent_pred, latent_pred, p=2)
        return torch.log(
            torch.exp(
                -self._latent_uniformity_temperature * pairwise_dist.pow(2)
            ).mean()
        )

    def _update_mi_encoder(self, rollout_flat: TensorDict) -> dict[str, float]:
        """One gradient step on the MI encoder (vMF alignment + regularisation)."""
        if self.mi_encoder is None or self.mi_encoder_optim is None:
            return {}
        if self._mi_loss_coeff <= 0.0:
            return {}

        obs_features, latents = self._mi_features_and_latents(rollout_flat)
        if obs_features.shape[0] == 0:
            return {}

        # vMF alignment: maximise cosine similarity to the ground-truth latent
        latent_pred = self.mi_encoder(obs_features)
        similarity = (latent_pred * latents).sum(dim=-1)
        mi_loss = -self._mi_vmf_kappa * similarity.mean()
        grad_penalty = self._mi_grad_penalty(obs_features, latents)

        weight_decay = torch.zeros((), device=self.device)
        if self._mi_weight_decay_coeff > 0.0:
            for p in self.mi_encoder.parameters():
                if p.ndim >= 2:
                    weight_decay = weight_decay + p.pow(2).mean()

        # Uniformity loss over rollout + expert embeddings (matching ASE / ProtoMotions)
        expert_td = self._next_expert_batch()
        expert_obs = self._latent_encoder_features_from_td(
            expert_td, detach=False, context="expert latent batch"
        ).to(self.device)
        uniformity = self._latent_uniformity(
            torch.cat([latent_pred, self.mi_encoder(expert_obs)], dim=0)
        )

        total_loss = (
            self._mi_loss_coeff * mi_loss
            + self._mi_encoder_grad_penalty_coeff * grad_penalty
            + self._mi_weight_decay_coeff * weight_decay
            + self._latent_uniformity_coeff * uniformity
        )

        self.mi_encoder_optim.zero_grad(set_to_none=True)
        total_loss.backward()
        if self._mi_grad_clip_norm > 0.0:
            clip_grad_norm_(self.mi_encoder.parameters(), self._mi_grad_clip_norm)
        self.mi_encoder_optim.step()

        return {
            "ipmd/mi_total_loss": total_loss.detach().item(),
            "ipmd/mi_loss": mi_loss.detach().item(),
            "ipmd/mi_similarity_mean": similarity.detach().mean().item(),
            "ipmd/mi_grad_penalty": grad_penalty.detach().item(),
            "ipmd/mi_weight_decay": weight_decay.detach().item(),
            "ipmd/latent_uniformity": uniformity.detach().item(),
        }

    def _mi_reward(self, obs_features: Tensor, latents: Tensor) -> Tensor:
        """Intrinsic MI reward: cosine alignment between encoder prediction and latent."""
        if self.mi_encoder is None:
            return torch.zeros(obs_features.shape[0], device=obs_features.device)
        with torch.no_grad():
            score = (self.mi_encoder(obs_features) * latents).sum(dim=-1)
        # ASE / ProtoMotions style: shift dot product from [-1, 1] to [0, 1]
        return (
            (score + 1.0) / 2.0
            if self._mi_hypersphere_reward_shift
            else score.clamp_min(0.0)
        )

    # -------------------------------------------------------------------------
    # Diversity loss (ProtoMotions style)
    # -------------------------------------------------------------------------

    def _diversity_loss(self, batch: TensorDict) -> Tensor:
        """Encourage action diversity across latent codes via mean-action delta."""
        if self._diversity_bonus_coeff <= 0.0 or batch.numel() <= 1:
            return torch.zeros((), device=self.device)

        # Old distribution parameters were stamped during the rollout
        old_latents = self._rollout_latents_from_td(batch, detach=True)
        old_dist = self._policy_operator.build_dist_from_params(
            batch.select("loc", "scale").clone()
        )
        old_action = self._policy_action_from_dist(old_dist)
        if old_action is None:
            msg = "IPMD diversity loss could not recover old policy mean action."
            raise RuntimeError(msg)
        old_mean_action = self._clip_policy_action(old_action.detach())

        # Replace latent with a freshly sampled code and re-evaluate
        new_latents = self._sample_unit_latents(
            batch.numel(), device=self.device, dtype=old_latents.dtype
        )
        policy_td = batch.select(*self._policy_obs_keys).clone()
        policy_td[self._latent_key] = new_latents.reshape(
            *batch.batch_size, self._latent_dim
        )  # type: ignore[index]
        new_action = self._policy_action_from_dist(
            self._policy_operator.get_dist(policy_td)
        )
        if new_action is None:
            msg = "IPMD diversity loss could not recover new policy mean action."
            raise RuntimeError(msg)
        new_mean_action = self._clip_policy_action(new_action)

        action_delta = (new_mean_action - old_mean_action).pow(2).mean(dim=-1)
        latent_delta = 0.5 - 0.5 * (new_latents * old_latents).sum(dim=-1)
        diversity_bonus = action_delta / (latent_delta + 1e-5)
        return (self._diversity_target - diversity_bonus).pow(2).mean()

    def _extra_actor_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        if self._diversity_bonus_coeff <= 0.0:
            return torch.zeros((), device=self.device), {}
        diversity_loss = self._diversity_loss(batch)
        return diversity_loss * self._diversity_bonus_coeff, {
            "loss_diversity": diversity_loss.detach()
        }

    # -------------------------------------------------------------------------
    # Reward loss helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _reward_grad_penalty_from_input(reward: Tensor, reward_input: Tensor) -> Tensor:
        """Squared gradient norm of the reward w.r.t. its input features."""
        (grads,) = torch.autograd.grad(
            reward.sum(),
            reward_input,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        return grads.pow(2).sum(dim=-1).mean()

    @staticmethod
    def _reward_tensor_stats(prefix: str, reward: Tensor) -> dict[str, float]:
        r = reward.detach().float()
        return {
            f"{prefix}_mean": r.mean().item(),
            f"{prefix}_std": r.std().item(),
            f"{prefix}_min": r.min().item(),
            f"{prefix}_max": r.max().item(),
        }

    @staticmethod
    def _reward_alignment_metrics(
        prefix: str, reward_pred: Tensor, reward_tgt: Tensor
    ) -> dict[str, float]:
        pred = reward_pred.detach().float().flatten()
        tgt = reward_tgt.detach().float().flatten()
        diff = pred - tgt

        pred_mean, tgt_mean = pred.mean(), tgt.mean()
        pred_c, tgt_c = pred - pred_mean, tgt - tgt_mean
        pred_var = pred_c.pow(2).mean()
        tgt_var = tgt_c.pow(2).mean()
        cov = (pred_c * tgt_c).mean()

        pearson_corr, corr_distance = epic_distance(pred, tgt)

        eps = 1e-8
        if pred_var <= eps:
            affine_scale = torch.zeros((), device=pred.device, dtype=pred.dtype)
            affine_bias = tgt_mean
            fitted = torch.full_like(tgt, tgt_mean)
        else:
            affine_scale = cov / pred_var
            affine_bias = tgt_mean - affine_scale * pred_mean
            fitted = affine_scale * pred + affine_bias

        resid = tgt - fitted
        ss_tot = tgt_c.pow(2).sum()
        affine_r2 = (
            torch.ones((), device=pred.device)
            if ss_tot <= eps
            else 1.0 - resid.pow(2).sum() / ss_tot
        )

        return {
            f"{prefix}/pearson_corr": pearson_corr.item(),
            f"{prefix}/corr_distance": corr_distance.item(),
            f"{prefix}/mae": diff.abs().mean().item(),
            f"{prefix}/rmse": diff.pow(2).mean().sqrt().item(),
            f"{prefix}/affine_scale": affine_scale.item(),
            f"{prefix}/affine_bias": affine_bias.item(),
            f"{prefix}/affine_r2": affine_r2.item(),
            f"{prefix}/target_std": tgt_var.sqrt().item(),
            f"{prefix}/pred_std": pred_var.sqrt().item(),
        }

    def _param_grad_norm(self, params: Iterable[torch.nn.Parameter]) -> Tensor:
        total_sq = sum(
            p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None
        )
        return (
            total_sq.sqrt()
            if isinstance(total_sq, Tensor)
            else torch.zeros((), device=self.device)
        )

    def _refresh_grad_clip_params(self) -> None:
        self._grad_clip_params = [  # type: ignore[attr-defined]
            p for group in self.optim.param_groups for p in group["params"]
        ]

    # -------------------------------------------------------------------------
    # Metric routing
    # -------------------------------------------------------------------------

    @property
    def _required_loss_metrics(self) -> list[str]:
        return [
            *super()._required_loss_metrics,
            "loss_reward_diff",
            "loss_reward_l2",
            "loss_reward_grad_penalty",
            "loss_reward_grad_penalty_batch",
            "loss_reward_grad_penalty_expert",
        ]

    @property
    def _optional_loss_metrics(self) -> list[str]:
        return [
            *super()._optional_loss_metrics,
            "loss_diversity",
            "mi_critic_loss",
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
        ]

    def _select_reported_loss_metrics(self, loss: TensorDict) -> TensorDict:
        all_keys = [*self._required_loss_metrics, *self._optional_loss_metrics]
        return loss.select(*[k for k in all_keys if k in loss])

    # -------------------------------------------------------------------------
    # Per-rollout reward preparation
    # -------------------------------------------------------------------------

    def _prepare_rollout_rewards(self, rollout: TensorDict) -> dict[str, float]:
        """Compute estimated rewards, mix with env rewards, and return diagnostics."""
        reward_key = ("next", "reward")
        with torch.no_grad():
            env_reward = rollout[reward_key]
            est_reward = (
                self._reward_from_batch(rollout, batch_role="rollout")  # type: ignore[arg-type]
                .detach()
                .clamp(min=self._est_reward_clamp_min, max=self._est_reward_clamp_max)
            )
            # MI reward flows through the separate mi_critic path (pre_iteration_compute),
            # matching ASE. It is NOT mixed into the direct reward here.
            mixed_reward = (
                self._env_reward_weight * env_reward
                + self._est_reward_weight * est_reward
            )

        rollout[("next", "env_reward")] = env_reward
        rollout[("next", "est_reward")] = est_reward
        rollout[reward_key] = mixed_reward

        return {
            "train/env_reward_mean": env_reward.mean().item(),
            "train/est_reward_mean": est_reward.mean().item(),
            **self._reward_alignment_metrics(
                "reward/env_vs_est", env_reward, est_reward
            ),
        }

    # -------------------------------------------------------------------------
    # MI critic
    # -------------------------------------------------------------------------

    def _mi_critic_input(self, td: TensorDict, *, detach: bool) -> Tensor:
        """Concatenate reward-obs features + latent as MI critic input."""
        obs = self._obs_features_from_td(
            td, self._reward_obs_keys, next_obs=False, detach=detach
        ).to(self.device)
        latents = self._rollout_latents_from_td(td, detach=detach)
        return torch.cat([obs, latents], dim=-1)

    def _update_mi_critic_batch(self, batch: TensorDict) -> dict[str, Tensor]:
        """One gradient step on the MI critic (MSE on MI returns, matching ASE)."""
        if (
            self.mi_critic is None
            or self.mi_critic_optim is None
            or "mi_returns" not in batch
        ):
            return {}

        mi_input = self._mi_critic_input(batch, detach=True)
        pred = self.mi_critic(mi_input).squeeze(-1)
        target = batch["mi_returns"].to(self.device)
        loss = F.mse_loss(pred, target)

        self.mi_critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        if self._mi_critic_grad_clip_norm > 0.0:
            clip_grad_norm_(self.mi_critic.parameters(), self._mi_critic_grad_clip_norm)
        self.mi_critic_optim.step()
        return {"mi_critic_loss": loss.detach()}

    def _get_expert_batch(self, batch: TensorDict) -> tuple[TensorDict, Tensor]:
        del batch
        return self._next_expert_batch(), torch.ones(
            (), device=self.device, dtype=torch.float32
        )

    def _attach_mi_targets(self, rollout: TensorDict) -> None:
        """Compute MI reward + value + GAE advantage for the separate MI baseline (like ASE)."""
        if self.mi_critic is None:
            return

        flat = rollout.reshape(-1)
        next_td = flat["next"]

        with torch.no_grad():
            # MI value at current step
            mi_value = (
                self.mi_critic(self._mi_critic_input(flat, detach=True))
                .squeeze(-1)
                .reshape(*rollout.batch_size)
            )

            # MI value at next step — use stamped next latent if available
            next_latent = self._latent_condition_from_td(next_td, detach=True)
            if next_latent is None:
                next_latent = self._rollout_latents_from_td(flat, detach=True)
            next_obs = self._obs_features_from_td(
                next_td, self._reward_obs_keys, next_obs=False, detach=True
            ).to(self.device)
            next_mi_input = torch.cat([next_obs, next_latent.to(self.device)], dim=-1)
            next_mi_value = (
                self.mi_critic(next_mi_input).squeeze(-1).reshape(*rollout.batch_size)
            )

            # MI intrinsic reward: cosine alignment
            mi_obs = self._latent_encoder_features_from_td(
                flat, detach=True, context="rollout latent batch"
            ).to(self.device)
            latents = self._rollout_latents_from_td(flat, detach=True)
            mi_reward = self._mi_reward(mi_obs, latents).reshape(*rollout.batch_size)

        done = rollout["next", "done"]
        if done.ndim == mi_reward.ndim + 1 and done.shape[-1] == 1:
            done = done.squeeze(-1)

        mi_advantages, mi_returns = generalized_advantage_estimate(
            mi_reward,
            mi_value,
            next_mi_value,
            done,
            gamma=float(self.config.loss.gamma),
            gae_lambda=float(self.config.ppo.gae_lambda),
        )
        rollout.set("mi_reward", mi_reward)
        rollout.set("mi_value", mi_value)
        rollout.set("mi_advantage", mi_advantages)
        rollout.set("mi_returns", mi_returns)

    def pre_iteration_compute(self, rollout: TensorDict) -> TensorDict:
        """GAE + MI advantage computation; populate data buffer for minibatch sampling."""
        with torch.no_grad():
            rollout = self.adv_module(rollout)
            self._attach_mi_targets(rollout)

            # Augment main advantage with MI advantage (matching ASE / ProtoMotions)
            mi_w = self._mi_reward_weight
            if mi_w > 0.0 and "mi_advantage" in rollout.keys(True):
                rollout["advantage"] = (
                    rollout["advantage"] + rollout["mi_advantage"] * mi_w
                )

            if getattr(self.config.compile, "compile_mode", None):
                rollout = rollout.clone()

        self.data_buffer.extend(rollout.reshape(-1))
        return rollout

    # -------------------------------------------------------------------------
    # Core update
    # -------------------------------------------------------------------------

    def update(  # type: ignore[override]
        self,
        batch: TensorDict,
        num_network_updates: int,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> tuple[TensorDict, int]:
        """PPO + BC + IPMD reward loss update.

        Three backward passes:
          1. PPO (actor-critic + optional diversity bonus)
          2. BC on expert actions  (if bc_coeff > 0)
          3. IPMD reward loss: E[r_π] − E[r_expert] + regularisation
        """
        self.optim.zero_grad(set_to_none=True)

        # --- 1. PPO loss ---
        ppo_loss: TensorDict = self.loss_module(batch)
        extra_actor_loss, extra_actor_metrics = self._extra_actor_loss(batch)
        (
            ppo_loss["loss_critic"]
            + ppo_loss["loss_objective"]
            + ppo_loss["loss_entropy"]
            + extra_actor_loss
        ).backward()

        output = ppo_loss.clone().detach_()
        for k, v in extra_actor_metrics.items():
            output[k] = v

        # --- 2. Behavior cloning on expert actions ---
        if self._bc_coeff > 0.0:
            expert_action = self._expert_action_from_td(expert_batch)
            if expert_action is None:
                msg = "BC loss requires expert actions, but none found in expert batch."
                raise RuntimeError(msg)

            # Stamp synthesized latent into expert obs if not already present
            expert_obs_td = expert_batch.clone(False)
            if self._latent_key not in expert_obs_td.keys(True):
                expert_latents = self._expert_latents_from_td(expert_batch, detach=True)
                expert_obs_td[self._latent_key] = expert_latents.reshape(
                    *expert_batch.batch_size, self._latent_dim
                )  # type: ignore[index]
            expert_obs_td = expert_obs_td.select(*self._policy_obs_keys)

            dist = self._policy_operator.get_dist(expert_obs_td)
            log_prob = self._reduce_log_prob(
                dist.log_prob(expert_action), expert_action
            )
            bc_nll = -log_prob.mean() * has_expert.to(dtype=log_prob.dtype)
            bc_loss = bc_nll * self._bc_coeff
            bc_loss.backward()

            # Diagnostic metrics
            policy_action = self._policy_action_from_dist(dist)
            log_prob_f = log_prob.detach().float()
            expert_action_f = expert_action.detach().float()

            output["loss_bc"] = bc_loss.detach()
            output["bc_nll"] = bc_nll.detach()
            output["bc_has_expert"] = has_expert.detach().float().mean()
            output["bc_log_prob_mean"] = log_prob_f.mean()
            output["bc_log_prob_nan_frac"] = log_prob_f.isnan().float().mean()
            output["bc_expert_action_abs_mean"] = expert_action_f.abs().mean()
            output["bc_expert_action_zero_frac"] = expert_action_f.eq(0).float().mean()
            output["bc_expert_action_nan_frac"] = expert_action_f.isnan().float().mean()
            if policy_action is not None:
                action_delta = (policy_action.detach() - expert_action.detach()).float()
                output["bc_policy_action_abs_mean"] = (
                    policy_action.detach().float().abs().mean()
                )
                output["bc_policy_action_mae"] = action_delta.abs().mean()
                output["bc_policy_action_rmse"] = action_delta.pow(2).mean().sqrt()
            output["bc_actor_grad_norm"] = self._param_grad_norm(
                self._policy_operator.parameters()
            )
            scale = getattr(dist, "scale", None)
            if isinstance(scale, Tensor):
                output["bc_policy_scale_mean"] = scale.detach().float().mean()

        # --- 3. IPMD reward loss: encourage r_π < r_expert ---
        if self._reward_grad_penalty_coeff > 0.0:
            r_pi_tuple: tuple[Tensor, Tensor] = self._reward_from_batch(
                batch, batch_role="rollout", requires_grad=True, return_input=True
            )  # type: ignore[assignment]
            r_exp_tuple: tuple[Tensor, Tensor] = self._reward_from_batch(
                expert_batch, batch_role="expert", requires_grad=True, return_input=True
            )  # type: ignore[assignment]
            r_pi, r_pi_x = r_pi_tuple
            r_exp, r_exp_x = r_exp_tuple
            gp_batch = self._reward_grad_penalty_from_input(r_pi, r_pi_x)
            gp_expert = self._reward_grad_penalty_from_input(r_exp, r_exp_x)
        else:
            r_pi: Tensor = self._reward_from_batch(batch, batch_role="rollout")  # type: ignore[assignment]
            r_exp: Tensor = self._reward_from_batch(expert_batch, batch_role="expert")  # type: ignore[assignment]
            gp_batch = gp_expert = torch.zeros((), device=self.device)

        diff = r_pi.mean() - r_exp.mean()
        l2 = r_pi.pow(2).mean() + r_exp.pow(2).mean()
        reward_gp = gp_batch + gp_expert
        (
            self._reward_loss_coeff * diff
            + self._reward_l2_coeff * l2.pow(0.5)
            + self._reward_grad_penalty_coeff * reward_gp
        ).backward()

        # --- Gradient clip + optimiser step ---
        grad_norm = clip_grad_norm_(self._grad_clip_params, self._max_grad_norm)
        self.optim.step()

        output["alpha"] = torch.ones((), device=self.device)
        output["grad_norm"] = grad_norm.detach()
        output["loss_reward_diff"] = diff.detach()
        output["loss_reward_l2"] = l2.detach()
        output["loss_reward_grad_penalty"] = reward_gp.detach()
        output["loss_reward_grad_penalty_batch"] = gp_batch.detach()
        output["loss_reward_grad_penalty_expert"] = gp_expert.detach()
        return output, num_network_updates + 1

    # -------------------------------------------------------------------------
    # Training loop hooks
    # -------------------------------------------------------------------------

    def prepare(  # type: ignore[override]
        self, iteration: PPOIterationData, metadata: PPOTrainingMetadata
    ) -> None:
        """Reward-model diagnostics and MI encoder update before the learning phase."""
        del metadata
        self._prepare_latent_rollout_batch_for_training(iteration.rollout)
        required = self._rollout_required_keys()
        self._log_batch_contract_once(
            flag_attr="_rollout_batch_contract_logged",
            context="rollout",
            batch=iteration.rollout,
            required_keys=required,
        )
        self._require_batch_keys(iteration.rollout, required, context="rollout batch")
        iteration.metrics.update(
            {
                f"train/{k}": v
                for k, v in self._update_mi_encoder(
                    iteration.rollout.reshape(-1)
                ).items()
            }
        )
        iteration.metrics.update(self._prepare_rollout_rewards(iteration.rollout))

    def iterate(
        self, iteration: PPOIterationData, metadata: PPOTrainingMetadata
    ) -> None:
        """PPO epochs over the prepared rollout with IPMD expert updates."""
        losses = TensorDict(
            batch_size=[metadata.epochs_per_rollout, metadata.minibatches_per_epoch]
        )
        learn_start = time.perf_counter()

        self.data_buffer.empty()
        self.actor_critic.train()
        self.adv_module.train()
        self.reward_estimator.train()
        if self.mi_critic is not None:
            self.mi_critic.train()

        with timeit("training"):
            iteration.rollout = self.pre_iteration_compute(iteration.rollout)

            if "mi_reward" in iteration.rollout.keys(True):
                iteration.metrics["train/mi_reward"] = (
                    iteration.rollout["mi_reward"].mean().item()
                )

            for epoch_idx in range(metadata.epochs_per_rollout):
                for batch_idx, batch in enumerate(self.data_buffer):
                    kl_context = None
                    if (self.config.optim.scheduler or "").lower() == "adaptive":
                        kl_context = self._prepare_kl_context(
                            batch, metadata.policy_operator
                        )

                    expert_batch, has_expert = self._get_expert_batch(batch)
                    with timeit("training/update"):
                        loss, metadata.updates_completed = self.update(
                            batch, metadata.updates_completed, expert_batch, has_expert
                        )

                    # MI critic update on the same minibatch
                    for k, v in self._update_mi_critic_batch(batch).items():
                        loss[k] = v

                    if self.lr_scheduler and self.lr_scheduler_step == "update":
                        self.lr_scheduler.step()
                    if kl_context is not None:
                        kl_approx = self._compute_kl_after_update(
                            kl_context, metadata.policy_operator
                        )
                        if kl_approx is not None:
                            loss["kl_approx"] = kl_approx.detach()
                            self._maybe_adjust_lr(kl_approx, self.config.optim)

                    losses[epoch_idx, batch_idx] = self._select_reported_loss_metrics(
                        loss
                    )

                if self.lr_scheduler and self.lr_scheduler_step == "epoch":
                    self.lr_scheduler.step()

        iteration.learn_time = time.perf_counter() - learn_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for k, v in losses_mean.items():  # type: ignore[attr-defined]
            iteration.metrics[f"train/{k}"] = v.item()  # type: ignore[attr-defined]

    def _record_env_metrics(self, iteration: PPOIterationData) -> None:
        """Drain IsaacLab env log queue into iteration metrics."""
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

    def record(
        self, iteration: PPOIterationData, metadata: PPOTrainingMetadata
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

        done_mask = rollout["next", "done"]
        episode_rewards = rollout["next", "episode_reward"][done_mask]
        if len(episode_rewards) > 0:
            ep_rewards = episode_rewards.cpu().tolist()
            self.episode_lengths.extend(
                rollout["next", "step_count"][done_mask].cpu().tolist()
            )
            self.episode_rewards.extend(ep_rewards)
            iteration.metrics.update(
                {
                    "episode/length": float(np.mean(self.episode_lengths)),
                    "episode/return": float(np.mean(self.episode_rewards)),
                    "train/reward": float(np.mean(ep_rewards)),
                }
            )

        iteration.metrics.update(self._build_control_metrics(metadata))
        iteration.metrics.update(self._build_timing_metrics(iteration, metadata))
        self._record_env_metrics(iteration)
        iteration.metrics.update(timeit.todict(prefix="time"))  # type: ignore[arg-type]
        self.log_metrics(iteration.metrics, step=metadata.frames_processed)
        self.collector.update_policy_weights_()
        self._refresh_progress_display(metadata, iteration)

        if (
            self.config.save_interval > 0
            and metadata.updates_completed % self.config.save_interval == 0
        ):
            self.save_model(
                path=self.log_dir / self.config.logger.save_path,
                step=metadata.frames_processed,
            )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict(self, obs: Tensor | np.ndarray | Mapping[Any, Any]) -> Tensor:
        """Return a deterministic action for the given observation(s)."""
        policy_op = self.actor_critic.get_policy_operator()
        policy_op.eval()

        with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
            if isinstance(obs, Mapping):
                td_data: dict[BatchKey, Tensor] = {}
                batch_shape: tuple[int, ...] | None = None
                for key in self._policy_obs_keys:
                    raw = mapping_get_obs_value(obs, key)
                    value = torch.as_tensor(raw, device=self.device)
                    ndim = self._obs_feature_ndims[key]
                    # Unsqueeze unbatched inputs to add a batch dimension
                    if (ndim == 0 and value.ndim == 0) or (
                        ndim > 0 and value.ndim == ndim
                    ):
                        value = value.unsqueeze(0)
                    cur_shape = infer_batch_shape(value, ndim)
                    if batch_shape is None:
                        batch_shape = cur_shape
                    elif cur_shape != batch_shape:
                        msg = f"predict() got inconsistent batch shapes: {batch_shape} vs {cur_shape} for key '{key}'."
                        raise ValueError(msg)
                    td_data[key] = value
            else:
                # Single-tensor observation: policy must expect exactly one key
                if len(self._policy_obs_keys) != 1:
                    raise ValueError(
                        f"predict() received a single tensor but policy expects multiple keys {self._policy_obs_keys}."
                        " Pass a dict instead."
                    )
                key = self._policy_obs_keys[0]
                value = torch.as_tensor(obs, device=self.device)
                ndim = self._obs_feature_ndims[key]
                if (ndim == 0 and value.ndim == 0) or (ndim > 0 and value.ndim == ndim):
                    value = value.unsqueeze(0)
                td_data = {key: value}
                batch_shape = infer_batch_shape(value, ndim)

            self._inject_predict_latents(td_data, batch_shape or (1,))
            td = TensorDict(
                td_data, batch_size=list(batch_shape or [1]), device=self.device
            )
            return policy_op(td)["action"]
