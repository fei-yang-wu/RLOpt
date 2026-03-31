from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

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
    LatentSkillCollectorPolicy,
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
    - ``"reference_posterior"`` - encode the current per-env reference observations
      with the learned MI posterior and use that latent for collection
    """

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
        self.config = cast(IPMDRLOptConfig, config)
        self.config: IPMDRLOptConfig
        self.env = env
        self._use_latent_command = bool(self.config.ipmd.use_latent_command)
        self._latent_key = cast(ObsKey, self.config.ipmd.latent_key)
        self._latent_dim = int(self.config.ipmd.latent_dim)
        self._collector_policy_wrapper = None
        self._collector_latents = None
        self._collector_latent_steps = None
        self._env_latent_setter = None
        self._command_source = self._normalize_command_source(
            self.config.ipmd.command_source
        )
        self._current_reference_obs_getter = self._discover_env_method(
            env,
            "get_current_reference_observations",
        )
        self._reference_posterior_fallback_logged = False

        self.config = cast(IPMDRLOptConfig, self.config)
        self._validate_env_latent_mode(env)
        if self._use_latent_command:
            self._init_latent_skills(
                env,
                latent_key=self._latent_key,
                latent_dim=self._latent_dim,
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
        self._validate_configured_obs_keys()

        all_obs_keys = dedupe_keys(
            self._policy_obs_keys + self._value_obs_keys + self._reward_obs_keys
        )
        self._obs_feature_ndims: dict[ObsKey, int] = {
            key: self._obs_key_feature_ndim(key) for key in all_obs_keys
        }
        self._obs_feature_dims: dict[ObsKey, int] = {
            key: self._obs_key_feature_dim(key) for key in all_obs_keys
        }
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        action_shape = tuple(int(dim) for dim in action_spec.shape)
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

        cfg = self.config.ipmd
        rit = cfg.reward_input_type
        lit = cfg.latent_input_type
        if lit not in ("s", "s'", "sa", "ss'"):
            msg = (
                "latent_input_type must be one of 's', 's'', 'sa', or 'ss''. "
                f"Got {cfg.latent_input_type!r}."
            )
            raise ValueError(msg)
        # Boolean flags for reward input assembly (avoids Python branching in update)
        self._rit_use_s: bool = rit in ("s", "sa", "sas")
        self._rit_use_a: bool = rit in ("sa", "sas")
        self._rit_use_sn: bool = rit in ("s'", "sas")
        self._lit_use_s: bool = lit in ("s", "sa", "ss'")
        self._lit_use_a: bool = lit == "sa"
        self._lit_use_sn: bool = lit in ("s'", "ss'")
        obs_dim = sum(self._obs_feature_dims[key] for key in self._reward_obs_keys)
        self._mi_obs_dim = 0
        if self._lit_use_s:
            self._mi_obs_dim += obs_dim
        if self._lit_use_a:
            self._mi_obs_dim += self._action_feature_dim
        if self._lit_use_sn:
            self._mi_obs_dim += obs_dim
        # Scalar caches
        self._reward_loss_coeff: float = float(cfg.reward_loss_coeff)
        self._reward_l2_coeff: float = float(cfg.reward_l2_coeff)
        self._reward_grad_penalty_coeff: float = float(cfg.reward_grad_penalty_coeff)
        self._bc_coeff: float = float(cfg.bc_coef)
        self._reward_detach_features: bool = bool(cfg.reward_detach_features)
        max_grad = getattr(self.config.optim, "max_grad_norm", None)
        self._max_grad_norm: float = float(max_grad) if max_grad else 1e10
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

    @property
    def collector_policy(self):
        """Return the collector policy, optionally stamping latent commands."""
        policy_operator = self.actor_critic.get_policy_operator()
        if not self._use_latent_command:
            return policy_operator
        if self._collector_policy_wrapper is None:
            self._collector_policy_wrapper = LatentSkillCollectorPolicy(
                self, policy_operator
            )
        return self._collector_policy_wrapper

    @staticmethod
    def _normalize_command_source(command_source: str) -> str:
        normalized = str(command_source).strip().lower()
        valid_sources = {"random", "reference_posterior"}
        if normalized not in valid_sources:
            msg = (
                f"Unsupported IPMD command_source={command_source!r}. "
                f"Expected one of {sorted(valid_sources)}."
            )
            raise ValueError(msg)
        return normalized

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

    def _reference_posterior_required_obs_keys(self) -> list[ObsKey] | None:
        if not self._lit_use_s or self._lit_use_a or self._lit_use_sn:
            return None
        return list(self._reward_obs_keys)

    def _reference_posterior_latents(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        env_ids: Tensor | None = None,
    ) -> Tensor | None:
        obs_keys = self._reference_posterior_required_obs_keys()
        if (
            obs_keys is None
            or self.mi_encoder is None
            or self._current_reference_obs_getter is None
        ):
            return None

        reference_obs = self._current_reference_obs_getter(
            required_keys=obs_keys,
            env_ids=env_ids,
        )
        if reference_obs is None or reference_obs.numel() != batch_size:
            return None

        with torch.no_grad():
            obs_features = self._obs_features_from_td(
                reference_obs,
                obs_keys,
                next_obs=False,
                detach=True,
            ).to(self.device)
            latents = self.mi_encoder(obs_features)
        return latents.to(device=device, dtype=dtype)

    def _refresh_collector_latents_from_reference_posterior(self) -> None:
        if self._command_source != "reference_posterior":
            return
        if self._collector_latents is None:
            return
        latents = self._reference_posterior_latents(
            int(self._collector_latents.shape[0]),
            device=self._collector_latents.device,
            dtype=self._collector_latents.dtype,
            env_ids=None,
        )
        if latents is None:
            if self._collector_latent_steps is not None:
                self._collector_latent_steps.zero_()
            return
        self._collector_latents.copy_(latents)
        self._publish_latents_to_env(self._collector_latents)

    def _sample_collector_latents(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        env_ids: Tensor | None = None,
    ) -> Tensor:
        if self._command_source != "reference_posterior":
            return super()._sample_collector_latents(
                batch_size,
                device=device,
                dtype=dtype,
                env_ids=env_ids,
            )

        latents = self._reference_posterior_latents(
            batch_size,
            device=device,
            dtype=dtype,
            env_ids=env_ids,
        )
        if latents is not None:
            return latents

        if not self._reference_posterior_fallback_logged:
            obs_keys = self._reference_posterior_required_obs_keys()
            if obs_keys is None:
                msg = (
                    "IPMD command_source=reference_posterior currently requires "
                    "ipmd.latent_input_type='s'; falling back to random latents."
                )
            else:
                msg = (
                    "IPMD command_source=reference_posterior fell back to random "
                    "latents."
                )
            self.log.warning(msg)
            self._reference_posterior_fallback_logged = True
        return super()._sample_collector_latents(
            batch_size,
            device=device,
            dtype=dtype,
            env_ids=env_ids,
        )

    def _validate_env_latent_mode(self, env) -> None:
        available_keys = set(env.observation_spec.keys(True))
        latent_present = self._latent_key in available_keys
        if self._use_latent_command and not latent_present:
            msg = (
                "IPMD use_latent_command=True requires the environment to expose "
                f"the latent observation key {self._latent_key!r}."
            )
            msg += self._latent_mode_hint()
            raise ValueError(msg)
        if not self._use_latent_command and latent_present:
            msg = (
                "IPMD use_latent_command=False requires the environment to omit "
                f"the latent observation key {self._latent_key!r}."
            )
            msg += self._latent_mode_hint()
            raise ValueError(msg)

    def _validate_configured_obs_keys(self) -> None:
        policy_has_latent = self._latent_key in self._policy_obs_keys
        value_has_latent = self._latent_key in self._value_obs_keys
        reward_has_latent = self._latent_key in self._reward_obs_keys

        if self._use_latent_command and not policy_has_latent:
            msg = (
                "IPMD use_latent_command=True requires the policy input keys to "
                f"contain {self._latent_key!r}."
            )
            msg += self._latent_mode_hint()
            raise ValueError(msg)
        if not self._use_latent_command and (
            policy_has_latent or value_has_latent or reward_has_latent
        ):
            msg = (
                "IPMD use_latent_command=False requires policy/value/reward input "
                "keys to exclude the latent command."
            )
            msg += self._latent_mode_hint()
            raise ValueError(msg)

    _REWARD_INPUT_TYPES = frozenset({"s", "s'", "sa", "sas"})

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
        """Return unbatched feature shape for an observation key."""
        shape = tuple(int(dim) for dim in self.env.observation_spec[key].shape)
        batch_prefix = tuple(int(dim) for dim in getattr(self.env, "batch_size", ()))
        if (
            len(batch_prefix) > 0
            and len(shape) >= len(batch_prefix)
            and shape[: len(batch_prefix)] == batch_prefix
        ):
            return shape[len(batch_prefix) :]
        return shape

    def _obs_key_feature_ndim(self, key: ObsKey) -> int:
        return len(self._obs_key_feature_shape(key))

    def _obs_key_feature_dim(self, key: ObsKey) -> int:
        shape = self._obs_key_feature_shape(key)
        return int(math.prod(shape)) if shape else 1

    def _obs_features_from_td(
        self,
        td: TensorDict | Any,
        keys: list[ObsKey],
        *,
        next_obs: bool,
        detach: bool,
    ) -> Tensor:
        parts: list[Tensor] = []
        for key in keys:
            obs = flatten_feature_tensor(
                td.get(next_obs_key(key) if next_obs else key),
                self._obs_feature_ndims[key],
            )
            parts.append(obs.detach() if detach else obs)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _action_features_from_td(self, td: TensorDict | Any, *, detach: bool) -> Tensor:
        action = flatten_feature_tensor(td.get("action"), self._action_feature_ndim)
        return action.detach() if detach else action

    def _latent_encoder_required_keys(self) -> list[BatchKey]:
        if not self._use_latent_command:
            return []
        required: list[BatchKey] = []
        if self._lit_use_s:
            required.extend(self._reward_obs_keys)
        if self._lit_use_a:
            required.append("action")
        if self._lit_use_sn:
            required.extend(next_obs_key(key) for key in self._reward_obs_keys)
        return dedupe_keys(required)

    def _rollout_required_keys(self) -> list[BatchKey]:
        required: list[BatchKey] = []
        if self._use_latent_command:
            required.append(cast(BatchKey, self._latent_key))
        if self._rit_use_s:
            required.extend(self._reward_obs_keys)
        if self._rit_use_a:
            required.append("action")
        if self._rit_use_sn:
            required.extend(next_obs_key(key) for key in self._reward_obs_keys)
        required.extend(self._latent_encoder_required_keys())
        return dedupe_keys(required)

    def _require_batch_keys(
        self,
        td: TensorDict | Any,
        required_keys: list[BatchKey],
        *,
        context: str,
    ) -> None:
        available_keys = set(td.keys(True))
        missing = [key for key in required_keys if key not in available_keys]
        if len(missing) == 0:
            return
        msg = (
            f"{context} is missing required keys: {missing}. "
            f"Available keys: {list(td.keys(True))}."
        )
        raise KeyError(msg)

    def _latent_encoder_features_from_td(
        self,
        td: TensorDict | Any,
        *,
        detach: bool,
        context: str,
    ) -> Tensor:
        if not self._use_latent_command:
            msg = "Latent encoder features requested while use_latent_command=False."
            raise RuntimeError(msg)
        required_keys = self._latent_encoder_required_keys()
        self._require_batch_keys(td, required_keys, context=context)
        parts: list[Tensor] = []
        if self._lit_use_s:
            parts.append(
                self._obs_features_from_td(
                    td,
                    self._reward_obs_keys,
                    next_obs=False,
                    detach=detach,
                )
            )
        if self._lit_use_a:
            parts.append(self._action_features_from_td(td, detach=detach))
        if self._lit_use_sn:
            parts.append(
                self._obs_features_from_td(
                    td,
                    self._reward_obs_keys,
                    next_obs=True,
                    detach=detach,
                )
            )
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _rollout_latents_from_td(
        self,
        td: TensorDict | Any,
        *,
        detach: bool,
    ) -> Tensor:
        if not self._use_latent_command:
            msg = "Rollout latent commands requested while use_latent_command=False."
            raise RuntimeError(msg)
        latent = self._latent_condition_from_td(cast(TensorDict, td), detach=detach)
        if latent is None:
            msg = "Rollout batch is missing stamped latent commands."
            raise RuntimeError(msg)
        return latent.to(self.device)

    def _expert_latents_from_td(
        self,
        td: TensorDict | Any,
        *,
        detach: bool,
    ) -> Tensor:
        if not self._use_latent_command:
            msg = "Expert latent commands requested while use_latent_command=False."
            raise RuntimeError(msg)
        latent = self._latent_condition_from_td(cast(TensorDict, td), detach=detach)
        if latent is not None:
            return latent.to(self.device)
        if self.mi_encoder is None:
            msg = "MI encoder must exist before synthesizing expert latents."
            raise RuntimeError(msg)
        obs_features = self._latent_encoder_features_from_td(
            td,
            detach=False,
            context="expert latent batch",
        )
        latent = self.mi_encoder(obs_features.to(self.device))
        return latent.detach() if detach else latent

    def _expert_required_keys(self) -> list[BatchKey]:
        """Return expert-batch keys required by current IPMD settings."""
        bc_enabled = float(self.config.ipmd.bc_coef) > 0.0

        required: list[BatchKey] = []
        if self._rit_use_s:
            required.extend(self._reward_obs_keys)
        if self._rit_use_sn:
            required.extend(next_obs_key(key) for key in self._reward_obs_keys)
        required.extend(self._latent_encoder_required_keys())
        if self._rit_use_a or bc_enabled:
            required.append("action")
        if bc_enabled:
            required.extend(
                key
                for key in self._policy_obs_keys
                if not self._use_latent_command or key != self._latent_key
            )
        return dedupe_keys(required)

    def _construct_reward_estimator(self) -> torch.nn.Module:
        """Create reward network whose input depends on ``reward_input_type``."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        rit = cfg.ipmd.reward_input_type
        if rit not in self._REWARD_INPUT_TYPES:
            msg = f"Unknown reward_input_type {rit!r}; expected one of {sorted(self._REWARD_INPUT_TYPES)}"
            raise ValueError(msg)
        out_act = cfg.ipmd.reward_output_activation
        if out_act not in self._REWARD_OUTPUT_ACTIVATIONS:
            msg = f"Unknown reward_output_activation {out_act!r}; expected one of {sorted(self._REWARD_OUTPUT_ACTIVATIONS)}"
            raise ValueError(msg)

        obs_dim = sum(self._obs_feature_dims[key] for key in self._reward_obs_keys)
        act_dim = self._action_feature_dim

        if rit in ("s", "s'"):
            in_dim = obs_dim
        elif rit == "sa":
            in_dim = obs_dim + act_dim
        else:  # "sas"
            in_dim = obs_dim * 2 + act_dim
        if self._use_latent_command:
            in_dim += self._latent_dim

        net = MLP(
            in_features=in_dim,
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
        """Create optimizers for PPO, the reward estimator, and the MI encoder."""
        if not hasattr(self, "reward_estimator"):
            return super()._set_optimizers(optimizer_cls, optimizer_kwargs)
        if not self._use_latent_command:
            all_params = list(self.actor_critic.parameters()) + list(
                self.reward_estimator.parameters()
            )
            return [optimizer_cls(all_params, **optimizer_kwargs)]
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

        if self.mi_critic is None:
            obs_dim = sum(self._obs_feature_dims[key] for key in self._reward_obs_keys)
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

        all_params = list(self.actor_critic.parameters()) + list(
            self.reward_estimator.parameters()
        )
        self._refresh_collector_latents_from_reference_posterior()
        return [optimizer_cls(all_params, **optimizer_kwargs)]

    def _require_expert_batch_keys(
        self,
        expert_batch: TensorDict,
        required_keys: list[BatchKey],
    ) -> None:
        available_keys = expert_batch.keys(True)
        missing = [
            key
            for key in required_keys
            if key not in available_keys
            and not (key == "action" and "expert_action" in available_keys)
        ]
        if len(missing) == 0:
            return
        msg = (
            f"Expert sampler contract violated. Missing keys: {missing}. "
            f"Required keys: {required_keys}. Available keys: {list(available_keys)}."
        )
        raise KeyError(msg)

    @staticmethod
    def _expert_action_from_td(td: TensorDict | Any) -> Tensor | None:
        action = td.get("expert_action")
        if action is not None:
            return cast(Tensor, action)
        action = td.get("action")
        if action is not None:
            return cast(Tensor, action)
        return None

    def _next_expert_batch(self, batch_size: int | None = None) -> TensorDict:
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
        required_keys = self._expert_required_keys()
        try:
            expert_batch = self._expert_batch_sampler(
                effective_batch_size,
                required_keys,
            )
        except Exception as err:
            msg = "Failed to sample expert batch from sampler."
            raise RuntimeError(msg) from err
        if expert_batch is None:
            msg = "Expert sampler returned None."
            raise RuntimeError(msg)
        if expert_batch.numel() > effective_batch_size:
            expert_batch = cast(TensorDict, expert_batch[:effective_batch_size])
        expert_batch = expert_batch.to(self.device)
        self._log_batch_contract_once(
            flag_attr="_expert_batch_contract_logged",
            context="expert",
            batch=expert_batch,
            required_keys=required_keys,
        )
        self._require_expert_batch_keys(expert_batch, required_keys)
        return expert_batch

    _REWARD_OUTPUT_ACTIVATIONS = frozenset({"none", "tanh", "sigmoid"})

    def _reward_condition_from_batch(
        self,
        td: TensorDict | Any,
        *,
        detach: bool,
        batch_role: str,
    ) -> Tensor:
        if not self._use_latent_command:
            msg = "Reward condition requested while use_latent_command=False."
            raise RuntimeError(msg)
        if batch_role == "rollout":
            return self._rollout_latents_from_td(td, detach=detach)
        if batch_role == "expert":
            return self._expert_latents_from_td(td, detach=detach)
        msg = f"Unknown batch_role {batch_role!r}."
        raise ValueError(msg)

    def _reward_input_from_batch(
        self,
        td: TensorDict | Any,
        *,
        batch_role: str,
        detach: bool | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Construct the reward-estimator input tensor from a transition batch."""
        # Uses pre-cached flags (_rit_use_s, _rit_use_a, _rit_use_sn) — no Python
        # branching on config strings at call time, making this compile-friendly.
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
        if self._use_latent_command:
            parts.append(
                self._reward_condition_from_batch(
                    td,
                    detach=True,
                    batch_role=batch_role,
                )
            )
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if requires_grad:
            x = x.detach().requires_grad_(True)
        return x

    def _reward_from_input(self, reward_input: Tensor) -> Tensor:
        """Evaluate the reward estimator on an already-assembled input tensor."""
        return self._reward_out_fn(self.reward_estimator(reward_input))

    def _reward_from_batch(
        self,
        td: TensorDict | Any,
        *,
        batch_role: str,
        detach: bool | None = None,
        requires_grad: bool = False,
        return_input: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute estimated reward for a batch of transitions."""
        x = self._reward_input_from_batch(
            td,
            batch_role=batch_role,
            detach=detach,
            requires_grad=requires_grad,
        )
        reward = self._reward_from_input(x)
        if return_input:
            return reward, x
        return reward

    def _mi_features_and_latents(self, batch: TensorDict) -> tuple[Tensor, Tensor]:
        if not self._use_latent_command:
            msg = "MI features requested while use_latent_command=False."
            raise RuntimeError(msg)
        obs_features = self._latent_encoder_features_from_td(
            batch,
            detach=False,
            context="rollout latent batch",
        ).to(self.device)
        latents = self._rollout_latents_from_td(batch, detach=True)
        return obs_features, latents.to(self.device)

    def _mi_grad_penalty(self, obs_features: Tensor, latents: Tensor) -> Tensor:
        if (
            self.mi_encoder is None
            or float(self.config.ipmd.mi_grad_penalty_coeff) <= 0.0
        ):
            return torch.zeros((), device=self.device)

        obs_req = obs_features.detach().requires_grad_(True)
        latent_pred = self.mi_encoder(obs_req)
        score = (latent_pred * latents.detach()).sum(dim=-1)
        grads = torch.autograd.grad(
            outputs=score.sum(),
            inputs=obs_req,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grads.pow(2).sum(dim=-1).mean()

    def _latent_uniformity(self, latent_pred: Tensor) -> Tensor:
        if latent_pred.shape[0] <= 1:
            return torch.zeros((), device=latent_pred.device, dtype=latent_pred.dtype)
        if latent_pred.shape[0] > 1024:
            # Full pairwise distances are quadratic in memory, so use a fixed-size
            # subset for IsaacLab-scale rollouts.
            sample_idx = torch.randperm(
                latent_pred.shape[0], device=latent_pred.device
            )[:1024]
            latent_pred = latent_pred[sample_idx]
        temperature = float(max(self.config.ipmd.latent_uniformity_temperature, 1.0e-6))
        # Use cdist (includes self-pairs), matching ASE / ProtoMotions compute_uniformity_loss.
        pairwise_dist = torch.cdist(latent_pred, latent_pred, p=2)
        kernel_values = torch.exp(-temperature * pairwise_dist.pow(2))
        return torch.log(kernel_values.mean())

    def _update_mi_encoder(self, rollout_flat: TensorDict) -> dict[str, float]:
        if not self._use_latent_command:
            return {}
        if self.mi_encoder is None or self.mi_encoder_optim is None:
            return {}
        if float(self.config.ipmd.mi_loss_coeff) <= 0.0:
            return {}

        obs_features, latents = self._mi_features_and_latents(rollout_flat)
        if obs_features.shape[0] == 0:
            return {}

        latent_pred = self.mi_encoder(obs_features)
        similarity = (latent_pred * latents).sum(dim=-1)
        mi_loss = -float(self.config.ipmd.latent_vmf_kappa) * similarity.mean()
        grad_penalty = self._mi_grad_penalty(obs_features, latents)

        weight_decay = torch.zeros((), device=self.device)
        if float(self.config.ipmd.mi_weight_decay_coeff) > 0.0:
            for param in self.mi_encoder.parameters():
                if param.ndim >= 2:
                    weight_decay = weight_decay + param.pow(2).mean()

        expert_td = self._next_expert_batch()
        expert_obs = self._latent_encoder_features_from_td(
            expert_td,
            detach=False,
            context="expert latent batch",
        ).to(self.device)
        uniformity_input = torch.cat(
            [latent_pred, self.mi_encoder(expert_obs)],
            dim=0,
        )
        uniformity = self._latent_uniformity(uniformity_input)

        total_loss = (
            float(self.config.ipmd.mi_loss_coeff) * mi_loss
            + float(self.config.ipmd.mi_grad_penalty_coeff) * grad_penalty
            + float(self.config.ipmd.mi_weight_decay_coeff) * weight_decay
            + float(self.config.ipmd.latent_uniformity_coeff) * uniformity
        )

        self.mi_encoder_optim.zero_grad(set_to_none=True)
        total_loss.backward()
        if float(self.config.ipmd.mi_grad_clip_norm) > 0.0:
            clip_grad_norm_(
                self.mi_encoder.parameters(), float(self.config.ipmd.mi_grad_clip_norm)
            )
        self.mi_encoder_optim.step()

        return {
            "ipmd/mi_total_loss": float(total_loss.detach().item()),
            "ipmd/mi_loss": float(mi_loss.detach().item()),
            "ipmd/mi_similarity_mean": float(similarity.detach().mean().item()),
            "ipmd/mi_grad_penalty": float(grad_penalty.detach().item()),
            "ipmd/mi_weight_decay": float(weight_decay.detach().item()),
            "ipmd/latent_uniformity": float(uniformity.detach().item()),
        }

    def _mi_reward(self, obs_features: Tensor, latents: Tensor) -> Tensor:
        if not self._use_latent_command:
            return torch.zeros(obs_features.shape[0], device=obs_features.device)
        if self.mi_encoder is None:
            return torch.zeros(obs_features.shape[0], device=obs_features.device)
        with torch.no_grad():
            latent_pred = self.mi_encoder(obs_features)
            score = (latent_pred * latents).sum(dim=-1)
        if self.config.ipmd.mi_hypersphere_reward_shift:
            # ASE / ProtoMotions-style: shift dot product from [-1, 1] to [0, 1].
            reward = (score + 1.0) / 2.0
        else:
            reward = score.clamp_min(0.0)
        # Weight applied at advantage level in pre_iteration_compute,
        # matching ASE which scales mi_advantages by mi_reward_weight there.
        return reward

    def _diversity_loss(self, batch: TensorDict) -> Tensor:
        """ProtoMotions-style diversity objective added directly to the actor loss."""
        if not self._use_latent_command:
            return torch.zeros((), device=self.device)
        if float(self.config.ipmd.diversity_bonus_coeff) <= 0.0 or batch.numel() <= 1:
            return torch.zeros((), device=self.device)
        self._require_batch_keys(
            batch,
            [cast(BatchKey, self._latent_key), "loc", "scale", *self._policy_obs_keys],
            context="ipmd diversity minibatch",
        )

        old_latents = self._rollout_latents_from_td(batch, detach=True)
        old_dist = self._policy_operator.build_dist_from_params(
            batch.select("loc", "scale").clone()
        )
        old_mean_action = self._policy_action_from_dist(old_dist)
        if old_mean_action is None:
            msg = "IPMD diversity loss could not recover the old policy mean action."
            raise RuntimeError(msg)
        old_mean_action = self._clip_policy_action(old_mean_action.detach())

        new_latents = self._sample_unit_latents(
            batch.numel(),
            device=self.device,
            dtype=old_latents.dtype,
        )
        policy_td = batch.select(*self._policy_obs_keys).clone()
        policy_td.set(
            cast(BatchKey, self._latent_key),
            new_latents.reshape(*batch.batch_size, self._latent_dim),
        )
        new_dist = self._policy_operator.get_dist(policy_td)
        new_mean_action = self._policy_action_from_dist(new_dist)
        if new_mean_action is None:
            msg = "IPMD diversity loss could not recover the new policy mean action."
            raise RuntimeError(msg)
        new_mean_action = self._clip_policy_action(new_mean_action)

        action_delta = (new_mean_action - old_mean_action).pow(2).mean(dim=-1)
        latent_delta = 0.5 - 0.5 * (new_latents * old_latents).sum(dim=-1)
        diversity_bonus = action_delta / (latent_delta + 1.0e-5)
        return (
            (float(self.config.ipmd.diversity_target) - diversity_bonus).pow(2).mean()
        )

    def _extra_actor_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        if float(self.config.ipmd.diversity_bonus_coeff) <= 0.0:
            return torch.zeros((), device=self.device), {}
        diversity_loss = self._diversity_loss(batch)
        weighted_loss = diversity_loss * float(self.config.ipmd.diversity_bonus_coeff)
        return weighted_loss, {"loss_diversity": diversity_loss.detach()}

    @staticmethod
    def _reward_grad_penalty_from_input(reward: Tensor, reward_input: Tensor) -> Tensor:
        """Squared gradient norm of reward with respect to its input features."""
        reward_grad = torch.autograd.grad(
            outputs=reward.sum(),
            inputs=reward_input,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return reward_grad.pow(2).sum(dim=-1).mean()

    @staticmethod
    def _reward_tensor_stats(prefix: str, reward: Tensor) -> dict[str, float]:
        reward_f = reward.detach().float()
        return {
            f"{prefix}_mean": reward_f.mean().item(),
            f"{prefix}_std": reward_f.std().item(),
            f"{prefix}_min": reward_f.min().item(),
            f"{prefix}_max": reward_f.max().item(),
        }

    @staticmethod
    def _reward_alignment_metrics(
        prefix: str, reward_pred: Tensor, reward_true: Tensor
    ) -> dict[str, float]:
        pred = reward_pred.detach().float().flatten()
        true = reward_true.detach().float().flatten()
        diff = pred - true

        pred_mean = pred.mean()
        true_mean = true.mean()
        pred_centered = pred - pred_mean
        true_centered = true - true_mean

        pred_var = pred_centered.pow(2).mean()
        true_var = true_centered.pow(2).mean()
        cov = (pred_centered * true_centered).mean()

        pearson_corr, corr_distance = epic_distance(pred, true)

        eps = 1e-8
        if pred_var <= eps:
            affine_scale = torch.zeros((), device=pred.device, dtype=pred.dtype)
            affine_bias = true_mean
            fitted = torch.full_like(true, true_mean)
        else:
            affine_scale = cov / pred_var
            affine_bias = true_mean - affine_scale * pred_mean
            fitted = affine_scale * pred + affine_bias

        resid = true - fitted
        ss_res = resid.pow(2).sum()
        ss_tot = true_centered.pow(2).sum()
        if ss_tot <= eps:
            affine_r2 = torch.ones((), device=pred.device, dtype=pred.dtype)
        else:
            affine_r2 = 1.0 - ss_res / ss_tot

        return {
            f"{prefix}/pearson_corr": pearson_corr.item(),
            f"{prefix}/corr_distance": corr_distance.item(),
            f"{prefix}/mae": diff.abs().mean().item(),
            f"{prefix}/rmse": diff.pow(2).mean().sqrt().item(),
            f"{prefix}/affine_scale": affine_scale.item(),
            f"{prefix}/affine_bias": affine_bias.item(),
            f"{prefix}/affine_r2": affine_r2.item(),
            f"{prefix}/target_std": true_var.sqrt().item(),
            f"{prefix}/pred_std": pred_var.sqrt().item(),
        }

    def _param_grad_norm(self, params: Any) -> Tensor:
        total_sq: Tensor | None = None
        for param in params:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            grad_tensor = grad.detach().float()
            grad_sq = grad_tensor.pow(2).sum()
            total_sq = grad_sq if total_sq is None else total_sq + grad_sq
        if total_sq is None:
            return torch.zeros((), device=self.device)
        return total_sq.sqrt()

    def _refresh_grad_clip_params(self) -> None:
        """Refresh the cached optimizer parameter list used for grad clipping."""
        self._grad_clip_params = [  # type: ignore[attr-defined]
            param for group in self.optim.param_groups for param in group["params"]
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
            env_reward = rollout.get(reward_key)
            assert env_reward is not None
            est_reward = (
                self._reward_from_batch(
                    rollout,
                    batch_role="rollout",
                )
                .detach()  # type: ignore[attr-defined]
                .clamp(
                    min=self.config.ipmd.estimated_reward_clamp_min,
                    max=self.config.ipmd.estimated_reward_clamp_max,
                )
            )  # type: ignore[attr-defined]
            # MI reward flows through the separate mi_critic path in
            # pre_iteration_compute (mi_advantages added to main advantages),
            # matching ASE. It is NOT mixed into the direct reward here.
            mixed_reward = (
                self.config.ipmd.env_reward_weight * env_reward
                + self.config.ipmd.est_reward_weight * est_reward
            )

        rollout.set(("next", "env_reward"), env_reward)
        rollout.set(("next", "est_reward"), est_reward)
        rollout.set(reward_key, mixed_reward)

        metrics.update(
            {
                "train/env_reward_mean": env_reward.mean().item(),
                "train/est_reward_mean": est_reward.mean().item(),
            }
        )
        metrics.update(
            self._reward_alignment_metrics("reward/env_vs_est", env_reward, est_reward)
        )

        return metrics

    def _update_mi_critic_batch(self, batch: TensorDict) -> dict[str, Tensor]:
        """Train the MI critic on one minibatch of MI returns (like ASE)."""
        if not self._use_latent_command:
            return {}
        if (
            self.mi_critic is None
            or self.mi_critic_optim is None
            or "mi_returns" not in batch
        ):
            return {}

        mi_input = self._mi_critic_input(batch, detach=True)
        pred = self.mi_critic(mi_input).squeeze(-1)
        target = cast(Tensor, batch.get("mi_returns")).to(self.device)
        loss = F.mse_loss(pred, target)

        self.mi_critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        if float(self.config.ipmd.mi_critic_grad_clip_norm) > 0.0:
            clip_grad_norm_(
                self.mi_critic.parameters(),
                float(self.config.ipmd.mi_critic_grad_clip_norm),
            )
        self.mi_critic_optim.step()
        return {"mi_critic_loss": loss.detach()}

    def _expert_batch_for_update(self, batch: TensorDict) -> tuple[TensorDict, Tensor]:
        """Return an expert batch aligned with one PPO minibatch update."""
        del batch
        expert_batch = self._next_expert_batch()
        return (
            expert_batch,
            torch.ones((), device=self.device, dtype=torch.float32),
        )

    def _mi_critic_input(self, td: TensorDict | Any, *, detach: bool) -> Tensor:
        """Concatenate reward-obs features + latent as MI critic input."""
        if not self._use_latent_command:
            msg = "MI critic input requested while use_latent_command=False."
            raise RuntimeError(msg)
        obs = self._obs_features_from_td(
            td, self._reward_obs_keys, next_obs=False, detach=detach
        ).to(self.device)
        latents = self._rollout_latents_from_td(td, detach=detach)
        return torch.cat([obs, latents], dim=-1)

    def _attach_mi_targets(self, rollout: TensorDict) -> None:
        """Compute MI value, advantage, and returns via a separate GAE pass (like ASE)."""
        if not self._use_latent_command:
            return
        if self.mi_critic is None:
            return

        flat_rollout = rollout.reshape(-1)
        next_td = cast(TensorDict, flat_rollout.get("next"))

        with torch.no_grad():
            mi_input = self._mi_critic_input(flat_rollout, detach=True)
            mi_value = self.mi_critic(mi_input).squeeze(-1).reshape(*rollout.batch_size)

            # Use next latent from the nested "next" tensordict (set by
            # _prepare_latent_rollout_batch_for_training).
            next_latent = self._latent_condition_from_td(next_td, detach=True)
            if next_latent is None:
                next_latent = self._rollout_latents_from_td(flat_rollout, detach=True)
            next_obs = self._obs_features_from_td(
                next_td, self._reward_obs_keys, next_obs=False, detach=True
            ).to(self.device)
            next_input = torch.cat([next_obs, next_latent.to(self.device)], dim=-1)
            next_mi_value = (
                self.mi_critic(next_input).squeeze(-1).reshape(*rollout.batch_size)
            )

            mi_obs = self._latent_encoder_features_from_td(
                flat_rollout, detach=True, context="rollout latent batch"
            ).to(self.device)
            latents = self._rollout_latents_from_td(flat_rollout, detach=True)
            mi_reward = self._mi_reward(mi_obs, latents).reshape(*rollout.batch_size)

        done = cast(Tensor, rollout["next", "done"])
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
        with torch.no_grad():
            rollout = self.adv_module(rollout)
            self._attach_mi_targets(rollout)

            # Add MI advantages to main task advantages, matching ASE / ProtoMotions:
            #   advantages += mi_advantages * mi_reward_weight
            mi_reward_w = float(self.config.ipmd.mi_reward_weight)
            if mi_reward_w > 0.0 and "mi_advantage" in rollout.keys(True):
                adv = rollout.get("advantage")
                mi_adv = rollout.get("mi_advantage").unsqueeze(-1)
                rollout.set("advantage", adv + mi_adv * mi_reward_w)

            if getattr(self.config.compile, "compile", False):
                rollout = rollout.clone()

        self.data_buffer.extend(rollout.reshape(-1))
        return rollout

    def update(
        self,
        batch: TensorDict,
        num_network_updates: int,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> tuple[TensorDict, int]:
        """PPO update plus optional BC loss and IPMD reward loss."""
        self.optim.zero_grad(set_to_none=True)

        # 1) PPO loss
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

        # 2) Behavior cloning on expert actions.
        bc_loss = torch.zeros((), device=self.device)

        if self._bc_coeff > 0.0:
            expert_action = self._expert_action_from_td(expert_batch)
            expert_obs_td = expert_batch.clone(False)
            if cast(BatchKey, self._latent_key) not in expert_obs_td.keys(True):
                expert_latents = self._expert_latents_from_td(
                    expert_batch,
                    detach=True,
                ).reshape(*expert_batch.batch_size, self._latent_dim)
                expert_obs_td.set(cast(BatchKey, self._latent_key), expert_latents)
            expert_obs_td = expert_obs_td.select(*self._policy_obs_keys)
            dist = self._policy_operator.get_dist(expert_obs_td)
            log_prob = dist.log_prob(expert_action)
            log_prob = self._reduce_log_prob(log_prob, expert_action)
            has_expert_float = has_expert.to(dtype=log_prob.dtype)
            bc_nll = -log_prob.mean() * has_expert_float
            bc_loss = bc_nll * self._bc_coeff
            bc_loss.backward()

        # 3) IPMD reward loss
        reward_grad_penalty_batch = torch.zeros((), device=self.device)
        reward_grad_penalty_expert = torch.zeros((), device=self.device)
        if self._reward_grad_penalty_coeff > 0.0:
            r_pi, r_pi_input = cast(
                tuple[Tensor, Tensor],
                self._reward_from_batch(
                    batch,
                    batch_role="rollout",
                    requires_grad=True,
                    return_input=True,
                ),
            )
            r_exp, r_exp_input = cast(
                tuple[Tensor, Tensor],
                self._reward_from_batch(
                    expert_batch,
                    batch_role="expert",
                    requires_grad=True,
                    return_input=True,
                ),
            )
            reward_grad_penalty_batch = self._reward_grad_penalty_from_input(
                r_pi, r_pi_input
            )
            reward_grad_penalty_expert = self._reward_grad_penalty_from_input(
                r_exp, r_exp_input
            )
        else:
            r_pi = self._reward_from_batch(batch, batch_role="rollout")  # type: ignore[attr-defined]
            r_exp = self._reward_from_batch(expert_batch, batch_role="expert")  # type: ignore[attr-defined]
        diff = r_pi.mean() - r_exp.mean()
        l2 = r_pi.pow(2).mean() + r_exp.pow(2).mean()
        reward_grad_penalty = reward_grad_penalty_batch + reward_grad_penalty_expert
        (
            self._reward_loss_coeff * diff
            + self._reward_l2_coeff * l2.pow(0.5)
            + self._reward_grad_penalty_coeff * reward_grad_penalty
        ).backward()

        # Gradient clipping — always call for a fixed graph
        grad_norm_tensor = clip_grad_norm_(self._grad_clip_params, self._max_grad_norm)

        self.optim.step()

        output_loss.set("alpha", torch.ones((), device=self.device))
        output_loss.set("loss_reward_diff", diff.detach())
        output_loss.set("loss_reward_l2", l2.detach())
        output_loss.set("loss_reward_grad_penalty", reward_grad_penalty.detach())

        if self._bc_coeff > 0.0:
            output_loss.set("loss_bc", bc_loss.detach())
            output_loss.set("bc_nll", bc_nll.detach())

        output_loss.set("grad_norm", grad_norm_tensor.detach())

        return output_loss, num_network_updates + 1

    def prepare(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,  # noqa: ARG002
    ) -> None:
        """Attach reward-model diagnostics and replace the PPO reward for this rollout."""
        if self._use_latent_command:
            self._prepare_latent_rollout_batch_for_training(iteration.rollout)
        self._log_batch_contract_once(
            flag_attr="_rollout_batch_contract_logged",
            context="rollout",
            batch=iteration.rollout,
            required_keys=self._rollout_required_keys(),
        )
        self._require_batch_keys(
            iteration.rollout,
            self._rollout_required_keys(),
            context="rollout batch",
        )
        iteration.metrics.update(self._prepare_rollout_rewards(iteration.rollout))

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
        if self.mi_critic is not None:
            self.mi_critic.train()

        with timeit("training"):
            # MI encoder trains once per rollout, before pre_iteration_compute which
            # uses the encoder in _attach_mi_targets.
            mi_encoder_metrics = self._update_mi_encoder(iteration.rollout.reshape(-1))
            if mi_encoder_metrics:
                iteration.metrics.update(
                    {f"train/{key}": value for key, value in mi_encoder_metrics.items()}
                )
            self._refresh_collector_latents_from_reference_posterior()

            iteration.rollout = self.pre_iteration_compute(iteration.rollout)

            if "mi_reward" in iteration.rollout.keys(True):
                iteration.metrics["train/mi_reward_mean"] = float(
                    cast(Tensor, iteration.rollout.get("mi_reward")).mean().item()
                )

            for epoch_idx in range(metadata.epochs_per_rollout):
                for batch_idx, batch in enumerate(self.data_buffer):
                    kl_context = None
                    if (self.config.optim.scheduler or "").lower() == "adaptive":
                        kl_context = self._prepare_kl_context(
                            batch, metadata.policy_operator
                        )

                    expert_batch, has_expert = self._expert_batch_for_update(batch)
                    with timeit("training/update"):
                        loss, metadata.updates_completed = self.update(
                            batch,
                            metadata.updates_completed,
                            expert_batch,
                            has_expert,
                        )

                    mi_critic_stats = self._update_mi_critic_batch(batch)
                    for key, value in mi_critic_stats.items():
                        loss.set(key, value)

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

            if self._use_latent_command:
                self._inject_predict_latents(td_data, batch_shape or (1,))
            td = TensorDict(
                td_data, batch_size=list(batch_shape or [1]), device=self.device
            )
            td = policy_op(td)
            return td.get("action")
