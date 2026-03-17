from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import InteractionType
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.data import (
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import MLP
from torchrl.record.loggers import Logger

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
    strip_next_prefix,
)
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info


@dataclass
class IPMDConfig(PPOConfig):
    """IPMD-specific configuration (PPO-based)."""

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

    estimated_reward_clamp_min: float | None = 0.0
    """Optional lower bound applied to estimated rewards before PPO reward mixing.

    Set to ``None`` to disable lower clipping.
    """

    estimated_reward_clamp_max: float | None = 0.25
    """Optional upper bound applied to estimated rewards before PPO reward mixing.

    Set to ``None`` to disable upper clipping.
    """

    estimated_reward_mix_coeff: float = 0.3
    """Linear mixing coefficient for estimated rewards in PPO.

    Mixed reward is computed as:
    ``reward = estimated_reward_mix_coeff * est_rew + (1 - estimated_reward_mix_coeff) * env_reward``.
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


class IPMD(PPO):
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

        self.config = cast(IPMDRLOptConfig, self.config)

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

        # Expert data source
        self._expert_buffer: TensorDictReplayBuffer | None = None
        self._expert_batch_sampler: (
            Callable[[int, list[BatchKey]], TensorDict | None] | None
        ) = None
        self._warned_no_expert = False
        self._warned_missing_expert_batch = False
        self._auto_attach_env_expert_sampler()

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

        self._refresh_grad_clip_params()
        self._policy_operator = self.actor_critic.get_policy_operator()
        self._bc_debug_anomaly_prints = 0

    def _cache_ipmd_scalars(self) -> None:
        """Pre-cache config scalars and reward-input flags for compile-friendly hot paths."""
        cfg = self.config.ipmd
        rit = cfg.reward_input_type
        # Boolean flags for reward input assembly (avoids Python branching in update)
        self._rit_use_s: bool = rit in ("s", "sa", "sas")
        self._rit_use_a: bool = rit in ("sa", "sas")
        self._rit_use_sn: bool = rit in ("s'", "sas")
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

    def _expert_required_keys(self) -> list[BatchKey]:
        """Return expert-batch keys required by current IPMD settings."""
        assert isinstance(self.config, IPMDRLOptConfig)
        rit = self.config.ipmd.reward_input_type
        bc_enabled = float(self.config.ipmd.bc_coef) > 0.0

        required: list[BatchKey] = []
        if rit in ("s", "sa", "sas"):
            required.extend(self._reward_obs_keys)
        if rit in ("s'", "sas"):
            required.extend([next_obs_key(key) for key in self._reward_obs_keys])
        if rit in ("sa", "sas") or bc_enabled:
            required.append("action")
        if bc_enabled:
            required.extend(self._policy_obs_keys)
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
        """Single optimizer for actor-critic and reward estimator (PPO + IPMD).

        During the first call from BaseAlgorithm.__init__, reward_estimator
        doesn't exist yet — fall back to the PPO optimizer.  IPMD.__init__
        re-creates the optimizer after the reward network is built.
        """
        if not hasattr(self, "reward_estimator"):
            return super()._set_optimizers(optimizer_cls, optimizer_kwargs)
        # PPO uses one optimizer for actor_critic; add reward_estimator to the same group
        all_params = list(self.actor_critic.parameters()) + list(
            self.reward_estimator.parameters()
        )
        return [optimizer_cls(all_params, **optimizer_kwargs)]

    @staticmethod
    def _discover_env_method(env: object, method_name: str) -> Callable | None:
        """Discover a callable by walking common wrapper attributes."""
        stack: list[object] = [env]
        visited: set[int] = set()

        while len(stack) > 0:
            current = stack.pop()
            obj_id = id(current)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            method = getattr(current, method_name, None)
            if callable(method):
                return method

            for attr_name in ("base_env", "env", "_env", "unwrapped"):
                try:
                    next_obj = getattr(current, attr_name, None)
                except Exception:
                    continue
                if next_obj is None:
                    continue
                if isinstance(next_obj, list | tuple):
                    stack.extend(next_obj)
                else:
                    stack.append(next_obj)
        return None

    def _auto_attach_env_expert_sampler(self) -> None:
        """Auto-attach expert sampler from env if available."""
        sampler = self._discover_env_method(self.env, "sample_expert_batch")
        if sampler is None:
            return

        def _wrapped_sampler(
            batch_size: int, required_keys: list[BatchKey]
        ) -> TensorDict | None:
            return cast(
                TensorDict | None,
                sampler(batch_size=batch_size, required_keys=required_keys),
            )

        self._expert_batch_sampler = _wrapped_sampler

    def _check_expert_batch_keys(self, expert_batch: TensorDict) -> bool:
        required_keys = self._expert_required_keys()
        available_keys = expert_batch.keys(True)
        missing = [
            key
            for key in required_keys
            if key not in available_keys
            and not (key == "action" and "expert_action" in available_keys)
        ]
        if missing:
            self.log.warning("Expert batch missing required keys: %s", missing)
            return False
        return True

    @staticmethod
    def _expert_action_from_td(td: TensorDict | Any) -> Tensor | None:
        action = td.get("expert_action")
        if action is not None:
            return cast(Tensor, action)
        action = td.get("action")
        if action is not None:
            return cast(Tensor, action)
        return None

    def _next_expert_batch(self) -> TensorDict | None:
        assert isinstance(self.config, IPMDRLOptConfig)
        effective_batch_size = int(
            self.config.ipmd.expert_batch_size or self.config.loss.mini_batch_size
        )

        if self._expert_buffer is not None:
            try:
                expert_batch = cast(
                    TensorDict,
                    self._expert_buffer.sample(
                        batch_size=self.config.ipmd.expert_batch_size
                    ),
                )
                if (
                    self.config.ipmd.expert_batch_size is not None
                    and expert_batch.numel() > self.config.ipmd.expert_batch_size
                ):
                    expert_batch = cast(
                        TensorDict,
                        expert_batch[: self.config.ipmd.expert_batch_size],
                    )
                return expert_batch
            except Exception:
                return None

        if self._expert_batch_sampler is None:
            return None
        try:
            expert_batch = self._expert_batch_sampler(
                effective_batch_size, self._expert_required_keys()
            )
        except Exception as err:
            self.log.warning("Failed to sample expert batch from sampler: %s", err)
            return None
        if expert_batch is None:
            return None
        if expert_batch.numel() > effective_batch_size:
            expert_batch = cast(TensorDict, expert_batch[:effective_batch_size])
        return expert_batch

    def _dummy_expert_batch(self, batch: TensorDict) -> TensorDict:
        """Return a single-transition expert batch with same structure as batch (for compile/CUDA graph)."""
        required_keys = self._expert_required_keys()
        available_keys = batch.keys(True)
        if all(key in available_keys for key in required_keys):
            return batch.select(*required_keys).clone()[:1]

        one = batch[:1]
        dummy = TensorDict({}, batch_size=[1], device=batch.device)
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        action_shape = tuple(int(dim) for dim in action_spec.shape)
        action_dtype = cast(
            torch.dtype, getattr(self.env.action_spec, "dtype", torch.float32)
        )
        for key in required_keys:
            if key in available_keys:
                dummy.set(key, one.get(key).clone())
                continue
            if key == "action":
                dummy_action = torch.zeros(
                    (1, *action_shape),
                    device=batch.device,
                    dtype=action_dtype,
                )
                dummy.set("action", dummy_action)
                continue
            obs_key = strip_next_prefix(key)
            obs_shape = self._obs_key_feature_shape(obs_key)
            obs_dtype = cast(
                torch.dtype,
                getattr(self.env.observation_spec[obs_key], "dtype", torch.float32),
            )
            dummy_obs = torch.zeros(
                (1, *obs_shape),
                device=batch.device,
                dtype=obs_dtype,
            )
            dummy.set(key, dummy_obs)
        return dummy

    def _dummy_loss_tensordict(self) -> TensorDict:
        """Dummy loss TensorDict with same keys as update output (for compile/fallback)."""
        return TensorDict(
            {
                "loss_critic": torch.tensor(0.0, device=self.device),
                "loss_objective": torch.tensor(0.0, device=self.device),
                "loss_entropy": torch.tensor(0.0, device=self.device),
                "loss_reward_diff": torch.tensor(0.0, device=self.device),
                "loss_reward_l2": torch.tensor(0.0, device=self.device),
                "loss_reward_grad_penalty": torch.tensor(0.0, device=self.device),
                "loss_reward_grad_penalty_batch": torch.tensor(0.0, device=self.device),
                "loss_reward_grad_penalty_expert": torch.tensor(
                    0.0, device=self.device
                ),
                "loss_bc": torch.tensor(0.0, device=self.device),
                "bc_nll": torch.tensor(0.0, device=self.device),
                "bc_has_expert": torch.tensor(0.0, device=self.device),
                "bc_log_prob_mean": torch.tensor(0.0, device=self.device),
                "bc_log_prob_nan_frac": torch.tensor(0.0, device=self.device),
                "bc_expert_action_abs_mean": torch.tensor(0.0, device=self.device),
                "bc_expert_action_zero_frac": torch.tensor(0.0, device=self.device),
                "bc_expert_action_nan_frac": torch.tensor(0.0, device=self.device),
                "bc_policy_action_abs_mean": torch.tensor(0.0, device=self.device),
                "bc_policy_action_mae": torch.tensor(0.0, device=self.device),
                "bc_policy_action_rmse": torch.tensor(0.0, device=self.device),
                "bc_actor_grad_norm": torch.tensor(0.0, device=self.device),
                "bc_policy_scale_mean": torch.tensor(0.0, device=self.device),
                "estimated_reward_mean": torch.tensor(0.0, device=self.device),
                "estimated_reward_std": torch.tensor(0.0, device=self.device),
                "expert_reward_mean": torch.tensor(0.0, device=self.device),
                "expert_reward_std": torch.tensor(0.0, device=self.device),
            },
            batch_size=[],
        )

    _REWARD_OUTPUT_ACTIVATIONS = frozenset({"none", "tanh", "sigmoid"})

    def _reward_input_from_batch(
        self,
        td: TensorDict | Any,
        *,
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
        detach: bool | None = None,
        requires_grad: bool = False,
        return_input: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute estimated reward for a batch of transitions."""
        x = self._reward_input_from_batch(
            td, detach=detach, requires_grad=requires_grad
        )
        reward = self._reward_from_input(x)
        if return_input:
            return reward, x
        return reward

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

    @staticmethod
    def _policy_action_from_dist(dist: Any) -> Tensor | None:
        for attr_name in ("mean", "loc", "mode"):
            try:
                value = getattr(dist, attr_name, None)
            except Exception:
                continue
            if value is None:
                continue
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if isinstance(value, Tensor):
                return value
        return None

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
        if reward_key not in rollout.keys(True):
            return metrics

        with torch.no_grad():
            env_reward = cast(Tensor, rollout.get(reward_key))
            est_rew_raw = cast(Tensor, self._reward_from_batch(rollout)).detach()

            clamp_min = self.config.ipmd.estimated_reward_clamp_min
            clamp_max = self.config.ipmd.estimated_reward_clamp_max
            done_penalty = float(self.config.ipmd.estimated_reward_done_penalty)
            mix_coeff = (
                float(self.config.ipmd.estimated_reward_mix_coeff)
                if self.config.ipmd.use_estimated_rewards_for_ppo
                else 0.0
            )

            est_rew = est_rew_raw
            if clamp_min is not None or clamp_max is not None:
                est_rew = torch.clamp(est_rew_raw, min=clamp_min, max=clamp_max)

            terminal_penalty_mask = None
            if done_penalty != 0.0 and ("next", "done") in rollout.keys(True):
                terminal_penalty_mask = rollout["next", "done"].bool()
                if ("next", "truncated") in rollout.keys(True):
                    terminal_penalty_mask = (
                        terminal_penalty_mask & ~rollout["next", "truncated"].bool()
                    )
                est_rew = est_rew - done_penalty * terminal_penalty_mask.to(
                    dtype=est_rew.dtype
                )

            mixed_reward = mix_coeff * est_rew + (1 - mix_coeff) * env_reward

        rollout.set(("next", "env_reward"), env_reward)
        rollout.set(("next", "estimated_reward_raw"), est_rew_raw)
        rollout.set(("next", "estimated_reward_clamped"), est_rew)
        rollout.set(reward_key, mixed_reward)

        metrics.update(
            {
                "train/step_reward_mean": env_reward.mean().item(),
                "train/step_reward_std": env_reward.std().item(),
                "train/step_reward_max": env_reward.max().item(),
                "train/step_reward_min": env_reward.min().item(),
                "train/estimated_reward_mix_coeff": mix_coeff,
                "train/estimated_reward_done_penalty": done_penalty,
            }
        )
        metrics.update(self._reward_tensor_stats("train/env_reward", env_reward))
        metrics.update(
            self._reward_tensor_stats("train/estimated_reward_raw", est_rew_raw)
        )
        metrics.update(
            self._reward_tensor_stats("train/estimated_reward_clamped", est_rew)
        )
        metrics.update(self._reward_tensor_stats("train/ppo_reward", mixed_reward))
        metrics.update(
            self._reward_alignment_metrics("reward/raw_vs_env", est_rew_raw, env_reward)
        )
        metrics.update(
            self._reward_alignment_metrics(
                "reward/clamped_vs_env",
                est_rew,
                env_reward,
            )
        )
        metrics.update(
            self._reward_alignment_metrics(
                "reward/mixed_vs_env",
                mixed_reward,
                env_reward,
            )
        )

        if terminal_penalty_mask is not None:
            metrics["reward/done_penalty_frac"] = (
                terminal_penalty_mask.float().mean().item()
            )
        if clamp_min is not None:
            metrics["reward/clip_low_frac"] = (
                (est_rew_raw <= clamp_min).float().mean().item()
            )
            metrics["reward/clamp_min"] = float(clamp_min)
        if clamp_max is not None:
            metrics["reward/clip_high_frac"] = (
                (est_rew_raw >= clamp_max).float().mean().item()
            )
            metrics["reward/clamp_max"] = float(clamp_max)

        return metrics

    def _expert_batch_for_update(self, batch: TensorDict) -> tuple[TensorDict, Tensor]:
        """Return an expert batch aligned with one PPO minibatch update."""
        expert_batch_raw = self._next_expert_batch()
        if expert_batch_raw is None:
            if not self._warned_missing_expert_batch:
                self.log.warning(
                    "Expert batch unavailable during training; falling back to a dummy batch."
                )
                self._warned_missing_expert_batch = True
            return (
                self._dummy_expert_batch(batch).to(self.device),
                torch.zeros((), device=self.device, dtype=torch.float32),
            )

        if not self._check_expert_batch_keys(expert_batch_raw):
            return (
                self._dummy_expert_batch(batch).to(self.device),
                torch.zeros((), device=self.device, dtype=torch.float32),
            )

        return (
            expert_batch_raw.to(self.device),
            torch.ones((), device=self.device, dtype=torch.float32),
        )

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
        (loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]).backward()
        output_loss = loss.clone().detach_()

        # 2) Behavior cloning on expert actions.
        bc_loss = torch.zeros((), device=self.device)

        if self._bc_coeff > 0.0:
            expert_action = self._expert_action_from_td(expert_batch)
            expert_obs_td = expert_batch.select(*self._policy_obs_keys)
            dist = self._policy_operator.get_dist(expert_obs_td)
            log_prob = dist.log_prob(expert_action)
            log_prob = self._reduce_log_prob(log_prob, expert_action)
            has_expert_float = has_expert.to(dtype=log_prob.dtype)
            bc_nll = -log_prob.mean() * has_expert_float
            bc_loss = bc_nll * self._bc_coeff
            bc_loss.backward()
            policy_action = self._policy_action_from_dist(dist)
            action_delta = policy_action.detach() - expert_action.detach()
            expert_action_f = expert_action.detach().float()
            policy_action_f = policy_action.detach().float()
            log_prob_f = log_prob.detach().float()
            actor_grad_norm = self._param_grad_norm(self._policy_operator.parameters())

        # 3) IPMD reward loss
        reward_grad_penalty_batch = torch.zeros((), device=self.device)
        reward_grad_penalty_expert = torch.zeros((), device=self.device)
        if self._reward_grad_penalty_coeff > 0.0:
            r_pi, r_pi_input = cast(
                tuple[Tensor, Tensor],
                self._reward_from_batch(
                    batch,
                    requires_grad=True,
                    return_input=True,
                ),
            )
            r_exp, r_exp_input = cast(
                tuple[Tensor, Tensor],
                self._reward_from_batch(
                    expert_batch,
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
            r_pi = cast(Tensor, self._reward_from_batch(batch))
            r_exp = cast(Tensor, self._reward_from_batch(expert_batch))
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
        output_loss.set(
            "loss_reward_grad_penalty_batch", reward_grad_penalty_batch.detach()
        )
        output_loss.set(
            "loss_reward_grad_penalty_expert", reward_grad_penalty_expert.detach()
        )
        if self._bc_coeff > 0.0:
            output_loss.set("loss_bc", bc_loss.detach())
            output_loss.set("bc_nll", bc_nll.detach())
            output_loss.set("bc_has_expert", has_expert_float.detach())
            output_loss.set("bc_log_prob_mean", log_prob_f.mean())
            output_loss.set(
                "bc_log_prob_nan_frac", torch.isnan(log_prob_f).float().mean()
            )
            output_loss.set("bc_expert_action_abs_mean", expert_action_f.abs().mean())
            output_loss.set(
                "bc_expert_action_zero_frac",
                expert_action_f.abs().lt(1e-6).float().mean(),
            )
            output_loss.set(
                "bc_expert_action_nan_frac", torch.isnan(expert_action_f).float().mean()
            )
            output_loss.set("bc_policy_action_abs_mean", policy_action_f.abs().mean())
            output_loss.set("bc_policy_action_mae", action_delta.abs().mean())
            output_loss.set("bc_policy_action_rmse", action_delta.pow(2).mean().sqrt())
            output_loss.set("bc_actor_grad_norm", actor_grad_norm.detach())
        output_loss.set("grad_norm", grad_norm_tensor.detach())
        output_loss.set(
            "lr",
            torch.tensor(
                self.optim.param_groups[0]["lr"],
                device=self.device,
                dtype=torch.float32,
            ),
        )
        with torch.no_grad():
            output_loss.set("estimated_reward_mean", r_pi.mean().detach())
            output_loss.set("estimated_reward_std", r_pi.std().detach())
            output_loss.set("expert_reward_mean", r_exp.mean().detach().nan_to_num(0.0))
            output_loss.set("expert_reward_std", r_exp.std().detach().nan_to_num(0.0))

        return output_loss, num_network_updates + 1

    def validate_training(self) -> None:
        """Warn when IPMD is about to train without a real expert source."""
        super().validate_training()
        if (
            self._expert_buffer is None
            and self._expert_batch_sampler is None
            and not self._warned_no_expert
        ):
            self.log.warning(
                "Expert source not set; reward estimator updates will use a dummy expert batch."
            )
            self._warned_no_expert = True

    def prepare(
        self,
        iteration: PPOIterationData,
        _metadata: PPOTrainingMetadata,
    ) -> None:
        """Attach reward-model diagnostics and replace the PPO reward for this rollout."""
        self.reward_estimator.eval()
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

        with timeit("training"):
            iteration.rollout = self.pre_iteration_compute(iteration.rollout)

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
                    loss = loss.clone()

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

    def _build_progress_postfix(self, iteration: PPOIterationData) -> dict[str, str]:
        postfix: dict[str, str] = {}
        if "train/step_reward_mean" in iteration.metrics:
            postfix["r_step"] = f"{iteration.metrics['train/step_reward_mean']:.2f}"
        if "episode/return" in iteration.metrics:
            postfix["r_ep"] = f"{iteration.metrics['episode/return']:.1f}"
        if "train/loss_objective" in iteration.metrics:
            postfix["pi_loss"] = f"{iteration.metrics['train/loss_objective']:.3f}"
        if "train/loss_reward_diff" in iteration.metrics:
            postfix["reward_diff"] = (
                f"{iteration.metrics['train/loss_reward_diff']:.3f}"
            )
        return postfix

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
            td = policy_op(td)
            return td.get("action")
