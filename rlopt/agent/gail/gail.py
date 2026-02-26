"""PPO-based GAIL and AMP implementations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler

from rlopt.agent.ppo import PPO, PPOConfig, PPORLOptConfig
from rlopt.config_utils import BatchKey, ObsKey, dedupe_keys, flatten_feature_tensor
from rlopt.utils import log_info

from .discriminator import Discriminator


class ExpertReplayBuffer(Protocol):
    def sample(self, batch_size: int | None = None) -> TensorDict: ...

    def __len__(self) -> int: ...


@dataclass
class GAILConfig:
    """Configuration for PPO-based GAIL."""

    discriminator_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    discriminator_activation: str = "relu"
    discriminator_lr: float = 3e-4
    discriminator_steps: int = 1
    discriminator_updates_per_policy_update: int | None = None
    discriminator_optimization_ratio: int = 1
    discriminator_update_warmup_updates: int = 0
    discriminator_grad_clip_norm: float = 1.0
    discriminator_loss_coeff: float = 1.0
    discriminator_batch_size: int | None = None
    expert_batch_size: int = 256
    discriminator_balance_policy_and_expert: bool = False

    discriminator_grad_penalty_coeff: float = 0.0
    discriminator_logit_reg_coeff: float = 0.0
    discriminator_weight_decay_coeff: float = 0.0

    discriminator_replay_size: int = 0
    discriminator_replay_ratio: float = 0.0
    discriminator_replay_batch_size: int | None = None
    discriminator_replay_keep_prob: float = 1.0

    normalize_discriminator_input: bool = False
    discriminator_input_noise_std: float = 0.0
    discriminator_input_dropout_prob: float = 0.0
    discriminator_input_norm_momentum: float = 0.01
    discriminator_input_norm_eps: float = 1.0e-5
    discriminator_input_norm_clip: float | None = None

    use_gail_reward: bool = True
    gail_reward_coeff: float = 1.0
    proportion_env_reward: float = 0.0
    normalize_discriminator_reward: bool = False
    reward_norm_momentum: float = 0.01
    reward_norm_eps: float = 1.0e-5
    reward_norm_clip: float | None = None

    reward_mix_alpha_start: float | None = None
    reward_mix_alpha_end: float | None = None
    reward_mix_anneal_updates: int = 0
    reward_mix_gate_estimated_std_min: float = 0.0
    reward_mix_alpha_when_unstable: float = 1.0
    reward_mix_gate_after_updates: int = 0
    reward_mix_gate_abs_gap_max: float = 0.0
    reward_mix_alpha_when_gap_large: float = 1.0

    amp_reward_clip: bool = True
    amp_reward_scale: float = 1.0

    discriminator_input_group: ObsKey | None = None
    """Optional observation group key for discriminator inputs (e.g. ``"policy"``)."""

    discriminator_input_keys: list[ObsKey] | None = None
    """Optional subset of observation keys for discriminator/reward input."""

    discriminator_use_action: bool = False
    """Whether discriminator additionally consumes ``action`` features."""

    # Deprecated aliases kept for config compatibility.
    discriminator_input_key: ObsKey | None = None
    discriminator_hidden_dim: int | None = None
    discriminator_num_layers: int | None = None
    env_reward_proportion: float | None = None

    def __post_init__(self) -> None:
        if self.discriminator_input_key is not None and self.discriminator_input_keys is None:
            self.discriminator_input_keys = [self.discriminator_input_key]
            warnings.warn(
                "`gail.discriminator_input_key` is deprecated. Use `gail.discriminator_input_keys`.",
                stacklevel=2,
            )
        if (
            self.discriminator_hidden_dim is not None
            and self.discriminator_num_layers is not None
        ):
            self.discriminator_hidden_dims = [
                int(self.discriminator_hidden_dim)
            ] * int(self.discriminator_num_layers)
            warnings.warn(
                "`gail.discriminator_hidden_dim/num_layers` are deprecated. Use `gail.discriminator_hidden_dims`.",
                stacklevel=2,
            )
        if self.env_reward_proportion is not None:
            self.proportion_env_reward = float(self.env_reward_proportion)
            warnings.warn(
                "`gail.env_reward_proportion` is deprecated. Use `gail.proportion_env_reward`.",
                stacklevel=2,
            )


@dataclass
class GAILRLOptConfig(PPORLOptConfig):
    """PPO config extended with GAIL parameters."""

    ppo: PPOConfig = field(default_factory=PPOConfig)
    gail: GAILConfig = field(default_factory=GAILConfig)


@dataclass
class AMPRLOptConfig(GAILRLOptConfig):
    """Configuration alias for AMP (same fields as GAIL)."""


class GAIL(PPO):
    """GAIL with PPO policy optimization and discriminator reward."""

    def __init__(self, env, config: GAILRLOptConfig):
        self.config = cast(GAILRLOptConfig, config)
        self._expert_buffer: ExpertReplayBuffer | None = None
        self._warned_no_expert = False

        self._disc_obs_keys = self._resolve_discriminator_obs_keys(env, self.config)
        self._disc_obs_feature_ndims = {
            key: self._obs_key_feature_ndim(env, key) for key in self._disc_obs_keys
        }
        self._disc_obs_dim = sum(
            self._obs_key_feature_dim(env, key) for key in self._disc_obs_keys
        )
        self._disc_condition_dim = int(self._discriminator_condition_dim())
        self._disc_obs_total_dim = self._disc_obs_dim + self._disc_condition_dim

        action_spec = getattr(env, "action_spec_unbatched", env.action_spec)
        action_shape = tuple(int(dim) for dim in action_spec.shape)
        self._action_feature_ndim = len(action_shape)
        self._action_dim = int(np.prod(action_shape)) if action_shape else 1

        self.discriminator: Discriminator | None = None
        self.discriminator_optim: torch.optim.Optimizer | None = None
        self._disc_replay_buffer: TensorDictReplayBuffer | None = None

        self._disc_input_running_mean: Tensor | None = None
        self._disc_input_running_var: Tensor | None = None
        self._disc_input_stats_initialized = False

        self._reward_running_mean: Tensor | None = None
        self._reward_running_var: Tensor | None = None
        self._reward_stats_initialized = False

        super().__init__(env, config)

        self.log.info(
            "Initialized PPO-based GAIL (policy params=%d, discriminator params=%d, disc_keys=%s, cond_dim=%d, use_action=%s)",
            sum(p.numel() for p in self.actor_critic.get_policy_operator().parameters()),
            sum(p.numel() for p in self.discriminator.parameters()) if self.discriminator else 0,
            self._disc_obs_keys,
            self._disc_condition_dim,
            self.config.gail.discriminator_use_action,
        )

    @staticmethod
    def _compose_grouped_key(group: ObsKey, key: ObsKey) -> ObsKey:
        if isinstance(group, tuple):
            if isinstance(key, tuple):
                return (*group, *key)
            return (*group, key)
        if isinstance(key, tuple):
            return (group, *key)
        return (group, key)

    @classmethod
    def _resolve_discriminator_obs_keys(
        cls, env, cfg: GAILRLOptConfig
    ) -> list[ObsKey]:
        gail_cfg = cfg.gail
        group = gail_cfg.discriminator_input_group
        keys = gail_cfg.discriminator_input_keys
        if keys:
            base = list(keys)
            if group is not None:
                base = [cls._compose_grouped_key(group, key) for key in base]
        elif group is not None:
            base = [group]
        else:
            value_cfg = cfg.value_function
            base = (
                value_cfg.get_input_keys()
                if value_cfg is not None
                else cfg.policy.get_input_keys()
            )

        available = set(env.observation_spec.keys(True))
        resolved: list[ObsKey] = []
        for key in dedupe_keys(list(base)):
            if key in available:
                resolved.append(cast(ObsKey, key))
                continue
            if isinstance(key, tuple) and len(key) > 0:
                flat_candidate = cast(ObsKey, key[-1])
                if flat_candidate in available:
                    resolved.append(flat_candidate)
                    continue
            if key == "policy" and "observation" in available:
                resolved.append(cast(ObsKey, "observation"))
                continue
            msg = f"Discriminator observation key '{key}' not found in env.observation_spec."
            raise KeyError(msg)
        return resolved

    @staticmethod
    def _obs_key_feature_shape(env, key: ObsKey) -> tuple[int, ...]:
        shape = tuple(int(dim) for dim in env.observation_spec[key].shape)
        batch_prefix = tuple(int(dim) for dim in getattr(env, "batch_size", ()))
        if (
            len(batch_prefix) > 0
            and len(shape) >= len(batch_prefix)
            and shape[: len(batch_prefix)] == batch_prefix
        ):
            return shape[len(batch_prefix) :]
        return shape

    @classmethod
    def _obs_key_feature_ndim(cls, env, key: ObsKey) -> int:
        return len(cls._obs_key_feature_shape(env, key))

    @classmethod
    def _obs_key_feature_dim(cls, env, key: ObsKey) -> int:
        shape = cls._obs_key_feature_shape(env, key)
        return int(np.prod(shape)) if shape else 1

    @staticmethod
    def _counter_as_int(value: int | Tensor) -> int:
        if isinstance(value, Tensor):
            return int(value.detach().item())
        return int(value)

    @staticmethod
    def _linear_schedule(
        start: float, end: float, step: int, total_steps: int
    ) -> float:
        if total_steps <= 0:
            return end
        step_clamped = min(max(step, 0), total_steps)
        alpha = step_clamped / float(total_steps)
        return float(start + alpha * (end - start))

    def _discriminator_condition_dim(self) -> int:
        """Extra discriminator conditioning feature dimension."""
        return 0

    def _discriminator_condition_required_keys(self) -> list[BatchKey]:
        """Additional batch keys needed by discriminator updates/replay."""
        return []

    def _discriminator_condition_from_td(
        self, td: TensorDict, *, detach: bool
    ) -> Tensor | None:
        """Return optional discriminator conditioning features."""
        return None

    def _prepare_rollout_batch_for_training(self, data: TensorDict) -> None:
        """Hook for subclasses to mutate collected batches before updates."""
        return None

    def _set_optimizers(
        self, optimizer_cls: type[torch.optim.Optimizer], optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        optimizers = super()._set_optimizers(optimizer_cls, optimizer_kwargs)
        if self.discriminator is None:
            self.discriminator = Discriminator(
                observation_dim=self._disc_obs_total_dim,
                action_dim=(
                    self._action_dim if self.config.gail.discriminator_use_action else 0
                ),
                hidden_dims=self.config.gail.discriminator_hidden_dims,
                activation=self.config.gail.discriminator_activation,
            ).to(self.device)
        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.gail.discriminator_lr,
        )
        if self._disc_replay_buffer is None:
            self._disc_replay_buffer = self._construct_discriminator_replay_buffer()
        return optimizers

    def _construct_discriminator_replay_buffer(self) -> TensorDictReplayBuffer | None:
        cfg = self.config.gail
        replay_size = int(cfg.discriminator_replay_size)
        if replay_size <= 0:
            return None
        replay_batch_size = (
            int(cfg.discriminator_replay_batch_size)
            if cfg.discriminator_replay_batch_size is not None
            else int(cfg.expert_batch_size)
        )
        return TensorDictReplayBuffer(
            pin_memory=False,
            sampler=RandomSampler(),
            storage=LazyTensorStorage(
                max_size=replay_size,
                compilable=self.config.compile.compile,
                device="cpu",
            ),
            batch_size=max(1, replay_batch_size),
            shared=self.config.collector.shared,
        )

    def _discriminator_policy_required_keys(self) -> list[BatchKey]:
        required: list[BatchKey] = list(self._disc_obs_keys)
        if self.config.gail.discriminator_use_action:
            required.append("action")
        required.extend(self._discriminator_condition_required_keys())
        return dedupe_keys(required)

    def _store_discriminator_replay_samples(self, batch: TensorDict) -> None:
        if self._disc_replay_buffer is None:
            return
        required_keys = self._discriminator_policy_required_keys()
        available = batch.keys(True)
        if any(key not in available for key in required_keys):
            return
        replay_batch = batch.select(*required_keys).reshape(-1).detach().to("cpu")
        keep_prob = float(np.clip(self.config.gail.discriminator_replay_keep_prob, 0.0, 1.0))
        if keep_prob < 1.0 and replay_batch.numel() > 0:
            keep_mask = torch.rand(replay_batch.numel()) < keep_prob
            if not bool(keep_mask.any()):
                return
            replay_batch = replay_batch[keep_mask]
        if replay_batch.numel() > 0:
            self._disc_replay_buffer.extend(replay_batch.clone())

    def _sample_discriminator_replay_batch(
        self, current_batch_size: int
    ) -> TensorDict | None:
        if self._disc_replay_buffer is None:
            return None
        replay_batch_size_cfg = self.config.gail.discriminator_replay_batch_size
        if replay_batch_size_cfg is not None:
            replay_batch_size = int(replay_batch_size_cfg)
        else:
            replay_batch_size = round(
                current_batch_size * max(0.0, float(self.config.gail.discriminator_replay_ratio))
            )
        if replay_batch_size <= 0:
            return None
        try:
            replay_batch = cast(
                TensorDict,
                self._disc_replay_buffer.sample(batch_size=replay_batch_size),
            )
        except Exception:
            return None
        if replay_batch.numel() <= 0:
            return None
        return replay_batch.to(self.device)

    def _discriminator_policy_batch_with_replay(self, batch: TensorDict) -> TensorDict:
        required_keys = self._discriminator_policy_required_keys()
        available = batch.keys(True)
        if any(key not in available for key in required_keys):
            return batch
        policy_batch = batch.select(*required_keys)
        replay_batch = self._sample_discriminator_replay_batch(policy_batch.numel())
        if replay_batch is None:
            return policy_batch
        replay_available = replay_batch.keys(True)
        if any(key not in replay_available for key in required_keys):
            return policy_batch
        replay_batch = replay_batch.select(*required_keys)
        return cast(TensorDict, torch.cat([policy_batch, replay_batch], dim=0))

    def set_expert_buffer(self, expert_buffer: ExpertReplayBuffer) -> None:
        self._expert_buffer = expert_buffer
        self.log.info("Expert buffer attached: %d samples", len(expert_buffer))

    def _next_expert_batch(self, batch_size: int | None = None) -> TensorDict | None:
        if self._expert_buffer is None:
            return None
        if batch_size is None:
            batch_size = int(self.config.gail.expert_batch_size)

        try:
            batch = cast(TensorDict, self._expert_buffer.sample(batch_size=batch_size))
        except TypeError:
            try:
                batch = cast(TensorDict, self._expert_buffer.sample())
            except Exception as exc:  # pragma: no cover - defensive
                self.log.warning("Failed to sample expert batch: %s", exc)
                return None
        except Exception as exc:  # pragma: no cover - defensive
            self.log.warning("Failed to sample expert batch: %s", exc)
            return None

        if batch_size is not None and batch.numel() > batch_size:
            batch = cast(TensorDict, batch[:batch_size])
        return batch.to(self.device)

    def _obs_features_from_td(self, td: TensorDict, *, detach: bool) -> Tensor:
        parts: list[Tensor] = []
        for key in self._disc_obs_keys:
            obs = cast(Tensor, td.get(cast(BatchKey, key)))
            obs = flatten_feature_tensor(obs, self._disc_obs_feature_ndims[key])
            parts.append(obs.detach() if detach else obs)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def _action_features_from_td(self, td: TensorDict, *, detach: bool) -> Tensor:
        action = cast(Tensor, td.get("action"))
        action = flatten_feature_tensor(action, self._action_feature_ndim)
        return action.detach() if detach else action

    def _compose_discriminator_inputs(
        self,
        obs: Tensor,
        action: Tensor | None,
        condition: Tensor | None = None,
    ) -> Tensor:
        parts: list[Tensor] = [obs]
        if condition is not None and condition.shape[-1] > 0:
            parts.append(condition)
        if action is None:
            action = self._empty_action_like(obs)
        parts.append(action)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _empty_action_like(obs: Tensor) -> Tensor:
        return torch.zeros(*obs.shape[:-1], 0, dtype=obs.dtype, device=obs.device)

    def _split_discriminator_inputs(
        self, inputs: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        obs_with_condition = inputs[..., : self._disc_obs_total_dim]
        if not self.config.gail.discriminator_use_action:
            return obs_with_condition, None
        action = inputs[..., self._disc_obs_total_dim :]
        return obs_with_condition, action

    def _normalize_discriminator_inputs(
        self, inputs: Tensor, *, update_stats: bool
    ) -> Tensor:
        cfg = self.config.gail
        if not cfg.normalize_discriminator_input:
            return inputs

        eps = float(cfg.discriminator_input_norm_eps)
        reduce_dims = tuple(range(inputs.ndim - 1))
        if len(reduce_dims) == 0:
            reduce_dims = (0,)
        if (
            not self._disc_input_stats_initialized
            or self._disc_input_running_mean is None
            or self._disc_input_running_var is None
            or self._disc_input_running_mean.shape[-1] != inputs.shape[-1]
        ):
            self._disc_input_running_mean = inputs.detach().mean(dim=reduce_dims)
            self._disc_input_running_var = inputs.detach().var(
                dim=reduce_dims, unbiased=False
            )
            self._disc_input_stats_initialized = True

        assert self._disc_input_running_mean is not None
        assert self._disc_input_running_var is not None

        if update_stats:
            momentum = float(np.clip(cfg.discriminator_input_norm_momentum, 0.0, 1.0))
            batch_mean = inputs.detach().mean(dim=reduce_dims)
            batch_var = inputs.detach().var(dim=reduce_dims, unbiased=False)
            self._disc_input_running_mean = (1.0 - momentum) * self._disc_input_running_mean + momentum * batch_mean
            self._disc_input_running_var = (1.0 - momentum) * self._disc_input_running_var + momentum * batch_var

        normalized = (inputs - self._disc_input_running_mean) / torch.sqrt(
            self._disc_input_running_var + eps
        )
        clip_value = cfg.discriminator_input_norm_clip
        if clip_value is not None and clip_value > 0.0:
            normalized = normalized.clamp(-float(clip_value), float(clip_value))
        return normalized

    def _apply_discriminator_input_augmentation(self, inputs: Tensor) -> Tensor:
        cfg = self.config.gail
        noise_std = float(cfg.discriminator_input_noise_std)
        if noise_std > 0.0:
            inputs = inputs + noise_std * torch.randn_like(inputs)

        dropout_prob = float(cfg.discriminator_input_dropout_prob)
        if dropout_prob > 0.0:
            keep_prob = float(np.clip(1.0 - dropout_prob, 1.0e-6, 1.0))
            mask = (torch.rand_like(inputs) < keep_prob).to(inputs.dtype)
            inputs = inputs * (mask / keep_prob)
        return inputs

    def _prepare_disc_obs_action(
        self,
        obs: Tensor,
        action: Tensor | None,
        condition: Tensor | None,
        *,
        update_stats: bool,
        apply_augmentation: bool,
    ) -> tuple[Tensor, Tensor | None]:
        inputs = self._compose_discriminator_inputs(obs, action, condition)
        inputs = self._normalize_discriminator_inputs(inputs, update_stats=update_stats)
        if apply_augmentation:
            inputs = self._apply_discriminator_input_augmentation(inputs)
        return self._split_discriminator_inputs(inputs)

    def _apply_discriminator_reward_normalization(
        self, rewards: Tensor, *, update_stats: bool
    ) -> Tensor:
        cfg = self.config.gail
        if not cfg.normalize_discriminator_reward:
            return rewards

        eps = float(cfg.reward_norm_eps)
        flat = rewards.reshape(-1)
        if (
            not self._reward_stats_initialized
            or self._reward_running_mean is None
            or self._reward_running_var is None
        ):
            self._reward_running_mean = flat.detach().mean()
            self._reward_running_var = flat.detach().var(unbiased=False)
            self._reward_stats_initialized = True

        assert self._reward_running_mean is not None
        assert self._reward_running_var is not None

        if update_stats:
            momentum = float(np.clip(cfg.reward_norm_momentum, 0.0, 1.0))
            batch_mean = flat.detach().mean()
            batch_var = flat.detach().var(unbiased=False)
            self._reward_running_mean = (1.0 - momentum) * self._reward_running_mean + momentum * batch_mean
            self._reward_running_var = (1.0 - momentum) * self._reward_running_var + momentum * batch_var

        normalized = (rewards - self._reward_running_mean) / torch.sqrt(
            self._reward_running_var + eps
        )
        clip_value = cfg.reward_norm_clip
        if clip_value is not None and clip_value > 0.0:
            normalized = normalized.clamp(-float(clip_value), float(clip_value))
        return normalized

    def _gail_discriminator_loss(
        self,
        expert_obs: Tensor,
        expert_action: Tensor | None,
        policy_obs: Tensor,
        policy_action: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        assert self.discriminator is not None
        if expert_action is None:
            expert_action = self._empty_action_like(expert_obs)
        if policy_action is None:
            policy_action = self._empty_action_like(policy_obs)
        return self.discriminator.compute_loss(
            expert_obs=expert_obs,
            expert_action=expert_action,
            policy_obs=policy_obs,
            policy_action=policy_action,
        )

    def _gail_reward(self, obs: Tensor, action: Tensor | None) -> Tensor:
        assert self.discriminator is not None
        if action is None:
            action = self._empty_action_like(obs)
        return self.discriminator.compute_reward(obs, action)

    def _discriminator_grad_penalty(
        self, expert_obs: Tensor, expert_action: Tensor | None
    ) -> Tensor:
        if float(self.config.gail.discriminator_grad_penalty_coeff) <= 0.0:
            return torch.zeros((), dtype=expert_obs.dtype, device=expert_obs.device)

        assert self.discriminator is not None
        obs_req = expert_obs.detach().requires_grad_(True)
        grad_inputs: list[Tensor] = [obs_req]

        if expert_action is not None and expert_action.shape[-1] > 0:
            action_req = expert_action.detach().requires_grad_(True)
            grad_inputs.append(action_req)
            logits = self.discriminator(obs_req, action_req).squeeze(-1)
        else:
            empty_action = self._empty_action_like(expert_obs)
            logits = self.discriminator(obs_req, empty_action).squeeze(-1)

        grads = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=grad_inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        grad_norm_sq = grads[0].pow(2).sum(dim=-1)
        for grad in grads[1:]:
            grad_norm_sq = grad_norm_sq + grad.pow(2).sum(dim=-1)
        return grad_norm_sq.mean()

    def _discriminator_weight_decay(self) -> Tensor:
        coeff = float(self.config.gail.discriminator_weight_decay_coeff)
        assert self.discriminator is not None
        if coeff <= 0.0:
            return torch.zeros((), device=self.device)
        weights = self.discriminator.all_weights()
        if len(weights) == 0:
            return torch.zeros((), device=self.device)
        total = torch.zeros((), device=self.device)
        for weight in weights:
            total = total + weight.pow(2).mean()
        return total

    def _discriminator_step_count(self) -> int:
        configured = self.config.gail.discriminator_updates_per_policy_update
        if configured is not None:
            return max(1, int(configured))
        return max(1, int(self.config.gail.discriminator_steps))

    def _discriminator_batch_size(self, policy_count: int, expert_count: int) -> int:
        cfg = self.config.gail
        target = cfg.discriminator_batch_size
        if target is None:
            target = cfg.expert_batch_size
        if cfg.discriminator_balance_policy_and_expert:
            return max(1, int(min(target, policy_count, expert_count)))
        return max(1, int(target))

    def _base_reward_mix_alpha(self) -> float:
        p_env = float(np.clip(self.config.gail.proportion_env_reward, 0.0, 1.0))
        return float(np.clip(1.0 - p_env, 0.0, 1.0))

    def _current_reward_mix_alpha(
        self,
        update_idx: int,
        estimated_reward_std: float,
        estimated_env_reward_abs_gap: float | None = None,
        *,
        default_alpha: float,
    ) -> float:
        cfg = self.config.gail

        start = cfg.reward_mix_alpha_start
        end = cfg.reward_mix_alpha_end
        if start is None:
            start = default_alpha
        if end is None:
            end = start

        alpha = self._linear_schedule(
            float(start),
            float(end),
            update_idx,
            int(cfg.reward_mix_anneal_updates),
        )

        if (
            update_idx >= int(cfg.reward_mix_gate_after_updates)
            and float(cfg.reward_mix_gate_estimated_std_min) > 0.0
            and estimated_reward_std < float(cfg.reward_mix_gate_estimated_std_min)
        ):
            alpha = min(alpha, float(cfg.reward_mix_alpha_when_unstable))

        if (
            estimated_env_reward_abs_gap is not None
            and update_idx >= int(cfg.reward_mix_gate_after_updates)
            and float(cfg.reward_mix_gate_abs_gap_max) > 0.0
            and estimated_env_reward_abs_gap > float(cfg.reward_mix_gate_abs_gap_max)
        ):
            alpha = min(alpha, float(cfg.reward_mix_alpha_when_gap_large))

        return float(np.clip(alpha, 0.0, 1.0))

    def _update_discriminator(
        self, rollout_flat: TensorDict, update_idx: int
    ) -> dict[str, float]:
        if self._expert_buffer is None:
            if not self._warned_no_expert:
                self.log.warning(
                    "No expert buffer set. Training falls back to environment reward."
                )
                self._warned_no_expert = True
            return {}

        if update_idx < int(self.config.gail.discriminator_update_warmup_updates):
            return {"discriminator_update_mask": 0.0}
        ratio = max(1, int(self.config.gail.discriminator_optimization_ratio))
        if (update_idx % ratio) != 0:
            return {"discriminator_update_mask": 0.0}

        assert self.discriminator is not None
        assert self.discriminator_optim is not None

        policy_batch = self._discriminator_policy_batch_with_replay(rollout_flat.reshape(-1))
        self._store_discriminator_replay_samples(rollout_flat.reshape(-1))

        policy_obs = self._obs_features_from_td(policy_batch, detach=False).to(self.device)
        policy_action: Tensor | None = None
        if self.config.gail.discriminator_use_action:
            policy_action = self._action_features_from_td(policy_batch, detach=False).to(
                self.device
            )
        policy_condition = self._discriminator_condition_from_td(
            policy_batch, detach=False
        )
        if policy_condition is not None:
            policy_condition = policy_condition.to(self.device)
        if policy_obs.shape[0] == 0:
            return {}

        stats_accum: dict[str, float] = {"discriminator_update_mask": 1.0}
        performed_steps = 0

        for _ in range(self._discriminator_step_count()):
            expert_td = self._next_expert_batch(batch_size=self.config.gail.expert_batch_size)
            if expert_td is None:
                continue

            expert_obs = self._obs_features_from_td(expert_td, detach=False).to(self.device)
            expert_action: Tensor | None = None
            if self.config.gail.discriminator_use_action:
                expert_action = self._action_features_from_td(expert_td, detach=False).to(
                    self.device
                )
            expert_condition = self._discriminator_condition_from_td(
                expert_td, detach=False
            )
            if expert_condition is not None:
                expert_condition = expert_condition.to(self.device)
            if expert_obs.shape[0] == 0:
                continue

            batch_size = self._discriminator_batch_size(policy_obs.shape[0], expert_obs.shape[0])
            if batch_size <= 0:
                continue

            p_idx = torch.randint(policy_obs.shape[0], (batch_size,), device=self.device)
            e_idx = torch.randint(expert_obs.shape[0], (batch_size,), device=self.device)

            policy_obs_b = policy_obs[p_idx].detach()
            expert_obs_b = expert_obs[e_idx]
            policy_action_b = (
                policy_action[p_idx].detach() if policy_action is not None else None
            )
            expert_action_b = (
                expert_action[e_idx] if expert_action is not None else None
            )
            policy_condition_b = (
                policy_condition[p_idx].detach()
                if policy_condition is not None
                else None
            )
            expert_condition_b = (
                expert_condition[e_idx] if expert_condition is not None else None
            )

            expert_obs_b, expert_action_b = self._prepare_disc_obs_action(
                expert_obs_b,
                expert_action_b,
                expert_condition_b,
                update_stats=True,
                apply_augmentation=True,
            )
            policy_obs_b, policy_action_b = self._prepare_disc_obs_action(
                policy_obs_b,
                policy_action_b,
                policy_condition_b,
                update_stats=True,
                apply_augmentation=True,
            )

            loss_main, info = self._gail_discriminator_loss(
                expert_obs_b,
                expert_action_b,
                policy_obs_b,
                policy_action_b,
            )

            gp = self._discriminator_grad_penalty(expert_obs_b, expert_action_b)

            expert_action_for_logits = expert_action_b
            policy_action_for_logits = policy_action_b
            if expert_action_for_logits is None:
                expert_action_for_logits = self._empty_action_like(expert_obs_b)
            if policy_action_for_logits is None:
                policy_action_for_logits = self._empty_action_like(policy_obs_b)
            expert_logits = self.discriminator(expert_obs_b, expert_action_for_logits).squeeze(-1)
            policy_logits = self.discriminator(policy_obs_b, policy_action_for_logits).squeeze(-1)
            logit_reg = 0.5 * (expert_logits.pow(2).mean() + policy_logits.pow(2).mean())
            weight_decay = self._discriminator_weight_decay()

            total_loss = (
                float(self.config.gail.discriminator_loss_coeff) * loss_main
                + float(self.config.gail.discriminator_grad_penalty_coeff) * gp
                + float(self.config.gail.discriminator_logit_reg_coeff) * logit_reg
                + float(self.config.gail.discriminator_weight_decay_coeff) * weight_decay
            )

            self.discriminator_optim.zero_grad(set_to_none=True)
            total_loss.backward()
            clip_grad_norm_(
                self.discriminator.parameters(),
                float(self.config.gail.discriminator_grad_clip_norm),
            )
            self.discriminator_optim.step()
            performed_steps += 1

            for key, value in info.items():
                stats_accum[key] = stats_accum.get(key, 0.0) + float(value.detach().item())
            stats_accum["discriminator_total_loss"] = stats_accum.get("discriminator_total_loss", 0.0) + float(total_loss.detach().item())
            stats_accum["discriminator_grad_penalty"] = stats_accum.get("discriminator_grad_penalty", 0.0) + float(gp.detach().item())
            stats_accum["discriminator_logit_reg"] = stats_accum.get("discriminator_logit_reg", 0.0) + float(logit_reg.detach().item())
            stats_accum["discriminator_weight_decay"] = stats_accum.get("discriminator_weight_decay", 0.0) + float(weight_decay.detach().item())

        if performed_steps <= 0:
            return {"discriminator_update_mask": 0.0}

        stats_accum["discriminator_updates_performed"] = float(performed_steps)
        for key in list(stats_accum.keys()):
            if key in {"discriminator_update_mask", "discriminator_updates_performed"}:
                continue
            stats_accum[key] = stats_accum[key] / float(performed_steps)
        return stats_accum

    def _replace_rewards_with_discriminator(
        self, data: TensorDict, update_idx: int
    ) -> dict[str, float]:
        if not self.config.gail.use_gail_reward or self.discriminator is None:
            return {}

        with torch.no_grad():
            obs = self._obs_features_from_td(data, detach=True).to(self.device)
            action: Tensor | None = None
            if self.config.gail.discriminator_use_action:
                action = self._action_features_from_td(data, detach=True).to(self.device)
            condition = self._discriminator_condition_from_td(data, detach=True)
            if condition is not None:
                condition = condition.to(self.device)

            obs, action = self._prepare_disc_obs_action(
                obs,
                action,
                condition,
                update_stats=False,
                apply_augmentation=False,
            )
            disc_reward = self._gail_reward(obs, action)
            disc_reward = self.config.gail.gail_reward_coeff * disc_reward
            disc_reward = self._apply_discriminator_reward_normalization(
                disc_reward,
                update_stats=True,
            )

            env_reward = data[("next", "reward")].to(self.device)
            squeeze_last = env_reward.ndim == disc_reward.ndim + 1 and env_reward.shape[-1] == 1
            env_reward_base = env_reward.squeeze(-1) if squeeze_last else env_reward

            reward_abs_gap = float((disc_reward.mean() - env_reward_base.mean()).abs().item())
            alpha = self._current_reward_mix_alpha(
                update_idx,
                float(disc_reward.std().item()),
                reward_abs_gap,
                default_alpha=self._base_reward_mix_alpha(),
            )
            mixed_reward = (1.0 - alpha) * env_reward_base + alpha * disc_reward
            if squeeze_last:
                mixed_reward = mixed_reward.unsqueeze(-1)
            data.set(("next", "reward"), mixed_reward)

        return {
            "gail/reward_mean": float(disc_reward.mean().item()),
            "gail/reward_std": float(disc_reward.std().item()),
            "gail/reward_mix_alpha": float(alpha),
            "gail/reward_mix_abs_gap": float(reward_abs_gap),
        }

    def train(self) -> None:  # type: ignore[override]
        cfg = self.config
        assert isinstance(cfg, GAILRLOptConfig)

        collected_frames = 0
        num_network_updates = torch.zeros((), dtype=torch.int64, device=self.device)
        pbar = tqdm.tqdm(total=cfg.collector.total_frames)

        num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
        if cfg.collector.frames_per_batch % cfg.loss.mini_batch_size != 0:
            num_mini_batches += 1
        self.total_network_updates = (
            (cfg.collector.total_frames // cfg.collector.frames_per_batch)
            * cfg.loss.epochs
            * num_mini_batches
        )

        cfg_loss_ppo_epochs = cfg.loss.epochs
        cfg_loss_anneal_clip_eps = cfg.ppo.anneal_clip_epsilon
        cfg_loss_clip_epsilon = cfg.ppo.clip_epsilon
        losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

        self.collector = cast(SyncDataCollector, self.collector)
        collector_iter = iter(self.collector)
        total_iter = len(self.collector)
        policy_op = self.actor_critic.get_policy_operator()

        for _ in range(total_iter):
            self.actor_critic.eval()
            self.adv_module.eval()
            with timeit("collecting"):
                data = next(collector_iter)
            self._prepare_rollout_batch_for_training(data)

            metrics_to_log: dict[str, Any] = {}
            frames_in_batch = data.numel()
            collected_frames += frames_in_batch
            pbar.update(frames_in_batch)

            if ("next", "reward") in data.keys(True):
                step_rewards = data["next", "reward"]
                metrics_to_log.update(
                    {
                        "train/step_reward_mean": step_rewards.mean().item(),
                        "train/step_reward_std": step_rewards.std().item(),
                        "train/step_reward_max": step_rewards.max().item(),
                        "train/step_reward_min": step_rewards.min().item(),
                    }
                )
            if ("next", "episode_reward") in data.keys(True):
                episode_rewards = data["next", "episode_reward"][data["next", "done"]]
                if len(episode_rewards) > 0:
                    episode_length = data["next", "step_count"][data["next", "done"]]
                    self.episode_lengths.extend(episode_length.cpu().tolist())
                    self.episode_rewards.extend(episode_rewards.cpu().tolist())
                    metrics_to_log.update(
                        {
                            "episode/length": float(np.mean(self.episode_lengths)),
                            "episode/return": float(np.mean(self.episode_rewards)),
                            "train/reward": float(np.mean(episode_rewards.cpu().tolist())),
                        }
                    )

            rollout_update_idx = self._counter_as_int(num_network_updates)
            discriminator_metrics = self._update_discriminator(data.reshape(-1), rollout_update_idx)
            if discriminator_metrics:
                metrics_to_log.update({f"train/{k}": v for k, v in discriminator_metrics.items()})
            reward_metrics = self._replace_rewards_with_discriminator(data, rollout_update_idx)
            if reward_metrics:
                metrics_to_log.update({f"train/{k}": v for k, v in reward_metrics.items()})

            self.data_buffer.empty()
            self.actor_critic.train()
            self.adv_module.train()
            with timeit("training"):
                for j in range(cfg_loss_ppo_epochs):
                    with torch.no_grad(), timeit("adv"):
                        data = self.adv_module(data)
                        if self.config.compile.compile_mode:
                            data = data.clone()

                    with timeit("rb - extend"):
                        data_reshape = data.reshape(-1)
                        self.data_buffer.extend(data_reshape)

                    for k, batch in enumerate(self.data_buffer):
                        kl_context = None
                        if (self.config.optim.scheduler or "").lower() == "adaptive":
                            kl_context = self._prepare_kl_context(batch, policy_op)
                        with timeit("update"):
                            loss, num_network_updates = self.update(  # type: ignore[misc]
                                batch, num_network_updates=num_network_updates
                            )
                            loss = loss.clone()
                        if self.lr_scheduler and self.lr_scheduler_step == "update":
                            self.lr_scheduler.step()
                        if kl_context is not None:
                            kl_approx = self._compute_kl_after_update(kl_context, policy_op)
                            if kl_approx is not None:
                                loss.set("kl_approx", kl_approx.detach())
                                self._maybe_adjust_lr(kl_approx, self.config.optim)

                        num_network_updates = num_network_updates.clone()
                        loss_keys = ["loss_critic", "loss_entropy", "loss_objective"]
                        optional_keys = [
                            "entropy",
                            "explained_variance",
                            "clip_fraction",
                            "value_clip_fraction",
                            "ESS",
                            "kl_approx",
                            "grad_norm",
                        ]
                        for key in optional_keys:
                            if key in loss:
                                loss_keys.append(key)
                        losses[j, k] = loss.select(*loss_keys)

                    if self.lr_scheduler and self.lr_scheduler_step == "epoch":
                        self.lr_scheduler.step()

            losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in losses_mean.items():
                metrics_to_log[f"train/{key}"] = value.item()
            metrics_to_log["train/lr"] = self.optim.param_groups[0]["lr"]

            clip_attr = getattr(self.loss_module, "clip_epsilon", None)
            if cfg_loss_anneal_clip_eps and isinstance(clip_attr, torch.Tensor):
                clip_epsilon_value = clip_attr.detach()
            else:
                clip_epsilon_value = torch.tensor(
                    cfg_loss_clip_epsilon,
                    device=self.device,
                    dtype=torch.float32,
                )
            metrics_to_log["train/clip_epsilon"] = clip_epsilon_value

            if "Isaac" in self.config.env.env_name and hasattr(self.env, "log_infos"):
                log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                log_info(log_info_dict, metrics_to_log)

            metrics_to_log.update(timeit.todict(prefix="time"))
            rate = pbar.format_dict.get("rate")
            if rate is not None:
                metrics_to_log["time/speed"] = rate

            self.log_metrics(metrics_to_log, step=collected_frames)
            self.collector.update_policy_weights_()

            if (
                self.config.save_interval > 0
                and num_network_updates % self.config.save_interval == 0
            ):
                self.save_model(
                    path=self.log_dir / self.config.logger.save_path,
                    step=int(collected_frames),
                )

        pbar.close()
        self.collector.shutdown()

    def _gail_state_dict(self) -> dict[str, Any]:
        return {
            "disc_input_running_mean": self._disc_input_running_mean,
            "disc_input_running_var": self._disc_input_running_var,
            "disc_input_stats_initialized": self._disc_input_stats_initialized,
            "reward_running_mean": self._reward_running_mean,
            "reward_running_var": self._reward_running_var,
            "reward_stats_initialized": self._reward_stats_initialized,
        }

    def _load_gail_state_dict(self, state: dict[str, Any]) -> None:
        self._disc_input_running_mean = state.get("disc_input_running_mean")
        self._disc_input_running_var = state.get("disc_input_running_var")
        self._disc_input_stats_initialized = bool(state.get("disc_input_stats_initialized", False))
        self._reward_running_mean = state.get("reward_running_mean")
        self._reward_running_var = state.get("reward_running_var")
        self._reward_stats_initialized = bool(state.get("reward_stats_initialized", False))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = {
            "actor_critic": self.actor_critic.state_dict(),
            "discriminator": self.discriminator.state_dict() if self.discriminator else {},
            "config": self.config,
            "gail_state": self._gail_state_dict(),
        }
        if self.discriminator_optim is not None:
            checkpoint["discriminator_optim"] = self.discriminator_optim.state_dict()
        torch.save(checkpoint, path)
        self.log.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        if self.discriminator is not None and "discriminator" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator"])
        if self.discriminator_optim is not None and "discriminator_optim" in checkpoint:
            self.discriminator_optim.load_state_dict(checkpoint["discriminator_optim"])
        gail_state = checkpoint.get("gail_state")
        if isinstance(gail_state, dict):
            self._load_gail_state_dict(gail_state)
        self.log.info("Model loaded from %s", path)


class AMP(GAIL):
    """Adversarial Motion Prior using PPO + least-squares discriminator."""

    def _gail_discriminator_loss(
        self,
        expert_obs: Tensor,
        expert_action: Tensor | None,
        policy_obs: Tensor,
        policy_action: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        assert self.discriminator is not None
        if expert_action is None:
            expert_action = self._empty_action_like(expert_obs)
        if policy_action is None:
            policy_action = self._empty_action_like(policy_obs)

        expert_logits = self.discriminator(expert_obs, expert_action).squeeze(-1)
        policy_logits = self.discriminator(policy_obs, policy_action).squeeze(-1)

        expert_target = torch.ones_like(expert_logits)
        policy_target = -torch.ones_like(policy_logits)
        loss = torch.mean((expert_logits - expert_target) ** 2) + torch.mean(
            (policy_logits - policy_target) ** 2
        )

        info = {
            "discriminator_loss": loss.detach(),
            "expert_d_mean": expert_logits.mean().detach(),
            "policy_d_mean": policy_logits.mean().detach(),
        }
        return loss, info

    def _gail_reward(self, obs: Tensor, action: Tensor | None) -> Tensor:
        assert self.discriminator is not None
        if action is None:
            action = self._empty_action_like(obs)
        logits = self.discriminator(obs, action).squeeze(-1)
        reward = 1.0 - 0.25 * (logits - 1.0) ** 2
        if self.config.gail.amp_reward_clip:
            reward = torch.clamp_min(reward, 0.0)
        reward = reward * float(self.config.gail.amp_reward_scale)
        return reward
