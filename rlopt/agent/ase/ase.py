"""ASE (Adversarial Skill Embeddings) on top of PPO adversarial training."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from rlopt.agent.gail.gail import AMP, AMPRLOptConfig
from rlopt.config_utils import BatchKey, ObsKey, dedupe_keys, flatten_feature_tensor
from rlopt.utils import get_activation_class


class _ASECollectorPolicy(nn.Module):
    """Collector policy wrapper that injects per-env latents before action sampling."""

    def __init__(self, agent: "ASE", policy_module: nn.Module):
        super().__init__()
        self.agent = agent
        self.policy_module = policy_module

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self.agent._inject_collector_latents(tensordict)
        return self.policy_module(tensordict)


class LatentEncoder(nn.Module):
    """Simple MLP encoder used for ASE MI objectives."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        activation: str,
    ) -> None:
        super().__init__()
        act_cls = get_activation_class(activation)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(act_cls())
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs_features: Tensor) -> Tensor:
        latent = self.network(obs_features)
        return F.normalize(latent, dim=-1, eps=1.0e-6)


@dataclass
class ASEConfig:
    """ASE-specific configuration layered on top of AMP/GAIL settings."""

    latent_dim: int = 16
    latent_key: str = "ase_latent"
    latent_steps_min: int = 30
    latent_steps_max: int = 120

    # MI / encoder settings
    latent_vmf_kappa: float = 1.0
    mi_reward_coeff: float = 0.25
    mi_loss_coeff: float = 1.0
    mi_encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    mi_encoder_activation: str = "elu"
    mi_encoder_lr: float = 3e-4
    mi_grad_clip_norm: float = 1.0
    mi_weight_decay_coeff: float = 0.0
    mi_grad_penalty_coeff: float = 0.0

    # Diversity / uniformity
    diversity_bonus_coeff: float = 0.05
    diversity_reward_scale: float = 1.0
    latent_uniformity_coeff: float = 0.0
    latent_uniformity_temperature: float = 2.0

    # Backward-compatible aliases from older ASE configs.
    num_skills: int | None = None
    style_coeff: float | None = None
    diversity_coeff: float | None = None
    task_coeff: float | None = None
    skill_sampling: str | None = None

    def __post_init__(self) -> None:
        if self.num_skills is not None:
            self.latent_dim = int(self.num_skills)
            warnings.warn(
                "`ase.num_skills` is deprecated. Use `ase.latent_dim`.",
                stacklevel=2,
            )
        if self.diversity_coeff is not None:
            self.diversity_bonus_coeff = float(self.diversity_coeff)
            warnings.warn(
                "`ase.diversity_coeff` is deprecated. Use `ase.diversity_bonus_coeff`.",
                stacklevel=2,
            )


@dataclass
class ASERLOptConfig(AMPRLOptConfig):
    """RLOpt config for ASE."""

    ase: ASEConfig = field(default_factory=ASEConfig)


class ASE(AMP):
    """ASE built on top of AMP's PPO-adversarial training core."""

    def __init__(self, env, config: ASERLOptConfig):
        self.config = cast(ASERLOptConfig, config)

        # Keep old ASE keys operational where practical.
        if self.config.ase.style_coeff is not None:
            self.config.gail.gail_reward_coeff = float(self.config.ase.style_coeff)
            warnings.warn(
                "`ase.style_coeff` is deprecated. Use `gail.gail_reward_coeff`.",
                stacklevel=2,
            )
        if self.config.ase.task_coeff is not None:
            self.config.gail.proportion_env_reward = float(
                np.clip(self.config.ase.task_coeff, 0.0, 1.0)
            )
            warnings.warn(
                "`ase.task_coeff` is deprecated. Use `gail.proportion_env_reward`.",
                stacklevel=2,
            )
        if self.config.ase.skill_sampling not in (None, "uniform"):
            warnings.warn(
                "Only uniform ASE latent sampling is supported in this implementation.",
                stacklevel=2,
            )

        self._latent_key = cast(ObsKey, self.config.ase.latent_key)
        self._latent_dim = int(self.config.ase.latent_dim)

        # Keys whose features are augmented with the sampled latent.
        value_keys = (
            self.config.value_function.get_input_keys()
            if self.config.value_function is not None
            else []
        )
        self._latent_aug_keys: list[ObsKey] = dedupe_keys(
            cast(list[ObsKey], self.config.policy.get_input_keys() + value_keys)
        )

        self._latent_aug_feature_ndims: dict[ObsKey, int] = {}
        self._latent_aug_base_dims: dict[ObsKey, int] = {}
        available = set(env.observation_spec.keys(True))
        for key in self._latent_aug_keys:
            if key not in available:
                continue
            self._latent_aug_feature_ndims[key] = self._obs_key_feature_ndim(env, key)
            self._latent_aug_base_dims[key] = self._obs_key_feature_dim(env, key)

        self._collector_latents: Tensor | None = None
        self._collector_latent_horizon: Tensor | None = None
        self._collector_policy_wrapper: _ASECollectorPolicy | None = None

        self.mi_encoder: LatentEncoder | None = None
        self.mi_encoder_optim: torch.optim.Optimizer | None = None

        super().__init__(env, config)

        self.log.info(
            "Initialized ASE (latent_dim=%d, latent_key=%s, latent_steps=[%d, %d], aug_keys=%s)",
            self._latent_dim,
            self._latent_key,
            int(self.config.ase.latent_steps_min),
            int(self.config.ase.latent_steps_max),
            self._latent_aug_keys,
        )

    @property
    def collector_policy(self):
        policy_operator = self.actor_critic.get_policy_operator()
        if self._collector_policy_wrapper is None:
            self._collector_policy_wrapper = _ASECollectorPolicy(self, policy_operator)
        return self._collector_policy_wrapper

    def _discriminator_condition_dim(self) -> int:
        return self._latent_dim

    def _discriminator_condition_required_keys(self) -> list[BatchKey]:
        return [cast(BatchKey, self._latent_key)]

    def _discriminator_condition_from_td(
        self, td: TensorDict, *, detach: bool
    ) -> Tensor | None:
        keys = td.keys(True)
        if self._latent_key in keys:
            latent = cast(Tensor, td.get(cast(BatchKey, self._latent_key)))
        else:
            latent = self._sample_unit_latents(
                td.numel(),
                device=self.device,
                dtype=torch.float32,
            )
            latent = latent.reshape(*td.batch_size, self._latent_dim)

        latent_flat = flatten_feature_tensor(latent, 1)
        return latent_flat.detach() if detach else latent_flat

    def _set_optimizers(
        self,
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_kwargs: dict[str, Any],
    ) -> list[torch.optim.Optimizer]:
        optimizers = super()._set_optimizers(optimizer_cls, optimizer_kwargs)

        if self.mi_encoder is None:
            self.mi_encoder = LatentEncoder(
                input_dim=self._disc_obs_dim,
                latent_dim=self._latent_dim,
                hidden_dims=list(self.config.ase.mi_encoder_hidden_dims),
                activation=self.config.ase.mi_encoder_activation,
            ).to(self.device)

        self.mi_encoder_optim = torch.optim.Adam(
            self.mi_encoder.parameters(),
            lr=float(self.config.ase.mi_encoder_lr),
        )
        return optimizers

    def _sample_unit_latents(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        z = torch.randn(batch_size, self._latent_dim, device=device, dtype=dtype)
        return F.normalize(z, dim=-1, eps=1.0e-6)

    def _sample_latent_horizon(self, batch_size: int, *, device: torch.device) -> Tensor:
        lo = max(1, int(self.config.ase.latent_steps_min))
        hi = max(lo, int(self.config.ase.latent_steps_max))
        if lo == hi:
            return torch.full((batch_size,), lo, device=device, dtype=torch.long)
        return torch.randint(lo, hi + 1, (batch_size,), device=device)

    def _ensure_collector_latent_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (
            self._collector_latents is None
            or self._collector_latent_horizon is None
            or self._collector_latents.shape[0] != batch_size
            or self._collector_latents.device != device
            or self._collector_latents.dtype != dtype
        ):
            self._collector_latents = self._sample_unit_latents(
                batch_size, device=device, dtype=dtype
            )
            self._collector_latent_horizon = self._sample_latent_horizon(
                batch_size, device=device
            )

    def _done_mask_from_td(self, td: TensorDict, *, batch_size: int) -> Tensor:
        keys = td.keys(True)
        done_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        candidate_keys: list[BatchKey] = [
            cast(BatchKey, "done"),
            cast(BatchKey, "terminated"),
            cast(BatchKey, "truncated"),
            cast(BatchKey, "is_init"),
            cast(BatchKey, ("next", "done")),
            cast(BatchKey, ("next", "terminated")),
            cast(BatchKey, ("next", "truncated")),
            cast(BatchKey, ("next", "is_init")),
        ]

        for key in candidate_keys:
            if key not in keys:
                continue
            value = cast(Tensor, td.get(key)).reshape(-1).to(self.device).bool()
            if value.numel() == batch_size:
                done_mask = done_mask | value
        return done_mask

    def _concat_latent_feature(
        self,
        key: ObsKey,
        feature: Tensor,
        latent: Tensor,
    ) -> Tensor:
        feature_ndim = self._latent_aug_feature_ndims.get(key)
        base_dim = self._latent_aug_base_dims.get(key)
        if feature_ndim is None or base_dim is None:
            return feature
        if feature_ndim != 1:
            return feature

        flat_feature = flatten_feature_tensor(feature, feature_ndim)
        flat_latent = flatten_feature_tensor(latent, 1)
        if flat_feature.shape[0] != flat_latent.shape[0]:
            return feature

        feat_dim = int(flat_feature.shape[-1])
        if feat_dim == base_dim + self._latent_dim:
            return feature
        if feat_dim != base_dim:
            return feature

        augmented = torch.cat([flat_feature, flat_latent], dim=-1)
        return augmented.reshape(*feature.shape[:-1], augmented.shape[-1])

    def _augment_obs_keys_with_latent(
        self,
        td: TensorDict,
        *,
        include_next: bool,
    ) -> None:
        keys = td.keys(True)
        if self._latent_key not in keys:
            return

        latent = cast(Tensor, td.get(cast(BatchKey, self._latent_key)))

        for key in self._latent_aug_keys:
            batch_key = cast(BatchKey, key)
            if batch_key in keys:
                feature = cast(Tensor, td.get(batch_key))
                td.set(batch_key, self._concat_latent_feature(key, feature, latent))

            if not include_next:
                continue

            next_key = cast(BatchKey, ("next", key) if not isinstance(key, tuple) else ("next", *key))
            next_latent_key = cast(BatchKey, ("next", self._latent_key))
            if next_key in keys and next_latent_key in keys:
                next_feature = cast(Tensor, td.get(next_key))
                next_latent = cast(Tensor, td.get(next_latent_key))
                td.set(
                    next_key,
                    self._concat_latent_feature(key, next_feature, next_latent),
                )

    def _inject_collector_latents(self, td: TensorDict) -> None:
        batch_size = int(td.numel())
        if batch_size <= 0:
            return

        # infer dtype from the first available augmented key
        dtype = torch.float32
        for key in self._latent_aug_keys:
            if cast(BatchKey, key) in td.keys(True):
                dtype = cast(Tensor, td.get(cast(BatchKey, key))).dtype
                break

        self._ensure_collector_latent_state(batch_size, device=self.device, dtype=dtype)
        assert self._collector_latents is not None
        assert self._collector_latent_horizon is not None

        self._collector_latent_horizon = self._collector_latent_horizon - 1
        done_mask = self._done_mask_from_td(td, batch_size=batch_size)
        renew_mask = done_mask | (self._collector_latent_horizon <= 0)

        if bool(renew_mask.any()):
            renew_count = int(renew_mask.sum().item())
            self._collector_latents[renew_mask] = self._sample_unit_latents(
                renew_count,
                device=self.device,
                dtype=dtype,
            )
            self._collector_latent_horizon[renew_mask] = self._sample_latent_horizon(
                renew_count,
                device=self.device,
            )

        latents = self._collector_latents.reshape(*td.batch_size, self._latent_dim)
        td.set(cast(BatchKey, self._latent_key), latents)

        if "next" in td.keys(True):
            td.set(cast(BatchKey, ("next", self._latent_key)), latents.clone())

        self._augment_obs_keys_with_latent(td, include_next=True)

    def _prepare_rollout_batch_for_training(self, data: TensorDict) -> None:
        keys = data.keys(True)

        if self._latent_key not in keys:
            latents = self._sample_unit_latents(
                data.numel(), device=self.device, dtype=torch.float32
            ).reshape(*data.batch_size, self._latent_dim)
            data.set(cast(BatchKey, self._latent_key), latents)

        next_latent_key = cast(BatchKey, ("next", self._latent_key))
        if next_latent_key not in keys:
            current_latents = cast(Tensor, data.get(cast(BatchKey, self._latent_key)))
            data.set(next_latent_key, current_latents.clone())

        self._augment_obs_keys_with_latent(data, include_next=True)

    def _mi_features_and_latents(self, batch: TensorDict) -> tuple[Tensor, Tensor]:
        obs_features = self._obs_features_from_td(batch, detach=False).to(self.device)
        latents = self._discriminator_condition_from_td(batch, detach=True)
        assert latents is not None
        return obs_features, latents.to(self.device)

    def _mi_grad_penalty(self, obs_features: Tensor, latents: Tensor) -> Tensor:
        coeff = float(self.config.ase.mi_grad_penalty_coeff)
        if coeff <= 0.0 or self.mi_encoder is None:
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
        temperature = float(max(self.config.ase.latent_uniformity_temperature, 1.0e-6))
        pairwise_sq = torch.pdist(latent_pred, p=2).pow(2)
        return torch.log(torch.exp(-temperature * pairwise_sq).mean() + 1.0e-8)

    def _update_mi_encoder(self, rollout_flat: TensorDict) -> dict[str, float]:
        if self.mi_encoder is None or self.mi_encoder_optim is None:
            return {}

        if float(self.config.ase.mi_loss_coeff) <= 0.0:
            return {}

        obs_features, latents = self._mi_features_and_latents(rollout_flat)
        if obs_features.shape[0] == 0:
            return {}

        latent_pred = self.mi_encoder(obs_features)
        similarity = (latent_pred * latents).sum(dim=-1)

        kappa = float(self.config.ase.latent_vmf_kappa)
        mi_loss = -(kappa * similarity.mean())
        grad_penalty = self._mi_grad_penalty(obs_features, latents)

        weight_decay_coeff = float(self.config.ase.mi_weight_decay_coeff)
        weight_decay = torch.zeros((), device=self.device)
        if weight_decay_coeff > 0.0:
            for param in self.mi_encoder.parameters():
                if param.ndim >= 2:
                    weight_decay = weight_decay + param.pow(2).mean()

        uniformity_coeff = float(self.config.ase.latent_uniformity_coeff)
        uniformity = self._latent_uniformity(latent_pred)

        total_loss = (
            float(self.config.ase.mi_loss_coeff) * mi_loss
            + float(self.config.ase.mi_grad_penalty_coeff) * grad_penalty
            + weight_decay_coeff * weight_decay
            + uniformity_coeff * uniformity
        )

        self.mi_encoder_optim.zero_grad(set_to_none=True)
        total_loss.backward()

        clip_norm = float(self.config.ase.mi_grad_clip_norm)
        if clip_norm > 0.0:
            clip_grad_norm_(self.mi_encoder.parameters(), clip_norm)

        self.mi_encoder_optim.step()

        return {
            "ase/mi_total_loss": float(total_loss.detach().item()),
            "ase/mi_loss": float(mi_loss.detach().item()),
            "ase/mi_similarity_mean": float(similarity.detach().mean().item()),
            "ase/mi_grad_penalty": float(grad_penalty.detach().item()),
            "ase/mi_weight_decay": float(weight_decay.detach().item()),
            "ase/latent_uniformity": float(uniformity.detach().item()),
        }

    def _update_discriminator(
        self, rollout_flat: TensorDict, update_idx: int
    ) -> dict[str, float]:
        stats = super()._update_discriminator(rollout_flat, update_idx)
        stats.update(self._update_mi_encoder(rollout_flat.reshape(-1)))
        return stats

    def _mi_reward(self, obs_features: Tensor, latents: Tensor) -> Tensor:
        if self.mi_encoder is None:
            return torch.zeros(obs_features.shape[0], device=obs_features.device)

        with torch.no_grad():
            latent_pred = self.mi_encoder(obs_features)
            score = (latent_pred * latents).sum(dim=-1)
        return (
            float(self.config.ase.mi_reward_coeff)
            * float(self.config.ase.latent_vmf_kappa)
            * score
        )

    def _diversity_reward(
        self,
        obs_features: Tensor,
        action_features: Tensor | None,
        latents: Tensor,
    ) -> Tensor:
        coeff = float(self.config.ase.diversity_bonus_coeff)
        if coeff <= 0.0 or latents.shape[0] <= 1:
            return torch.zeros(obs_features.shape[0], device=obs_features.device)

        with torch.no_grad():
            obs_main, action_main = self._prepare_disc_obs_action(
                obs_features,
                action_features,
                latents,
                update_stats=False,
                apply_augmentation=False,
            )
            reward_main = self._gail_reward(obs_main, action_main)

            perm = torch.randperm(latents.shape[0], device=latents.device)
            shuffled_latent = latents[perm]
            obs_alt, action_alt = self._prepare_disc_obs_action(
                obs_features,
                action_features,
                shuffled_latent,
                update_stats=False,
                apply_augmentation=False,
            )
            reward_alt = self._gail_reward(obs_alt, action_alt)

        return (
            coeff
            * float(self.config.ase.diversity_reward_scale)
            * (reward_main - reward_alt).abs()
        )

    def _replace_rewards_with_discriminator(
        self, data: TensorDict, update_idx: int
    ) -> dict[str, float]:
        if not self.config.gail.use_gail_reward or self.discriminator is None:
            return {}

        with torch.no_grad():
            obs_features = self._obs_features_from_td(data, detach=True).to(self.device)
            action_features: Tensor | None = None
            if self.config.gail.discriminator_use_action:
                action_features = self._action_features_from_td(data, detach=True).to(
                    self.device
                )

            latents = self._discriminator_condition_from_td(data, detach=True)
            assert latents is not None
            latents = latents.to(self.device)

            obs_disc, action_disc = self._prepare_disc_obs_action(
                obs_features,
                action_features,
                latents,
                update_stats=False,
                apply_augmentation=False,
            )
            style_reward = float(self.config.gail.gail_reward_coeff) * self._gail_reward(
                obs_disc, action_disc
            )

            mi_reward = self._mi_reward(obs_features, latents)
            diversity_reward = self._diversity_reward(
                obs_features,
                action_features,
                latents,
            )

            adv_reward = style_reward + mi_reward + diversity_reward
            adv_reward = self._apply_discriminator_reward_normalization(
                adv_reward,
                update_stats=True,
            )

            env_reward = data[("next", "reward")].to(self.device)
            squeeze_last = env_reward.ndim == adv_reward.ndim + 1 and env_reward.shape[-1] == 1
            env_reward_base = env_reward.squeeze(-1) if squeeze_last else env_reward

            reward_abs_gap = float((adv_reward.mean() - env_reward_base.mean()).abs().item())
            alpha = self._current_reward_mix_alpha(
                update_idx,
                float(adv_reward.std().item()),
                reward_abs_gap,
                default_alpha=self._base_reward_mix_alpha(),
            )
            mixed_reward = (1.0 - alpha) * env_reward_base + alpha * adv_reward
            if squeeze_last:
                mixed_reward = mixed_reward.unsqueeze(-1)
            data.set(("next", "reward"), mixed_reward)

        return {
            "ase/style_reward_mean": float(style_reward.mean().item()),
            "ase/mi_reward_mean": float(mi_reward.mean().item()),
            "ase/diversity_reward_mean": float(diversity_reward.mean().item()),
            "ase/reward_mean": float(adv_reward.mean().item()),
            "ase/reward_std": float(adv_reward.std().item()),
            "ase/reward_mix_alpha": float(alpha),
            "ase/reward_mix_abs_gap": float(reward_abs_gap),
        }

    def _gail_state_dict(self) -> dict[str, Any]:
        state = super()._gail_state_dict()
        state.update(
            {
                "mi_encoder": (
                    self.mi_encoder.state_dict() if self.mi_encoder is not None else None
                ),
                "mi_encoder_optim": (
                    self.mi_encoder_optim.state_dict()
                    if self.mi_encoder_optim is not None
                    else None
                ),
            }
        )
        return state

    def _load_gail_state_dict(self, state: dict[str, Any]) -> None:
        super()._load_gail_state_dict(state)
        if self.mi_encoder is not None and state.get("mi_encoder") is not None:
            self.mi_encoder.load_state_dict(cast(dict[str, Any], state["mi_encoder"]))
        if self.mi_encoder_optim is not None and state.get("mi_encoder_optim") is not None:
            self.mi_encoder_optim.load_state_dict(
                cast(dict[str, Any], state["mi_encoder_optim"])
            )

    def save(self, path: str | Path) -> None:
        super().save(path)

    def load(self, path: str | Path) -> None:
        super().load(path)
