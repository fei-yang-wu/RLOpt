"""ASE (Adversarial Skill Embeddings) with agent-managed latent commands."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import MLP

from rlopt.agent.gail.gail import AMP, AMPRLOptConfig
from rlopt.agent.imitation.latent_skill import (
    LatentEncoder,
    LatentSkillMixin,
    generalized_advantage_estimate,
)
from rlopt.agent.ppo.ppo import (
    PPO,
    CatInputs,
    PPOIterationData,
    PPOTrainingMetadata,
)
from rlopt.config_utils import (
    BatchKey,
    ObsKey,
    infer_batch_shape,
    mapping_get_obs_value,
)
from rlopt.utils import get_activation_class


@dataclass
class ASEConfig:
    """ASE-specific configuration layered on top of AMP/GAIL settings."""

    latent_dim: int = 16
    latent_key: ObsKey = "latent_command"
    latent_steps_min: int = 30
    latent_steps_max: int = 120

    latent_vmf_kappa: float = 1.0
    mi_reward_weight: float = 0.25
    mi_loss_coeff: float = 1.0
    mi_encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    mi_encoder_activation: str = "elu"
    mi_encoder_lr: float = 3e-4
    mi_grad_clip_norm: float = 1.0
    mi_weight_decay_coeff: float = 0.0
    mi_grad_penalty_coeff: float = 0.0

    mi_critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    mi_critic_activation: str = "elu"
    mi_critic_lr: float = 3e-4
    mi_critic_grad_clip_norm: float = 1.0

    diversity_bonus_coeff: float = 0.05
    diversity_target: float = 1.0
    latent_uniformity_coeff: float = 0.0
    latent_uniformity_temperature: float = 2.0

    # Backward-compatible aliases kept for old configs.
    num_skills: int | None = None
    style_coeff: float | None = None
    diversity_coeff: float | None = None
    task_coeff: float | None = None

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


class ASE(LatentSkillMixin, AMP[ASERLOptConfig]):
    """ASE built on top of AMP with explicit latent-command observations."""

    def __init__(self, env, config: ASERLOptConfig):
        self.config: ASERLOptConfig = config

        if self.config.ase.style_coeff is not None:
            self.config.gail.gail_reward_coeff = float(self.config.ase.style_coeff)
        if self.config.ase.task_coeff is not None:
            self.config.gail.proportion_env_reward = float(
                np.clip(self.config.ase.task_coeff, 0.0, 1.0)
            )

        self._init_latent_skills(
            env,
            latent_key=cast(ObsKey, self.config.ase.latent_key),
            latent_dim=int(self.config.ase.latent_dim),
            latent_steps_min=int(self.config.ase.latent_steps_min),
            latent_steps_max=int(self.config.ase.latent_steps_max),
        )

        self.mi_encoder: LatentEncoder | None = None
        self.mi_encoder_optim: torch.optim.Optimizer | None = None
        self.mi_critic: TensorDictModule | None = None
        self.mi_critic_optim: torch.optim.Optimizer | None = None
        self._policy_obs_keys: list[ObsKey] = []
        self._policy_obs_feature_ndims: dict[ObsKey, int] = {}
        self._mi_critic_in_keys: list[ObsKey] = []

        super().__init__(env, config)

        self._policy_obs_keys = list(self.config.policy.get_input_keys())
        self._policy_obs_feature_ndims = {
            key: self.observation_feature_rank(key) for key in self._policy_obs_keys
        }
        self._mi_critic_in_keys = [*self._disc_obs_keys, self._latent_key]

        self.log.info(
            "Initialized ASE (latent_dim=%d, latent_key=%s, disc_keys=%s)",
            self._latent_dim,
            self._latent_key,
            self._disc_obs_keys,
        )

    def _discriminator_condition_dim(self) -> int:
        return self._latent_dim

    def _discriminator_condition_required_keys(self) -> list[BatchKey]:
        return [cast(BatchKey, self._latent_key)]

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

        if self.mi_critic is None:
            mi_mlp = MLP(
                in_features=self._disc_obs_dim + self._latent_dim,
                out_features=1,
                num_cells=list(self.config.ase.mi_critic_hidden_dims),
                activation_class=get_activation_class(
                    self.config.ase.mi_critic_activation
                ),
                device=self.device,
            )
            self.mi_critic = TensorDictModule(
                module=CatInputs(mi_mlp),
                in_keys=[*self._disc_obs_keys, self._latent_key],
                out_keys=["mi_value_pred"],
            ).to(self.device)

        self.mi_encoder_optim = torch.optim.Adam(
            self.mi_encoder.parameters(),
            lr=float(self.config.ase.mi_encoder_lr),
        )
        self.mi_critic_optim = torch.optim.Adam(
            self.mi_critic.parameters(),
            lr=float(self.config.ase.mi_critic_lr),
        )
        return optimizers

    def _discriminator_condition_from_td(
        self, td: TensorDict, *, detach: bool
    ) -> Tensor | None:
        latent = self._latent_condition_from_td(td, detach=detach)
        if latent is not None:
            return latent
        if self.mi_encoder is None:
            return self._sample_unit_latents(
                td.numel(),
                device=self.device,
                dtype=torch.float32,
            )
        obs_features = self._obs_features_from_td(td, detach=False).to(self.device)
        latent = self.mi_encoder(obs_features)
        return latent.detach() if detach else latent

    def _mi_features_and_latents(self, batch: TensorDict) -> tuple[Tensor, Tensor]:
        obs_features = self._obs_features_from_td(batch, detach=False).to(self.device)
        latents = self._discriminator_condition_from_td(batch, detach=True)
        if latents is None:
            raise RuntimeError("ASE rollout batch is missing latent commands.")
        return obs_features, latents.to(self.device)

    def _mi_grad_penalty(self, obs_features: Tensor, latents: Tensor) -> Tensor:
        if (
            self.mi_encoder is None
            or float(self.config.ase.mi_grad_penalty_coeff) <= 0.0
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

        mi_loss = -float(self.config.ase.latent_vmf_kappa) * similarity.mean()
        grad_penalty = self._mi_grad_penalty(obs_features, latents)

        weight_decay = torch.zeros((), device=self.device)
        if float(self.config.ase.mi_weight_decay_coeff) > 0.0:
            for param in self.mi_encoder.parameters():
                if param.ndim >= 2:
                    weight_decay = weight_decay + param.pow(2).mean()

        expert_uniformity = latent_pred
        expert_td = self._next_expert_batch(
            batch_size=self.config.gail.expert_batch_size
        )
        if expert_td is not None:
            expert_obs = self._obs_features_from_td(expert_td, detach=False).to(
                self.device
            )
            expert_uniformity = torch.cat(
                [latent_pred, self.mi_encoder(expert_obs)], dim=0
            )

        uniformity = self._latent_uniformity(expert_uniformity)
        total_loss = (
            float(self.config.ase.mi_loss_coeff) * mi_loss
            + float(self.config.ase.mi_grad_penalty_coeff) * grad_penalty
            + float(self.config.ase.mi_weight_decay_coeff) * weight_decay
            + float(self.config.ase.latent_uniformity_coeff) * uniformity
        )

        self.mi_encoder_optim.zero_grad(set_to_none=True)
        total_loss.backward()
        if float(self.config.ase.mi_grad_clip_norm) > 0.0:
            clip_grad_norm_(
                self.mi_encoder.parameters(), float(self.config.ase.mi_grad_clip_norm)
            )
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
            float(self.config.ase.mi_reward_weight)
            * float(self.config.ase.latent_vmf_kappa)
            * score
        )

    def _diversity_loss(self, batch: TensorDict) -> Tensor:
        """ProtoMotions-style diversity objective added directly to the actor loss."""
        if float(self.config.ase.diversity_bonus_coeff) <= 0.0 or batch.numel() <= 1:
            return torch.zeros((), device=self.device)
        if "loc" not in batch.keys(True) or "scale" not in batch.keys(True):
            raise KeyError("ASE diversity loss requires PPO rollout loc/scale keys.")

        old_latents = self._latent_condition_from_td(batch, detach=True)
        if old_latents is None:
            raise RuntimeError("ASE minibatch is missing latent commands.")
        old_latents = old_latents.to(self.device)

        policy_operator = self.actor_critic.get_policy_operator()
        old_dist = policy_operator.build_dist_from_params(
            batch.select("loc", "scale").clone()
        )
        old_mean_action = self._policy_action_from_dist(old_dist)
        if old_mean_action is None:
            raise RuntimeError(
                "ASE diversity loss could not recover the old policy mean action."
            )
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
        new_dist = policy_operator.get_dist(policy_td)
        new_mean_action = self._policy_action_from_dist(new_dist)
        if new_mean_action is None:
            raise RuntimeError(
                "ASE diversity loss could not recover the new policy mean action."
            )
        new_mean_action = self._clip_policy_action(new_mean_action)

        action_delta = (new_mean_action - old_mean_action).pow(2).mean(dim=-1)
        latent_delta = 0.5 - 0.5 * (new_latents * old_latents).sum(dim=-1)
        diversity_bonus = action_delta / (latent_delta + 1.0e-5)
        return (float(self.config.ase.diversity_target) - diversity_bonus).pow(2).mean()

    def _extra_actor_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        if float(self.config.ase.diversity_bonus_coeff) <= 0.0:
            return super()._extra_actor_loss(batch)
        extra_loss, extra_metrics = super()._extra_actor_loss(batch)
        diversity_loss = self._diversity_loss(batch)
        extra_metrics["loss_diversity"] = diversity_loss.detach()
        weighted_loss = diversity_loss * float(self.config.ase.diversity_bonus_coeff)
        return extra_loss + weighted_loss, extra_metrics

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
            if latents is None:
                raise RuntimeError("ASE rollout batch is missing latent commands.")
            latents = latents.to(self.device)

            obs_disc, action_disc = self._prepare_disc_obs_action(
                obs_features,
                action_features,
                latents,
                update_stats=False,
                apply_augmentation=False,
            )
            style_reward = float(
                self.config.gail.gail_reward_coeff
            ) * self._gail_reward(obs_disc, action_disc)
            mi_reward = self._mi_reward(obs_features, latents)

            adv_reward = style_reward + mi_reward
            adv_reward = self._apply_discriminator_reward_normalization(
                adv_reward,
                update_stats=True,
            )

            env_reward = data[("next", "reward")].to(self.device)
            squeeze_last = (
                env_reward.ndim == adv_reward.ndim + 1 and env_reward.shape[-1] == 1
            )
            env_reward_base = env_reward.squeeze(-1) if squeeze_last else env_reward

            reward_abs_gap = float(
                (adv_reward.mean() - env_reward_base.mean()).abs().item()
            )
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
            "ase/reward_mean": float(adv_reward.mean().item()),
            "ase/reward_std": float(adv_reward.std().item()),
            "ase/reward_mix_alpha": float(alpha),
            "ase/reward_mix_abs_gap": float(reward_abs_gap),
        }

    def _attach_mi_targets(self, rollout: TensorDict) -> None:
        if self.mi_critic is None:
            return

        flat_rollout = rollout.reshape(-1)
        next_td = cast(TensorDict, flat_rollout.get("next"))

        with torch.no_grad():
            mi_value = self.mi_critic(
                flat_rollout.select(*self._mi_critic_in_keys).clone()
            )["mi_value_pred"].reshape(*rollout.batch_size)
            next_value = self.mi_critic(
                next_td.select(*self._mi_critic_in_keys).clone()
            )["mi_value_pred"].reshape(*rollout.batch_size)

            obs_features = self._obs_features_from_td(flat_rollout, detach=True).to(
                self.device
            )
            latents = self._discriminator_condition_from_td(flat_rollout, detach=True)
            if latents is None:
                raise RuntimeError("ASE rollout batch is missing latent commands.")
            mi_reward = self._mi_reward(obs_features, latents).reshape(
                *rollout.batch_size
            )

        done = rollout["next", "done"]
        if done.ndim == mi_reward.ndim + 1 and done.shape[-1] == 1:
            done = done.squeeze(-1)
        mi_advantages, mi_returns = generalized_advantage_estimate(
            mi_reward,
            mi_value.squeeze(-1),
            next_value.squeeze(-1),
            done,
            gamma=float(self.config.loss.gamma),
            gae_lambda=float(self.config.ppo.gae_lambda),
        )
        rollout.set("mi_reward", mi_reward)
        rollout.set("mi_value", mi_value.squeeze(-1))
        rollout.set("mi_advantage", mi_advantages)
        rollout.set("mi_returns", mi_returns)

    def pre_iteration_compute(self, rollout: TensorDict) -> TensorDict:
        with torch.no_grad():
            rollout = self.adv_module(rollout)
            self._attach_mi_targets(rollout)
            if getattr(self.config.compile, "compile_mode", None):
                rollout = rollout.clone()

        self.data_buffer.extend(rollout.reshape(-1))
        return rollout

    def _update_mi_critic_batch(self, batch: TensorDict) -> dict[str, Tensor]:
        if (
            self.mi_critic is None
            or self.mi_critic_optim is None
            or "mi_returns" not in batch
        ):
            return {}

        pred_td = self.mi_critic(batch.select(*self._mi_critic_in_keys).clone())
        pred = cast(Tensor, pred_td.get("mi_value_pred")).squeeze(-1)
        target = cast(Tensor, batch.get("mi_returns")).to(self.device)
        loss = F.mse_loss(pred, target)

        self.mi_critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        if float(self.config.ase.mi_critic_grad_clip_norm) > 0.0:
            clip_grad_norm_(
                self.mi_critic.parameters(),
                float(self.config.ase.mi_critic_grad_clip_norm),
            )
        self.mi_critic_optim.step()
        return {"mi_critic_loss": loss.detach()}

    @property
    def _optional_loss_metrics(self) -> list[str]:
        return [*super()._optional_loss_metrics, "mi_critic_loss", "loss_diversity"]

    def prepare(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,
    ) -> None:
        self._prepare_latent_rollout_batch_for_training(iteration.rollout)
        rollout_update_idx = self._counter_as_int(metadata.updates_completed)

        discriminator_metrics = self._update_discriminator(
            iteration.rollout.reshape(-1), rollout_update_idx
        )
        if discriminator_metrics:
            iteration.metrics.update(
                {f"train/{key}": value for key, value in discriminator_metrics.items()}
            )

        reward_metrics = self._replace_rewards_with_discriminator(
            iteration.rollout,
            rollout_update_idx,
        )
        if reward_metrics:
            iteration.metrics.update(
                {f"train/{key}": value for key, value in reward_metrics.items()}
            )

    def iterate(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,
    ) -> None:
        losses = TensorDict(
            batch_size=[metadata.epochs_per_rollout, metadata.minibatches_per_epoch]
        )
        learn_start = time.perf_counter()

        self.data_buffer.empty()
        self.actor_critic.train()
        self.adv_module.train()
        if self.mi_critic is not None:
            self.mi_critic.train()

        with timeit("training"):
            iteration.rollout = self.pre_iteration_compute(iteration.rollout)

            for epoch_idx in range(metadata.epochs_per_rollout):
                for batch_idx, batch in enumerate(self.data_buffer):
                    kl_context = None
                    if (self.config.optim.scheduler or "").lower() == "adaptive":
                        kl_context = self._prepare_kl_context(
                            batch, metadata.policy_operator
                        )

                    loss, metadata.updates_completed = PPO.update(
                        self, batch, metadata.updates_completed
                    )
                    loss = loss.clone()

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

    def train(self) -> None:  # type: ignore[override]
        PPO.train(self)

    def predict(self, obs: Tensor | np.ndarray | Mapping[Any, Any]) -> Tensor:  # type: ignore[override]
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
                    feature_ndim = self._policy_obs_feature_ndims[key]
                    if (feature_ndim == 0 and value.ndim == 0) or (
                        feature_ndim > 0 and value.ndim == feature_ndim
                    ):
                        value = value.unsqueeze(0)
                    current_batch_shape = infer_batch_shape(value, feature_ndim)
                    if batch_shape is None:
                        batch_shape = current_batch_shape
                    elif current_batch_shape != batch_shape:
                        raise ValueError(
                            "All observation tensors passed to predict() must share the same batch shape."
                        )
                    td_data[cast(BatchKey, key)] = value
            else:
                if len(input_keys) != 1:
                    raise ValueError(
                        "predict() received a single tensor observation, but the policy expects keyed observations."
                    )
                key = input_keys[0]
                value = torch.as_tensor(obs, device=self.device)
                feature_ndim = self._policy_obs_feature_ndims[key]
                if (feature_ndim == 0 and value.ndim == 0) or (
                    feature_ndim > 0 and value.ndim == feature_ndim
                ):
                    value = value.unsqueeze(0)
                td_data[cast(BatchKey, key)] = value
                batch_shape = infer_batch_shape(value, feature_ndim)

            batch_shape = batch_shape or (1,)
            self._inject_predict_latents(td_data, batch_shape)
            td = TensorDict(td_data, batch_size=list(batch_shape), device=self.device)
            td = policy_op(td)
            return td.get("action")

    def _gail_state_dict(self) -> dict[str, Any]:
        state = super()._gail_state_dict()
        state.update(
            {
                "mi_encoder": (
                    self.mi_encoder.state_dict()
                    if self.mi_encoder is not None
                    else None
                ),
                "mi_encoder_optim": (
                    self.mi_encoder_optim.state_dict()
                    if self.mi_encoder_optim is not None
                    else None
                ),
                "mi_critic": (
                    self.mi_critic.state_dict() if self.mi_critic is not None else None
                ),
                "mi_critic_optim": (
                    self.mi_critic_optim.state_dict()
                    if self.mi_critic_optim is not None
                    else None
                ),
            }
        )
        return state

    def _load_gail_state_dict(self, state: dict[str, Any]) -> None:
        super()._load_gail_state_dict(state)
        if self.mi_encoder is not None and state.get("mi_encoder") is not None:
            self.mi_encoder.load_state_dict(cast(dict[str, Any], state["mi_encoder"]))
        if (
            self.mi_encoder_optim is not None
            and state.get("mi_encoder_optim") is not None
        ):
            self.mi_encoder_optim.load_state_dict(
                cast(dict[str, Any], state["mi_encoder_optim"])
            )
        if self.mi_critic is not None and state.get("mi_critic") is not None:
            self.mi_critic.load_state_dict(cast(dict[str, Any], state["mi_critic"]))
        if (
            self.mi_critic_optim is not None
            and state.get("mi_critic_optim") is not None
        ):
            self.mi_critic_optim.load_state_dict(
                cast(dict[str, Any], state["mi_critic_optim"])
            )

    def save(self, path: str | Path) -> None:
        super().save(path)

    def load(self, path: str | Path) -> None:
        super().load(path)
