"""ProtoMotions-style ASE with agent-managed latent commands."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

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

from rlopt.agent.ase.model import ASEDiscriminatorEncoder
from rlopt.agent.gail.gail import AMP, AMPRLOptConfig
from rlopt.agent.imitation.latent_skill import (
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
    """ASE-specific configuration with ProtoMotions-style naming."""

    # Latent skill space config
    latent_dim: int = 64
    latent_key: ObsKey = "latent_command"
    latent_steps_min: int = 30
    latent_steps_max: int = 120
    command_source: str = "random"

    # Reward weights
    task_reward_w: float = 1.0
    discriminator_reward_w: float = 1.0
    mi_reward_w: float = 0.25
    mi_hypersphere_reward_shift: bool = True

    # MI encoder configuration
    mi_enc_weight_decay: float = 0.0
    mi_enc_grad_penalty: float = 0.0
    diversity_bonus: float = 0.05
    diversity_tar: float = 1.0
    latent_uniformity_weight: float = 0.0
    uniformity_kernel_scale: float = 2.0
    conditional_discriminator: bool = True

    # MI critic configuration
    mi_critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    mi_critic_activation: str = "elu"
    mi_critic_lr: float = 3e-4
    mi_critic_grad_clip_norm: float = 1.0

    # Discriminator critic configuration
    discriminator_critic_hidden_dims: list[int] = field(
        default_factory=lambda: [256, 256]
    )
    discriminator_critic_activation: str = "elu"
    discriminator_critic_lr: float = 3e-4
    discriminator_critic_grad_clip_norm: float = 1.0


@dataclass
class ASERLOptConfig(AMPRLOptConfig):
    """RLOpt config for ASE."""

    ase: ASEConfig = field(default_factory=ASEConfig)


class ASE(LatentSkillMixin, AMP[ASERLOptConfig]):
    """ASE built on top of AMP with explicit latent-command observations."""

    discriminator: ASEDiscriminatorEncoder | None

    def __init__(self, env, config: ASERLOptConfig):
        self.config: ASERLOptConfig = config

        self._init_latent_skills(
            env,
            latent_key=cast(ObsKey, self.config.ase.latent_key),
            latent_dim=int(self.config.ase.latent_dim),
            latent_steps_min=int(self.config.ase.latent_steps_min),
            latent_steps_max=int(self.config.ase.latent_steps_max),
        )

        self.discriminator_critic: TensorDictModule | None = None
        self.discriminator_critic_optim: torch.optim.Optimizer | None = None
        self.mi_critic: TensorDictModule | None = None
        self.mi_critic_optim: torch.optim.Optimizer | None = None
        self._policy_obs_keys: list[ObsKey] = []
        self._policy_obs_feature_ndims: dict[ObsKey, int] = {}
        self._style_critic_in_keys: list[ObsKey] = []
        self._mi_critic_in_keys: list[ObsKey] = []
        self._command_source = self._normalize_command_source(
            self.config.ase.command_source
        )

        super().__init__(env, config)

        self._policy_obs_keys = list(self.config.policy.get_input_keys())
        self._policy_obs_feature_ndims = {
            key: self.observation_feature_rank(key) for key in self._policy_obs_keys
        }
        self._style_critic_in_keys = [*self._disc_obs_keys, self._latent_key]
        self._mi_critic_in_keys = [*self._disc_obs_keys, self._latent_key]

        self.log.info(
            "Initialized ASE (latent_dim=%d, latent_key=%s, command_source=%s, disc_keys=%s)",
            self._latent_dim,
            self._latent_key,
            self._command_source,
            self._disc_obs_keys,
        )

    @staticmethod
    def _normalize_command_source(command_source: str) -> str:
        normalized = str(command_source).strip().lower()
        valid_sources = {"random"}
        if normalized not in valid_sources:
            msg = (
                f"Unsupported ASE command_source={command_source!r}. "
                f"Expected one of {sorted(valid_sources)}."
            )
            raise ValueError(msg)
        return normalized

    def _discriminator_condition_dim(self) -> int:
        return self._latent_dim

    def _discriminator_condition_required_keys(self) -> list[BatchKey]:
        return [cast(BatchKey, self._latent_key)]

    def _set_optimizers(
        self,
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_kwargs: dict[str, Any],
    ) -> list[torch.optim.Optimizer]:
        optimizers = PPO._set_optimizers(self, optimizer_cls, optimizer_kwargs)

        if self.discriminator is None:
            self.discriminator = ASEDiscriminatorEncoder(
                observation_dim=self._disc_obs_total_dim,
                action_dim=(
                    self._action_dim if self.config.gail.discriminator_use_action else 0
                ),
                latent_dim=self._latent_dim,
                hidden_dims=list(self.config.gail.discriminator_hidden_dims),
                activation=self.config.gail.discriminator_activation,
            ).to(self.device)
        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(self.config.gail.discriminator_lr),
        )
        if self._disc_replay_buffer is None:
            self._disc_replay_buffer = self._construct_discriminator_replay_buffer()

        if self.discriminator_critic is None:
            disc_value_mlp = MLP(
                in_features=self._disc_obs_dim + self._latent_dim,
                out_features=1,
                num_cells=list(self.config.ase.discriminator_critic_hidden_dims),
                activation_class=get_activation_class(
                    self.config.ase.discriminator_critic_activation
                ),
                device=self.device,
            )
            self.discriminator_critic = TensorDictModule(
                module=CatInputs(disc_value_mlp),
                in_keys=[*self._disc_obs_keys, self._latent_key],
                out_keys=["style_value_pred"],
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

        self.discriminator_critic_optim = torch.optim.Adam(
            self.discriminator_critic.parameters(),
            lr=float(self.config.ase.discriminator_critic_lr),
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
        if self.discriminator is None:
            return self._sample_unit_latents(
                td.numel(),
                device=self.device,
                dtype=torch.float32,
            )
        obs_features = self._obs_features_from_td(td, detach=False).to(self.device)
        latent = self.discriminator.mi_encode_from_obs(obs_features)
        return latent.detach() if detach else latent

    def _style_reward(self, batch: TensorDict) -> Tensor:
        if not self.config.gail.use_gail_reward or self.discriminator is None:
            return torch.zeros(batch.numel(), device=self.device)

        obs_features = self._obs_features_from_td(batch, detach=True).to(self.device)
        action_features: Tensor | None = None
        if self.config.gail.discriminator_use_action:
            action_features = self._action_features_from_td(batch, detach=True).to(
                self.device
            )
        latents = self._discriminator_condition_from_td(batch, detach=True)
        if latents is None:
            msg = "ASE rollout batch is missing latent commands."
            raise RuntimeError(msg)

        obs_disc, action_disc = self._prepare_disc_obs_action(
            obs_features,
            action_features,
            latents.to(self.device),
            update_stats=False,
            apply_augmentation=False,
        )
        return self._gail_reward(obs_disc, action_disc)

    def _mi_reward(self, obs_features: Tensor, latents: Tensor) -> Tensor:
        if self.discriminator is None:
            return torch.zeros(obs_features.shape[0], device=obs_features.device)
        with torch.no_grad():
            return self.discriminator.compute_mi_reward(
                obs_features,
                latents,
                bool(self.config.ase.mi_hypersphere_reward_shift),
            )

    def _uniformity_loss(self, encodings: Tensor) -> Tensor:
        if encodings.shape[0] <= 1:
            return torch.zeros((), device=encodings.device, dtype=encodings.dtype)
        if encodings.shape[0] > 1024:
            sample_idx = torch.randperm(encodings.shape[0], device=encodings.device)[
                :1024
            ]
            encodings = encodings[sample_idx]
        scale = float(max(self.config.ase.uniformity_kernel_scale, 1.0e-6))
        pairwise_dist = torch.cdist(encodings, encodings, p=2)
        kernel_values = torch.exp(-scale * pairwise_dist.pow(2))
        return torch.log(kernel_values.mean())

    def _sample_td_indices(self, td: TensorDict, batch_size: int) -> TensorDict:
        if td.numel() == batch_size:
            return td
        indices = torch.randint(td.numel(), (batch_size,), device=self.device)
        return cast(TensorDict, td[indices])

    def _critic_value_loss(
        self, pred: Tensor, old_pred: Tensor, target: Tensor
    ) -> Tensor:
        if not self.config.ppo.clip_value:
            return F.mse_loss(pred, target)
        clip_eps = float(self.config.ppo.clip_epsilon)
        old_pred = old_pred.to(self.device)
        unclipped = (pred - target).pow(2)
        clipped_pred = old_pred + torch.clamp(pred - old_pred, -clip_eps, clip_eps)
        clipped = (clipped_pred - target).pow(2)
        return torch.max(unclipped, clipped).mean()

    @staticmethod
    def _align_advantage_shape(base_advantage: Tensor, aux_advantage: Tensor) -> Tensor:
        if (
            base_advantage.ndim == aux_advantage.ndim + 1
            and base_advantage.shape[-1] == 1
        ):
            return aux_advantage.unsqueeze(-1)
        if (
            aux_advantage.ndim == base_advantage.ndim + 1
            and aux_advantage.shape[-1] == 1
        ):
            return aux_advantage.squeeze(-1)
        return aux_advantage

    def _update_discriminator(
        self, rollout_flat: TensorDict, update_idx: int
    ) -> dict[str, float]:
        if update_idx < int(self.config.gail.discriminator_update_warmup_updates):
            return {"discriminator/update_mask": 0.0}
        ratio = max(1, int(self.config.gail.discriminator_optimization_ratio))
        if (update_idx % ratio) != 0:
            return {"discriminator/update_mask": 0.0}

        assert self.discriminator is not None
        assert self.discriminator_optim is not None

        policy_required = self._discriminator_policy_required_keys()
        policy_batch = rollout_flat.reshape(-1).select(*policy_required)
        self._store_discriminator_replay_samples(rollout_flat.reshape(-1))

        stats_accum: dict[str, float] = {"discriminator/update_mask": 1.0}
        performed_steps = 0

        for _ in range(self._discriminator_step_count()):
            expert_td = self._next_expert_batch(
                batch_size=self.config.gail.expert_batch_size
            )
            batch_size = self._discriminator_batch_size(
                policy_batch.numel(),
                expert_td.numel(),
            )
            if batch_size <= 0:
                continue

            agent_td = self._sample_td_indices(policy_batch, batch_size)
            replay_td = self._sample_discriminator_replay_batch(batch_size)
            if replay_td is None:
                replay_td = agent_td
            else:
                replay_td = replay_td.select(*policy_required)
                replay_td = self._sample_td_indices(replay_td, batch_size)
            expert_td_b = self._sample_td_indices(expert_td, batch_size)

            agent_obs = self._obs_features_from_td(agent_td, detach=False).to(
                self.device
            )
            replay_obs = self._obs_features_from_td(replay_td, detach=False).to(
                self.device
            )
            expert_obs = self._obs_features_from_td(expert_td_b, detach=False).to(
                self.device
            )

            agent_action: Tensor | None = None
            replay_action: Tensor | None = None
            expert_action: Tensor | None = None
            if self.config.gail.discriminator_use_action:
                agent_action = self._action_features_from_td(agent_td, detach=False).to(
                    self.device
                )
                replay_action = self._action_features_from_td(
                    replay_td, detach=False
                ).to(self.device)
                expert_action = self._action_features_from_td(
                    expert_td_b, detach=False
                ).to(self.device)

            agent_latents = self._discriminator_condition_from_td(agent_td, detach=True)
            replay_latents = self._discriminator_condition_from_td(
                replay_td, detach=True
            )
            expert_latents = self._discriminator_condition_from_td(
                expert_td_b, detach=False
            )
            if (
                agent_latents is None
                or replay_latents is None
                or expert_latents is None
            ):
                msg = "ASE discriminator step requires latent conditioning."
                raise RuntimeError(msg)
            agent_latents = agent_latents.to(self.device)
            replay_latents = replay_latents.to(self.device)
            expert_latents = expert_latents.detach().to(self.device)

            expert_obs_disc, expert_action_disc = self._prepare_disc_obs_action(
                expert_obs,
                expert_action,
                expert_latents,
                update_stats=True,
                apply_augmentation=True,
            )
            agent_obs_disc, agent_action_disc = self._prepare_disc_obs_action(
                agent_obs,
                agent_action,
                agent_latents,
                update_stats=True,
                apply_augmentation=True,
            )
            replay_obs_disc, replay_action_disc = self._prepare_disc_obs_action(
                replay_obs,
                replay_action,
                replay_latents,
                update_stats=True,
                apply_augmentation=True,
            )

            negative_expert_logits: Tensor | None = None
            if self.config.ase.conditional_discriminator:
                negative_latents = self._sample_unit_latents(
                    batch_size,
                    device=self.device,
                    dtype=expert_obs.dtype,
                )
                negative_expert_obs_disc, negative_expert_action_disc = (
                    self._prepare_disc_obs_action(
                        expert_obs,
                        expert_action,
                        negative_latents,
                        update_stats=True,
                        apply_augmentation=True,
                    )
                )
                negative_expert_logits = self.discriminator.forward_logits(
                    negative_expert_obs_disc,
                    negative_expert_action_disc,
                )

            expert_logits = self.discriminator.forward_logits(
                expert_obs_disc,
                expert_action_disc,
            )
            agent_logits = self.discriminator.forward_logits(
                agent_obs_disc,
                agent_action_disc,
            )
            replay_logits = self.discriminator.forward_logits(
                replay_obs_disc,
                replay_action_disc,
            )

            expert_loss = -F.logsigmoid(expert_logits).mean()
            unlabeled_loss = F.softplus(agent_logits).mean()
            replay_loss = F.softplus(replay_logits).mean()
            if self.config.ase.conditional_discriminator:
                assert negative_expert_logits is not None
                negative_expert_loss = F.softplus(negative_expert_logits).mean()
                neg_loss = 0.5 * (unlabeled_loss + replay_loss + negative_expert_loss)
            else:
                neg_loss = 0.5 * (unlabeled_loss + replay_loss)
            class_loss = 0.5 * (expert_loss + neg_loss)

            expert_obs_gp = expert_obs.detach().requires_grad_(True)
            expert_obs_gp_disc, expert_action_gp_disc = self._prepare_disc_obs_action(
                expert_obs_gp,
                expert_action,
                expert_latents.detach(),
                update_stats=False,
                apply_augmentation=False,
            )
            expert_logits_gp = self.discriminator.forward_logits(
                expert_obs_gp_disc,
                expert_action_gp_disc,
            )
            disc_grad = torch.autograd.grad(
                expert_logits_gp,
                expert_obs_gp,
                grad_outputs=torch.ones_like(expert_logits_gp),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            disc_grad_penalty = disc_grad.pow(2).sum(dim=-1).mean()
            grad_loss = (
                float(self.config.gail.discriminator_grad_penalty_coeff)
                * disc_grad_penalty
            )

            discriminator_l2_total = torch.zeros((), device=self.device)
            if float(self.config.gail.discriminator_weight_decay_coeff) > 0.0:
                discriminator_l2_total = sum(
                    weight.pow(2).sum()
                    for weight in self.discriminator.all_discriminator_weights()
                )
            discriminator_l2_loss = discriminator_l2_total * float(
                self.config.gail.discriminator_weight_decay_coeff
            )

            logit_l2_total = torch.zeros((), device=self.device)
            if float(self.config.gail.discriminator_logit_reg_coeff) > 0.0:
                logit_l2_total = sum(
                    weight.pow(2).sum() for weight in self.discriminator.logit_weights()
                )
            logit_l2_loss = logit_l2_total * float(
                self.config.gail.discriminator_logit_reg_coeff
            )

            agent_obs_mi = agent_obs.detach().requires_grad_(
                float(self.config.ase.mi_enc_grad_penalty) > 0.0
            )
            agent_obs_mi_disc, _ = self._prepare_disc_obs_action(
                agent_obs_mi,
                agent_action,
                agent_latents.detach(),
                update_stats=False,
                apply_augmentation=False,
            )
            mi_enc_pred_agent = self.discriminator.mi_encode(agent_obs_mi_disc)
            mi_enc_pred_expert = self.discriminator.mi_encode(expert_obs_disc.detach())

            mi_enc_err = self.discriminator.calc_von_mises_fisher_enc_error(
                mi_enc_pred_agent,
                agent_latents.detach(),
            )
            mi_enc_loss = mi_enc_err.mean()

            all_encodings = torch.cat([mi_enc_pred_agent, mi_enc_pred_expert], dim=0)
            uniformity_loss = self._uniformity_loss(all_encodings)

            encoder_l2_total = torch.zeros((), device=self.device)
            if float(self.config.ase.mi_enc_weight_decay) > 0.0:
                encoder_l2_total = sum(
                    weight.pow(2).sum() for weight in self.discriminator.enc_weights()
                )
            encoder_l2_loss = encoder_l2_total * float(
                self.config.ase.mi_enc_weight_decay
            )

            encoder_grad_penalty = torch.zeros((), device=self.device)
            if float(self.config.ase.mi_enc_grad_penalty) > 0.0:
                mi_enc_grad = torch.autograd.grad(
                    mi_enc_err,
                    agent_obs_mi,
                    grad_outputs=torch.ones_like(mi_enc_err),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                encoder_grad_penalty = mi_enc_grad.pow(2).sum(dim=-1).mean()
            encoder_grad_loss = encoder_grad_penalty * float(
                self.config.ase.mi_enc_grad_penalty
            )

            encoder_loss = (
                mi_enc_loss
                + uniformity_loss * float(self.config.ase.latent_uniformity_weight)
                + encoder_l2_loss
                + encoder_grad_loss
            )

            total_loss = (
                class_loss
                + grad_loss
                + discriminator_l2_loss
                + logit_l2_loss
                + encoder_loss
            )

            self.discriminator_optim.zero_grad(set_to_none=True)
            total_loss.backward()
            clip_grad_norm_(
                self.discriminator.parameters(),
                float(self.config.gail.discriminator_grad_clip_norm),
            )
            self.discriminator_optim.step()
            performed_steps += 1

            agent_acc = (agent_logits.squeeze(-1) < 0.0).float().mean()
            replay_acc = (replay_logits.squeeze(-1) < 0.0).float().mean()
            neg_acc = 0.5 * (agent_acc + replay_acc)
            negative_logit_mean = 0.5 * (
                agent_logits.detach().mean() + replay_logits.detach().mean()
            )
            if self.config.ase.conditional_discriminator:
                assert negative_expert_logits is not None
                negative_logit_mean = 0.5 * (
                    negative_logit_mean + negative_expert_logits.detach().mean()
                )

            step_stats = {
                "discriminator/loss": float(total_loss.detach().item()),
                "discriminator/pos_acc": float(
                    (expert_logits.squeeze(-1) > 0.0).float().mean().item()
                ),
                "discriminator/agent_acc": float(agent_acc.item()),
                "discriminator/replay_acc": float(replay_acc.item()),
                "discriminator/neg_acc": float(neg_acc.item()),
                "discriminator/grad_penalty": float(disc_grad_penalty.detach().item()),
                "discriminator/grad_loss": float(grad_loss.detach().item()),
                "discriminator/class_loss": float(class_loss.detach().item()),
                "discriminator/l2_logit_total": float(logit_l2_total.detach().item()),
                "discriminator/l2_logit_loss": float(logit_l2_loss.detach().item()),
                "discriminator/l2_total": float(discriminator_l2_total.detach().item()),
                "discriminator/l2_loss": float(discriminator_l2_loss.detach().item()),
                "discriminator/expert_logit_mean": float(
                    expert_logits.detach().mean().item()
                ),
                "discriminator/agent_logit_mean": float(
                    agent_logits.detach().mean().item()
                ),
                "discriminator/replay_logit_mean": float(
                    replay_logits.detach().mean().item()
                ),
                "discriminator/negative_logit_mean": float(
                    negative_logit_mean.detach().item()
                ),
                "encoder/loss": float(encoder_loss.detach().item()),
                "encoder/mi_enc_loss": float(mi_enc_loss.detach().item()),
                "encoder/uniformity_loss": float(uniformity_loss.detach().item()),
                "encoder/l2_loss": float(encoder_l2_loss.detach().item()),
                "encoder/grad_penalty": float(encoder_grad_loss.detach().item()),
                "encoder/similarity_mean": float((-mi_enc_err.detach()).mean().item()),
            }
            if self.config.ase.conditional_discriminator:
                assert negative_expert_logits is not None
                step_stats["discriminator/negative_expert_logit_mean"] = float(
                    negative_expert_logits.detach().mean().item()
                )

            for key, value in step_stats.items():
                stats_accum[key] = stats_accum.get(key, 0.0) + value

        if performed_steps <= 0:
            return {"discriminator/update_mask": 0.0}

        stats_accum["discriminator/updates_performed"] = float(performed_steps)
        for key in list(stats_accum.keys()):
            if key in {"discriminator/update_mask", "discriminator/updates_performed"}:
                continue
            stats_accum[key] = stats_accum[key] / float(performed_steps)
        return stats_accum

    def _attach_style_targets(self, rollout: TensorDict) -> None:
        if (
            self.discriminator_critic is None
            or not self.config.gail.use_gail_reward
            or self.discriminator is None
        ):
            return

        flat_rollout = rollout.reshape(-1)
        next_td = cast(TensorDict, flat_rollout.get("next"))

        with torch.no_grad():
            style_value = self.discriminator_critic(
                flat_rollout.select(*self._style_critic_in_keys).clone()
            )["style_value_pred"].reshape(*rollout.batch_size)
            next_style_value = self.discriminator_critic(
                next_td.select(*self._style_critic_in_keys).clone()
            )["style_value_pred"].reshape(*rollout.batch_size)
            style_reward = self._style_reward(next_td).reshape(*rollout.batch_size)

        done = rollout["next", "done"]
        if done.ndim == style_reward.ndim + 1 and done.shape[-1] == 1:
            done = done.squeeze(-1)
        style_advantages, style_returns = generalized_advantage_estimate(
            style_reward,
            style_value.squeeze(-1),
            next_style_value.squeeze(-1),
            done,
            gamma=float(self.config.loss.gamma),
            gae_lambda=float(self.config.ppo.gae_lambda),
        )
        rollout.set("style_reward", style_reward)
        rollout.set("style_value", style_value.squeeze(-1))
        rollout.set("style_advantage", style_advantages)
        rollout.set("style_returns", style_returns)

    def _attach_mi_targets(self, rollout: TensorDict) -> None:
        if self.mi_critic is None or self.discriminator is None:
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

            obs_features = self._obs_features_from_td(next_td, detach=True).to(
                self.device
            )
            latents = self._discriminator_condition_from_td(next_td, detach=True)
            if latents is None:
                msg = "ASE rollout batch is missing latent commands."
                raise RuntimeError(msg)
            mi_reward = self._mi_reward(obs_features, latents.to(self.device)).reshape(
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

    def _diversity_loss(self, batch: TensorDict) -> Tensor:
        if float(self.config.ase.diversity_bonus) <= 0.0 or batch.numel() <= 1:
            return torch.zeros((), device=self.device)
        if "loc" not in batch.keys(True) or "scale" not in batch.keys(True):
            msg = "ASE diversity loss requires PPO rollout loc/scale keys."
            raise KeyError(msg)

        old_latents = self._latent_condition_from_td(batch, detach=True)
        if old_latents is None:
            msg = "ASE minibatch is missing latent commands."
            raise RuntimeError(msg)
        old_latents = old_latents.to(self.device)

        policy_operator = self.actor_critic.get_policy_operator()
        old_dist = policy_operator.build_dist_from_params(
            batch.select("loc", "scale").clone()
        )
        old_mean_action = self._policy_action_from_dist(old_dist)
        if old_mean_action is None:
            msg = "ASE diversity loss could not recover the old policy mean action."
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
        new_dist = policy_operator.get_dist(policy_td)
        new_mean_action = self._policy_action_from_dist(new_dist)
        if new_mean_action is None:
            msg = "ASE diversity loss could not recover the new policy mean action."
            raise RuntimeError(msg)
        new_mean_action = self._clip_policy_action(new_mean_action)

        action_delta = (new_mean_action - old_mean_action).pow(2).mean(dim=-1)
        latent_delta = 0.5 - 0.5 * (new_latents * old_latents).sum(dim=-1)
        diversity_bonus = action_delta / (latent_delta + 1.0e-5)
        return (float(self.config.ase.diversity_tar) - diversity_bonus).pow(2).mean()

    def _extra_actor_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        if float(self.config.ase.diversity_bonus) <= 0.0:
            return super()._extra_actor_loss(batch)
        extra_loss, extra_metrics = super()._extra_actor_loss(batch)
        diversity_loss = self._diversity_loss(batch)
        extra_metrics["loss_diversity"] = diversity_loss.detach()
        weighted_loss = diversity_loss * float(self.config.ase.diversity_bonus)
        return extra_loss + weighted_loss, extra_metrics

    def pre_iteration_compute(self, rollout: TensorDict) -> TensorDict:
        with torch.no_grad():
            rollout = self.adv_module(rollout)
            adv = cast(Tensor, rollout.get("advantage"))
            rollout.set("advantage", adv * float(self.config.ase.task_reward_w))

            self._attach_style_targets(rollout)
            self._attach_mi_targets(rollout)

            advantage = rollout["advantage"]
            style_reward_w = self.config.ase.discriminator_reward_w
            if style_reward_w > 0.0 and "style_advantage" in rollout.keys(True):
                style_adv = rollout["style_advantage"].unsqueeze(-1)
                advantage = advantage + style_adv * style_reward_w

            mi_reward_w = self.config.ase.mi_reward_w
            if mi_reward_w > 0.0 and "mi_advantage" in rollout.keys(True):
                mi_adv = rollout["mi_advantage"].unsqueeze(-1)
                advantage = advantage + mi_adv * mi_reward_w
            rollout.set("advantage", advantage)

            if getattr(self.config.compile, "compile", False):
                rollout = rollout.clone()

        self.data_buffer.extend(rollout.reshape(-1))
        return rollout

    def _update_discriminator_critic_batch(
        self, batch: TensorDict
    ) -> dict[str, Tensor]:
        if (
            self.discriminator_critic is None
            or self.discriminator_critic_optim is None
            or "style_returns" not in batch
        ):
            return {}

        pred_td = self.discriminator_critic(
            batch.select(*self._style_critic_in_keys).clone()
        )
        pred = cast(Tensor, pred_td.get("style_value_pred")).squeeze(-1)
        target = cast(Tensor, batch.get("style_returns")).to(self.device)
        old_pred = cast(Tensor, batch.get("style_value")).to(self.device)
        loss = self._critic_value_loss(pred, old_pred, target)

        self.discriminator_critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        if float(self.config.ase.discriminator_critic_grad_clip_norm) > 0.0:
            clip_grad_norm_(
                self.discriminator_critic.parameters(),
                float(self.config.ase.discriminator_critic_grad_clip_norm),
            )
        self.discriminator_critic_optim.step()
        return {"discriminator_critic_loss": loss.detach()}

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
        old_pred = cast(Tensor, batch.get("mi_value")).to(self.device)
        loss = self._critic_value_loss(pred, old_pred, target)

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
        return [
            *super()._optional_loss_metrics,
            "discriminator_critic_loss",
            "mi_critic_loss",
            "loss_diversity",
        ]

    def prepare(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,  # noqa: ARG002
    ) -> None:
        self._prepare_latent_rollout_batch_for_training(iteration.rollout)

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
        if self.discriminator_critic is not None:
            self.discriminator_critic.train()
        if self.mi_critic is not None:
            self.mi_critic.train()

        with timeit("training"):
            # Pre-iteration compute, attach style and MI targets, and compute advantages.
            iteration.rollout = self.pre_iteration_compute(iteration.rollout)

            # Update discriminator and MI encoders
            rollout_update_idx = self._counter_as_int(metadata.updates_completed)
            discriminator_metrics = self._update_discriminator(
                iteration.rollout.reshape(-1),
                rollout_update_idx,
            )
            if discriminator_metrics:
                iteration.metrics.update(
                    {
                        f"train/{key}": value
                        for key, value in discriminator_metrics.items()
                    }
                )

            # Log style and MI rewards.
            if "style_reward" in iteration.rollout.keys(True):
                iteration.metrics["train/ase/style_reward_mean"] = float(
                    cast(Tensor, iteration.rollout.get("style_reward")).mean().item()
                )
            if "mi_reward" in iteration.rollout.keys(True):
                iteration.metrics["train/ase/mi_reward_mean"] = float(
                    cast(Tensor, iteration.rollout.get("mi_reward")).mean().item()
                )

            # Run PPO epochs over the rollout and update additional critics.
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

                    # Update discriminator critic.
                    disc_critic_stats = self._update_discriminator_critic_batch(batch)
                    for key, value in disc_critic_stats.items():
                        loss.set(key, value)

                    # Update MI critic.
                    mi_critic_stats = self._update_mi_critic_batch(batch)
                    for key, value in mi_critic_stats.items():
                        loss.set(key, value)

                    # Update learning rate scheduler.
                    if self.lr_scheduler and self.lr_scheduler_step == "update":
                        self.lr_scheduler.step()

                    # Compute KL approximation.
                    if kl_context is not None:
                        kl_approx = self._compute_kl_after_update(
                            kl_context, metadata.policy_operator
                        )
                        if kl_approx is not None:
                            loss.set("kl_approx", kl_approx.detach())
                            self._maybe_adjust_lr(kl_approx, self.config.optim)

                    # Select reported loss metrics.
                    losses[epoch_idx, batch_idx] = self._select_reported_loss_metrics(
                        loss
                    )

                # Update learning rate scheduler.
                if self.lr_scheduler and self.lr_scheduler_step == "epoch":
                    self.lr_scheduler.step()

        iteration.learn_time = time.perf_counter() - learn_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():  # type: ignore[attr-defined]
            iteration.metrics[f"train/{key}"] = value.item()  # type: ignore[attr-defined]

    def train(self) -> None:  # type: ignore
        """Train the agent with the shared on-policy rollout-to-update workflow."""
        self.validate_training()
        metadata = self.init_metadata()

        try:
            for iteration_idx in range(metadata.total_iterations):
                # 1) Collect a rollout and create the iteration state.
                iteration = self.collect(metadata, iteration_idx)

                # 2) Let the algorithm reshape rewards or attach extra data.
                self.prepare(iteration, metadata)

                # 3) Run the shared learning phase over the prepared rollout.
                self.iterate(iteration, metadata)

                # 4) Log, refresh collector weights, and checkpoint if needed.
                self.record(iteration, metadata)
        finally:
            metadata.progress_bar.close()  # type: ignore[attr-defined]
            self.collector.shutdown()

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
                        msg = (
                            "All observation tensors passed to predict() must share "
                            "the same batch shape."
                        )
                        raise ValueError(msg)
                    td_data[cast(BatchKey, key)] = value
            else:
                if len(input_keys) != 1:
                    msg = (
                        "predict() received a single tensor observation, but the "
                        "policy expects keyed observations."
                    )
                    raise ValueError(msg)
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
            td = TensorDict(td_data, batch_size=list(batch_shape), device=self.device)
            self._inject_latent_command(td)
            td = policy_op(td)
            return td.get("action")

    def _gail_state_dict(self) -> dict[str, Any]:
        state = super()._gail_state_dict()
        state.update(
            {
                "discriminator_critic": (
                    self.discriminator_critic.state_dict()
                    if self.discriminator_critic is not None
                    else None
                ),
                "discriminator_critic_optim": (
                    self.discriminator_critic_optim.state_dict()
                    if self.discriminator_critic_optim is not None
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
        if (
            self.discriminator_critic is not None
            and state.get("discriminator_critic") is not None
        ):
            self.discriminator_critic.load_state_dict(
                cast(dict[str, Any], state["discriminator_critic"])
            )
        if (
            self.discriminator_critic_optim is not None
            and state.get("discriminator_critic_optim") is not None
        ):
            self.discriminator_critic_optim.load_state_dict(
                cast(dict[str, Any], state["discriminator_critic_optim"])
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
