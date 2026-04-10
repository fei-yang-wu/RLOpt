"""Pluggable latent-learning methods for latent-conditioned imitation agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from rlopt.agent.imitation.latent_skill import LatentEncoder
from rlopt.config_utils import BatchKey, dedupe_keys
from rlopt.utils import get_activation_class

if TYPE_CHECKING:
    from rlopt.agent.ipmd.ipmd import IPMD


class LatentDecoder(nn.Module):
    """Simple MLP decoder used by deterministic latent autoencoders."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str,
    ) -> None:
        super().__init__()
        act_cls = get_activation_class(activation)

        layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(act_cls())
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        return self.network(latent)


class GaussianLatentModel(nn.Module):
    """Diagonal-Gaussian latent model with configurable MLP backbone."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        activation: str,
        *,
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        act_cls = get_activation_class(activation)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(act_cls())
            prev_dim = int(hidden_dim)

        self.backbone = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()
        self.mean_head = nn.Linear(prev_dim, latent_dim)
        self.log_std_head = nn.Linear(prev_dim, latent_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(features)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std


class DeterministicLatentModel(nn.Module):
    """Simple MLP latent embedding without distributional outputs."""

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

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class FeatureProbe(nn.Module):
    """Simple MLP probe for detached latent diagnostics."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class BaseLatentLearner:
    """Interface for latent learning modules used by IPMD."""

    uses_aux_reward: bool = False
    uses_joint_policy_loss: bool = False

    def __init__(self) -> None:
        self.agent: IPMD | None = None

    def initialize(self, agent: IPMD) -> None:
        self.agent = agent

    def required_patch_obs_keys(self) -> list[BatchKey]:
        return []

    def required_expert_batch_keys(self) -> list[BatchKey]:
        return []

    def joint_parameters(self) -> list[nn.Parameter]:
        return []

    def infer_current_reference_latents(
        self,
        *,
        env_ids: Tensor | None = None,
    ) -> Tensor | None:
        del env_ids
        return None

    def infer_expert_latents(
        self,
        expert_batch: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        del expert_batch, detach
        msg = "Active latent learner does not support expert latent inference."
        raise RuntimeError(msg)

    def infer_batch_latents(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        del batch, detach, context
        return None

    def sample_expert_prior_latents(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        del batch_size, device, dtype
        return None

    def update(self, rollout_flat: TensorDict) -> dict[str, float]:
        del rollout_flat
        return {}

    def joint_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        device = getattr(self.agent, "device", batch.device)
        return torch.zeros((), device=device), {}

    def aux_reward_from_rollout(self, rollout_flat: TensorDict) -> Tensor | None:
        del rollout_flat
        return None


class ExpertMLEPosteriorLatentLearner(BaseLatentLearner):
    """Legacy latent learner that fits q(z | transition) through expert-action MLE."""

    uses_aux_reward = True

    def __init__(self) -> None:
        super().__init__()
        self.encoder: LatentEncoder | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def initialize(self, agent: IPMD) -> None:
        super().initialize(agent)
        self._ensure_encoder(int(agent._mi_obs_dim))

    def _ensure_encoder(self, input_dim: int) -> None:
        assert self.agent is not None
        if self.encoder is None:
            self.encoder = LatentEncoder(
                input_dim=input_dim,
                latent_dim=self.agent._latent_dim,
                hidden_dims=list(self.agent.config.ipmd.mi_encoder_hidden_dims),
                activation=self.agent.config.ipmd.mi_encoder_activation,
            ).to(self.agent.device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.encoder.parameters(),
                lr=float(self.agent.config.ipmd.mi_encoder_lr),
            )
        self.agent.mi_encoder = self.encoder
        self.agent.mi_encoder_optim = self.optimizer

    def _reference_obs_keys(self) -> list[BatchKey]:
        assert self.agent is not None
        if not self.agent._lit_use_s or self.agent._lit_use_a or self.agent._lit_use_sn:
            return []
        return list(self.agent._reward_obs_keys)

    def _inference_encoder(self) -> nn.Module:
        assert self.agent is not None
        if self.agent.mi_encoder is not None:
            return self.agent.mi_encoder
        if self.encoder is None:
            msg = "Legacy latent learner encoder has not been initialized."
            raise RuntimeError(msg)
        return self.encoder

    def required_expert_batch_keys(self) -> list[BatchKey]:
        assert self.agent is not None
        required: list[BatchKey] = []
        required.extend(self.agent._latent_encoder_required_keys())
        required.extend(self.agent._policy_obs_keys_without_latent)
        required.append("action")
        return dedupe_keys(required)

    def infer_current_reference_latents(
        self,
        *,
        env_ids: Tensor | None = None,
    ) -> Tensor | None:
        assert self.agent is not None
        obs_keys = self._reference_obs_keys()
        if len(obs_keys) == 0 or self.agent._current_reference_obs_getter is None:
            return None
        reference_obs = self.agent._current_reference_obs_getter(
            required_keys=obs_keys,
            env_ids=env_ids,
        )
        if reference_obs is None or reference_obs.numel() == 0:
            return None
        obs_features = self.agent._obs_features_from_td(
            reference_obs,
            cast(list, obs_keys),
            next_obs=False,
            detach=True,
        ).to(self.agent.device)
        self._ensure_encoder(int(obs_features.shape[-1]))
        with torch.no_grad():
            return self._inference_encoder()(obs_features)

    def infer_expert_latents(
        self,
        expert_batch: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        assert self.agent is not None
        latent = self.agent._latent_condition_from_td(expert_batch, detach=detach)
        if latent is not None:
            return latent.to(self.agent.device)
        obs_features = self.agent._latent_encoder_features_from_td(
            expert_batch,
            detach=False,
            context="expert latent batch",
        ).to(self.agent.device)
        self._ensure_encoder(int(obs_features.shape[-1]))
        latent = self._inference_encoder()(obs_features)
        return latent.detach() if detach else latent

    def infer_batch_latents(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        assert self.agent is not None
        obs_features = self.agent._latent_encoder_features_from_td(
            batch,
            detach=detach,
            context=context,
        ).to(self.agent.device)
        if self.encoder is None and self.agent.mi_encoder is None:
            self._ensure_encoder(int(obs_features.shape[-1]))
        encoder = self._inference_encoder()
        if detach:
            with torch.no_grad():
                return encoder(obs_features)
        return encoder(obs_features)

    def sample_expert_prior_latents(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        assert self.agent is not None
        obs_keys = self._reference_obs_keys()
        if len(obs_keys) == 0:
            return None
        expert_batch = self.agent._next_expert_batch(
            batch_size,
            required_keys=obs_keys,
        )
        obs_features = self.agent._obs_features_from_td(
            expert_batch,
            cast(list, obs_keys),
            next_obs=False,
            detach=True,
        ).to(self.agent.device)
        self._ensure_encoder(int(obs_features.shape[-1]))
        with torch.no_grad():
            latents = self._inference_encoder()(obs_features)
        return latents.to(device=device, dtype=dtype)

    def update(self, rollout_flat: TensorDict) -> dict[str, float]:
        del rollout_flat
        assert self.agent is not None
        if float(self.agent.config.ipmd.mi_loss_coeff) <= 0.0:
            return {}
        expert_td = self.agent._next_expert_batch(
            required_keys=self.required_expert_batch_keys()
        )
        expert_obs = self.agent._latent_encoder_features_from_td(
            expert_td,
            detach=False,
            context="expert latent mle",
        ).to(self.agent.device)
        if expert_obs.shape[0] == 0:
            return {}
        self._ensure_encoder(int(expert_obs.shape[-1]))
        assert self.encoder is not None
        assert self.optimizer is not None

        expert_action = self.agent._expert_action_from_td(expert_td)
        if expert_action is None:
            msg = "Expert batch is missing expert actions required for latent MLE."
            raise RuntimeError(msg)
        expert_action = expert_action.to(self.agent.device)

        latent_pred = self.encoder(expert_obs)
        policy_td = self.agent._expert_policy_td_with_latents(expert_td, latent_pred)
        dist = self.agent._policy_operator.get_dist(policy_td)
        log_prob = dist.log_prob(expert_action)
        log_prob = self.agent._reduce_log_prob(log_prob, expert_action)
        mle_loss = -log_prob.mean()

        grad_penalty = torch.zeros((), device=self.agent.device)
        if float(self.agent.config.ipmd.mi_grad_penalty_coeff) > 0.0:
            obs_req = expert_obs.detach().requires_grad_(True)
            latent_req = self.encoder(obs_req)
            policy_td = self.agent._expert_policy_td_with_latents(expert_td, latent_req)
            dist = self.agent._policy_operator.get_dist(policy_td)
            log_prob_req = dist.log_prob(expert_action)
            log_prob_req = self.agent._reduce_log_prob(log_prob_req, expert_action)
            grads = torch.autograd.grad(
                outputs=log_prob_req.sum(),
                inputs=obs_req,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_penalty = grads.pow(2).sum(dim=-1).mean()

        weight_decay = torch.zeros((), device=self.agent.device)
        if float(self.agent.config.ipmd.mi_weight_decay_coeff) > 0.0:
            for param in self.encoder.parameters():
                if param.ndim >= 2:
                    weight_decay = weight_decay + param.pow(2).mean()

        uniformity = self.agent._latent_uniformity(latent_pred)
        total_loss = (
            float(self.agent.config.ipmd.mi_loss_coeff) * mle_loss
            + float(self.agent.config.ipmd.mi_grad_penalty_coeff) * grad_penalty
            + float(self.agent.config.ipmd.mi_weight_decay_coeff) * weight_decay
            + float(self.agent.config.ipmd.latent_uniformity_coeff) * uniformity
        )

        self.optimizer.zero_grad(set_to_none=True)
        self.agent.optim.zero_grad(set_to_none=True)
        total_loss.backward()
        if float(self.agent.config.ipmd.mi_grad_clip_norm) > 0.0:
            clip_grad_norm_(
                self.encoder.parameters(),
                float(self.agent.config.ipmd.mi_grad_clip_norm),
            )
        self.optimizer.step()
        self.agent.optim.zero_grad(set_to_none=True)

        return {
            "ipmd/mi_total_loss": float(total_loss.detach().item()),
            "ipmd/mi_loss": float(mle_loss.detach().item()),
            "ipmd/mi_expert_log_prob_mean": float(log_prob.detach().mean().item()),
            "ipmd/mi_grad_penalty": float(grad_penalty.detach().item()),
            "ipmd/mi_weight_decay": float(weight_decay.detach().item()),
            "ipmd/latent_uniformity": float(uniformity.detach().item()),
        }

    def aux_reward_from_rollout(self, rollout_flat: TensorDict) -> Tensor | None:
        assert self.agent is not None
        if self.encoder is None and self.agent.mi_encoder is None:
            return None
        obs_features = self.agent._latent_encoder_features_from_td(
            rollout_flat,
            detach=True,
            context="rollout latent batch",
        ).to(self.agent.device)
        latents = self.agent._rollout_latents_from_td(rollout_flat, detach=True)
        with torch.no_grad():
            latent_pred = self._inference_encoder()(obs_features)
            score = (latent_pred * latents).sum(dim=-1)
        if self.agent.config.ipmd.mi_hypersphere_reward_shift:
            return (score + 1.0) / 2.0
        return score.clamp_min(0.0)


class PatchAutoencoderLatentLearner(BaseLatentLearner):
    """Reference-patch autoencoder that embeds motion windows into latent commands."""

    uses_aux_reward = False

    def __init__(self) -> None:
        super().__init__()
        self.encoder: LatentEncoder | None = None
        self.decoder: LatentDecoder | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def _config(self):
        assert self.agent is not None
        return self.agent.config.ipmd.latent_learning

    def _posterior_input_keys(self) -> list[BatchKey]:
        assert self.agent is not None
        return list(self.agent._posterior_obs_keys)

    def required_expert_batch_keys(self) -> list[BatchKey]:
        return self._posterior_input_keys()

    def _patch_features_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor:
        assert self.agent is not None
        required_keys = self._posterior_input_keys()
        self.agent._require_batch_keys(td, required_keys, context=context)
        parts: list[Tensor] = []
        for key in required_keys:
            value = cast(Tensor, td.get(key)).to(self.agent.device)
            value = value.detach() if detach else value
            parts.append(value.reshape(value.shape[0], -1))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _ensure_modules(self, input_dim: int) -> None:
        assert self.agent is not None
        cfg = self._config()
        if self.encoder is None:
            self.encoder = LatentEncoder(
                input_dim=input_dim,
                latent_dim=self.agent._latent_dim,
                hidden_dims=list(cfg.encoder_hidden_dims),
                activation=cfg.encoder_activation,
            ).to(self.agent.device)
        if self.decoder is None:
            self.decoder = LatentDecoder(
                latent_dim=self.agent._latent_dim,
                output_dim=input_dim,
                hidden_dims=list(cfg.decoder_hidden_dims),
                activation=cfg.decoder_activation,
            ).to(self.agent.device)
        if self.optimizer is None:
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            self.optimizer = torch.optim.Adam(params, lr=float(cfg.lr))
        self.agent.mi_encoder = self.encoder
        self.agent.mi_encoder_optim = self.optimizer

    def infer_batch_latents(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        patch_features = self._patch_features_from_td(
            batch,
            detach=detach,
            context=context,
        )
        self._ensure_modules(int(patch_features.shape[-1]))
        assert self.encoder is not None
        if detach:
            with torch.no_grad():
                return self.encoder(patch_features)
        return self.encoder(patch_features)

    def infer_expert_latents(
        self,
        expert_batch: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        assert self.agent is not None
        latent = self.agent._latent_condition_from_td(expert_batch, detach=detach)
        if latent is not None:
            return latent.to(self.agent.device)
        patch_features = self._patch_features_from_td(
            expert_batch,
            detach=False,
            context="expert patch batch",
        )
        self._ensure_modules(int(patch_features.shape[-1]))
        assert self.encoder is not None
        latent = self.encoder(patch_features)
        return latent.detach() if detach else latent

    def sample_expert_prior_latents(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        assert self.agent is not None
        expert_batch = self.agent._next_expert_batch(
            batch_size,
            required_keys=self.required_expert_batch_keys(),
        )
        patch_features = self._patch_features_from_td(
            expert_batch,
            detach=True,
            context="expert patch prior",
        )
        self._ensure_modules(int(patch_features.shape[-1]))
        assert self.encoder is not None
        with torch.no_grad():
            latents = self.encoder(patch_features)
        return latents.to(device=device, dtype=dtype)

    def update(self, rollout_flat: TensorDict) -> dict[str, float]:
        del rollout_flat
        assert self.agent is not None
        cfg = self._config()
        if float(cfg.recon_coeff) <= 0.0:
            return {}
        expert_td = self.agent._next_expert_batch(
            required_keys=self.required_expert_batch_keys()
        )
        patch_features = self._patch_features_from_td(
            expert_td,
            detach=False,
            context="expert patch autoencoder",
        )
        if patch_features.shape[0] == 0:
            return {}
        self._ensure_modules(int(patch_features.shape[-1]))
        assert self.encoder is not None
        assert self.decoder is not None
        assert self.optimizer is not None

        latent_pred = self.encoder(patch_features)
        patch_recon = self.decoder(latent_pred)
        recon_loss = F.mse_loss(patch_recon, patch_features)

        uniformity = torch.zeros((), device=self.agent.device)
        if float(cfg.uniformity_coeff) > 0.0:
            uniformity = self.agent._latent_uniformity(latent_pred)

        weight_decay = torch.zeros((), device=self.agent.device)
        if float(cfg.weight_decay_coeff) > 0.0:
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            for param in params:
                if param.ndim >= 2:
                    weight_decay = weight_decay + param.pow(2).mean()

        total_loss = (
            float(cfg.recon_coeff) * recon_loss
            + float(cfg.uniformity_coeff) * uniformity
            + float(cfg.weight_decay_coeff) * weight_decay
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if float(cfg.grad_clip_norm) > 0.0:
            clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                float(cfg.grad_clip_norm),
            )
        self.optimizer.step()

        return {
            "ipmd/latent_total_loss": float(total_loss.detach().item()),
            "ipmd/latent_recon_loss": float(recon_loss.detach().item()),
            "ipmd/latent_uniformity": float(uniformity.detach().item()),
            "ipmd/latent_weight_decay": float(weight_decay.detach().item()),
        }


class PolicyKLBottleneckLatentLearner(BaseLatentLearner):
    """Variational latent bottleneck trained jointly with the PPO actor."""

    uses_joint_policy_loss = True

    def __init__(self) -> None:
        super().__init__()
        self.posterior: GaussianLatentModel | None = None
        self.prior: GaussianLatentModel | None = None
        self.probe: FeatureProbe | None = None
        self.probe_optimizer: torch.optim.Optimizer | None = None

    def _config(self):
        assert self.agent is not None
        return self.agent.config.ipmd.latent_learning

    def _posterior_input_keys(self) -> list[BatchKey]:
        assert self.agent is not None
        cfg = self._config()
        if cfg.posterior_input_keys:
            return dedupe_keys(list(cfg.posterior_input_keys))
        return dedupe_keys(
            [*self.agent._policy_obs_keys_without_latent, *self.agent._reward_obs_keys]
        )

    def _prior_input_keys(self) -> list[BatchKey]:
        assert self.agent is not None
        cfg = self._config()
        if cfg.prior_input_keys:
            return dedupe_keys(list(cfg.prior_input_keys))
        return dedupe_keys(list(self.agent._policy_obs_keys_without_latent))

    def _probe_target_keys(self) -> list[BatchKey]:
        assert self.agent is not None
        cfg = self._config()
        if cfg.probe_target_keys:
            return dedupe_keys(list(cfg.probe_target_keys))
        return dedupe_keys(list(self.agent._reward_obs_keys))

    def _probe_input_keys(self) -> list[BatchKey]:
        cfg = self._config()
        if not bool(cfg.probe_condition_on_state):
            return []
        return self._prior_input_keys()

    def _ensure_probe(self) -> None:
        assert self.agent is not None
        cfg = self._config()
        if not bool(cfg.probe_enabled):
            return
        probe_input_dim = self.agent._latent_dim + sum(
            self.agent._obs_feature_dims[key] for key in self._probe_input_keys()
        )
        probe_output_dim = sum(
            self.agent._obs_feature_dims[key] for key in self._probe_target_keys()
        )
        if self.probe is None:
            self.probe = FeatureProbe(
                input_dim=probe_input_dim,
                output_dim=probe_output_dim,
                hidden_dims=list(cfg.probe_hidden_dims),
                activation=cfg.probe_activation,
            ).to(self.agent.device)
        if self.probe_optimizer is None:
            self.probe_optimizer = torch.optim.Adam(
                self.probe.parameters(),
                lr=float(cfg.probe_lr),
            )

    def _features_from_td(
        self,
        td: TensorDict,
        keys: list[BatchKey],
        *,
        detach: bool,
        context: str,
    ) -> Tensor:
        assert self.agent is not None
        self.agent._require_batch_keys(td, keys, context=context)
        return self.agent._obs_features_from_td(
            td,
            cast(list, keys),
            next_obs=False,
            detach=detach,
        ).to(self.agent.device)

    def _ensure_modules(self) -> None:
        assert self.agent is not None
        cfg = self._config()
        posterior_input_dim = sum(
            self.agent._obs_feature_dims[key] for key in self._posterior_input_keys()
        )
        prior_input_dim = sum(
            self.agent._obs_feature_dims[key] for key in self._prior_input_keys()
        )
        if self.posterior is None:
            self.posterior = GaussianLatentModel(
                input_dim=posterior_input_dim,
                latent_dim=self.agent._latent_dim,
                hidden_dims=list(cfg.encoder_hidden_dims),
                activation=cfg.encoder_activation,
                log_std_min=float(cfg.log_std_min),
                log_std_max=float(cfg.log_std_max),
            ).to(self.agent.device)
        if self.prior is None:
            self.prior = GaussianLatentModel(
                input_dim=prior_input_dim,
                latent_dim=self.agent._latent_dim,
                hidden_dims=list(cfg.prior_hidden_dims),
                activation=cfg.prior_activation,
                log_std_min=float(cfg.log_std_min),
                log_std_max=float(cfg.log_std_max),
            ).to(self.agent.device)
        self._ensure_probe()

    def initialize(self, agent: IPMD) -> None:
        super().initialize(agent)
        self._ensure_modules()

    def joint_parameters(self) -> list[nn.Parameter]:
        self._ensure_modules()
        params: list[nn.Parameter] = []
        if self.posterior is not None:
            params.extend(list(self.posterior.parameters()))
        if self.prior is not None:
            params.extend(list(self.prior.parameters()))
        return params

    def infer_batch_latents(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        self._ensure_modules()
        posterior_input = self._features_from_td(
            batch,
            self._posterior_input_keys(),
            detach=detach,
            context=context,
        )
        assert self.posterior is not None
        latent_mean, _ = self.posterior(posterior_input)
        return latent_mean.detach() if detach else latent_mean

    def infer_expert_latents(
        self,
        expert_batch: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        latent = self.infer_batch_latents(
            expert_batch,
            detach=detach,
            context="expert latent batch",
        )
        if latent is None:
            msg = "Policy KL latent learner could not infer expert latents."
            raise RuntimeError(msg)
        return latent

    def joint_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        assert self.agent is not None
        self._ensure_modules()
        cfg = self._config()
        posterior_input = self._features_from_td(
            batch,
            self._posterior_input_keys(),
            detach=False,
            context="policy latent posterior batch",
        )
        prior_input = self._features_from_td(
            batch,
            self._prior_input_keys(),
            detach=False,
            context="policy latent prior batch",
        )
        assert self.posterior is not None
        assert self.prior is not None
        q_mean, q_log_std = self.posterior(posterior_input)
        p_mean, p_log_std = self.prior(prior_input)
        q_var = torch.exp(2.0 * q_log_std)
        p_var = torch.exp(2.0 * p_log_std)
        kl = 0.5 * (
            2.0 * (p_log_std - q_log_std)
            + (q_var + (q_mean - p_mean).pow(2)) / p_var
            - 1.0
        )
        kl_loss = kl.sum(dim=-1).mean()
        weighted_loss = kl_loss * float(cfg.kl_coeff)
        return weighted_loss, {
            "loss_latent_kl": kl_loss.detach(),
            "latent_posterior_std": q_log_std.exp().mean().detach(),
            "latent_prior_std": p_log_std.exp().mean().detach(),
            "latent_posterior_mean_abs": q_mean.abs().mean().detach(),
        }

    def update(self, rollout_flat: TensorDict) -> dict[str, float]:
        assert self.agent is not None
        self._ensure_modules()
        cfg = self._config()
        if not bool(cfg.probe_enabled):
            return {}
        if self.probe is None or self.probe_optimizer is None:
            return {}
        if rollout_flat.numel() <= 0:
            return {}

        probe_batch = rollout_flat
        probe_batch_size = cfg.probe_batch_size
        if (
            probe_batch_size is not None
            and int(probe_batch_size) > 0
            and probe_batch.numel() > int(probe_batch_size)
        ):
            sample_idx = torch.randperm(probe_batch.numel(), device=self.agent.device)[
                : int(probe_batch_size)
            ]
            probe_batch = cast(TensorDict, probe_batch[sample_idx])

        with torch.no_grad():
            latents = self.infer_batch_latents(
                probe_batch,
                detach=True,
                context="latent probe batch",
            )
        if latents is None:
            return {}

        probe_inputs: list[Tensor] = [latents.to(self.agent.device)]
        probe_state_keys = self._probe_input_keys()
        if len(probe_state_keys) > 0:
            probe_inputs.append(
                self._features_from_td(
                    probe_batch,
                    probe_state_keys,
                    detach=True,
                    context="latent probe state batch",
                )
            )
        probe_input = (
            probe_inputs[0]
            if len(probe_inputs) == 1
            else torch.cat(probe_inputs, dim=-1)
        )
        target = self._features_from_td(
            probe_batch,
            self._probe_target_keys(),
            detach=True,
            context="latent probe target batch",
        )

        pred = self.probe(probe_input)
        probe_loss = F.mse_loss(pred, target)
        probe_rmse = torch.sqrt(probe_loss)
        probe_mae = (pred.detach() - target).abs().mean()

        self.probe_optimizer.zero_grad(set_to_none=True)
        probe_loss.backward()
        if float(cfg.probe_grad_clip_norm) > 0.0:
            clip_grad_norm_(self.probe.parameters(), float(cfg.probe_grad_clip_norm))
        self.probe_optimizer.step()

        return {
            "ipmd/latent_probe_loss": float(probe_loss.detach().item()),
            "ipmd/latent_probe_rmse": float(probe_rmse.detach().item()),
            "ipmd/latent_probe_mae": float(probe_mae.detach().item()),
            "ipmd/latent_probe_target_std": float(target.detach().std().item()),
            "ipmd/latent_probe_pred_std": float(pred.detach().std().item()),
        }


class PolicyMLPEmbeddingLatentLearner(PolicyKLBottleneckLatentLearner):
    """Deterministic MLP latent embedding.

    When ``recon_coeff <= 0`` (default), the encoder is trained jointly through
    PPO (``uses_joint_policy_loss = True``).

    When ``recon_coeff > 0``, the encoder is trained **only** by a
    reconstruction loss in :meth:`update` and the PPO policy sees detached
    latents (``uses_joint_policy_loss = False``).  This prevents the
    representation drift that destabilises PPO when the encoder co-adapts
    with the actor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder: DeterministicLatentModel | None = None
        self.decoder: LatentDecoder | None = None
        self.recon_optimizer: torch.optim.Optimizer | None = None

    def initialize(self, agent: IPMD) -> None:
        super().initialize(agent)
        cfg = self._config()
        if float(cfg.recon_coeff) > 0.0:
            self.uses_joint_policy_loss = False

    def _ensure_modules(self) -> None:
        assert self.agent is not None
        cfg = self._config()
        encoder_input_dim = sum(
            self.agent._obs_feature_dims[key] for key in self._posterior_input_keys()
        )
        if self.encoder is None:
            self.encoder = DeterministicLatentModel(
                input_dim=encoder_input_dim,
                latent_dim=self.agent._latent_dim,
                hidden_dims=list(cfg.encoder_hidden_dims),
                activation=cfg.encoder_activation,
            ).to(self.agent.device)
        if float(cfg.recon_coeff) > 0.0:
            if self.decoder is None:
                self.decoder = LatentDecoder(
                    latent_dim=self.agent._latent_dim,
                    output_dim=encoder_input_dim,
                    hidden_dims=list(cfg.decoder_hidden_dims),
                    activation=cfg.decoder_activation,
                ).to(self.agent.device)
            if self.recon_optimizer is None:
                params = list(self.encoder.parameters()) + list(
                    self.decoder.parameters()
                )
                self.recon_optimizer = torch.optim.Adam(params, lr=float(cfg.lr))
        self._ensure_probe()

    def joint_parameters(self) -> list[nn.Parameter]:
        self._ensure_modules()
        if self.encoder is None or not self.uses_joint_policy_loss:
            return []
        return list(self.encoder.parameters())

    def infer_batch_latents(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        self._ensure_modules()
        encoder_input = self._features_from_td(
            batch,
            self._posterior_input_keys(),
            detach=detach,
            context=context,
        )
        assert self.encoder is not None
        latents = F.normalize(self.encoder(encoder_input), dim=-1, eps=1.0e-6)
        return latents.detach() if detach else latents

    def joint_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        assert self.agent is not None
        latents = self.infer_batch_latents(
            batch,
            detach=False,
            context="policy latent embedding batch",
        )
        if latents is None:
            msg = (
                "Deterministic policy latent learner failed to infer minibatch latents."
            )
            raise RuntimeError(msg)
        zero = torch.zeros((), device=self.agent.device)
        return zero, {
            "latent_embedding_norm_mean": latents.norm(dim=-1).mean().detach(),
            "latent_embedding_std": latents.std().detach(),
            "latent_embedding_abs_mean": latents.abs().mean().detach(),
        }

    def update(self, rollout_flat: TensorDict) -> dict[str, float]:
        # Run probe update from parent first.
        probe_metrics = super().update(rollout_flat)

        assert self.agent is not None
        cfg = self._config()
        if float(cfg.recon_coeff) <= 0.0:
            return probe_metrics

        self._ensure_modules()
        assert self.encoder is not None
        assert self.decoder is not None
        assert self.recon_optimizer is not None

        # Use rollout observations as reconstruction targets.
        features = self._features_from_td(
            rollout_flat,
            self._posterior_input_keys(),
            detach=True,
            context="recon update batch",
        )
        if features.shape[0] == 0:
            return probe_metrics

        latent_pred = F.normalize(
            self.encoder(features),
            dim=-1,
            eps=1.0e-6,
        )
        recon = self.decoder(latent_pred)
        recon_loss = F.mse_loss(recon, features)

        uniformity = torch.zeros((), device=self.agent.device)
        if float(cfg.uniformity_coeff) > 0.0:
            uniformity = self.agent._latent_uniformity(latent_pred)

        weight_decay = torch.zeros((), device=self.agent.device)
        if float(cfg.weight_decay_coeff) > 0.0:
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            for param in params:
                if param.ndim >= 2:
                    weight_decay = weight_decay + param.pow(2).mean()

        total_loss = (
            float(cfg.recon_coeff) * recon_loss
            + float(cfg.uniformity_coeff) * uniformity
            + float(cfg.weight_decay_coeff) * weight_decay
        )

        self.recon_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if float(cfg.grad_clip_norm) > 0.0:
            clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                float(cfg.grad_clip_norm),
            )
        self.recon_optimizer.step()

        recon_metrics = {
            "ipmd/latent_total_loss": float(total_loss.detach().item()),
            "ipmd/latent_recon_loss": float(recon_loss.detach().item()),
            "ipmd/latent_uniformity": float(uniformity.detach().item()),
            "ipmd/latent_weight_decay": float(weight_decay.detach().item()),
        }
        return {**probe_metrics, **recon_metrics}


class FixedProjectionLatentLearner(BaseLatentLearner):
    """Frozen random orthogonal projection from s^ref to z.

    No learnable parameters — the projection matrix is fixed at initialization.
    Diagnostic tool: if the policy can learn with this, the problem is encoder
    instability, not latent capacity.
    """

    def __init__(self) -> None:
        super().__init__()
        self._projection: Tensor | None = None

    def _posterior_input_keys(self) -> list[BatchKey]:
        assert self.agent is not None
        cfg = self.agent.config.ipmd.latent_learning
        if cfg.posterior_input_keys:
            return dedupe_keys(list(cfg.posterior_input_keys))
        return dedupe_keys(
            [*self.agent._policy_obs_keys_without_latent, *self.agent._reward_obs_keys]
        )

    def _features_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor:
        assert self.agent is not None
        keys = self._posterior_input_keys()
        self.agent._require_batch_keys(td, keys, context=context)
        return self.agent._obs_features_from_td(
            td,
            cast(list, keys),
            next_obs=False,
            detach=detach,
        ).to(self.agent.device)

    def _ensure_projection(self, input_dim: int) -> None:
        assert self.agent is not None
        if self._projection is not None:
            return
        latent_dim = self.agent._latent_dim
        # Random orthogonal matrix via QR decomposition of a Gaussian matrix.
        random_matrix = torch.randn(latent_dim, input_dim, device=self.agent.device)
        q, _ = torch.linalg.qr(random_matrix.T)
        # q is (input_dim, latent_dim); we want (latent_dim, input_dim).
        self._projection = q.T.contiguous()

    def _project(self, features: Tensor) -> Tensor:
        assert self._projection is not None
        return F.normalize(
            features @ self._projection.T,
            dim=-1,
            eps=1.0e-6,
        )

    def infer_current_reference_latents(
        self,
        *,
        env_ids: Tensor | None = None,
    ) -> Tensor | None:
        assert self.agent is not None
        if self.agent._current_reference_obs_getter is None:
            return None
        keys = self._posterior_input_keys()
        reference_obs = self.agent._current_reference_obs_getter(
            required_keys=keys,
            env_ids=env_ids,
        )
        if reference_obs is None or reference_obs.numel() == 0:
            return None
        features = self.agent._obs_features_from_td(
            reference_obs,
            cast(list, keys),
            next_obs=False,
            detach=True,
        ).to(self.agent.device)
        self._ensure_projection(int(features.shape[-1]))
        with torch.no_grad():
            return self._project(features)

    def infer_batch_latents(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        del detach
        features = self._features_from_td(batch, detach=True, context=context)
        self._ensure_projection(int(features.shape[-1]))
        with torch.no_grad():
            return self._project(features)

    def infer_expert_latents(
        self,
        expert_batch: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        del detach
        latent = self.infer_batch_latents(
            expert_batch,
            detach=True,
            context="expert latent batch",
        )
        if latent is None:
            msg = "Fixed projection latent learner failed to infer expert latents."
            raise RuntimeError(msg)
        return latent

    def sample_expert_prior_latents(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        assert self.agent is not None
        expert_batch = self.agent._next_expert_batch(
            batch_size,
            required_keys=list(self._posterior_input_keys()),
        )
        features = self._features_from_td(
            expert_batch,
            detach=True,
            context="expert fixed projection prior",
        )
        self._ensure_projection(int(features.shape[-1]))
        with torch.no_grad():
            latents = self._project(features)
        return latents.to(device=device, dtype=dtype)


def _build_registry() -> dict[str, Callable[[], BaseLatentLearner]]:
    return {
        "expert_mle_posterior": ExpertMLEPosteriorLatentLearner,
        "legacy_mi_encoder": ExpertMLEPosteriorLatentLearner,
        "patch_autoencoder": PatchAutoencoderLatentLearner,
        "policy_kl_bottleneck": PolicyKLBottleneckLatentLearner,
        "policy_mlp_embedding": PolicyMLPEmbeddingLatentLearner,
        "fixed_projection": FixedProjectionLatentLearner,
    }


_LATENT_LEARNER_REGISTRY = _build_registry()


def build_latent_learner(method: str) -> BaseLatentLearner:
    """Construct a registered latent learner implementation."""
    normalized = str(method).strip().lower()
    try:
        factory = _LATENT_LEARNER_REGISTRY[normalized]
    except KeyError as err:
        msg = (
            f"Unknown latent_learning.method={method!r}. "
            f"Expected one of {sorted(_LATENT_LEARNER_REGISTRY)}."
        )
        raise ValueError(msg) from err
    return factory()
