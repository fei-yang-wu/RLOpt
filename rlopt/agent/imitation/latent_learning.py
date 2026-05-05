"""Pluggable latent-learning methods for latent-conditioned imitation agents."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from rlopt.agent.imitation.utils import LatentEncoder
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

    def checkpoint_state_dict(self) -> dict[str, Any]:
        modules = {
            name: module.state_dict()
            for name, module in vars(self).items()
            if isinstance(module, nn.Module)
        }
        optimizers = {
            name: optimizer.state_dict()
            for name, optimizer in vars(self).items()
            if isinstance(optimizer, torch.optim.Optimizer)
        }
        return {
            "modules": modules,
            "optimizers": optimizers,
            "extra": self._checkpoint_extra_state_dict(),
        }

    def load_checkpoint_state_dict(self, state: dict[str, Any]) -> None:
        for name, module_state in state["modules"].items():
            module = getattr(self, name)
            module.load_state_dict(module_state)
        for name, optimizer_state in state["optimizers"].items():
            optimizer = getattr(self, name)
            optimizer.load_state_dict(optimizer_state)
        self._load_checkpoint_extra_state_dict(state["extra"])

    def _checkpoint_extra_state_dict(self) -> dict[str, Any]:
        return {}

    def _load_checkpoint_extra_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            msg = f"{type(self).__name__} does not load extra checkpoint state."
            raise NotImplementedError(msg)

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

    def reconstruct_batch_features(
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

    def initialize(self, agent: IPMD) -> None:
        super().initialize(agent)
        input_dim = sum(
            agent._obs_feature_dims[key] for key in agent._posterior_obs_keys
        )
        self._ensure_modules(int(input_dim))

    def _patch_features_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        assert self.agent is not None
        required_keys = self._posterior_input_keys()
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
            if bool(cfg.freeze_encoder):
                params = list(self.decoder.parameters())
            else:
                params = list(self.encoder.parameters()) + list(
                    self.decoder.parameters()
                )
            self.optimizer = torch.optim.Adam(params, lr=float(cfg.lr))

    def joint_parameters(self) -> list[nn.Parameter]:
        assert self.agent is not None
        cfg = self._config()
        if not bool(cfg.train_posterior_through_policy) or self.encoder is None:
            return []
        return list(self.encoder.parameters())

    def infer_batch_latents(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        del context
        patch_features = self._patch_features_from_td(
            batch,
            detach=detach,
        )
        self._ensure_modules(int(patch_features.shape[-1]))
        assert self.encoder is not None
        if detach:
            with torch.no_grad():
                return self.encoder(patch_features)
        return self.encoder(patch_features)

    def reconstruct_batch_features(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        del context
        patch_features = self._patch_features_from_td(
            batch,
            detach=detach,
        )
        self._ensure_modules(int(patch_features.shape[-1]))
        assert self.encoder is not None
        assert self.decoder is not None
        if detach:
            with torch.no_grad():
                return self.decoder(self.encoder(patch_features))
        return self.decoder(self.encoder(patch_features))

    def infer_expert_latents(
        self,
        expert_batch: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        """Sample latent from the posterior distribution z \sim p(\cdot|s').
            Patch here means a windows of expert transitions that centered at the current timestep.

        Args:
            expert_batch (TensorDict): Batch of expert transitions containing the required observation keys for posterior inference.
            detach (bool): Whether to detach the inferred latents.

        Returns:
            Tensor: Inferred latent tensor of shape (batch_size, latent_dim).
        """
        assert self.agent is not None
        patch_features = self._patch_features_from_td(
            expert_batch,
            detach=False,
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
        )
        if patch_features.shape[0] == 0:
            return {}
        self._ensure_modules(int(patch_features.shape[-1]))
        assert self.encoder is not None
        assert self.decoder is not None
        assert self.optimizer is not None

        freeze_encoder = bool(cfg.freeze_encoder)
        if freeze_encoder:
            with torch.no_grad():
                latent_pred = self.encoder(patch_features)
            latent_pred = latent_pred.detach()
        else:
            latent_pred = self.encoder(patch_features)
        patch_recon = self.decoder(latent_pred)
        recon_error = patch_recon - patch_features
        recon_loss = F.mse_loss(patch_recon, patch_features)
        recon_mae = recon_error.abs().mean()
        recon_max_abs = recon_error.abs().max()

        weight_decay = torch.zeros((), device=self.agent.device)
        if float(cfg.weight_decay_coeff) > 0.0:
            wd_modules: list[nn.Module] = [self.decoder]
            if not freeze_encoder:
                wd_modules.append(self.encoder)
            for module in wd_modules:
                for param in module.parameters():
                    if param.ndim >= 2:
                        weight_decay = weight_decay + param.pow(2).mean()

        total_loss = (
            float(cfg.recon_coeff) * recon_loss
            + float(cfg.weight_decay_coeff) * weight_decay
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if float(cfg.grad_clip_norm) > 0.0:
            clip_params = list(self.decoder.parameters())
            if not freeze_encoder:
                clip_params = list(self.encoder.parameters()) + clip_params
            clip_grad_norm_(clip_params, float(cfg.grad_clip_norm))
        self.optimizer.step()

        return {
            "ipmd/latent_total_loss": float(total_loss.detach().item()),
            "ipmd/latent_recon_loss": float(recon_loss.detach().item()),
            "ipmd/latent_posterior_recon_mse": float(recon_loss.detach().item()),
            "ipmd/latent_posterior_recon_mae": float(recon_mae.detach().item()),
            "ipmd/latent_posterior_recon_max_abs": float(recon_max_abs.detach().item()),
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

    def required_expert_batch_keys(self) -> list[BatchKey]:
        return self._posterior_input_keys()

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
    ) -> Tensor:
        assert self.agent is not None
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
        del context
        self._ensure_modules()
        posterior_input = self._features_from_td(
            batch,
            self._posterior_input_keys(),
            detach=detach,
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
        )
        prior_input = self._features_from_td(
            batch,
            self._prior_input_keys(),
            detach=False,
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
                context="",
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
        del context
        self._ensure_modules()
        encoder_input = self._features_from_td(
            batch,
            self._posterior_input_keys(),
            detach=detach,
        )
        assert self.encoder is not None
        latents = F.normalize(self.encoder(encoder_input), dim=-1, eps=1.0e-6)
        return latents.detach() if detach else latents

    def reconstruct_batch_features(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        del context
        self._ensure_modules()
        if self.decoder is None:
            return None
        encoder_input = self._features_from_td(
            batch,
            self._posterior_input_keys(),
            detach=detach,
        )
        assert self.encoder is not None
        if detach:
            with torch.no_grad():
                latents = F.normalize(self.encoder(encoder_input), dim=-1, eps=1.0e-6)
                return self.decoder(latents)
        latents = F.normalize(self.encoder(encoder_input), dim=-1, eps=1.0e-6)
        return self.decoder(latents)

    def joint_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        assert self.agent is not None
        latents = self.infer_batch_latents(
            batch,
            detach=False,
            context="",
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

    def _checkpoint_extra_state_dict(self) -> dict[str, Any]:
        if self._projection is None:
            return {}
        return {"projection": self._projection}

    def _load_checkpoint_extra_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        assert self.agent is not None
        self._projection = cast(Tensor, state["projection"]).to(self.agent.device)

    def _posterior_input_keys(self) -> list[BatchKey]:
        assert self.agent is not None
        cfg = self.agent.config.ipmd.latent_learning
        if cfg.posterior_input_keys:
            return dedupe_keys(list(cfg.posterior_input_keys))
        return dedupe_keys(
            [*self.agent._policy_obs_keys_without_latent, *self.agent._reward_obs_keys]
        )

    def required_expert_batch_keys(self) -> list[BatchKey]:
        return self._posterior_input_keys()

    def _features_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        assert self.agent is not None
        keys = self._posterior_input_keys()
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
        del context
        features = self._features_from_td(batch, detach=True)
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
        features = self._features_from_td(expert_batch, detach=True)
        self._ensure_projection(int(features.shape[-1]))
        with torch.no_grad():
            latents = self._project(features)
        return latents.to(device=device, dtype=dtype)


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization (Mentzer 2023).

    Per-dim bounded scalar quantization. No codebook params, no aux loss, no
    collapse risk. Effective codebook size = ``prod(levels)``.
    """

    def __init__(self, levels: list[int]) -> None:
        super().__init__()
        if len(levels) == 0:
            msg = "FSQQuantizer requires at least one level."
            raise ValueError(msg)
        if any(int(level) < 2 for level in levels):
            msg = f"FSQ levels must each be >= 2, got {levels!r}."
            raise ValueError(msg)
        levels_i = torch.tensor([int(level) for level in levels], dtype=torch.long)
        levels_t = levels_i.to(torch.float32)
        # Half range per dim: integer indices range over [-half, half] (with one extra
        # for even L).  We follow the canonical implementation.
        self.register_buffer("levels", levels_t)
        self.register_buffer("_levels_i", levels_i)
        self.register_buffer(
            "_basis",
            torch.cumprod(
                torch.cat([torch.ones(1), levels_t[:-1]]),
                dim=0,
            ).to(torch.long),
        )
        self.code_dim = len(levels)

    @property
    def codebook_size(self) -> int:
        return int(self.levels.prod().item())

    def _bound(self, z: Tensor) -> Tensor:
        # Squash to [-1, 1] with extra slack so the rounded grid stays inside.
        half_l = (self.levels - 1) * 0.5
        offset = torch.where(
            self.levels % 2 == 0,
            torch.tensor(0.5, device=z.device),
            torch.tensor(0.0, device=z.device),
        )
        shift = torch.atanh(offset / half_l.clamp(min=1.0))
        return (z + shift).tanh() * half_l - offset

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bounded = self._bound(z_e)
        rounded = torch.round(bounded)
        # Straight-through estimator.
        z_q = bounded + (rounded - bounded).detach()
        # Code id: shift to non-negative integer indices then mix with basis.
        levels_i = self._levels_i.to(rounded.device)
        zhat = rounded.to(torch.long) + levels_i // 2
        zhat = zhat.clamp(min=0)
        zhat = torch.minimum(zhat, levels_i - 1)
        code = (zhat * self._basis.to(zhat.device)).sum(dim=-1)
        return z_q, code, bounded


class EMAVQQuantizer(nn.Module):
    """Standard VQ-VAE quantizer with EMA codebook updates + commitment loss."""

    def __init__(
        self,
        codebook_size: int,
        embed_dim: int,
        *,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_coeff: float = 0.25,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.decay = float(decay)
        self.eps = float(eps)
        self.commitment_coeff = float(commitment_coeff)

        embedding = torch.randn(self.codebook_size, self.embed_dim) * 0.01
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("cluster_sum", embedding.clone())
        self.register_buffer("_initialized", torch.tensor(False))

    @torch.no_grad()
    def _kmeans_init(self, z_e: Tensor) -> None:
        flat = z_e.reshape(-1, self.embed_dim)
        if flat.shape[0] == 0:
            return
        if flat.shape[0] >= self.codebook_size:
            idx = torch.randperm(flat.shape[0], device=flat.device)[
                : self.codebook_size
            ]
        else:
            idx = torch.randint(
                0, flat.shape[0], (self.codebook_size,), device=flat.device
            )
        self.embedding.copy_(flat[idx])
        self.cluster_sum.copy_(self.embedding)
        self.cluster_size.fill_(1.0)
        self._initialized.fill_(True)

    def _quantize(self, z_e: Tensor) -> tuple[Tensor, Tensor]:
        flat = z_e.reshape(-1, self.embed_dim)
        # Squared distance to each code.
        dist = (
            flat.pow(2).sum(-1, keepdim=True)
            - 2.0 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(-1)
        )
        code = dist.argmin(dim=-1)
        z_q = F.embedding(code, self.embedding)
        z_q = z_q.reshape_as(z_e)
        code = code.reshape(z_e.shape[:-1])
        return z_q, code

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.training and not bool(self._initialized.item()):
            self._kmeans_init(z_e.detach())
        z_q, code = self._quantize(z_e)
        commitment = F.mse_loss(z_e, z_q.detach())
        # Straight-through.
        z_q_st = z_e + (z_q - z_e).detach()

        if self.training:
            with torch.no_grad():
                flat = z_e.detach().reshape(-1, self.embed_dim)
                code_flat = code.reshape(-1)
                onehot = F.one_hot(code_flat, self.codebook_size).to(flat.dtype)
                cluster_size_new = onehot.sum(dim=0)
                cluster_sum_new = onehot.t() @ flat
                self.cluster_size.mul_(self.decay).add_(
                    cluster_size_new, alpha=1.0 - self.decay
                )
                self.cluster_sum.mul_(self.decay).add_(
                    cluster_sum_new, alpha=1.0 - self.decay
                )
                n = self.cluster_size.sum()
                cluster_size_smoothed = (
                    (self.cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )
                self.embedding.copy_(
                    self.cluster_sum / cluster_size_smoothed.unsqueeze(-1)
                )

        return z_q_st, code, commitment * self.commitment_coeff

    @torch.no_grad()
    def revive_dead_codes(self, z_e: Tensor, *, threshold: float = 1e-3) -> int:
        """Reinit unused codes from the current encoder distribution."""
        flat = z_e.detach().reshape(-1, self.embed_dim)
        if flat.shape[0] == 0:
            return 0
        dead = self.cluster_size < threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        idx = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        self.embedding[dead] = flat[idx]
        self.cluster_sum[dead] = self.embedding[dead]
        self.cluster_size[dead] = 1.0
        return n_dead


class GumbelQuantizer(nn.Module):
    """Categorical bottleneck via Gumbel-Softmax with optional straight-through."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.logit_head = nn.Linear(input_dim, self.codebook_size)
        self.codebook = nn.Embedding(self.codebook_size, self.embed_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def forward(
        self,
        z_e: Tensor,
        *,
        tau: float,
        hard: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits = self.logit_head(z_e)
        if self.training:
            y = F.gumbel_softmax(logits, tau=max(float(tau), 1e-3), hard=hard, dim=-1)
        else:
            idx = logits.argmax(dim=-1)
            y = F.one_hot(idx, self.codebook_size).to(logits.dtype)
        z_q = y @ self.codebook.weight
        code = logits.argmax(dim=-1)
        log_q = F.log_softmax(logits, dim=-1)
        # KL(q || Uniform) = sum q*(log q - log(1/K)) = sum q*log q + log K
        kl_uniform = (log_q.exp() * log_q).sum(-1).mean() + math.log(self.codebook_size)
        return z_q, code, kl_uniform, log_q


class PatchVQVAELatentLearner(BaseLatentLearner):
    """Vector-quantized autoencoder over expert windows.

    - Encoder: MLP over flattened posterior-input features (typically a window
      of expert states; configure ``patch_past_steps`` / ``patch_future_steps``
      and point ``posterior_input_keys`` at the windowed observation terms).
    - Quantizer: ``"fsq" | "vq_ema" | "gumbel" | "identity"`` selectable via cfg.
    - Decoder: MLP reconstructing the flattened window features. Optional
      action decoder reconstructs ``expert_action`` from the quantized latent
      to ground codes in behavior, not just kinematics.

    During posterior-command rollout, each env encodes a fresh expert window at
    code boundaries and holds that code for ``code_period`` env steps.
    """

    uses_aux_reward = False

    def __init__(self) -> None:
        super().__init__()
        self.encoder: DeterministicLatentModel | None = None
        self.decoder: LatentDecoder | None = None
        self.action_decoder: LatentDecoder | None = None
        self.code_to_latent: nn.Linear | None = None
        self.fsq: FSQQuantizer | None = None
        self.vq: EMAVQQuantizer | None = None
        self.gumbel: GumbelQuantizer | None = None
        self.optimizer: torch.optim.Optimizer | None = None

        self._update_calls: int = 0
        self._collector_z_q: Tensor | None = None
        self._collector_steps: Tensor | None = None

    def _checkpoint_extra_state_dict(self) -> dict[str, Any]:
        return {"update_calls": self._update_calls}

    def _load_checkpoint_extra_state_dict(self, state: dict[str, Any]) -> None:
        self._update_calls = int(state["update_calls"])

    # ----- config / key helpers --------------------------------------------------
    def _config(self):
        assert self.agent is not None
        return self.agent.config.ipmd.latent_learning

    def _quantizer_kind(self) -> str:
        return str(self._config().quantizer).strip().lower()

    def _phase_dim(self) -> int:
        cfg = self._config()
        return 2 if str(cfg.command_phase_mode).strip().lower() == "sin_cos" else 0

    def _code_latent_dim(self) -> int:
        assert self.agent is not None
        cfg = self._config()
        if cfg.code_latent_dim is not None and int(cfg.code_latent_dim) > 0:
            return int(cfg.code_latent_dim)
        return int(self.agent._latent_dim) - self._phase_dim()

    def _embed_dim(self) -> int:
        cfg = self._config()
        if cfg.codebook_embed_dim is not None and int(cfg.codebook_embed_dim) > 0:
            return int(cfg.codebook_embed_dim)
        return self._code_latent_dim()

    def _posterior_input_keys(self) -> list[BatchKey]:
        assert self.agent is not None
        cfg = self._config()
        if cfg.posterior_input_keys:
            return dedupe_keys(list(cfg.posterior_input_keys))
        return list(self.agent._posterior_obs_keys)

    def required_expert_batch_keys(self) -> list[BatchKey]:
        keys = self._posterior_input_keys()
        if float(self._config().action_recon_coeff) > 0.0:
            keys = [*list(keys), cast(BatchKey, "expert_action")]
        return dedupe_keys(keys)

    # ----- features / module construction ---------------------------------------
    def _features_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        assert self.agent is not None
        parts: list[Tensor] = []
        for key in self._posterior_input_keys():
            value = cast(Tensor, td.get(key)).to(self.agent.device)
            value = value.detach() if detach else value
            parts.append(value.reshape(value.shape[0], -1))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _ensure_modules(self, input_dim: int) -> None:
        assert self.agent is not None
        cfg = self._config()
        kind = self._quantizer_kind()
        embed_dim = self._embed_dim()
        code_latent_dim = self._code_latent_dim()

        if kind == "fsq":
            if self.fsq is None:
                self.fsq = FSQQuantizer(list(cfg.fsq_levels)).to(self.agent.device)
            encoder_out_dim = self.fsq.code_dim
        elif kind == "vq_ema":
            encoder_out_dim = embed_dim
            if self.vq is None:
                self.vq = EMAVQQuantizer(
                    codebook_size=int(cfg.codebook_size),
                    embed_dim=embed_dim,
                    decay=float(cfg.ema_decay),
                    commitment_coeff=float(cfg.commitment_coeff),
                ).to(self.agent.device)
        elif kind == "gumbel":
            encoder_out_dim = max(
                int(cfg.encoder_hidden_dims[-1]) if cfg.encoder_hidden_dims else 256, 1
            )
            if self.gumbel is None:
                self.gumbel = GumbelQuantizer(
                    input_dim=encoder_out_dim,
                    codebook_size=int(cfg.codebook_size),
                    embed_dim=embed_dim,
                ).to(self.agent.device)
        elif kind == "identity":
            encoder_out_dim = code_latent_dim
        else:
            msg = (
                f"Unknown quantizer={kind!r} for patch_vqvae. "
                "Expected 'fsq', 'vq_ema', 'gumbel', or 'identity'."
            )
            raise ValueError(msg)

        if self.encoder is None:
            self.encoder = DeterministicLatentModel(
                input_dim=input_dim,
                latent_dim=encoder_out_dim,
                hidden_dims=list(cfg.encoder_hidden_dims),
                activation=cfg.encoder_activation,
            ).to(self.agent.device)

        if kind in {"vq_ema", "gumbel"}:
            quantized_dim = embed_dim
        elif kind == "identity":
            quantized_dim = code_latent_dim
        else:
            assert self.fsq is not None
            quantized_dim = self.fsq.code_dim
        if self.code_to_latent is None and quantized_dim != code_latent_dim:
            self.code_to_latent = nn.Linear(quantized_dim, code_latent_dim).to(
                self.agent.device
            )

        decoder_in_dim = quantized_dim
        if self.decoder is None:
            self.decoder = LatentDecoder(
                latent_dim=decoder_in_dim,
                output_dim=input_dim,
                hidden_dims=list(cfg.decoder_hidden_dims),
                activation=cfg.decoder_activation,
            ).to(self.agent.device)

        if float(cfg.action_recon_coeff) > 0.0 and self.action_decoder is None:
            action_dim = int(self.agent._action_feature_dim)
            self.action_decoder = LatentDecoder(
                latent_dim=decoder_in_dim,
                output_dim=action_dim,
                hidden_dims=list(cfg.decoder_hidden_dims),
                activation=cfg.decoder_activation,
            ).to(self.agent.device)

        if self.optimizer is None:
            params: list[nn.Parameter] = []
            params += list(self.encoder.parameters())
            params += list(self.decoder.parameters())
            if self.action_decoder is not None:
                params += list(self.action_decoder.parameters())
            if self.code_to_latent is not None:
                params += list(self.code_to_latent.parameters())
            if self.gumbel is not None:
                params += list(self.gumbel.parameters())
            # vq_ema codebook updates are EMA-only, no optimizer entry required.
            self.optimizer = torch.optim.Adam(params, lr=float(cfg.lr))

    def initialize(self, agent: IPMD) -> None:
        super().initialize(agent)
        input_dim = sum(
            agent._obs_feature_dims[key] for key in self._posterior_input_keys()
        )
        self._ensure_modules(int(input_dim))

    # ----- core encode + quantize -----------------------------------------------
    def _current_tau(self) -> float:
        cfg = self._config()
        if int(cfg.gumbel_tau_anneal_iters) <= 0:
            return float(cfg.gumbel_tau_end)
        progress = min(
            1.0,
            float(self._update_calls) / float(cfg.gumbel_tau_anneal_iters),
        )
        return float(cfg.gumbel_tau_start) + progress * (
            float(cfg.gumbel_tau_end) - float(cfg.gumbel_tau_start)
        )

    def _quantize(
        self,
        z_e: Tensor,
    ) -> dict[str, Tensor]:
        kind = self._quantizer_kind()
        out: dict[str, Tensor] = {}
        if kind == "fsq":
            assert self.fsq is not None
            z_q, code, _ = self.fsq(z_e)
            out["z_q"] = z_q
            out["code"] = code
            out["aux_loss"] = torch.zeros((), device=z_e.device)
        elif kind == "vq_ema":
            assert self.vq is not None
            z_q, code, commitment = self.vq(z_e)
            out["z_q"] = z_q
            out["code"] = code
            out["aux_loss"] = commitment
        elif kind == "gumbel":
            assert self.gumbel is not None
            tau = self._current_tau()
            z_q, code, kl_uniform, log_q = self.gumbel(
                z_e, tau=tau, hard=bool(self._config().gumbel_hard)
            )
            out["z_q"] = z_q
            out["code"] = code
            out["aux_loss"] = kl_uniform
            out["log_q"] = log_q
            out["tau"] = torch.tensor(tau, device=z_e.device)
        elif kind == "identity":
            out["z_q"] = z_e
            out["code"] = torch.zeros(
                z_e.shape[:-1],
                device=z_e.device,
                dtype=torch.long,
            )
            out["aux_loss"] = torch.zeros((), device=z_e.device)
        return out

    def _project_to_code_latent(self, z_q: Tensor) -> Tensor:
        if self.code_to_latent is None:
            return z_q
        return self.code_to_latent(z_q)

    def _append_command_phase(
        self,
        code_latents: Tensor,
        phase: Tensor,
    ) -> Tensor:
        if self._phase_dim() == 0:
            return code_latents
        angle = phase.to(device=code_latents.device, dtype=code_latents.dtype)
        angle = angle.reshape(-1) * (2.0 * math.pi)
        phase_features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        return torch.cat((code_latents, phase_features), dim=-1)

    def _infer_code_latents(
        self,
        batch: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        features = self._features_from_td(batch, detach=detach)
        self._ensure_modules(int(features.shape[-1]))
        assert self.encoder is not None
        if detach:
            with torch.no_grad():
                z_e = self.encoder(features)
                quant = self._quantize(z_e)
                return self._project_to_code_latent(quant["z_q"])
        z_e = self.encoder(features)
        quant = self._quantize(z_e)
        return self._project_to_code_latent(quant["z_q"])

    # ----- BaseLatentLearner interface ------------------------------------------
    def infer_batch_latents(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        del context
        code_latents = self._infer_code_latents(batch, detach=detach)
        phase = torch.zeros(
            code_latents.shape[0],
            device=code_latents.device,
            dtype=code_latents.dtype,
        )
        return self._append_command_phase(code_latents, phase)

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
            msg = "patch_vqvae failed to infer expert latents."
            raise RuntimeError(msg)
        return latent

    def reconstruct_batch_features(
        self,
        batch: TensorDict,
        *,
        detach: bool,
        context: str,
    ) -> Tensor | None:
        del context
        features = self._features_from_td(batch, detach=detach)
        self._ensure_modules(int(features.shape[-1]))
        assert self.encoder is not None
        assert self.decoder is not None
        if detach:
            with torch.no_grad():
                z_e = self.encoder(features)
                quant = self._quantize(z_e)
                return self.decoder(quant["z_q"])
        z_e = self.encoder(features)
        quant = self._quantize(z_e)
        return self.decoder(quant["z_q"])

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
        latents = self.infer_expert_latents(expert_batch, detach=True)
        return latents.to(device=device, dtype=dtype)

    # ----- collector hold-across-steps ------------------------------------------
    def _renew_collector_codes(
        self,
        td: TensorDict,
        renew_mask: Tensor,
    ) -> None:
        assert self.agent is not None
        latents = self._infer_code_latents(
            td,
            detach=True,
        )
        latents = latents.to(self.agent.device).reshape(-1, self._code_latent_dim())
        if self._collector_z_q is None or self._collector_z_q.shape != latents.shape:
            self._collector_z_q = latents.clone()
        else:
            self._collector_z_q[renew_mask] = latents[renew_mask]

    def infer_collector_latents(self, td: TensorDict) -> Tensor:
        """Publish one held code per env, plus optional within-code phase."""
        assert self.agent is not None
        cfg = self._config()
        device = self.agent.device
        command_dim = int(self.agent._latent_dim)
        code_latent_dim = self._code_latent_dim()
        batch_size = int(td.numel())
        if batch_size <= 0:
            return torch.empty(0, command_dim, device=device)

        if (
            self._collector_z_q is None
            or self._collector_z_q.shape[0] != batch_size
            or self._collector_z_q.shape[1] != code_latent_dim
            or self._collector_z_q.device != device
        ):
            self._collector_z_q = torch.zeros(
                batch_size, code_latent_dim, device=device
            )
            self._collector_steps = torch.zeros(
                batch_size, device=device, dtype=torch.long
            )

        assert self._collector_steps is not None
        period = max(1, int(cfg.code_period))
        done_mask = self._build_done_mask(td, batch_size=batch_size, device=device)
        renew_mask = done_mask | (self._collector_steps <= 0)
        if bool(renew_mask.any()):
            self._renew_collector_codes(td, renew_mask)
            self._collector_steps[renew_mask] = period
        phase = (period - self._collector_steps).clamp(min=0).to(torch.float32)
        phase = phase / float(period)
        command = self._append_command_phase(self._collector_z_q, phase)
        self._collector_steps -= 1
        return command

    @staticmethod
    def _build_done_mask(
        td: TensorDict, *, batch_size: int, device: torch.device
    ) -> Tensor:
        mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        candidates: list[BatchKey] = [
            cast(BatchKey, "done"),
            cast(BatchKey, "terminated"),
            cast(BatchKey, "truncated"),
            cast(BatchKey, "is_init"),
            cast(BatchKey, ("next", "done")),
            cast(BatchKey, ("next", "terminated")),
            cast(BatchKey, ("next", "truncated")),
        ]
        keys = td.keys(True)
        for key in candidates:
            if key not in keys:
                continue
            value = cast(Tensor, td.get(key)).reshape(-1).to(device=device).bool()
            if value.numel() == batch_size:
                mask |= value
        return mask

    # ----- training update -------------------------------------------------------
    def update(self, rollout_flat: TensorDict) -> dict[str, float]:
        del rollout_flat
        assert self.agent is not None
        cfg = self._config()
        if float(cfg.recon_coeff) <= 0.0 and float(cfg.action_recon_coeff) <= 0.0:
            return {}

        expert_td = self.agent._next_expert_batch(
            required_keys=self.required_expert_batch_keys()
        )
        features = self._features_from_td(expert_td, detach=False)
        if features.shape[0] == 0:
            return {}

        self._ensure_modules(int(features.shape[-1]))
        assert self.encoder is not None
        assert self.decoder is not None
        assert self.optimizer is not None

        z_e = self.encoder(features)
        quant = self._quantize(z_e)
        z_q = quant["z_q"]
        code = quant["code"]
        aux_loss = quant["aux_loss"]

        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, features)

        action_loss = torch.zeros((), device=self.agent.device)
        if (
            float(cfg.action_recon_coeff) > 0.0
            and self.action_decoder is not None
            and "expert_action" in expert_td.keys(True)
        ):
            expert_action = cast(Tensor, expert_td.get("expert_action")).to(
                self.agent.device
            )
            expert_action_flat = expert_action.reshape(expert_action.shape[0], -1)
            pred_action = self.action_decoder(z_q)
            action_loss = F.mse_loss(pred_action, expert_action_flat)

        weight_decay = torch.zeros((), device=self.agent.device)
        if float(cfg.weight_decay_coeff) > 0.0:
            for module in (self.encoder, self.decoder):
                for param in module.parameters():
                    if param.ndim >= 2:
                        weight_decay = weight_decay + param.pow(2).mean()

        gumbel_kl = torch.zeros((), device=self.agent.device)
        usage_entropy = torch.zeros((), device=self.agent.device)
        if self._quantizer_kind() == "gumbel":
            gumbel_kl = aux_loss
            if float(cfg.code_usage_entropy_coeff) > 0.0 and "log_q" in quant:
                log_q = quant["log_q"]
                marginal = log_q.exp().mean(dim=0).clamp(min=1e-8)
                usage_entropy = -(marginal * marginal.log()).sum()
        commit_loss = (
            aux_loss
            if self._quantizer_kind() == "vq_ema"
            else torch.zeros((), device=self.agent.device)
        )

        total = (
            float(cfg.recon_coeff) * recon_loss
            + float(cfg.action_recon_coeff) * action_loss
            + commit_loss
            + float(cfg.gumbel_kl_to_uniform_coeff) * gumbel_kl
            - float(cfg.code_usage_entropy_coeff) * usage_entropy
            + float(cfg.weight_decay_coeff) * weight_decay
        )

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if float(cfg.grad_clip_norm) > 0.0:
            params: list[nn.Parameter] = []
            params += list(self.encoder.parameters())
            params += list(self.decoder.parameters())
            if self.action_decoder is not None:
                params += list(self.action_decoder.parameters())
            if self.code_to_latent is not None:
                params += list(self.code_to_latent.parameters())
            if self.gumbel is not None:
                params += list(self.gumbel.parameters())
            clip_grad_norm_(params, float(cfg.grad_clip_norm))
        self.optimizer.step()

        # Code-usage diagnostics + dead-code revival.
        with torch.no_grad():
            code_flat = code.reshape(-1)
            codebook_size = self._codebook_size()
            usage_hist = torch.bincount(code_flat, minlength=codebook_size).float()
            usage_prob = usage_hist / usage_hist.sum().clamp(min=1.0)
            perplexity = torch.exp(
                -(usage_prob * (usage_prob.clamp(min=1e-12)).log()).sum()
            )
            dead_codes = int((usage_hist <= 0).sum().item())

        revived = 0
        if (
            self._quantizer_kind() == "vq_ema"
            and int(cfg.dead_code_reset_iters) > 0
            and self._update_calls > 0
            and self._update_calls % int(cfg.dead_code_reset_iters) == 0
        ):
            assert self.vq is not None
            revived = self.vq.revive_dead_codes(z_e.detach())

        self._update_calls += 1

        metrics: dict[str, float] = {
            "ipmd/vqvae_total_loss": float(total.detach().item()),
            "ipmd/vqvae_recon_mse": float(recon_loss.detach().item()),
            "ipmd/vqvae_action_recon_mse": float(action_loss.detach().item()),
            "ipmd/vqvae_commit_loss": float(commit_loss.detach().item()),
            "ipmd/vqvae_gumbel_kl_uniform": float(gumbel_kl.detach().item()),
            "ipmd/vqvae_code_usage_entropy": float(usage_entropy.detach().item()),
            "ipmd/vqvae_codebook_perplexity": float(perplexity.item()),
            "ipmd/vqvae_dead_codes": float(dead_codes),
            "ipmd/vqvae_codebook_size": float(self._codebook_size()),
            "ipmd/vqvae_codes_revived": float(revived),
            "ipmd/vqvae_weight_decay": float(weight_decay.detach().item()),
        }
        if self._quantizer_kind() == "gumbel":
            metrics["ipmd/vqvae_gumbel_tau"] = float(self._current_tau())
        return metrics

    def _codebook_size(self) -> int:
        kind = self._quantizer_kind()
        if kind == "fsq":
            assert self.fsq is not None
            return self.fsq.codebook_size
        if kind == "vq_ema":
            assert self.vq is not None
            return self.vq.codebook_size
        if kind == "gumbel":
            assert self.gumbel is not None
            return self.gumbel.codebook_size
        if kind == "identity":
            return 1
        return 0


def _build_registry() -> dict[str, Callable[[], BaseLatentLearner]]:
    return {
        "patch_autoencoder": PatchAutoencoderLatentLearner,
        "policy_kl_bottleneck": PolicyKLBottleneckLatentLearner,
        "policy_mlp_embedding": PolicyMLPEmbeddingLatentLearner,
        "fixed_projection": FixedProjectionLatentLearner,
        "patch_vqvae": PatchVQVAELatentLearner,
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
