from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import ReplayBuffer
from torchrl.record.loggers import Logger

from rlopt.agent.ipmd.ipmd_simple import IPMD, IPMDRLOptConfig
from rlopt.agent.ipmd.network import DDPM, FactorizedNCE


@dataclass
class SRConfig:
    """Spectral representation learning configuration.

    Supports both DiffSR (diffusion-based) and CTRL-SR (contrastive-based) methods.
    """

    method: str = "diffsr"  # "diffsr" or "ctrlsr"

    # Shared fields
    feature_dim: int = 64
    phi_hidden_dims: tuple[int, ...] = (512, 512)
    mu_hidden_dims: tuple[int, ...] = (512, 512)
    reward_hidden_dims: tuple[int, ...] = (512,)
    rff_dim: int | None = None
    feature_lr: float = 1e-4
    reward_loss_coeff: float = 0.0
    update_steps: int = 1

    # DiffSR-specific
    sample_steps: int = 16
    sample_eval_interval: int = 50  # 0 = disabled; run sampling check every N SR updates

    # CTRL-SR-specific
    num_noises: int = 16
    linear: bool = False
    ranking: bool = True


@dataclass
class IPMDSRRLOptConfig(IPMDRLOptConfig):
    """IPMD + Spectral Representation configuration."""

    sr: SRConfig = field(default_factory=SRConfig)

    def __post_init__(self) -> None:
        super().__post_init__()


class IPMDSR(IPMD):
    """IPMD with spectral representation learning trained alongside.

    The SR model is trained on the same rollout data used for IPMD updates,
    using a separate optimizer. The SR update runs before each IPMD update step.
    """

    def __init__(
        self,
        env,
        config: IPMDSRRLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        self.config = cast(IPMDSRRLOptConfig, config)
        self.env = env

        self.sr_model: DDPM | FactorizedNCE = self._construct_sr_model()
        self.sr_model.to(self.device)

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

        self.sr_optim = torch.optim.Adam(
            self.sr_model.parameters(), lr=self.config.sr.feature_lr
        )
        self._sr_update_count = 0
        self._pending_sr_metrics: dict[str, list[float]] = {}

    def _construct_sr_model(self) -> DDPM | FactorizedNCE:
        """Construct the spectral representation model."""
        cfg = self.config.sr
        # Use first policy obs key for obs_dim
        obs_key = self.config.policy.get_input_keys()[0]
        obs_dim = self.env.observation_spec[obs_key].shape[-1]
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        action_dim = action_spec.shape[-1]

        if cfg.method == "diffsr":
            return DDPM(
                obs_dim=obs_dim,
                action_dim=action_dim,
                feature_dim=cfg.feature_dim,
                phi_hidden_dims=cfg.phi_hidden_dims,
                mu_hidden_dims=cfg.mu_hidden_dims,
                reward_hidden_dims=cfg.reward_hidden_dims,
                rff_dim=cfg.rff_dim,
                sample_steps=cfg.sample_steps,
                device=self.device,
            )
        elif cfg.method == "ctrlsr":
            return FactorizedNCE(
                obs_dim=obs_dim,
                action_dim=action_dim,
                feature_dim=cfg.feature_dim,
                phi_hidden_dims=cfg.phi_hidden_dims,
                mu_hidden_dims=cfg.mu_hidden_dims,
                reward_hidden_dims=cfg.reward_hidden_dims,
                rff_dim=cfg.rff_dim,
                num_noises=cfg.num_noises,
                linear=cfg.linear,
                ranking=cfg.ranking,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unknown SR method: {cfg.method!r}. Choose 'diffsr' or 'ctrlsr'."
            )

    def _sr_obs_key(self) -> str:
        """Return the primary observation key used for SR inputs."""
        return self.config.policy.get_input_keys()[0]

    def _sr_loss_from_batch(
        self, batch: TensorDict
    ) -> tuple[dict[str, float], Tensor]:
        """Compute SR loss from a batch of transitions."""
        batch = batch.to(self.device)
        obs_key = self._sr_obs_key()
        obs = cast(Tensor, batch.get(obs_key))
        action = cast(Tensor, batch.get("action"))
        next_obs = cast(Tensor, batch.get(("next", obs_key)))

        reward = batch.get("reward", None)
        if reward is None:
            reward = batch.get(("next", "reward"))
        if reward is None:
            reward = torch.zeros(obs.shape[0], 1, device=self.device)
        reward = cast(Tensor, reward).view(-1, 1)

        metrics, model_loss, reward_loss = self.sr_model.compute_loss(
            obs, action, next_obs, reward
        )
        loss = model_loss + self.config.sr.reward_loss_coeff * reward_loss
        return metrics, loss

    @torch.no_grad()
    def _sr_sample_eval(self, batch: TensorDict) -> dict[str, float]:
        """Run diffusion sampling to reconstruct s from s', measure distance."""
        if not isinstance(self.sr_model, DDPM):
            return {}
        self.sr_model.eval()
        batch = batch.to(self.device)
        obs_key = self._sr_obs_key()
        obs = cast(Tensor, batch.get(obs_key))
        next_obs = cast(Tensor, batch.get(("next", obs_key)))
        s_recon, _ = self.sr_model.sample(sp=next_obs)
        mse = (s_recon - obs).pow(2).sum(-1).mean()
        l1 = (s_recon - obs).abs().sum(-1).mean()
        self.sr_model.train()
        return {
            "sample/recon_mse": mse.item(),
            "sample/recon_l1": l1.item(),
        }

    def _train_sr_steps(
        self, batch: TensorDict, steps: int
    ) -> dict[str, float]:
        """Train the SR model for the given number of steps on mini-batches."""
        metrics_accum: dict[str, float] = {}
        if steps <= 0:
            return metrics_accum

        self.sr_model.train()
        batch_size = batch.batch_size[0]
        mini_batch_size = max(1, int(batch_size // steps))

        for i in range(steps):
            start = i * mini_batch_size
            end = batch_size if i == steps - 1 else (i + 1) * mini_batch_size
            mini_batch = batch[start:end]

            metrics, loss = self._sr_loss_from_batch(mini_batch)
            self.sr_optim.zero_grad(set_to_none=True)
            loss.backward()
            self.sr_optim.step()
            self._sr_update_count += 1

            for key, value in metrics.items():
                metrics_accum[key] = metrics_accum.get(key, 0.0) + float(value)

        for key in list(metrics_accum.keys()):
            metrics_accum[key] /= float(steps)

        # Periodic sampling evaluation (DiffSR only)
        interval = self.config.sr.sample_eval_interval
        if interval > 0 and self._sr_update_count % interval == 0:
            sample_metrics = self._sr_sample_eval(batch)
            metrics_accum.update(sample_metrics)

        return metrics_accum

    def update(
        self,
        batch: TensorDict,
        num_network_updates: int,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> tuple[TensorDict, int]:
        """SR update followed by IPMD update; SR metrics accumulated for logging."""
        sr_metrics = self._train_sr_steps(batch, self.config.sr.update_steps)

        # Accumulate SR metrics (averaged at log time)
        for key, value in sr_metrics.items():
            self._pending_sr_metrics.setdefault(key, []).append(value)

        return super().update(
            batch, num_network_updates, expert_batch, has_expert
        )

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        **kwargs,
    ) -> None:
        """Inject accumulated SR metrics before delegating to parent logger."""
        if self._pending_sr_metrics:
            merged = dict(metrics)
            for key, values in self._pending_sr_metrics.items():
                merged[f"sr/{key}"] = sum(values) / len(values)
            self._pending_sr_metrics.clear()
            super().log_metrics(merged, **kwargs)
        else:
            super().log_metrics(metrics, **kwargs)
