"""IPMD with bilinear spectral representation learning.

Implements the bottom-up skill composition approach where spectral components
are parameterized as phi_i(s,a) = f(s)^T W_i g(a) with low-rank factorization
W_i = u_i v_i^T. The policy uses f(s) @ W(z) as input to a neural network
that outputs a Gaussian action distribution (PPO on top of the representation).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs.utils import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal
from torchrl.record.loggers import Logger

from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.ipmd.network import (
    PositionalFeature,
    ResidualMLP,
    get_noise_schedule,
)
from rlopt.models.gaussian_policy import GaussianPolicyHead
from rlopt.utils import get_activation_class

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BilinearSRConfig:
    """Bilinear spectral representation configuration."""

    feature_dim: int = 512
    """Number of spectral components (d)."""

    embed_dim: int = 256
    """Shared embedding dimension for state and action encoders."""

    f_hidden_dims: tuple[int, ...] = (512, 512)
    """Hidden layers for state encoder f(s)."""

    g_hidden_dims: tuple[int, ...] = (512, 512)
    """Hidden layers for action encoder g(a)."""

    mu_hidden_dims: tuple[int, ...] = (512, 512)
    """Hidden layers for next-state encoder mu(s')."""

    feature_lr: float = 1e-4
    """Learning rate for the bilinear SR model."""

    update_steps: int = 16
    """Number of SR gradient steps per IPMD update."""

    history_buffer_size: int = 10_000_000
    """Max transitions in the SR history replay buffer."""

    sr_batch_size: int = 4096
    """Batch size for SR training mini-batches."""

    num_noises: int = 64
    """Number of diffusion noise steps."""

    x_min: float = -10.0
    """Clamp min for reverse diffusion."""

    x_max: float = 10.0
    """Clamp max for reverse diffusion."""

    sample_eval_interval: int = 50
    """Run sampling check every N SR updates."""

    detach_features_for_policy: bool = True
    """Detach representation features before feeding to policy (no grad to SR)."""

    ema_tau: float = 0.005
    """EMA update rate for the policy target state_net. Lower = slower tracking."""

    use_ema_for_policy: bool = True
    """Use a slow-moving EMA copy of state_net for policy input (stabilizes PPO)."""


@dataclass
class IPMDBilinearRLOptConfig(IPMDRLOptConfig):
    """IPMD + Bilinear spectral representation configuration."""

    bilinear: BilinearSRConfig = field(default_factory=BilinearSRConfig)

    def __post_init__(self) -> None:
        super().__post_init__()


# ---------------------------------------------------------------------------
# Bilinear Representation Model
# ---------------------------------------------------------------------------


class EmpiricalNormalization(nn.Module):
    """Running mean/std normalization using Welford's online algorithm."""

    def __init__(self, shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Update running statistics with a batch of observations."""
        x = x.reshape(-1, x.shape[-1])
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=0)
        batch_count = x.shape[0]

        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / new_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / new_count
        self.var = m2 / new_count
        self.count = new_count

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize input using running statistics."""
        if self.count < 2:
            return x
        return (x - self.mean) / (self.var.sqrt() + self.eps)

    def inv_normalize(self, x: Tensor) -> Tensor:
        """Inverse normalize input using running statistics."""
        return x * (self.var.sqrt() + self.eps) + self.mean


class BilinearRepresentation(nn.Module):
    r"""Bilinear spectral representation with low-rank factorization.

    Each spectral component is parameterized as:
        phi_i(s, a) = f(s)^T W_i g(a),  with W_i = u_i v_i^T

    Stacked form:
        phi(s, a) = (f(s) @ U^T) * (g(a) @ V^T)   (element-wise product)

    where U \in R^{d x embed_dim}, V \in R^{d x embed_dim}.

    The next-state encoder mu(s') is trained via diffusion so that
    phi(s, a)^T mu(s') predicts the noise added to s'.
    """

    def __init__(
        self,
        obs_dim: int,
        next_obs_dim: int,
        action_dim: int,
        feature_dim: int,
        embed_dim: int,
        f_hidden_dims: tuple[int, ...],
        g_hidden_dims: tuple[int, ...],
        mu_hidden_dims: tuple[int, ...],
        num_noises: int = 16,
        use_ema_for_policy: bool = True,
        x_min: float = -10.0,
        x_max: float = 10.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.next_obs_dim = next_obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_noises = num_noises
        self.x_min, self.x_max = x_min, x_max

        # Empirical normalization for observations
        self.obs_norm = EmpiricalNormalization(
            next_obs_dim
        )  # normalize next obs for better noise mixing

        # Noise schedule from the original DiffSR implementation (VP schedule)
        betas, alphas, alphabars = get_noise_schedule("vp", num_noises)
        alphabars_prev = F.pad(alphabars[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas[..., None])
        self.register_buffer("alphas", alphas[..., None])
        self.register_buffer("alphabars", alphabars[..., None])
        self.register_buffer("alphabars_prev", alphabars_prev[..., None])

        self.mlp_t = nn.Sequential(
            PositionalFeature(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
        )

        # self.mlp_phi = ResidualMLP(
        #     input_dim=obs_dim + action_dim,
        #     output_dim=feature_dim,
        #     hidden_dims=[512, 512],
        #     activation=nn.Mish(),
        # )

        self.state_net = ResidualMLP(
            input_dim=obs_dim,
            output_dim=embed_dim,
            hidden_dims=f_hidden_dims,
            activation=nn.Mish(),
        )
        self.action_net = ResidualMLP(
            input_dim=action_dim,
            output_dim=embed_dim,
            hidden_dims=g_hidden_dims,
            activation=nn.Mish(),
        )

        # --- Low-rank bilinear factors U, V ---
        # W_i = u_i v_i^T;  U = [u_1; ...; u_d], V = [v_1; ...; v_d]
        self.U = nn.Parameter(torch.randn(feature_dim, embed_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(feature_dim, embed_dim) * 0.01)

        # --- Next-state encoder mu(s', t) -> Jacobian (feature_dim x obs_dim) ---
        self.mu_net = ResidualMLP(
            input_dim=next_obs_dim + 128,
            output_dim=feature_dim * next_obs_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
        )

        # --- EMA target state_net for policy (slow-moving copy) ---
        if use_ema_for_policy:
            self.state_net_ema = copy.deepcopy(self.state_net).to(device)
            for p in self.state_net_ema.parameters():
                p.requires_grad_(False)
        else:
            self.state_net_ema = None

    def add_noise(
        self, x0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise_idx = torch.randint(
            0, self.num_noises, (x0.shape[0],), device=x0.device
        ).long()
        alphabars = self.alphabars[noise_idx]
        eps = torch.randn_like(x0)
        xt = alphabars.sqrt() * x0 + (1 - alphabars).sqrt() * eps
        return xt, noise_idx, eps

    @torch.no_grad()
    def update_ema(self, tau: float) -> None:
        """Polyak-average state_net into state_net_ema."""
        if self.state_net_ema is None:
            return
        for p_ema, p_online in zip(
            self.state_net_ema.parameters(), self.state_net.parameters()
        ):
            p_ema.data.lerp_(p_online.data, tau)

    def encode_state(self, s: Tensor) -> Tensor:
        return self.state_net(s)

    def encode_action(self, a: Tensor) -> Tensor:
        return self.action_net(a)

    def forward_phi(self, s: Tensor, a: Tensor) -> Tensor:
        """Compute phi(s,a) = (f(s) @ U^T) * (g(a) @ V^T) -> R^d.

        Args:
            s: Raw state observations (..., obs_dim).
            a: Raw actions (..., action_dim).

        Returns:
            Spectral features (..., d).
        """
        s = self.encode_state(s)
        a = self.encode_action(a)
        return (s @ self.U.T) * (a @ self.V.T)
        # return self.mlp_phi(torch.concat([s, a], dim=-1))

    def forward_mu(self, sp: Tensor, t: Tensor) -> Tensor:
        """mu(s', t) -> Jacobian (B, feature_dim, obs_dim)."""
        t_ff = self.mlp_t(t)
        all = torch.concat([sp, t_ff], dim=-1)
        all = self.mu_net(all)
        return all.reshape(-1, self.feature_dim, self.next_obs_dim)

    def forward_eps(
        self,
        s: Tensor | None = None,
        a: Tensor | None = None,
        sp: Tensor | None = None,
        t: Tensor | None = None,
        z_phi: Tensor | None = None,
        z_mu: Tensor | None = None,
    ) -> Tensor:
        """Predict noise: eps_pred = phi(s,a)^T @ mu(s',t) -> (B, obs_dim)."""
        if z_phi is None:
            z_phi = self.forward_phi(s, a)
        if z_mu is None:
            z_mu = self.forward_mu(sp, t)
        # z_phi: (B, d), z_mu: (B, d, obs_dim)
        # eps_pred: (B, obs_dim)
        return torch.bmm(z_phi.unsqueeze(1), z_mu).squeeze(1)

    def compute_loss(
        self,
        s: Tensor,
        a: Tensor,
        sp: Tensor,
        r: Tensor,
    ) -> tuple[dict[str, float], Tensor, Tensor]:
        """Diffusion loss for bilinear spectral representation learning."""
        s = s  # no normalization for state
        x0 = self.obs_norm.normalize(sp)
        xt, t, eps = self.add_noise(x0)

        z_phi = self.forward_phi(s=s, a=a)
        z_mu = self.forward_mu(sp=xt.detach(), t=t.unsqueeze(-1))
        eps_pred = self.forward_eps(z_phi=z_phi, z_mu=z_mu)
        diffusion_loss = (eps_pred - eps).pow(2).sum(-1).mean()

        metrics = {
            "info/eps_l1_norm": eps.abs().mean().item(),
            "info/eps_pred_l1_norm": eps_pred.abs().mean().item(),
            "info/x0_mean": x0.mean().item(),
            "info/x0_std": x0.std(0).mean().item(),
            "info/x0_l1_norm": x0.abs().mean().item(),
            "info/obs_norm_mean": self.obs_norm.mean.abs().mean().item(),
            "info/obs_norm_std": self.obs_norm.var.sqrt().mean().item(),
            "info/obs_norm_count": float(self.obs_norm.count.item()),
            "loss/dynamics_loss": diffusion_loss.item(),
        }
        return metrics, diffusion_loss, torch.tensor(0.0, device=s.device)

    @torch.no_grad()
    def sample(
        self,
        s: Tensor,
        a: Tensor,
        preserve_history: bool = False,
    ) -> tuple[Tensor, dict]:
        info = {}
        shape = (s.shape[0], self.next_obs_dim)
        xt = torch.randn(shape, device=s.device)
        z_phi = self.forward_phi(s, a)  # (B, feature_dim) — already normalized inside

        if preserve_history:
            info["sample_history"] = [xt.clone()]

        for t in reversed(range(self.num_noises)):
            z = torch.randn_like(xt)
            timestep = torch.full((xt.shape[0],), t, dtype=torch.int64, device=s.device)
            z_mu = self.forward_mu(sp=xt, t=timestep.unsqueeze(-1))
            eps_pred = torch.bmm(z_phi.unsqueeze(1), z_mu).squeeze(1)

            sigma_t = 0
            if t > 0:
                sigma_t_sq = (
                    self.betas[timestep]
                    * (1 - self.alphabars_prev[timestep])
                    / (1 - self.alphabars[timestep])
                )
                sigma_t = sigma_t_sq.clip(1e-20).sqrt()

            xt = (
                1.0
                / self.alphas[timestep].sqrt()
                * (
                    xt
                    - self.betas[timestep]
                    / (1 - self.alphabars[timestep]).sqrt()
                    * eps_pred
                )
                + sigma_t * z
            )
            xt = xt.clip(self.x_min, self.x_max)

            if preserve_history:
                info["sample_history"].append(xt.clone())

        xt = self.obs_norm.inv_normalize(xt)
        return xt, info

    def compute_q_representation(self, s: Tensor, a: Tensor) -> Tensor:
        """Return phi(s, a) for Q-function use."""
        return self.forward_phi(s, a)

    def compute_policy_representation(self, s: Tensor, z: Tensor = None) -> Tensor:
        r"""Compute policy input: f_ema(s) (detached).

        Uses the slow-moving EMA copy of state_net when available, so that the
        policy's input distribution stays approximately stationary even as the
        online state_net is updated by SR training.

        Args:
            s: State observations (B, obs_dim).
            z: Skill coefficients (d,) or (B, d). Currently unused.

        Returns:
            Policy representation (B, embed_dim).
        """
        if self.state_net_ema is None:
            f_s = self.state_net(s).detach()  # (B, embed_dim)
        else:
            f_s = self.state_net_ema(s).detach()  # (B, embed_dim)
        return f_s


# ---------------------------------------------------------------------------
# Policy Head
# ---------------------------------------------------------------------------


class BilinearPolicyHead(GaussianPolicyHead):
    """Policy head using bilinear spectral representation.

    Inherits from GaussianPolicyHead. The only difference is that it first
    computes f(s) @ W(z) from the bilinear representation, then passes
    the result through the base MLP (inherited from GaussianPolicyHead)
    to produce (loc, scale) for a Gaussian action distribution.

    The bilinear representation's features are detached by default so PPO
    gradients do not flow into the representation (trained separately).
    """

    def __init__(
        self,
        bilinear_rep: BilinearRepresentation,
        num_cells: list[int],
        activation_fn: str,
        action_dim: int,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        clip_log_std: bool = False,
        detach_features: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        base_mlp = MLP(
            in_features=None,  # lazy initialization
            out_features=action_dim,
            num_cells=num_cells,
            activation_class=get_activation_class(activation_fn),
            device=device,
        )

        super().__init__(
            base=base_mlp,
            action_dim=action_dim,
            log_std_init=log_std_init,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            clip_log_std=clip_log_std,
            device=device,
        )

        # Store bilinear_rep as submodule (shared with agent).
        # Detach prevents PPO gradients flowing into the representation.
        self.bilinear_rep = bilinear_rep
        self.detach_features = detach_features

    def forward(self, *obs: Tensor, z: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Compute (loc, scale) from observations via bilinear representation.

        1. Concatenate multi-key observations.
        2. Compute policy representation f(s) @ W(z).
        3. (Optionally) detach from SR computation graph.
        4. Feed through base MLP -> loc, then log_std_module -> scale.
        """
        if len(obs) == 0:
            raise ValueError(
                "BilinearPolicyHead.forward() expected at least one observation tensor."
            )
        combined = obs[0] if len(obs) == 1 else torch.cat(list(obs), dim=-1)
        rep = self.bilinear_rep.compute_policy_representation(combined, z)
        if self.detach_features:
            rep = rep.detach()
        # Delegate to GaussianPolicyHead: base MLP + log_std_module
        loc = self.base(rep)
        scale = self.log_std_module(loc)
        return loc, scale


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class IPMDBilinear(IPMD):
    """IPMD with bilinear spectral representation learning.

    Diffusion-based SR training on (s, a, s') via a history buffer,
    bilinear policy f(s) @ W(z) -> MLP -> Gaussian, plus standard
    IPMD reward estimator and BC loss.
    """

    def __init__(
        self,
        env,
        config: IPMDBilinearRLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        self.config = cast(IPMDBilinearRLOptConfig, config)
        self.env = env

        # Build before super().__init__ so _construct_policy can use it.
        self.bilinear_rep = self._construct_bilinear_model()
        self.bilinear_rep.to(self.device)

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
            self.bilinear_rep.parameters(), lr=self.config.bilinear.feature_lr
        )

        self._sr_update_count = 0
        self._pending_sr_metrics: dict[str, list[float]] = {}
        self._sr_history_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.config.bilinear.history_buffer_size,
                device=self.device,
            ),
            sampler=RandomSampler(),
            batch_size=self.config.bilinear.sr_batch_size,
        )

    # -- Construction --

    def _concat_obs_from_td(self, td: TensorDict) -> Tensor:
        """Concatenate all policy obs keys from a TensorDict."""
        parts = [td.get(k).flatten(-1) for k in self._get_obs_keys()]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _concat_next_obs_from_td(self, td: TensorDict) -> Tensor:
        """Concatenate all next policy obs keys from a TensorDict."""
        parts = [td.get(("next", k)).flatten(-1) for k in self._get_next_obs_keys()]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _get_obs_keys(self) -> list[str]:
        return list(self.config.policy.get_input_keys())

    def _get_next_obs_keys(self) -> list[str]:
        return list(self.config.policy.get_input_keys())[
            2:5
        ]  # only ["base_ang_vel", "joint_pos_rel", "joint_vel_rel"]

    def _construct_bilinear_model(self) -> BilinearRepresentation:
        cfg = self.config.bilinear
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        obs_dim = sum(
            int(self.env.observation_spec[k].shape[-1]) for k in self._get_obs_keys()
        )
        next_obs_dim = sum(
            int(self.env.observation_spec[k].shape[-1])
            for k in self._get_next_obs_keys()
        )
        return BilinearRepresentation(
            obs_dim=obs_dim,
            next_obs_dim=next_obs_dim,
            action_dim=action_spec.shape[-1],
            feature_dim=cfg.feature_dim,
            embed_dim=cfg.embed_dim,
            f_hidden_dims=cfg.f_hidden_dims,
            g_hidden_dims=cfg.g_hidden_dims,
            mu_hidden_dims=cfg.mu_hidden_dims,
            num_noises=cfg.num_noises,
            use_ema_for_policy=cfg.use_ema_for_policy,
            x_min=cfg.x_min,
            x_max=cfg.x_max,
            device=self.device,
        )

    def _construct_policy(
        self, policy_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        from torchrl.data import Bounded

        action_spec = self.env.action_spec_unbatched
        action_dim = int(action_spec.shape[-1])

        if isinstance(action_spec, Bounded):
            dist_cls, dist_kw = (
                TanhNormal,
                {
                    "low": action_spec.space.low,
                    "high": action_spec.space.high,
                    "tanh_loc": False,
                },
            )
        else:
            from torchrl.modules import IndependentNormal

            dist_cls, dist_kw = IndependentNormal, {}

        policy_head = policy_net or BilinearPolicyHead(
            bilinear_rep=self.bilinear_rep,
            num_cells=list(self.config.policy.num_cells),
            activation_fn=self.config.policy.activation_fn,
            action_dim=action_dim,
            log_std_init=self.config.ppo.log_std_init,
            log_std_min=self.config.ppo.log_std_min,
            log_std_max=self.config.ppo.log_std_max,
            clip_log_std=self.config.ppo.clip_log_std,
            detach_features=self.config.bilinear.detach_features_for_policy,
            device=self.device,
        )

        return ProbabilisticActor(
            TensorDictModule(
                policy_head,
                in_keys=self.config.policy.get_input_keys(),
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),
            distribution_class=dist_cls,
            distribution_kwargs=dist_kw,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )

    # -- SR training --

    def _store_transitions(self, batch: TensorDict) -> None:
        obs = self._concat_obs_from_td(batch)
        next_obs = self._concat_next_obs_from_td(batch)

        # Update running normalization statistics with new observations
        self.bilinear_rep.obs_norm.update(next_obs)

        transitions = TensorDict(
            {
                "obs": obs,
                "action": batch.get("action"),
                ("next", "obs"): next_obs,
            },
            batch_size=batch.batch_size,
        )
        self._sr_history_buffer.extend(transitions.reshape(-1).detach())

    def _sr_loss_from_batch(self, batch: TensorDict) -> tuple[dict[str, float], Tensor]:
        batch = batch.to(self.device)
        obs = cast(Tensor, batch.get("obs"))
        action = cast(Tensor, batch.get("action"))
        next_obs = cast(Tensor, batch.get(("next", "obs")))
        reward = torch.zeros(obs.shape[0], 1, device=self.device)
        metrics, loss, _ = self.bilinear_rep.compute_loss(obs, action, next_obs, reward)
        return metrics, loss

    def _train_sr_steps(self, steps: int) -> dict[str, float]:
        metrics_accum: dict[str, float] = {}
        if steps <= 0 or len(self._sr_history_buffer) == 0:
            return metrics_accum

        self.bilinear_rep.train()
        for _ in range(steps):
            metrics, loss = self._sr_loss_from_batch(self._sr_history_buffer.sample())
            self.sr_optim.zero_grad(set_to_none=True)
            loss.backward()
            self.sr_optim.step()
            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)

        self._sr_update_count += 1
        for k in metrics_accum:
            metrics_accum[k] /= float(steps)

        # Periodic sampling evaluation (DiffSR only)
        interval = self.config.bilinear.sample_eval_interval
        if interval > 0 and self._sr_update_count % interval == 0:
            sample_batch = self._sr_history_buffer.sample()
            sample_metrics = self._sr_sample_eval(sample_batch)
            metrics_accum.update(sample_metrics)
        return metrics_accum

    @torch.no_grad()
    def _sr_sample_eval(self, batch: TensorDict) -> dict[str, float]:
        """Run diffusion sampling to reconstruct s from s', measure distance."""
        self.bilinear_rep.eval()
        batch = batch.to(self.device)
        obs = cast(Tensor, batch.get("obs"))
        action = cast(Tensor, batch.get("action"))
        next_obs = cast(Tensor, batch.get(("next", "obs")))
        s_recon, _ = self.bilinear_rep.sample(s=obs, a=action, preserve_history=True)
        mse = (s_recon - next_obs).pow(2).sum(-1).mean()
        l1 = (s_recon - next_obs).abs().sum(-1).mean()
        self.bilinear_rep.train()
        return {
            "sample/recon_mse": mse.item(),
            "sample/recon_l1": l1.item(),
        }

    # -- Update / logging overrides --

    def update(
        self,
        batch: TensorDict,
        num_network_updates: int,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> tuple[TensorDict, int]:
        self._store_transitions(batch)

        # PPO first: update policy while representation is consistent with
        # the rollout log-probs.  SR update afterwards avoids shifting the
        # representation before PPO can use the batch, which would cause
        # large importance-sampling ratios and excessive clipping.
        result = super().update(batch, num_network_updates, expert_batch, has_expert)

        sr_metrics = self._train_sr_steps(self.config.bilinear.update_steps)

        self.bilinear_rep.update_ema(self.config.bilinear.ema_tau)
        # self.bilinear_rep.update_ema(1.0)

        # Collapse detection: compute feature statistics on a sample batch
        collapse_metrics = self._compute_collapse_metrics(batch)
        sr_metrics.update(collapse_metrics)

        for k, v in sr_metrics.items():
            self._pending_sr_metrics.setdefault(k, []).append(v)
        self._pending_sr_metrics.setdefault("history_buffer_size", []).append(
            float(len(self._sr_history_buffer))
        )
        return result

    @torch.no_grad()
    def _compute_collapse_metrics(self, batch: TensorDict) -> dict[str, float]:
        """Compute metrics to detect representation collapse in f(s) and bilinear phi."""
        obs = self._concat_obs_from_td(batch).reshape(-1, self.bilinear_rep.obs_dim)
        # Sub-sample for efficiency
        if obs.shape[0] > 4096:
            idx = torch.randperm(obs.shape[0], device=obs.device)[:4096]
            obs = obs[idx]

        # f(s) from online state_net
        f_s = self.bilinear_rep.compute_policy_representation(obs)  # (N, embed_dim)

        # Per-dimension std across the batch (low = collapse)
        f_std = f_s.std(dim=0)  # (embed_dim,)

        # Cosine similarity between random pairs (high = collapse)
        perm = torch.randperm(obs.shape[0], device=obs.device)
        cos_sim = F.cosine_similarity(f_s, f_s[perm], dim=-1).abs().mean()

        return {
            "collapse/f_s_std_mean": f_std.mean().item(),
            "collapse/f_s_cos_sim_random_pairs": cos_sim.item(),
            "collapse/f_s_l1_mean": f_s.abs().mean().item(),
        }

    def log_metrics(self, metrics: Mapping[str, Any], **kwargs) -> None:
        if self._pending_sr_metrics:
            merged = dict(metrics)
            for k, vals in self._pending_sr_metrics.items():
                merged[f"sr/{k}"] = sum(vals) / len(vals)
            self._pending_sr_metrics.clear()
            super().log_metrics(merged, **kwargs)
        else:
            super().log_metrics(metrics, **kwargs)
