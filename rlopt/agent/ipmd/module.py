"""Spectral representation modules for IPMD with bilinear structure.

All representations share the bilinear phi factorization:
    phi(s, a) = g(a)^T F(s),
where g(a) in R^{embed_dim} and F(s) in R^{embed_dim x feature_dim}.
The policy feature is F(s) z in R^{embed_dim}.

Different subclasses implement different spectral learning objectives:
    - DiffSR: Diffusion-based noise prediction loss
    - Speder: Spectral contrastive loss (maximize pos / minimize neg inner products)
    - CtrlSR: Classification-based (binary) NCE on perturbed s' at multiple noise levels
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rlopt.agent.ipmd.network import (
    PositionalFeature,
    ResidualMLP,
    get_noise_schedule,
)


# ---------------------------------------------------------------------------
# Empirical normalization
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
        """Recompute mean/var from the provided data (e.g. all effective buffer samples)."""
        x = x.reshape(-1, x.shape[-1])
        self.mean = x.mean(dim=0)
        self.var = x.var(dim=0, correction=0)
        self.count = torch.tensor(x.shape[0], dtype=torch.long, device=x.device)

    def normalize(self, x: Tensor) -> Tensor:
        if self.count < 2:
            return x
        return (x - self.mean) / (self.var.sqrt() + self.eps)

    def inv_normalize(self, x: Tensor) -> Tensor:
        return x * (self.var.sqrt() + self.eps) + self.mean


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BilinearSR(ABC, nn.Module):
    """Abstract base for bilinear spectral representation modules.

    phi(s, a) = g(a)^T F(s), with g(a) in R^{embed_dim} and
    F(s) in R^{embed_dim x feature_dim}. Subclasses implement mu and
    compute_loss for different SR objectives.
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
        use_ema_for_policy: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.next_obs_dim = next_obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        # --- Shared bilinear phi components ---
        # F(s): flat output is reshaped to (embed_dim, feature_dim) per sample.
        self.state_net = ResidualMLP(
            input_dim=obs_dim,
            output_dim=embed_dim * feature_dim,
            hidden_dims=f_hidden_dims,
            activation=nn.Mish(),
        )
        # Zero-init the final projection of F(s) so phi(s,a) = g(a)^T F(s) = 0
        # at step 0. This keeps the initial SR loss bounded (diffusion loss
        # starts at E[||eps||^2] = next_obs_dim; Speder inner products start at 0).
        nn.init.zeros_(self.state_net.fc2.weight)
        nn.init.zeros_(self.state_net.fc2.bias)
        # g(a): vector in R^{embed_dim}.
        self.action_net = ResidualMLP(
            input_dim=action_dim,
            output_dim=embed_dim,
            hidden_dims=g_hidden_dims,
            activation=nn.Mish(),
        )

        # --- EMA target state_net for policy ---
        if use_ema_for_policy:
            self.state_net_ema = copy.deepcopy(self.state_net).to(device)
            for p in self.state_net_ema.parameters():
                p.requires_grad_(False)
        else:
            self.state_net_ema = None

    # -- Bilinear phi (shared) --

    def _F(self, s: Tensor, use_ema: bool = False) -> Tensor:
        """Return F(s) with shape (B, embed_dim, feature_dim), tanh-bounded per element."""
        net = self.state_net_ema if (use_ema and self.state_net_ema is not None) else self.state_net
        # return torch.tanh(net(s).reshape(-1, self.embed_dim, self.feature_dim))
        return net(s).reshape(-1, self.embed_dim, self.feature_dim)

    def encode_state(self, s: Tensor) -> Tensor:
        return self._F(s)

    def encode_action(self, a: Tensor) -> Tensor:
        return self.action_net(a)

    def forward_phi(self, s: Tensor, a: Tensor) -> Tensor:
        """phi(s, a) = (1/sqrt(E)) * g(a)^T F(s) -> (B, feature_dim).

        F(s) and g(a) are both tanh-bounded; the 1/sqrt(embed_dim) factor
        keeps the bilinear inner product in sigmoid's linear regime, mirroring
        scaled dot-product attention.
        """
        F_s = self._F(s)                    # (B, E, D), tanh-bounded
        g_a = self.encode_action(a)         # (B, E),    tanh-bounded
        return torch.einsum("be,bef->bf", g_a, F_s) / math.sqrt(self.embed_dim)

    # -- EMA --

    @torch.no_grad()
    def update_ema(self, tau: float) -> None:
        if self.state_net_ema is None:
            return
        for p_ema, p_online in zip(
            self.state_net_ema.parameters(), self.state_net.parameters()
        ):
            p_ema.data.lerp_(p_online.data, tau)

    # -- Policy / Q representation (shared) --

    def compute_policy_representation(self, s: Tensor, z: Tensor = None) -> Tensor:
        """F(s) z -> (B, embed_dim). Detached; uses EMA state_net when enabled."""
        F_s = self._F(s, use_ema=True).detach()
        # return torch.einsum("bef,bf->be", F_s, z)
        component1 = torch.einsum("bef,bf->be", F_s, z)
        component2 = s.detach()
        return torch.concat([component1, component2], dim=-1)

    def compute_q_representation(self, s: Tensor, a: Tensor) -> Tensor:
        """Return phi(s, a) for Q-function use."""
        return self.forward_phi(s, a)

    # -- Abstract interface --

    @abstractmethod
    def compute_loss(
        self,
        s: Tensor,
        a: Tensor,
        sp: Tensor,
        r: Tensor,
    ) -> tuple[dict[str, float], Tensor, Tensor]:
        """Compute the spectral representation learning loss.

        Returns:
            metrics: Dict of scalar metrics for logging.
            main_loss: Primary SR loss (diffusion or spectral contrastive).
            aux_loss: Auxiliary loss (e.g. reward prediction), or zero tensor.
        """
        ...

    def sample(
        self,
        s: Tensor,
        a: Tensor,
        preserve_history: bool = False,
    ) -> tuple[Tensor, dict]:
        """Generate s' given (s, a). Only supported by diffusion-based SR."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support generative sampling."
        )

    @property
    def supports_sampling(self) -> bool:
        return False

    def update_obs_norm(self, next_obs: Tensor) -> None:
        """Update observation normalization statistics. No-op by default."""
        pass


# ---------------------------------------------------------------------------
# DiffSR implementation
# ---------------------------------------------------------------------------


class DiffSRBilinear(BilinearSR):
    r"""Bilinear spectral representation with diffusion-based learning.

    mu(s', t) outputs a Jacobian (feature_dim x next_obs_dim) so that
    phi(s,a)^T @ mu(s',t) predicts the noise added to s' in the forward
    diffusion process.
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
        super().__init__(
            obs_dim=obs_dim,
            next_obs_dim=next_obs_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            f_hidden_dims=f_hidden_dims,
            g_hidden_dims=g_hidden_dims,
            use_ema_for_policy=use_ema_for_policy,
            device=device,
        )
        self.num_noises = num_noises
        self.x_min, self.x_max = x_min, x_max

        # Observation normalization for diffusion
        self.obs_norm = EmpiricalNormalization(next_obs_dim)

        # VP noise schedule
        betas, alphas, alphabars = get_noise_schedule("vp", num_noises)
        alphabars_prev = F.pad(alphabars[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas[..., None])
        self.register_buffer("alphas", alphas[..., None])
        self.register_buffer("alphabars", alphabars[..., None])
        self.register_buffer("alphabars_prev", alphabars_prev[..., None])

        # Time embedding
        self.mlp_t = nn.Sequential(
            PositionalFeature(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
        )

        # mu(s', t) -> Jacobian (feature_dim x next_obs_dim)
        self.mu_net = ResidualMLP(
            input_dim=next_obs_dim + 128,
            output_dim=feature_dim * next_obs_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
        )

    def update_obs_norm(self, next_obs: Tensor) -> None:
        self.obs_norm.update(next_obs)

    def add_noise(self, x0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        noise_idx = torch.randint(
            0, self.num_noises, (x0.shape[0],), device=x0.device
        ).long()
        alphabars = self.alphabars[noise_idx]
        eps = torch.randn_like(x0)
        xt = alphabars.sqrt() * x0 + (1 - alphabars).sqrt() * eps
        return xt, noise_idx, eps

    def forward_mu(self, sp: Tensor, t: Tensor) -> Tensor:
        """mu(s', t) -> Jacobian (B, feature_dim, next_obs_dim)."""
        t_ff = self.mlp_t(t)
        x = torch.concat([sp, t_ff], dim=-1)
        x = self.mu_net(x)
        return x.reshape(-1, self.feature_dim, self.next_obs_dim)

    def forward_eps(
        self,
        s: Tensor | None = None,
        a: Tensor | None = None,
        sp: Tensor | None = None,
        t: Tensor | None = None,
        z_phi: Tensor | None = None,
        z_mu: Tensor | None = None,
    ) -> Tensor:
        """Predict noise: eps_pred = phi(s,a)^T @ mu(s',t) -> (B, next_obs_dim)."""
        if z_phi is None:
            z_phi = self.forward_phi(s, a)
        if z_mu is None:
            z_mu = self.forward_mu(sp, t)
        return torch.bmm(z_phi.unsqueeze(1), z_mu).squeeze(1)

    def compute_loss(
        self,
        s: Tensor,
        a: Tensor,
        sp: Tensor,
        r: Tensor,
    ) -> tuple[dict[str, float], Tensor, Tensor]:
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

    @property
    def supports_sampling(self) -> bool:
        return True

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
        z_phi = self.forward_phi(s, a)

        if preserve_history:
            info["sample_history"] = [xt.clone()]

        for t in reversed(range(self.num_noises)):
            z = torch.randn_like(xt)
            timestep = torch.full(
                (xt.shape[0],), t, dtype=torch.int64, device=s.device
            )
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
                1.0 / self.alphas[timestep].sqrt()
                * (xt - self.betas[timestep] / (1 - self.alphabars[timestep]).sqrt() * eps_pred)
                + sigma_t * z
            )
            xt = xt.clip(self.x_min, self.x_max)

            if preserve_history:
                info["sample_history"].append(xt.clone())

        xt = self.obs_norm.inv_normalize(xt)
        return xt, info


# ---------------------------------------------------------------------------
# Speder implementation
# ---------------------------------------------------------------------------


class SpederBilinear(BilinearSR):
    r"""Bilinear spectral representation with spectral contrastive learning.

    mu(s') outputs a feature vector in R^d (tanh-bounded). The loss maximizes
    the inner product phi(s,a)^T mu(s') for matching (s,a)->s' pairs and
    minimizes it for non-matching (negative) pairs from the same batch:

        L = lam * E[off-diag(inner^2)] / (B*(B-1)) - 2 * E[diag(inner)]

    Optional VP noise perturbation on s' creates multiple noise levels for
    robustness (as in the original Speder paper).
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
        num_noises: int = 25,
        lam: float = 1024.0,
        use_ema_for_policy: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            next_obs_dim=next_obs_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            f_hidden_dims=f_hidden_dims,
            g_hidden_dims=g_hidden_dims,
            use_ema_for_policy=use_ema_for_policy,
            device=device,
        )
        self.lam = lam
        self.num_noises = num_noises

        # Observation normalization for diffusion
        self.obs_norm = EmpiricalNormalization(next_obs_dim)

        if num_noises > 0:
            self.use_noise_perturbation = True
            betas, alphas, alphabars = get_noise_schedule("vp", num_noises)
            alphabars_prev = F.pad(alphabars[:-1], (1, 0), value=1.0)
            self.register_buffer("betas", betas[..., None])
            self.register_buffer("alphas", alphas[..., None])
            self.register_buffer("alphabars", alphabars[..., None])
            self.register_buffer("alphabars_prev", alphabars_prev[..., None])
        else:
            self.use_noise_perturbation = False

        # Time embedding (used when noise perturbation is enabled)
        self.mlp_t = nn.Sequential(
            PositionalFeature(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
        )

        # mu(s', [t]) -> feature_dim (tanh-bounded)
        self.mu_net = ResidualMLP(
            input_dim=next_obs_dim + (128 if self.use_noise_perturbation else 0),
            output_dim=feature_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
        )

    def forward_mu(self, sp: Tensor, t: Tensor | None = None) -> Tensor:
        """mu(s', [t]) -> (*, feature_dim), tanh-bounded."""
        leading = sp.shape[:-1]
        if t is not None:
            t_flat = t.reshape(-1, t.shape[-1])
            t_ff = self.mlp_t(t_flat).reshape(*leading, -1)
            sp = torch.concat([sp, t_ff], dim=-1)
        sp_flat = sp.reshape(-1, sp.shape[-1])
        out = self.mu_net(sp_flat)
        return torch.tanh(out).reshape(*leading, -1)

    def update_obs_norm(self, next_obs: Tensor) -> None:
        self.obs_norm.update(next_obs)

    def compute_loss(
        self,
        s: Tensor,
        a: Tensor,
        sp: Tensor,
        r: Tensor,
    ) -> tuple[dict[str, float], Tensor, Tensor]:
        B = sp.shape[0]
        N = self.num_noises if self.use_noise_perturbation else 1
        z_phi = self.forward_phi(s, a)  # (B, feature_dim)
        sp = self.obs_norm.normalize(sp)

        # Optionally perturb s' at multiple noise levels
        if self.use_noise_perturbation:
            sp_expanded = sp.unsqueeze(0).repeat(self.num_noises, 1, 1)  # (N, B, obs)
            t = torch.arange(0, self.num_noises, device=sp.device)
            t = t.repeat_interleave(B).reshape(self.num_noises, B)
            alphabars = self.alphabars[t]
            eps = torch.randn_like(sp_expanded)
            tilde_sp = alphabars.sqrt() * sp_expanded + (1 - alphabars).sqrt() * eps
            t = t.unsqueeze(-1)  # (N, B, 1)
        else:
            tilde_sp = sp.unsqueeze(0)  # (1, B, obs)
            t = None

        z_mu = self.forward_mu(tilde_sp, t)  # (N, B, feature_dim)
        z_phi = z_phi.unsqueeze(0).repeat(N, 1, 1)  # (N, B, feature_dim)
        inner = torch.bmm(z_phi, z_mu.transpose(-1, -2))  # (N, B, B)

        # Positive: diagonal (matching pairs)
        pos = torch.diagonal(inner, dim1=-2, dim2=-1)  # (N, B)
        pos_loss = pos.mean()

        # Negative: off-diagonal squared (non-matching pairs)
        tau = 1.0
        inner_logcosh = torch.log(torch.cosh(inner / tau + 1e-6)) * tau * tau
        neg_loss = inner_logcosh.sum(dim=[-2, -1]) / B / B
        neg_loss = neg_loss.mean()
        model_loss = 2 * neg_loss - 2 * pos_loss
        # neg_loss = neg_loss.mean()

        # inner_sq = inner.pow(2)
        # diag_sq_sum = torch.diagonal(inner_sq, dim1=-2, dim2=-1).sum(-1)
        # neg_loss = (inner_sq.sum(dim=[-2, -1]) - diag_sq_sum) / (B * (B - 1))
        # neg_loss = neg_loss.mean()

        # model_loss = self.lam * neg_loss - 2 * pos_loss

        # Logging metrics
        with torch.no_grad():
            pos_per_noise = pos.mean(dim=-1)  # (N,)
            neg_per_noise = (
                (inner.sum(dim=[-2, -1]) - pos.sum(dim=-1)) / (B * (B - 1))
            )

        metrics = {
            "loss/dynamics_loss": model_loss.item(),
            "info/obs_norm_mean": self.obs_norm.mean.abs().mean().item(),
            "info/obs_norm_std": self.obs_norm.var.sqrt().mean().item(),
            "misc/phi_norm": z_phi.abs().mean().item(),
            "misc/phi_std": z_phi[0].std(0).mean().item(),
            "misc/mu_norm": z_mu[0].abs().mean().item(),
            "misc/mu_std": z_mu[0].std(0).mean().item(),
        }
        checkpoints = list(range(0, N, max(1, N // 5)))
        for i in checkpoints:
            metrics[f"detail/pos_prob_{i}"] = pos_per_noise[i].item()
            metrics[f"detail/neg_prob_{i}"] = neg_per_noise[i].item()
            metrics[f"detail/prob_gap_{i}"] = (pos_per_noise[i] - neg_per_noise[i]).item()

        return metrics, model_loss, torch.tensor(0.0, device=s.device)


# ---------------------------------------------------------------------------
# CtrlSR (binary NCE) implementation
# ---------------------------------------------------------------------------


class CtrlSRBilinear(BilinearSR):
    r"""Bilinear spectral representation trained with classification-based (binary) NCE.

    phi(s, a) = g(a)^T F(s) (inherited bilinear factorization).
    mu(s', t) maps a noise-perturbed next state and its noise-level index t
    to a feature in R^{feature_dim} (tanh-bounded).

    For each minibatch of size B, we form an (N, B, B) logit matrix
        L_{n, i, j} = phi(s_i, a_i)^T mu(s'_{n,j}, t=n),
    where s'_{n,j} is the j-th next state perturbed at noise level n. The
    matching pairs lie on the diagonal. Each entry is treated as an
    independent binary classification problem (positive on the diagonal,
    negative off-diagonal). With K = B - 1 in-batch negatives per positive,
    the canonical NCE bias `- log K` makes the optimum match the true
    log-density-ratio and prevents the (B-1) negatives from dominating the
    per-row gradient at initialization:
        loss = BCE(logits - log(B - 1), eye(B)).
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
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            next_obs_dim=next_obs_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            f_hidden_dims=f_hidden_dims,
            g_hidden_dims=g_hidden_dims,
            use_ema_for_policy=use_ema_for_policy,
            device=device,
        )
        self.num_noises = num_noises
        self.use_noise_perturbation = num_noises > 0
        self.N = num_noises if self.use_noise_perturbation else 1

        self.obs_norm = EmpiricalNormalization(next_obs_dim)

        if self.use_noise_perturbation:
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

        self.mu_net = ResidualMLP(
            input_dim=next_obs_dim + (128 if self.use_noise_perturbation else 0),
            output_dim=feature_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
        )

    def forward_mu(self, sp: Tensor, t: Tensor | None = None) -> Tensor:
        """mu(s', [t]) -> (*, feature_dim), tanh-bounded."""
        leading = sp.shape[:-1]
        if t is not None:
            t_flat = t.reshape(-1, t.shape[-1])
            t_ff = self.mlp_t(t_flat).reshape(*leading, -1)
            sp = torch.concat([sp, t_ff], dim=-1)
        sp_flat = sp.reshape(-1, sp.shape[-1])
        out = self.mu_net(sp_flat)
        return torch.tanh(out).reshape(*leading, -1)

    def update_obs_norm(self, next_obs: Tensor) -> None:
        self.obs_norm.update(next_obs)

    def compute_loss(
        self,
        s: Tensor,
        a: Tensor,
        sp: Tensor,
        r: Tensor,
    ) -> tuple[dict[str, float], Tensor, Tensor]:
        B = sp.shape[0]
        z_phi = self.forward_phi(s, a)  # (B, feature_dim)
        sp = self.obs_norm.normalize(sp)

        if self.use_noise_perturbation:
            sp_expanded = sp.unsqueeze(0).repeat(self.N, 1, 1)  # (N, B, obs)
            t = torch.arange(0, self.N, device=sp.device)
            t = t.repeat_interleave(B).reshape(self.N, B)
            alphabars = self.alphabars[t]
            eps = torch.randn_like(sp_expanded)
            sp_t = alphabars.sqrt() * sp_expanded + (1 - alphabars).sqrt() * eps
            t = t.unsqueeze(-1)  # (N, B, 1)
        else:
            sp_t = sp.unsqueeze(0)  # (1, B, obs)
            t = None

        z_mu = self.forward_mu(sp_t, t)  # (N, B, feature_dim)
        z_phi_expanded = z_phi.unsqueeze(0).repeat(self.N, 1, 1)
        logits = torch.bmm(z_phi_expanded, z_mu.transpose(-1, -2))  # (N, B, B)

        labels = torch.eye(B, device=s.device).unsqueeze(0).expand(self.N, B, B)
        # NCE noise-prior correction: K = B - 1 negatives per positive.
        log_K = math.log(B - 1) if B > 1 else 0.0
        eff_logits = logits - log_K
        per_noise_loss = F.binary_cross_entropy_with_logits(
            eff_logits, labels, reduction="none"
        ).mean([-2, -1])  # (N,)
        model_loss = per_noise_loss.mean()

        with torch.no_grad():
            pos_logits = torch.diagonal(logits, dim1=-2, dim2=-1)  # (N, B)
            pos_per_noise = pos_logits.mean(dim=-1)  # (N,)
            neg_per_noise = (logits.sum(dim=[-2, -1]) - pos_logits.sum(dim=-1)) / (
                B * (B - 1)
            )

        metrics = {
            "loss/dynamics_loss": model_loss.item(),
            "info/obs_norm_mean": self.obs_norm.mean.abs().mean().item(),
            "info/obs_norm_std": self.obs_norm.var.sqrt().mean().item(),
            "info/obs_norm_count": float(self.obs_norm.count.item()),
            "misc/phi_norm": z_phi.abs().mean().item(),
            "misc/phi_std": z_phi.std(0).mean().item(),
            "misc/mu_norm": z_mu[0].abs().mean().item(),
            "misc/mu_std": z_mu[0].std(0).mean().item(),
        }
        checkpoints = list(range(0, self.N, max(1, self.N // 5)))
        for i in checkpoints:
            metrics[f"detail/pos_logits_{i}"] = pos_per_noise[i].item()
            metrics[f"detail/neg_logits_{i}"] = neg_per_noise[i].item()
            metrics[f"detail/logit_gap_{i}"] = (pos_per_noise[i] - neg_per_noise[i]).item()
            metrics[f"detail/model_loss_{i}"] = per_noise_loss[i].item()

        return metrics, model_loss, torch.tensor(0.0, device=s.device)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SR_REGISTRY: dict[str, type[BilinearSR]] = {
    "diffsr": DiffSRBilinear,
    "speder": SpederBilinear,
    "ctrlsr": CtrlSRBilinear,
}


def build_bilinear_sr(sr_type: str, **kwargs) -> BilinearSR:
    """Build a BilinearSR module by name.

    Args:
        sr_type: One of "diffsr" or "speder".
        **kwargs: Forwarded to the chosen class constructor.
    """
    cls = SR_REGISTRY.get(sr_type)
    if cls is None:
        raise ValueError(
            f"Unknown sr_type={sr_type!r}. Available: {list(SR_REGISTRY.keys())}"
        )
    return cls(**kwargs)
