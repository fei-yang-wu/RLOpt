from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchrl.modules import MLP


def linear_beta_schedule(beta_min: float = 1e-4, beta_max: float = 0.02, T: int = 1000):
    return np.linspace(beta_min, beta_max, T)


def cosine_beta_schedule(s: float = 0.008, T: int = 1000):
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2.) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clip(None, 0.999)


def vp_beta_schedule(T: int = 1000):
    t = np.arange(1, T + 1)
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas


def get_noise_schedule(
    noise_schedule: str = "linear",
    num_noises: int = 1000,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
    s: float = 0.008,
):
    if noise_schedule == "linear":
        betas = linear_beta_schedule(beta_min, beta_max, num_noises)
    elif noise_schedule == "cosine":
        betas = cosine_beta_schedule(s, num_noises)
    elif noise_schedule == "vp":
        betas = vp_beta_schedule(num_noises)
    else:
        raise NotImplementedError(f"Unknown noise schedule: {noise_schedule}")
    alphas = 1 - betas
    alphabars = np.cumprod(alphas, axis=0)
    return (
        torch.as_tensor(betas, dtype=torch.float32),
        torch.as_tensor(alphas, dtype=torch.float32),
        torch.as_tensor(alphabars, dtype=torch.float32),
    )


class PositionalFeature(nn.Module):
    """Sinusoidal positional embedding for diffusion time steps."""

    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1).float()
        freqs = torch.arange(0, self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x @ freqs.unsqueeze(0)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.activation = activation
        if input_dim != hidden_dim:
            self.residual = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual = nn.Identity()
        if dropout and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dropout(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + self.residual(residual)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_dims: Sequence[int] = (),
        activation: Callable = F.relu,
        dropout: float = None,
        device: str = "cpu",
    ):
        super().__init__()
        assert len(hidden_dims) > 0, "hidden_dims must be a list of at least one integer"
        self.activation = activation
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        hidden_dims = [hidden_dims[0]] + list(hidden_dims)
        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(
                hidden_dims[i],
                hidden_dims[i + 1],
                dropout,
                activation,
            ) for i in range(len(hidden_dims) - 1)
        ])

    def forward(self, x):
        x = self.fc1(x)
        x = self.blocks(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class DDPM(nn.Module):
    """Diffusion model for spectral representation learning.

    Factorizes p(s'|s) with perturbation on the s branch.
    phi(s_t, t) -> Jacobian (feature_dim x obs_dim), mu(s') -> feature_dim.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int,
        phi_hidden_dims: Iterable[int],
        mu_hidden_dims: Iterable[int],
        reward_hidden_dims: Iterable[int],
        rff_dim: int | None,
        sample_steps: int,
        x_min: float = -10.0,
        x_max: float = 10.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.sample_steps = sample_steps
        self.x_min, self.x_max = x_min, x_max
        self.rff_dim = rff_dim

        # Noise schedule from the original DiffSR implementation (VP schedule)
        betas, alphas, alphabars = get_noise_schedule("vp", sample_steps)
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
        self.mlp_phi = ResidualMLP(
            input_dim=obs_dim + action_dim,
            output_dim=feature_dim,
            hidden_dims=phi_hidden_dims,
            activation=nn.Mish(),
        )

        self.mlp_mu = ResidualMLP(
            input_dim=obs_dim + 128,
            output_dim=obs_dim * feature_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
        )
        self.reward_head = MLP(
            in_features=feature_dim,
            out_features=1,
            num_cells=list(reward_hidden_dims),
            activation_class=nn.Mish,
        )

    def add_noise(
        self, x0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise_idx = torch.randint(
            0, self.sample_steps, (x0.shape[0],), device=x0.device
        ).long()
        alphabars = self.alphabars[noise_idx]
        eps = torch.randn_like(x0)
        xt = alphabars.sqrt() * x0 + (1 - alphabars).sqrt() * eps
        return xt, noise_idx, eps

    def forward_phi(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.mlp_phi(x)

    def forward_mu(self, sp: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_ff = self.mlp_t(t)
        all = torch.concat([sp, t_ff], dim=-1)
        all = self.mlp_mu(all)
        return all.reshape(-1, self.feature_dim, self.obs_dim)

    def forward_eps(
        self,
        s: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        sp: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        z_phi: torch.Tensor | None = None,
        z_mu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_phi is None:
            z_phi = self.forward_phi(s=s, a=a)
        if z_mu is None:
            z_mu = self.forward_mu(sp=sp, t=t)
        eps = torch.bmm(z_phi.unsqueeze(1), z_mu).squeeze(1)
        return eps

    def compute_loss(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        sp: torch.Tensor,
        r: torch.Tensor,
    ) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        x0 = sp
        xt, t, eps = self.add_noise(x0)

        z_phi = self.forward_phi(s=s, a=a)
        z_mu = self.forward_mu(sp=xt.detach(), t=t.unsqueeze(-1))
        eps_pred = self.forward_eps(z_phi=z_phi, z_mu=z_mu)
        diffusion_loss = (eps_pred - eps).pow(2).sum(-1).mean()

        reward_loss = F.mse_loss(self.reward_head(z_phi), r)

        metrics = ({
            "info/eps_l1_norm": eps.abs().mean(),
            "info/eps_pred_l1_norm": eps_pred.abs().mean(),
            "info/x0_mean": x0.mean(),
            "info/x0_std": x0.std(0).mean(),
            "info/x0_l1_norm": x0.abs().mean(),
            "loss/dynamics_loss": diffusion_loss.item(),
            "loss/reward_loss": reward_loss.item(),
        })
        return metrics, diffusion_loss, reward_loss

    @torch.no_grad()
    def sample(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        preserve_history: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Generate sp given (s, a) via reverse diffusion.

        The forward process noises sp; the reverse process denoises from
        x_T ~ N(0,I) back to sp, conditioned on (s, a) through phi.
        """
        info = {}
        shape = (s.shape[0], self.obs_dim)
        xt = torch.randn(shape, device=s.device)
        z_phi = self.forward_phi(s, a)  # (B, feature_dim) — constant across steps

        if preserve_history:
            info["sample_history"] = [xt.clone()]

        for t in reversed(range(self.sample_steps)):
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

        return xt, info



class FactorizedNCE(nn.Module):
    """Contrastive learning model for spectral representation (CTRL-SR).

    Factorizes p(s'|s) with perturbation on the s branch.
    phi(s_t, t) -> feature_dim, mu(s') -> feature_dim.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int,
        phi_hidden_dims: Sequence[int],
        mu_hidden_dims: Sequence[int],
        reward_hidden_dims: Sequence[int],
        rff_dim: int | None = None,
        num_noises: int = 0,
        linear: bool = False,
        ranking: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.rff_dim = rff_dim
        self.linear = linear
        self.ranking = ranking
        self._device = torch.device(device)

        if num_noises > 0:
            self.use_noise_perturbation = True
            self.N = num_noises
            betas, alphas, alphabars = get_noise_schedule("vp", num_noises)
            alphabars_prev = F.pad(alphabars[:-1], (1, 0), value=1.0)
            self.register_buffer("betas", betas[..., None])
            self.register_buffer("alphas", alphas[..., None])
            self.register_buffer("alphabars", alphabars[..., None])
            self.register_buffer("alphabars_prev", alphabars_prev[..., None])
        else:
            self.use_noise_perturbation = False
            self.N = 1

        self.mlp_t = nn.Sequential(
            PositionalFeature(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
        )
        # phi takes (s_t, t) -> feature_dim (perturbation on s)
        self.mlp_phi = ResidualMLP(
            input_dim=obs_dim + (128 if self.use_noise_perturbation else 0),
            output_dim=feature_dim,
            hidden_dims=phi_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device,
        )
        # mu takes (s'_t, t) -> feature_dim (perturbation on s')
        self.mlp_mu = ResidualMLP(
            input_dim=obs_dim + (128 if self.use_noise_perturbation else 0),
            output_dim=feature_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device,
        )

        self.reward_head = MLP(
            in_features=feature_dim,
            out_features=1,
            num_cells=list(reward_hidden_dims),
            activation_class=nn.Mish,
        )

        if not self.ranking:
            self.normalizer = nn.Parameter(
                torch.zeros(self.N, dtype=torch.float32)
            )

    def forward_phi(self, s: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """phi(s_t, t) -> feature_dim. Perturbation on s."""
        leading = s.shape[:-1]
        if t is not None:
            t_flat = t.reshape(-1, t.shape[-1])
            t_ff = self.mlp_t(t_flat).reshape(*leading, -1)
            s = torch.cat([s, t_ff], dim=-1)
        s_flat = s.reshape(-1, s.shape[-1])
        out = self.mlp_phi(s_flat)
        return torch.tanh(out).reshape(*leading, -1)

    def forward_mu(self, sp: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """mu(s'_t, t) -> feature_dim. Optionally with noise perturbation."""
        leading = sp.shape[:-1]
        if self.use_noise_perturbation:
            if t is None:
                t = torch.zeros(*leading, 1, dtype=torch.int64, device=sp.device)
            t_flat = t.reshape(-1, t.shape[-1])
            t_ff = self.mlp_t(t_flat).reshape(*leading, -1)
            sp = torch.cat([sp, t_ff], dim=-1)
        sp_flat = sp.reshape(-1, sp.shape[-1])
        out = self.mlp_mu(sp_flat)
        return torch.tanh(out).reshape(*leading, -1)

    def compute_logits(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        sp: torch.Tensor,
        z_phi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = s.shape[0]
        # Perturbation on s and s'
        if self.use_noise_perturbation:
            s_expanded = s.unsqueeze(0).repeat(self.N, 1, 1)
            t = torch.arange(0, self.N, device=s.device)
            t = t.repeat_interleave(B).reshape(self.N, B)
            alphabars = self.alphabars[t]
            eps = torch.randn_like(s_expanded)
            s_t = alphabars.sqrt() * s_expanded + (1 - alphabars).sqrt() * eps

            # Noise on sp (same noise levels, independent noise)
            sp_expanded = sp.unsqueeze(0).repeat(self.N, 1, 1)
            eps_sp = torch.randn_like(sp_expanded)
            sp_t = alphabars.sqrt() * sp_expanded + (1 - alphabars).sqrt() * eps_sp

            t = t.unsqueeze(-1)
        else:
            s_t = s.unsqueeze(0)
            sp_t = sp.unsqueeze(0)
            t = None

        if z_phi is None:
            z_phi = self.forward_phi(s_t, t)  # (N, B, z_dim)
        else:
            z_phi = z_phi.unsqueeze(0).repeat(self.N, 1, 1)

        z_mu = self.forward_mu(sp_t, t)  # (N, B, z_dim)
        logits = torch.bmm(z_phi, z_mu.transpose(-1, -2))  # (N, LB, RB)
        return logits

    def compute_loss(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        sp: torch.Tensor,
        r: torch.Tensor,
    ) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        B = s.shape[0]
        z_mu = self.forward_mu(sp)
        logits = self.compute_logits(s, a, sp)

        # Dynamic labels computed from actual batch size
        if self.ranking:
            labels = torch.arange(B, device=s.device).unsqueeze(0).repeat(self.N, 1)
        else:
            labels = torch.eye(B, device=s.device).unsqueeze(0).repeat(self.N, 1, 1)

        if self.linear:
            eff_logits = (F.softplus(logits, beta=3.0) + 1e-6).log()
            if self.ranking:
                model_loss = F.cross_entropy(eff_logits, labels, reduction="none").mean(-1)
            else:
                eff_logits = eff_logits * self.normalizer.exp()[:, None, None] / B
                model_loss = F.binary_cross_entropy_with_logits(
                    eff_logits, labels, reduction="none"
                ).mean([-2, -1])
        else:
            eff_logits = logits
            if self.ranking:
                model_loss = F.cross_entropy(eff_logits, labels, reduction="none").mean(-1)
            else:
                eff_logits = eff_logits + self.normalizer[:, None, None] - math.log(B)
                model_loss = F.binary_cross_entropy_with_logits(
                    eff_logits, labels, reduction="none"
                ).mean([-2, -1])

        if r.ndim == 1:
            r = r.unsqueeze(-1)
        reward_loss = F.mse_loss(self.reward_head(z_mu), r)

        # Compute info metrics
        pos_logits = logits[
            torch.arange(self.N, device=s.device).unsqueeze(1),
            torch.arange(B, device=s.device),
            torch.arange(B, device=s.device).unsqueeze(0).repeat(self.N, 1),
        ]  # (N, B)
        pos_logits_per_noise = pos_logits.mean(dim=1)  # (N,)
        neg_logits = (logits.sum(dim=-1) - pos_logits) / (logits.shape[-1] - 1)
        neg_logits_per_noise = neg_logits.mean(dim=-1)  # (N,)

        metrics = {
            "loss/dynamics_loss": model_loss.mean().item(),
            "loss/reward_loss": reward_loss.item(),
            "misc/mu_norm": z_mu.abs().mean().item(),
            "misc/mu_std": z_mu.std(0).mean().item(),
        }
        checkpoints = list(range(0, self.N, 5))
        metrics.update({
            f"detail/pos_logits_{i}": pos_logits_per_noise[i].item() for i in checkpoints
        })
        metrics.update({
            f"detail/neg_logits_{i}": neg_logits_per_noise[i].item() for i in checkpoints
        })
        metrics.update({
            f"detail/logit_gap_{i}": (pos_logits_per_noise[i] - neg_logits_per_noise[i]).item()
            for i in checkpoints
        })
        metrics.update({
            f"detail/model_loss_{i}": model_loss[i].item() for i in checkpoints
        })
        return metrics, model_loss.mean(), reward_loss

    def compute_feature(self, sp: torch.Tensor) -> torch.Tensor:
        """Return the clean mu(s') representation for control."""
        return self.forward_mu(sp)
