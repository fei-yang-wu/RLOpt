from __future__ import annotations

from collections.abc import Iterable

import torch
import torch as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.models.embeddings import get_timestep_embedding
from torchrl.modules import MLP


class PositionalFeature(nn.Module):
    """Sinusoidal positional embedding for diffusion time steps."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        timesteps = t.reshape(-1).float()
        return get_timestep_embedding(timesteps, self.embed_dim)


class ResidualMLP(nn.Module):
    """Residual MLP used by the diffusion feature network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        activation_cls: type[nn.Module] = nn.Mish,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            self.net = nn.Linear(input_dim, output_dim)
            return
        self.blocks = nn.ModuleList()
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    activation_cls(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            in_dim = hidden_dim
        self.activation = activation_cls()
        self.out = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "blocks"):
            return self.net(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            if x.shape == residual.shape:
                x = x + residual
            x = self.activation(x)
        return self.out(x)


class DDPM(nn.Module):
    """Diffusion model for linear MDP feature learning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
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
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.sample_steps = sample_steps
        self.x_min, self.x_max = x_min, x_max
        self.rff_dim = rff_dim

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=sample_steps, beta_schedule="linear"
        )

        self.mlp_s = MLP(
            in_features=obs_dim,
            out_features=embed_dim,
            num_cells=[embed_dim * 2],
            activation_class=nn.Mish,
        )
        self.mlp_a = MLP(
            in_features=action_dim,
            out_features=embed_dim,
            num_cells=[embed_dim * 2],
            activation_class=nn.Mish,
        )
        self.mlp_t = nn.Sequential(
            PositionalFeature(embed_dim),
            MLP(
                in_features=embed_dim,
                out_features=embed_dim,
                num_cells=[embed_dim * 2],
                activation_class=nn.Mish,
            ),
        )
        self.mlp_phi = ResidualMLP(
            input_dim=embed_dim * 2,
            output_dim=feature_dim,
            hidden_dims=phi_hidden_dims,
            activation_cls=nn.Mish,
        )
        self.mlp_mu = ResidualMLP(
            input_dim=obs_dim + embed_dim,
            output_dim=obs_dim * feature_dim,
            hidden_dims=mu_hidden_dims,
            activation_cls=nn.Mish,
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
        eps = torch.randn_like(x0)
        xt = self.noise_scheduler.add_noise(x0, eps, noise_idx)
        return xt, noise_idx, eps

    def encode_state(self, s: torch.Tensor) -> torch.Tensor:
        return self.mlp_s(s)

    def forward_phi(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        s_ff = self.mlp_s(s)
        a_ff = self.mlp_a(a)
        x = torch.cat([s_ff, a_ff], dim=-1)
        return self.mlp_phi(x)

    def forward_mu(self, sp: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_ff = self.mlp_t(t)
        all_feats = torch.cat([sp, t_ff], dim=-1)
        all_feats = self.mlp_mu(all_feats)
        return all_feats.reshape(-1, self.feature_dim, self.obs_dim)

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
            if s is None or a is None:
                msg = "s and a must be provided when z_phi is None."
                raise ValueError(msg)
            z_phi = self.forward_phi(s=s, a=a)
        if z_mu is None:
            if sp is None or t is None:
                msg = "sp and t must be provided when z_mu is None."
                raise ValueError(msg)
            z_mu = self.forward_mu(sp=sp, t=t)
        return torch.bmm(z_phi.unsqueeze(1), z_mu).squeeze(1)

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

        if r.ndim == 1:
            r = r.unsqueeze(-1)
        reward_pred = self.reward_head(z_phi)
        reward_loss = F.mse_loss(reward_pred, r)

        metrics = {
            "info/eps_l1_norm": eps.abs().mean().item(),
            "info/eps_pred_l1_norm": eps_pred.abs().mean().item(),
            "info/x0_mean": x0.mean().item(),
            "info/x0_std": x0.std(0).mean().item(),
            "info/x0_l1_norm": x0.abs().mean().item(),
            "loss/diffusion_loss": diffusion_loss.item(),
            "loss/reward_loss": reward_loss.item(),
        }
        return metrics, diffusion_loss, reward_loss

    def compute_feature(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.forward_phi(s, a)
