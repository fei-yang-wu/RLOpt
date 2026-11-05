"""DiffSR as the dynamics model of the world-model agent.

The SuperMimic model ``DynamicsModel`` regresses the normalized state
increment with a deterministic MLP, f(s, a, xi) -> d. This module learns the
same conditional, p(d | s, a, xi), with DiffSR's factorized denoiser: for a
noised increment d_t at diffusion step t,

    eps_hat(d_t, t; s, a) = phi(s, a)^T mu(d_t, t),

phi in R^F and mu in R^{F x D}, trained by one-step denoising
(``DiffSRBilinear``). Inputs, increments and their normalizers are those of
``DynamicsModel``, so both models are fit to the same targets in the same
units and roll out through the same ``integrate``.

Two read-outs of the learned model:

* ``sample``: ancestral DDPM sampling, one increment draw per call;
* ``forward``: a linear read-out of the spectral feature, W phi(s, a) + b.
  Under the factorization any conditional mean of the next state is linear in
  phi, so this head is the model's mean prediction. It is deterministic and
  differentiable, so it is also the drop-in for ``DynamicsModel`` in an
  imagined rollout. ``readout_detach_phi`` trains it as a probe (no gradient
  into phi) or jointly with the denoiser.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from rlopt.agent.ipmd.module import DiffSRBilinear
from rlopt.agent.wm_bptt.networks import RunningNorm


class DiffSRDynamics(nn.Module):
    """p(increment | s, a, xi) by DiffSR, with a linear mean read-out.

    ``action_slice`` locates the action inside the world-model input vector;
    the remaining coordinates (state features and sim context) are phi's
    state branch.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        action_slice: tuple[int, int],
        *,
        feature_dim: int = 128,
        embed_dim: int = 512,
        g_hidden_dims: Sequence[int] = (512, 512),
        mu_hidden_dims: Sequence[int] = (512, 512),
        phi_parameterization: str = "concat",
        mu_conditioning: str = "next",
        num_noises: int = 16,
        readout_detach_phi: bool = True,
        x_clip: float = 10.0,
    ) -> None:
        super().__init__()
        lo, hi = (int(action_slice[0]), int(action_slice[1]))
        if not 0 <= lo < hi <= in_dim:
            msg = f"action_slice {action_slice} is outside an input of width {in_dim}."
            raise ValueError(msg)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.action_slice = (lo, hi)
        keep = torch.cat((torch.arange(0, lo), torch.arange(hi, in_dim)))
        self.register_buffer("source_index", keep, persistent=False)
        self.in_norm = RunningNorm(in_dim, eps=1e-2)
        self.out_norm = RunningNorm(out_dim, eps=1e-4)
        self.sr = DiffSRBilinear(
            obs_dim=in_dim - (hi - lo),
            next_obs_dim=out_dim,
            action_dim=hi - lo,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            g_hidden_dims=tuple(g_hidden_dims),
            mu_hidden_dims=tuple(mu_hidden_dims),
            phi_parameterization=phi_parameterization,
            num_noises=num_noises,
            use_ema_for_policy=False,
            x_min=-x_clip,
            x_max=x_clip,
            mu_conditioning=mu_conditioning,
        )
        self.readout = nn.Linear(feature_dim, out_dim)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.readout_detach_phi = bool(readout_detach_phi)
        self.num_noises = int(num_noises)

    # ------------------------------------------------------------ helpers
    def _split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.in_norm(x)
        lo, hi = self.action_slice
        return z.index_select(-1, self.source_index), z[..., lo:hi]

    def phi(self, x: Tensor, *, bf16: bool = False) -> Tensor:
        src, act = self._split(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and x.is_cuda):
            out = self.sr.forward_phi(src, act)
        return out.float()

    def normalize_increment(self, inc: Tensor) -> Tensor:
        return (inc - self.out_norm.mean) / self.out_norm.std()

    def denormalize_increment(self, d: Tensor) -> Tensor:
        return d * self.out_norm.std() + self.out_norm.mean

    def _eps(self, phi: Tensor, src: Tensor, d_t: Tensor, t: Tensor, bf16: bool) -> Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and d_t.is_cuda):
            mu = self.sr.forward_mu(d_t, t.unsqueeze(-1), s=src)
        return torch.bmm(phi.unsqueeze(1), mu.float()).squeeze(1)

    # ------------------------------------------------------------ objectives
    def denoising_loss(
        self, x: Tensor, inc: Tensor, mask: Tensor | None = None, *, bf16: bool = False
    ) -> tuple[Tensor, Tensor]:
        """DiffSR's one-step loss on the normalized increment (sum over dims,
        mean over the rows ``mask`` keeps, as ``DiffSRBilinear``) and phi."""
        src, act = self._split(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and x.is_cuda):
            phi = self.sr.forward_phi(src, act)
        phi = phi.float()
        d0 = self.normalize_increment(inc)
        d_t, t, eps = self.sr.add_noise(d0)
        eps_hat = self._eps(phi, src, d_t, t, bf16)
        per = (eps_hat - eps).square().sum(-1)
        if mask is None:
            return per.mean(), phi
        m = mask.float()
        return (per * m).sum() / m.sum().clamp_min(1.0), phi

    def readout_from_phi(self, phi: Tensor) -> Tensor:
        """Denormalized mean increment from (possibly detached) phi."""
        if self.readout_detach_phi:
            phi = phi.detach()
        return self.denormalize_increment(self.readout(phi))

    def forward(self, x: Tensor, *, bf16: bool = False) -> Tensor:
        """Mean increment by the linear read-out (``DynamicsModel`` signature)."""
        return self.readout_from_phi(self.phi(x, bf16=bf16))

    def denoised(self, x: Tensor, d_t: Tensor, step: int, *, bf16: bool = False) -> Tensor:
        """Denormalized posterior mean E[d_0 | d_t, s, a] at diffusion step
        ``step`` (Tweedie), one denoiser pass. For a fixed ``d_t`` it is affine
        in phi(s, a) with the row's own matrix mu(d_t, t): the model's own
        linear read-out, no probe head."""
        src, act = self._split(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and x.is_cuda):
            phi = self.sr.forward_phi(src, act)
        t = torch.full((x.shape[0],), int(step), dtype=torch.long, device=x.device)
        eps_hat = self._eps(phi.float(), src, d_t, t, bf16)
        abar = self.sr.alphabars[int(step)]
        return self.denormalize_increment((d_t - (1 - abar).sqrt() * eps_hat) / abar.sqrt())

    # ------------------------------------------------------------ sampling
    def sample(
        self,
        x: Tensor,
        *,
        bf16: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """One denormalized increment draw per row by ancestral DDPM
        sampling (the ``DiffSRBilinear.sample`` recursion, in this module's
        normalization). Differentiable in x when grad is enabled."""
        src, act = self._split(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and x.is_cuda):
            phi = self.sr.forward_phi(src, act)
        phi = phi.float()
        sr = self.sr
        shape = (x.shape[0], self.out_dim)
        d = torch.randn(shape, device=x.device, generator=generator)
        for step in reversed(range(self.num_noises)):
            t = torch.full((shape[0],), step, dtype=torch.long, device=x.device)
            eps_hat = self._eps(phi, src, d, t, bf16)
            alpha, beta, abar = sr.alphas[step], sr.betas[step], sr.alphabars[step]
            d = (d - beta / (1 - abar).sqrt() * eps_hat) / alpha.sqrt()
            if step > 0:
                var = beta * (1 - sr.alphabars_prev[step]) / (1 - abar)
                noise = torch.randn(shape, device=x.device, generator=generator)
                d = d + var.clamp_min(1e-20).sqrt() * noise
            d = d.clamp(sr.x_min, sr.x_max)
        return self.denormalize_increment(d)
