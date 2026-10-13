"""Paired denoising diagnostics without normalizer or optimizer updates."""

from __future__ import annotations

import torch
from torch import Tensor

from rlopt.agent.ipmd.module import DiffSRBilinear


@torch.no_grad()
def noise_level_losses(
    module: DiffSRBilinear, source: Tensor, z: Tensor, target: Tensor, *, seed: int
) -> dict[str, Tensor]:
    """Use identical Gaussian draws for real, shuffled, and zero-code losses.

    Losses sum over target coordinates, matching DiffSR's training objective.
    A local generator avoids perturbing training or expert-sampling RNG state.
    """
    if z.shape[0] < 2:
        msg = "Paired dynamics probes need at least two examples."
        raise ValueError(msg)
    generator = torch.Generator(device=target.device).manual_seed(seed)
    x0 = module.obs_norm.normalize(target)
    eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    # A nonzero cyclic shift guarantees no code is paired with its own row.
    shift = int(torch.randint(1, z.shape[0], (), generator=generator, device=z.device))
    codes = {"real_z": z, "shuffled_z": z.roll(shift, 0), "zero_z": torch.zeros_like(z)}
    phis = {name: module.forward_phi(source, code) for name, code in codes.items()}
    losses = {}
    for level in range(module.num_noises):
        alpha = module.alphabars[level]
        xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * eps
        time = torch.full((z.shape[0], 1), level, device=z.device, dtype=torch.long)
        fields = module.forward_mu(xt, time, s=source)
        for name, phi in phis.items():
            prediction = module.forward_eps(z_phi=phi, z_mu=fields)
            losses[f"noise_{level}/{name}"] = (prediction - eps).square().sum(-1).mean()
    for name in codes:
        losses[f"mean/{name}"] = torch.stack([
            losses[f"noise_{level}/{name}"] for level in range(module.num_noises)
        ]).mean()
    return losses
