"""Score-function correction of the real-window SVG(H) gradient.

REBAR's construction (Tucker et al. 2017) for a Gaussian action
``a = mu_theta(o) + eta``: for a control variate ``c`` that does not read the
sample's outcome,

    grad E[Q] = E[(Q - c) grad log pi] + E[grad_a c] grad mu.

SVG(H) supplies the second term. With noise inference the re-simulated return
equals the recorded one, so taking that as ``c`` makes the residual identically
zero. The control variate here is the linear ``c_t = q_t . eta_t`` instead, with
``q_t`` the action gradient of the objective on the policy's mean path (recorded
residuals kept, action noise zero). By Stein's lemma its pathwise term is the
mean-path SVG gradient, so the estimator is

    g = g_svg + sum_t gamma^t rho_t grad log pi(a_t | o_t),  rho_t = A_t - L_t,

with ``A_t`` the real advantage and ``L_t`` the linearized return change the
recorded noise caused from step t on. A slope read at the recorded action
contains curvature, ``d(eta) . eta = d(0) . eta + eta^T H eta``, which is why the
mean path is used. The score term runs PPO's clipped ratio, because windows come
from slightly older policies. It is far noisier than ``g_svg``, so
``CorrectionAverage`` averages it over iterations and caps it against ``g_svg``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def linearized_return(slopes: Tensor, eta: Tensor, mask: Tensor, decay: float) -> Tensor:
    """L_t = sum_{k >= t} decay^(k - t) q_k . eta_k, (B, H), over rows ``mask`` keeps."""
    effect = (slopes * eta).sum(-1) * mask
    out = torch.zeros_like(effect)
    run = torch.zeros_like(effect[:, 0])
    for t in range(effect.shape[1] - 1, -1, -1):
        run = effect[:, t] + decay * run
        out[:, t] = run
    return out


def masked_center(x: Tensor, mask: Tensor) -> Tensor:
    """``x`` minus its mean over the rows ``mask`` keeps. A constant offset has no
    expected effect on the score term but adds variance (critics fit on imagined
    rollouts undervalue real returns)."""
    return x - (x * mask).sum() / mask.sum().clamp_min(1.0)


def shuffle_rows(x: Tensor, mask: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """``x`` with its kept entries permuted among themselves: the same values with
    no relation to the row's noise (the control arm)."""
    flat = x.reshape(-1).clone()
    idx = mask.reshape(-1).nonzero().squeeze(-1)
    perm = torch.randperm(idx.numel(), device=x.device, generator=generator)
    flat[idx] = flat[idx[perm]]
    return flat.reshape(x.shape)


def clipped_score_surrogate(
    mu: Tensor,
    mu_old: Tensor,
    action: Tensor,
    residual: Tensor,
    weight: Tensor,
    sigma: float,
    clip_eps: float,
) -> tuple[Tensor, Tensor]:
    """PPO's clipped objective on the residual, mean over windows.

    ``mu`` (B, H, A) carries the actor's gradient, the rest are constants;
    ``weight`` (B, H) is gamma^t on kept rows. At ``mu == mu_old`` the gradient is
    ``sum_t weight_t residual_t grad log pi``. Returns the surrogate and the share
    of kept rows whose ratio left the clip range.
    """
    action, mu_old = action.detach(), mu_old.detach()
    residual, weight = residual.detach(), weight.detach()
    inv = 1.0 / (2.0 * sigma * sigma)
    log_ratio = ((action - mu_old).square().sum(-1) - (action - mu).square().sum(-1)) * inv
    ratio = log_ratio.clamp(-20.0, 20.0).exp()
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratio * residual, clipped * residual)
    keep = weight > 0
    outside = ((ratio - clipped).abs() > 0) & keep
    frac = outside.sum().float() / keep.sum().clamp_min(1).float()
    return (weight * surrogate).sum(1).mean(), frac


class CorrectionAverage:
    """Exponential average of the correction gradient, capped against ``g_svg``.

    One estimate is about ten times ``g_svg`` in norm and nearly all noise, so it
    is averaged over about ``2 / rate`` iterations. The cap keeps a noise-dominated
    average from setting the update's direction: the applied correction never
    exceeds ``cap`` times ``g_svg`` in norm, which also bounds how far this arm can
    leave plain SVG(H).
    """

    def __init__(self, rate: float, cap: float) -> None:
        if not 0.0 < rate <= 1.0:
            msg = f"rate must be in (0, 1], got {rate}."
            raise ValueError(msg)
        self.rate, self.cap = float(rate), float(cap)
        self.mean: Tensor | None = None
        self.steps = 0

    def update(self, correction: Tensor, svg: Tensor) -> tuple[Tensor, Tensor]:
        """Fold in one estimate; returns the correction to apply and its cap scale."""
        if self.mean is None:
            self.mean = torch.zeros_like(correction)
        self.steps += 1
        self.mean.mul_(1.0 - self.rate).add_(correction, alpha=self.rate)
        unbiased = self.mean / (1.0 - (1.0 - self.rate) ** self.steps)
        limit = self.cap * svg.norm()
        scale = (limit / unbiased.norm().clamp_min(1e-12)).clamp(max=1.0)
        return unbiased * scale, scale

    def state_dict(self) -> dict:
        return {"mean": self.mean, "steps": self.steps}

    def load_state_dict(self, state: dict) -> None:
        self.mean, self.steps = state["mean"], int(state["steps"])
