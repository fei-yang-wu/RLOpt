"""FlashSAC building blocks: norm-controlled layers with unit-norm weight projection.

Adapted from Holiday-Robot/FlashSAC (commit 87edc90).

MIT License. Copyright (c) 2026 Holiday Robotics.
See LICENSE-FlashSAC.txt in this directory for the full notice.

Every layer that owns a constrained parameter exposes ``normalize_parameters()``.
The learner calls it after each optimizer step:

- linear weights: every row is projected to unit norm,
- BatchNorm affine (scale and bias together): projected to norm ``sqrt(d)``,
- RMSNorm scale: projected to norm ``sqrt(d)``.

BatchNorm modes are passed as an argument (``training``) and never read from
``module.training``. This keeps the modules safe for CUDA-graph capture and
``torch.compile``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def safe_tanh_log_det_jacobian(x: Tensor) -> Tensor:
    """Return ``log(1 - tanh(x)^2)`` in a numerically stable form."""
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


def _project_affine(parts: list[Tensor]) -> None:
    """Rescale the joint vector made of ``parts`` (last dim) to norm sqrt(d), in place."""
    dim = parts[0].shape[-1]
    sqsum = sum(torch.sum(p * p, dim=-1, keepdim=True) for p in parts)
    factor = math.sqrt(dim) * torch.rsqrt(sqsum + 1e-8)
    for p in parts:
        p.copy_(p * factor)


class UnitLinear(nn.Module):
    """Bias-free linear layer whose rows are projected to unit norm."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.w(x)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """BatchNorm with momentum 0.01. Scale and bias are projected to norm sqrt(d)."""

    def __init__(
        self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: Tensor, training: bool) -> Tensor:
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        _project_affine([self.weight.data, self.bias.data])


class UnitRMSNorm(nn.Module):
    """RMSNorm whose scale is projected to norm sqrt(d)."""

    def __init__(self, input_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        _project_affine([self.weight.data])


class EnsembleUnitLinear(nn.Module):
    """``num_ensemble`` independent unit-row linear maps, evaluated as one einsum."""

    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for i in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[i], gain=1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    """BatchNorm per ensemble member, layout ``(N, batch, dim)``.

    In training mode the statistics come from the batch and the running
    buffers move by ``lerp`` with Bessel-corrected variance. The batch size
    is read from the tensor shape, so there is no Python-side constant.
    """

    def __init__(
        self,
        num_ensemble: int,
        input_dim: int,
        momentum: float = 0.01,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

    def forward(self, x: Tensor, training: bool) -> Tensor:
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                b = x.shape[1]
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_(
                    (var.squeeze(1) * (b / (b - 1))).float(), self.momentum
                )
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(
                self.running_var.unsqueeze(1) + self.eps
            )
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        _project_affine([self.weight.data, self.bias.data])


class EnsembleUnitRMSNorm(nn.Module):
    """RMSNorm per ensemble member, layout ``(N, batch, dim)``."""

    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        _project_affine([self.weight.data])


def collect_normalizable(module: nn.Module) -> list[nn.Module]:
    """Return all submodules that own a weight-norm projection."""
    return [m for m in module.modules() if hasattr(m, "normalize_parameters")]
