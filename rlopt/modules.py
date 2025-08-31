from __future__ import annotations

import weakref
from numbers import Number
from typing import Dict, Optional, Sequence, Union
import math

import numpy as np
import torch
from packaging import version
from torch import distributions as D, nn

from torch.distributions import constraints
from torch.distributions.transforms import _InverseTransform

from torchrl.modules.distributions.truncated_normal import (
    TruncatedNormal as _TruncatedNormal,
)

from torchrl.modules.distributions.utils import (
    _cast_device,
    FasterTransformedDistribution,
    safeatanh_noeps,
    safetanh_noeps,
)

# speeds up distribution construction
D.Distribution.set_default_validate_args(False)

try:
    from torch.compiler import assume_constant_result
except ImportError:
    from torch._dynamo import assume_constant_result

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling

TORCH_VERSION = version.parse(torch.__version__).base_version
TORCH_VERSION_PRE_2_6 = version.parse(TORCH_VERSION) < version.parse("2.6.0")

EPS = 1e-6


class CustomTanhTransform(D.transforms.TanhTransform):

    def _inverse(self, y):
        # from stable_baselines3's `common.distributions.TanhBijector`
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """

        y = y.clamp(-1.0 + EPS, 1.0 - EPS)
        return 0.5 * (y.log1p() - (-y).log1p())

    def log_abs_det_jacobian(self, x, y):
        # From PyTorch `TanhTransform`
        """
        tl;dr log(1-tanh^2(x)) = log(sech^2(x))
                               = 2log(2/(e^x + e^(-x)))
                               = 2(log2 - log(e^x/(1 + e^(-2x)))
                               = 2(log2 - x - log(1 + e^(-2x)))
                               = 2(log2 - x - softplus(-2x))
        """

        return 2.0 * (math.log(2.0) - x - nn.functional.softplus(-2.0 * x))


class TanhNormalStable(D.TransformedDistribution):
    def __init__(self, loc, scale, event_dims=1):
        self._event_dims = event_dims
        self._t = [CustomTanhTransform()]
        self.update(loc, scale)

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - D.utils._sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.domain.event_dim,
            )
            y = x

        log_prob = log_prob + D.utils._sum_rightmost(
            self.base_dist.log_prob(y), event_dim - len(self.base_dist.event_shape)
        )

        log_prob = torch.clamp(
            log_prob, min=math.log10(EPS)
        )  # <- **CLAMPING THIS SEEMS TO RESOLVE THE ISSUE**
        return log_prob

    def update(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        self.loc = loc
        self.scale = scale
        if (
            hasattr(self, "base_dist")
            and (self.base_dist.base_dist.loc.shape == self.loc.shape)
            and (self.base_dist.base_dist.scale.shape == self.scale.shape)
        ):
            self.base_dist.base_dist.loc = self.loc
            self.base_dist.base_dist.scale = self.scale
        else:
            base = D.Independent(D.Normal(self.loc, self.scale), self._event_dims)
            super().__init__(base, self._t)

    @property
    def mode(self):
        m = self.base_dist.base_dist.mean
        for t in self.transforms:
            m = t(m)
        return m
