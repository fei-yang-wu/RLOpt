"""Gaussian policy components for continuous action spaces.

This module provides components for building Gaussian (Normal) policies with
learned standard deviation parameters, supporting both PPO and SAC algorithms.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StateIndependentLogStd(nn.Module):
    """Learnable state-independent log standard deviation for Gaussian policies.

    This module maintains a learnable log_std parameter that is independent of
    the input state. The standard deviation is computed as exp(log_std) and
    can optionally be clamped to a specified range.

    This approach is commonly used in PPO and matches RSL-RL's implementation.

    Args:
        action_dim: Dimension of the action space.
        log_std_init: Initial value for log standard deviation. Default 0.0
            corresponds to std=1.0.
        log_std_min: Minimum value for log_std when clipping is enabled.
        log_std_max: Maximum value for log_std when clipping is enabled.
        clip_log_std: Whether to clamp log_std to [log_std_min, log_std_max].
        device: Device to place the parameter on.

    Example:
        >>> log_std_module = StateIndependentLogStd(action_dim=4, log_std_init=0.0)
        >>> loc = policy_mlp(obs)  # Shape: [batch, 4]
        >>> scale = log_std_module(loc)  # Shape: [batch, 4]
    """

    def __init__(
        self,
        action_dim: int,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        clip_log_std: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init, device=device)
        )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clip_log_std = clip_log_std

    def forward(self, loc: Tensor) -> Tensor:
        """Compute scale (std) expanded to match loc shape.

        Args:
            loc: Mean tensor from policy network, shape [..., action_dim].

        Returns:
            Scale tensor, shape [..., action_dim].
        """
        log_std = self.log_std
        if self.clip_log_std:
            log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        scale = torch.exp(log_std)
        return scale.expand_as(loc)

    def get_log_std(self) -> Tensor:
        """Get the (optionally clamped) log standard deviation."""
        log_std = self.log_std
        if self.clip_log_std:
            log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return log_std


class GaussianPolicyHead(nn.Module):
    """Gaussian policy head with learned state-independent log standard deviation.

    Wraps a base network (that outputs action means) with a learnable log_std
    parameter to output both mean (loc) and standard deviation (scale) for
    a Gaussian action distribution.

    This is the standard parameterization used in PPO and can also be used
    in SAC for consistency.

    Args:
        base: Base network that maps observations to action means.
        action_dim: Dimension of the action space.
        log_std_init: Initial value for log standard deviation.
        log_std_min: Minimum log_std when clipping is enabled.
        log_std_max: Maximum log_std when clipping is enabled.
        clip_log_std: Whether to clamp log_std values.
        device: Device for the log_std parameter.

    Example:
        >>> base_mlp = MLP(in_features=obs_dim, out_features=action_dim, ...)
        >>> policy = GaussianPolicyHead(base_mlp, action_dim=4)
        >>> loc, scale = policy(obs)
    """

    def __init__(
        self,
        base: nn.Module,
        action_dim: int,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        clip_log_std: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.log_std_module = StateIndependentLogStd(
            action_dim=action_dim,
            log_std_init=log_std_init,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            clip_log_std=clip_log_std,
            device=device,
        )

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute action mean and standard deviation.

        Args:
            obs: Observation tensor.

        Returns:
            Tuple of (loc, scale) tensors for the Gaussian distribution.
        """
        loc = self.base(obs)
        scale = self.log_std_module(loc)
        return loc, scale

    @property
    def log_std(self) -> nn.Parameter:
        """Access the log_std parameter directly."""
        return self.log_std_module.log_std
