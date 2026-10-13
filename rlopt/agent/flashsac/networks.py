"""FlashSAC actor, double categorical critic and temperature.

Adapted from Holiday-Robot/FlashSAC (commit 87edc90).

MIT License. Copyright (c) 2026 Holiday Robotics.
See LICENSE-FlashSAC.txt in this directory for the full notice.

The actor works in tanh space: it returns ``tanh(u)`` in ``[-1, 1]`` and the
log-probability of that tanh action. The physical action bound is applied by
the learner, outside these modules.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rlopt.agent.flashsac.layers import (
    EnsembleUnitBatchNorm,
    EnsembleUnitLinear,
    EnsembleUnitRMSNorm,
    UnitBatchNorm,
    UnitLinear,
    UnitRMSNorm,
    safe_tanh_log_det_jacobian,
)

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class FlashSACEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: Tensor, training: bool) -> Tensor:
        return self.w(self.norm(x, training=training))


class FlashSACBlock(nn.Module):
    """Inverted-residual block: linear, BN, ReLU, linear, BN, ReLU, skip."""

    def __init__(self, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: Tensor, training: bool) -> Tensor:
        residual = x
        x = F.relu(self.norm1(self.w1(x), training=training))
        x = F.relu(self.norm2(self.w2(x), training=training))
        return x + residual


class NormalTanhPolicy(nn.Module):
    """Gaussian head with ``log_std`` squashed into ``[LOG_STD_MIN, LOG_STD_MAX]``."""

    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_w = UnitLinear(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))

    def get_mean_and_std(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean = F.linear(x, self.mean_w.w.weight, self.mean_bias)
        raw_log_std = F.linear(x, self.std_w.w.weight, self.std_bias)
        log_std = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * 0.5 * (
            1 + torch.tanh(raw_log_std)
        )
        return mean, torch.exp(log_std)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean, std = self.get_mean_and_std(x)
        dist = torch.distributions.Normal(mean, std, validate_args=False)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action) - safe_tanh_log_det_jacobian(raw_action)
        return torch.tanh(raw_action), log_prob.sum(1)


class FlashSACActor(nn.Module):
    """Actor. ``forward`` returns ``(tanh_action, log_prob)`` for a reparameterised sample."""

    def __init__(
        self, num_blocks: int, input_dim: int, hidden_dim: int, action_dim: int
    ) -> None:
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim, hidden_dim)
        self.encoder = nn.ModuleList(
            [FlashSACBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim, action_dim)

    def features(self, observations: Tensor, training: bool) -> Tensor:
        x = self.embedder(observations, training)
        for block in self.encoder:
            x = block(x, training)
        return self.post_norm(x)

    def mean_and_std(
        self, observations: Tensor, training: bool
    ) -> tuple[Tensor, Tensor]:
        return self.predictor.get_mean_and_std(self.features(observations, training))

    def forward(self, observations: Tensor, training: bool) -> tuple[Tensor, Tensor]:
        return self.predictor(self.features(observations, training))


class EnsembleFlashSACEmbedder(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: Tensor, training: bool) -> Tensor:
        return self.w(self.norm(x, training=training))


class EnsembleFlashSACBlock(nn.Module):
    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: Tensor, training: bool) -> Tensor:
        residual = x
        x = F.relu(self.norm1(self.w1(x), training=training))
        x = F.relu(self.norm2(self.w2(x), training=training))
        return x + residual


class EnsembleCategoricalValue(nn.Module):
    """Categorical value head. Returns expected values and per-atom log-probs."""

    def __init__(
        self,
        num_ensemble: int,
        hidden_dim: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
    ) -> None:
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_atoms)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_atoms))
        self.register_buffer(
            "bin_values",
            torch.linspace(v_min, v_max, num_atoms, dtype=torch.float32).reshape(
                1, 1, -1
            ),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.w(x) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(logits, dim=-1)
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)
        return value, log_prob


class FlashSACDoubleCritic(nn.Module):
    """Two independent critic heads fused into batched ops, layout ``(N, batch, dim)``.

    ``forward(obs, action, training)`` returns ``(q_values (N, B), log_probs (N, B, atoms))``.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        num_qs: int = 2,
    ) -> None:
        super().__init__()
        self.num_qs = num_qs
        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList(
            [EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)]
        )
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_qs, hidden_dim, num_atoms, v_min, v_max
        )

    def forward(
        self, observations: Tensor, actions: Tensor, training: bool
    ) -> tuple[Tensor, Tensor]:
        x = torch.cat((observations, actions), dim=-1)
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        return self.predictor(self.post_norm(x))


class FlashSACTemperature(nn.Module):
    """Entropy temperature ``alpha = exp(log_temp)``."""

    def __init__(self, initial_value: float = 0.01) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(
            torch.tensor([math.log(initial_value)], dtype=torch.float32)
        )

    def forward(self) -> Tensor:
        return torch.exp(self.log_temp)
