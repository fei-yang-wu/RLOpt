"""Deterministic latent planner co-trained with the skill encoder.

The planner regresses the encoder's code from the causal past and the
instruction::

    z_hat = P(X, l)

where ``X`` is phi's source (the current macro state plus its past chunk) and
``l`` a fixed sentence embedding of the clip caption. Read as a unit-variance
Gaussian factor ``p(z | X, l) = N(z; P(X, l), I)`` of the joint model
``p(z, Y | X, l) = p(z | X, l) p(Y | X, z)``, its negative log-likelihood is
``0.5 * ||z - P(X, l)||^2`` summed over code dimensions, which is the loss
returned here.

Unlike the language-tilted flow planner, this factor may be trained JOINTLY
with the encoder: when the trainer does not detach the target, the gradient of
the Gaussian NLL reaches the encoder and pulls the code toward what the past
and the instruction already predict.

Caution (measured 2026-10-05, encoder-ablation-5b `joint_planner`): with a
non-detached target this fixed-variance NLL is NOT scale-invariant, so it is
lowered by shrinking the code. SIGReg's penalty is bounded and the affine_poe
NTP head absorbs any rescale of z, so SIGReg does not hold the scale: the
code's per-dim std ended ~12x below the control's. A joint variant should
detach the target or normalize the residual by a stop-gradient scale of z.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _require_positive_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_non_negative_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized < 0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


class LatentRegressionPlanner(nn.Module):
    """MLP ``z_hat = P(X, l)`` over the causal macro history and a language embedding."""

    def __init__(
        self,
        *,
        state_dim: int,
        lang_embed_dim: int,
        z_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 1024, 512),
    ) -> None:
        super().__init__()
        self.state_dim = _require_positive_int("state_dim", state_dim)
        self.lang_embed_dim = _require_non_negative_int("lang_embed_dim", lang_embed_dim)
        self.z_dim = _require_positive_int("z_dim", z_dim)
        self.hidden_dims = tuple(
            _require_positive_int("hidden_dims", dim) for dim in hidden_dims
        )
        if not self.hidden_dims:
            msg = "hidden_dims must not be empty."
            raise ValueError(msg)
        layers: list[nn.Module] = []
        previous = self.state_dim + self.lang_embed_dim
        for width in self.hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.SiLU()))
            previous = width
        layers.append(nn.Linear(previous, self.z_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: Tensor, lang: Tensor | None) -> Tensor:
        if self.lang_embed_dim > 0:
            if lang is None:
                msg = "This planner is language-conditioned; pass a language embedding."
                raise ValueError(msg)
            inputs = torch.cat((state, lang.to(dtype=state.dtype)), dim=-1)
        else:
            inputs = state
        return self.network(inputs)

    @staticmethod
    def gaussian_nll(z_hat: Tensor, z: Tensor) -> Tensor:
        """``0.5 * ||z - z_hat||^2`` summed over code dims, mean over rows."""
        return 0.5 * (z - z_hat).pow(2).sum(dim=-1).mean()

    @staticmethod
    @torch.no_grad()
    def fit_metrics(z_hat: Tensor, z: Tensor) -> dict[str, float]:
        """Scale-free fit diagnostics of a batch of predictions."""
        residual = (z - z_hat).pow(2).sum(dim=-1).mean()
        total = (z - z.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean()
        explained = 1.0 - float(residual.item()) / max(float(total.item()), 1.0e-12)
        return {
            "mse": float(F.mse_loss(z_hat, z).item()),
            "cosine": float(F.cosine_similarity(z_hat, z, dim=-1).mean().item()),
            "r2": explained,
            "z_hat_rms": float(z_hat.pow(2).mean().sqrt().item()),
        }

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "planner_type": "latent_regression",
            "state_dim": int(self.state_dim),
            "lang_embed_dim": int(self.lang_embed_dim),
            "z_dim": int(self.z_dim),
            "hidden_dims": list(self.hidden_dims),
            "planner_state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint_state_dict(cls, state: dict[str, Any]) -> LatentRegressionPlanner:
        planner = cls(
            state_dim=int(state["state_dim"]),
            lang_embed_dim=int(state["lang_embed_dim"]),
            z_dim=int(state["z_dim"]),
            hidden_dims=tuple(int(d) for d in state["hidden_dims"]),
        )
        planner.load_state_dict(state["planner_state_dict"])
        return planner
