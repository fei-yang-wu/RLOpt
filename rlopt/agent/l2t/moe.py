from __future__ import annotations

from typing import Callable, Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """
    Simple two-layer feed-forward expert used in MoE blocks.

    Maps from `d_model` -> `d_ff` -> `d_model` with activation.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: Callable[..., nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = activation()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TopKRouter(nn.Module):
    """
    Top-k routing with optional router noise and load-balancing aux loss.

    Returns mixing weights over experts and records an auxiliary loss that
    encourages balanced expert usage, following GShard/Switch-Transformer.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        k: int = 2,
        router_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        assert k >= 1 and k <= num_experts, "k must be in [1, num_experts]"
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(d_model, num_experts, bias=True)
        self.router_jitter = router_jitter
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (N, d_model)
        logits = self.router(x)
        if self.training and self.router_jitter and self.router_jitter > 0:
            logits = logits + torch.randn_like(logits) * self.router_jitter

        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        topk_probs, topk_indices = torch.topk(probs, k=self.k, dim=-1)
        # Renormalize top-k weights so they sum to 1
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Build dense [N, E] mixing matrix (zeros except top-k positions)
        N, E = probs.shape
        mixing = probs.new_zeros((N, E))
        mixing.scatter_(dim=-1, index=topk_indices, src=topk_weights)

        # Load-balancing auxiliary loss
        # fraction of tokens routed to each expert (based on top-k assignment)
        assignment = probs.new_zeros((N, E))
        one = probs.new_ones((N, self.k))
        assignment.scatter_(dim=-1, index=topk_indices, src=one)
        frac_tokens = assignment.mean(dim=0)  # [E]
        prob_mass = probs.mean(dim=0)  # [E]
        aux_loss = (self.num_experts * (frac_tokens * prob_mass).sum()).mean()
        self.last_aux_loss = aux_loss

        return mixing, probs, topk_indices, topk_weights


class MoE(nn.Module):
    """
    Dense-compute Mixture-of-Experts block with top-k routing.

    - Computes all experts in parallel (for simplicity) and mixes their outputs
      using top-k router weights.
    - Records a load-balancing auxiliary loss on the router.

    Note: This is compute-dense (does not drop/dispatch tokens), but follows
    the modern top-k routing and balancing objective used in LLM MoE layers.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        k: int = 2,
        ffn_expand: float = 4.0,
        activation: Callable[..., nn.Module] = nn.GELU,
        dropout: float = 0.0,
        router_jitter: float = 0.0,
        sparse_dispatch: bool = False,
        capacity_factor: float = 1.0,
    ) -> None:
        super().__init__()
        assert num_experts >= 1, "num_experts must be >= 1"
        assert k >= 1 and k <= num_experts, "k must be in [1, num_experts]"
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.ffn_expand = ffn_expand
        d_ff = max(1, int(d_model * ffn_expand))

        self.experts = nn.ModuleList(
            [
                ExpertFFN(d_model, d_ff, activation=activation, dropout=dropout)
                for _ in range(num_experts)
            ]
        )
        self.router = TopKRouter(
            d_model=d_model, num_experts=num_experts, k=k, router_jitter=router_jitter
        )
        self.sparse_dispatch = sparse_dispatch
        self.capacity_factor = capacity_factor
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept x of shape (..., d_model), flatten for processing
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])  # (N, d_model)

        # Router outputs dense mixing matrix [N, E] and top-k info
        mixing, probs, topk_idx, topk_w = self.router(x_flat)

        if not self.sparse_dispatch:
            # Dense compute and mixing
            expert_outputs = torch.stack(
                [expert(x_flat) for expert in self.experts], dim=1
            )
            y_flat = torch.sum(expert_outputs * mixing.unsqueeze(-1), dim=1)
        else:
            # Sparse dispatch with capacity and token dropping
            N, d_model = x_flat.shape
            E = self.num_experts
            k = self.k
            # Capacity per expert: ceil(capacity_factor * (N * k / E))
            cap = max(1, int(math.ceil(self.capacity_factor * float(N * k) / float(E))))

            # Flatten assignments
            flat_e = topk_idx.reshape(-1)  # [A]
            flat_w = topk_w.reshape(-1)  # [A]
            tok = (
                torch.arange(N, device=x_flat.device)
                .unsqueeze(1)
                .expand(N, k)
                .reshape(-1)
            )  # [A]

            # Compute position within each expert using cumulative counts
            one_hot = F.one_hot(flat_e, num_classes=E).to(x_flat.dtype)  # [A, E]
            cumsum = torch.cumsum(one_hot, dim=0) - one_hot  # [A, E]
            pos = cumsum[
                torch.arange(one_hot.size(0), device=x_flat.device), flat_e
            ].long()  # [A]

            keep = pos < cap
            if keep.any():
                e_k = flat_e[keep]
                pos_k = pos[keep]
                tok_k = tok[keep]
                w_k = flat_w[keep]

                # Renormalize weights per token among kept assignments
                sum_w = torch.zeros(N, device=x_flat.device, dtype=x_flat.dtype)
                sum_w.index_add_(0, tok_k, w_k)
                w_norm = w_k / (sum_w[tok_k] + 1e-9)

                # Build expert input buffers [E, cap, d_model]
                expert_in = x_flat.new_zeros((E, cap, d_model))
                expert_in[e_k, pos_k] = x_flat[tok_k]

                # Compute experts
                expert_out = x_flat.new_zeros((E, cap, d_model))
                for e in range(E):
                    expert_out[e] = self.experts[e](expert_in[e])

                # Gather contributions and combine by scattering back to tokens
                contrib = expert_out[e_k, pos_k]  # [Kept, d_model]
                y_flat = x_flat.new_zeros((N, d_model))
                y_flat.index_add_(0, tok_k, contrib * w_norm.unsqueeze(-1))
            else:
                # Nothing kept: return zeros
                y_flat = x_flat.new_zeros((N, d_model))

        # Record aux loss
        self.last_aux_loss = self.router.last_aux_loss

        y = y_flat.view(*orig_shape)
        return y
