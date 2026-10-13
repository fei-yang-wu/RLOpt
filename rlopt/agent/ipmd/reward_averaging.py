"""Averaging for the IPMD reward saddle (2026-10-12).

The reward-estimation problem is a saddle point,

    min_pi max_f  E_E[f] - E_pi[f] - psi(f) - alpha H(pi),

and alternating last-iterate updates can cycle or let the reward player win
outright (the estimator saturates). Two averages address it:

- ``polyak_update``: the policy is rewarded by an exponential moving average
  of the estimator's parameters (averaging the reward iterates).
- ``ReservoirFeatureReplay``: the estimator's policy rows mix the current
  batch with a uniform sample of every policy row seen so far (averaging the
  policy iterates). Against a mixture ``p_bar`` of past policies the logistic
  optimum is ``log p_E / p_bar``, which ranks states from bad old policies
  below those of recent good ones instead of only separating the reference
  from the current policy.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.no_grad()
def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """``target <- (1 - tau) target + tau source`` over parameters; buffers copied."""
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.lerp_(s_param.detach(), tau)
    for t_buf, s_buf in zip(target.buffers(), source.buffers()):
        t_buf.copy_(s_buf)


class ReservoirFeatureReplay(nn.Module):
    """Fixed-size uniform sample of every row ever offered (reservoir sampling).

    Row ``n`` (counting from 0 over the whole history) is kept with
    probability ``capacity / (n + 1)`` and, if kept, overwrites a uniformly
    chosen slot, so at any time the buffer is a uniform sample of all rows
    offered so far. Vectorized over a batch; two rows of one batch that draw
    the same slot leave one of them, which only perturbs uniformity by the
    collision rate. All state lives in buffers, so it travels with
    ``state_dict``.
    """

    def __init__(self, capacity: int, dim: int, device: torch.device | str) -> None:
        super().__init__()
        if capacity <= 0 or dim <= 0:
            raise ValueError("capacity and dim must be positive.")
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.register_buffer("rows", torch.zeros(self.capacity, self.dim, device=device))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long, device=device))

    @property
    def size(self) -> int:
        return int(min(int(self.seen.item()), self.capacity))

    @torch.no_grad()
    def add(self, rows: Tensor) -> None:
        rows = rows.detach().reshape(-1, self.dim).to(self.rows.device, self.rows.dtype)
        count = int(rows.shape[0])
        if count == 0:
            return
        seen = int(self.seen.item())
        index = torch.arange(seen, seen + count, device=rows.device)
        fill = index < self.capacity
        if bool(fill.any()):
            self.rows[index[fill]] = rows[fill]
        rest = ~fill
        if bool(rest.any()):
            # Row n (n >= capacity) is kept with probability capacity / (n + 1).
            draws = (torch.rand(int(rest.sum()), device=rows.device) * (index[rest] + 1).float()).long()
            keep = draws < self.capacity
            self.rows[draws[keep]] = rows[rest][keep]
        self.seen += count

    @torch.no_grad()
    def sample(self, count: int) -> Tensor:
        size = self.size
        if size == 0:
            raise RuntimeError("ReservoirFeatureReplay.sample called on an empty buffer.")
        index = torch.randint(0, size, (int(count),), device=self.rows.device)
        return self.rows[index]


__all__ = ["ReservoirFeatureReplay", "polyak_update"]
