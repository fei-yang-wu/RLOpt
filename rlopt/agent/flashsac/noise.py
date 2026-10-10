"""FlashSAC exploration noise and actor-weight publication.

``NoiseRepeat`` follows the FlashSAC idea (Holiday-Robot/FlashSAC, commit
87edc90, MIT, Copyright (c) 2026 Holiday Robotics): one N(0, I) tensor is held
for ``k`` steps with ``k`` drawn from a truncated zeta law on ``1..max_repeat``.

One departure from the reference: the reference uses one global clock for all
environments and never resets it. Here every environment has its own device
counter, and an environment's counter restarts when that environment resets.
No call reads a value on the host.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def truncated_zeta_pmf(s: float = 2.0, max_repeat: int = 16) -> Tensor:
    """Return ``p(k) ~ k^-s`` for ``k = 1..max_repeat`` as a float64 tensor."""
    k = torch.arange(1, max_repeat + 1, dtype=torch.float64)
    p = k.pow(-s)
    return p / p.sum()


class NoiseRepeat:
    """Per-environment temporally correlated Gaussian noise.

    Call it once per control step. It returns a ``(num_envs, act_dim)`` tensor.
    An environment keeps the same noise for ``k`` consecutive calls, then
    draws new noise and a new ``k``. Passing ``reset_mask`` (bool, ``(num_envs,)``)
    forces a fresh draw for those environments on this call.
    """

    def __init__(
        self,
        num_envs: int,
        act_dim: int,
        device: torch.device | str,
        zeta_s: float = 2.0,
        max_repeat: int = 16,
        generator: torch.Generator | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.act_dim = act_dim
        self.device = torch.device(device)
        self.max_repeat = max_repeat
        self.generator = generator
        self.cdf = torch.cumsum(truncated_zeta_pmf(zeta_s, max_repeat), 0).to(
            device=self.device, dtype=torch.float32
        )
        self.noise = torch.zeros(num_envs, act_dim, device=self.device)
        # Calls left for the current noise. 0 means "draw on the next call".
        self.remaining = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        # True for environments that drew new noise on the last call (for tests and logs).
        self.last_redrawn = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def _sample_repeat(self) -> Tensor:
        u = torch.rand(self.num_envs, device=self.device, generator=self.generator)
        idx = torch.searchsorted(self.cdf, u, right=True)
        return idx.clamp_(max=self.max_repeat - 1) + 1

    @torch.no_grad()
    def __call__(self, reset_mask: Tensor | None = None) -> Tensor:
        redraw = self.remaining <= 0
        if reset_mask is not None:
            redraw = redraw | reset_mask.to(self.device, torch.bool)
        fresh = torch.randn(
            self.num_envs,
            self.act_dim,
            device=self.device,
            generator=self.generator,
        )
        self.noise = torch.where(redraw.unsqueeze(1), fresh, self.noise)
        self.remaining = torch.where(redraw, self._sample_repeat(), self.remaining) - 1
        self.last_redrawn = redraw
        return self.noise

    def state_dict(self) -> dict[str, Tensor]:
        return {"noise": self.noise.clone(), "remaining": self.remaining.clone()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.noise = state["noise"].to(self.device).clone()
        self.remaining = state["remaining"].to(self.device).clone()


@torch.no_grad()
def publish_actor_state(learner: Any, clone: bool = True) -> dict[str, Tensor]:
    """Return the tensors a rollout actor needs, BatchNorm statistics included.

    Keys are ``actor.<name>`` for every actor parameter and buffer, plus
    ``action_bound``. A parameters-only copy would leave the rollout actor with
    stale BatchNorm statistics, so this helper always includes the buffers.
    With ``clone=True`` the result is a snapshot that later updates cannot change.
    """
    state: dict[str, Tensor] = {}
    for name, value in learner.actor.state_dict().items():
        state[f"actor.{name}"] = value.detach().clone() if clone else value.detach()
    bound = learner.action_bound.detach()
    state["action_bound"] = bound.clone() if clone else bound
    return state


@torch.no_grad()
def load_published_actor_state(
    actor: torch.nn.Module,
    state: dict[str, Tensor],
    action_bound: Tensor | None = None,
) -> None:
    """Copy a published state into a rollout actor, strictly.

    Raises ``KeyError`` when a parameter or BatchNorm buffer is missing, so a
    parameters-only publication cannot be loaded by mistake.
    """
    own = actor.state_dict()
    missing = [k for k in own if f"actor.{k}" not in state]
    if missing:
        msg = f"published actor state lacks: {missing[:4]} ..."
        raise KeyError(msg)
    for name, tensor in own.items():
        tensor.copy_(state[f"actor.{name}"])
    if action_bound is not None and "action_bound" in state:
        action_bound.copy_(state["action_bound"])
