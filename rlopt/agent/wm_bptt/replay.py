"""Per-env ring of simulator rows for the world-model agent.

All envs write one row per vectorized step, so a row is addressed by
(absolute step tau, env). Episode membership is read from the env's own
episode-step counter: rows tau and tau + 1 of one env are a transition iff
``episode_step[tau + 1] == episode_step[tau] + 1``. A terminal transition
(the env resets) therefore never counts, and neither does anything across
the ring's write head.
"""

from __future__ import annotations

import torch
from torch import Tensor


class SequenceReplay:
    def __init__(
        self,
        num_envs: int,
        capacity: int,
        state_dim: int,
        action_dim: int,
        context_dim: int,
        device: torch.device,
    ) -> None:
        self.num_envs = int(num_envs)
        self.capacity = int(capacity)
        self.device = device
        f = {"device": device}
        T, N = self.capacity, self.num_envs
        self.state = torch.zeros(T, N, state_dim, **f)
        self.action = torch.zeros(T, N, action_dim, **f)
        self.prev_action = torch.zeros(T, N, action_dim, **f)
        self.context = torch.zeros(T, N, context_dim, **f)
        self.rank = torch.zeros(T, N, dtype=torch.long, **f)
        self.frame = torch.zeros(T, N, dtype=torch.long, **f)
        self.episode_step = torch.full((T, N), -10, dtype=torch.long, **f)
        self.push = torch.zeros(T, N, dtype=torch.bool, **f)
        self.steps = 0  # absolute steps written

    def add(
        self,
        state: Tensor,
        action: Tensor,
        prev_action: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
        episode_step: Tensor,
        push: Tensor,
    ) -> None:
        i = self.steps % self.capacity
        self.state[i] = state
        self.action[i] = action
        self.prev_action[i] = prev_action
        self.context[i] = context
        self.rank[i] = rank
        self.frame[i] = frame
        self.episode_step[i] = episode_step
        self.push[i] = push
        self.steps += 1

    @property
    def oldest(self) -> int:
        return max(0, self.steps - self.capacity)

    def _slot(self, tau: Tensor) -> Tensor:
        return tau % self.capacity

    def sample_starts(self, batch: int, span: int, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Uniform (tau, env) with rows tau .. tau + span inside the ring."""
        lo = self.oldest
        hi = self.steps - 1 - span
        if hi < lo:
            msg = f"replay holds {self.steps} steps; a span of {span} needs more."
            raise RuntimeError(msg)
        tau = torch.randint(lo, hi + 1, (batch,), device=self.device, generator=generator)
        env = torch.randint(0, self.num_envs, (batch,), device=self.device, generator=generator)
        return tau, env

    def window(self, tau: Tensor, env: Tensor, length: int) -> dict[str, Tensor]:
        """Rows tau .. tau + length - 1, (B, L, ...), with transition validity
        ``valid`` (B, L - 1): row j -> j + 1 is one episode with no push."""
        steps = tau.unsqueeze(1) + torch.arange(length, device=self.device)
        slot = self._slot(steps)
        e = env.unsqueeze(1).expand_as(slot)
        out = {
            "state": self.state[slot, e],
            "action": self.action[slot, e],
            "prev_action": self.prev_action[slot, e],
            "context": self.context[slot, e],
            "rank": self.rank[slot, e],
            "frame": self.frame[slot, e],
            "episode_step": self.episode_step[slot, e],
            "push": self.push[slot, e],
        }
        es = out["episode_step"]
        out["valid"] = (es[:, 1:] == es[:, :-1] + 1) & ~out["push"][:, 1:]
        return out

    def history(self, tau: Tensor, env: Tensor, length: int) -> dict[str, Tensor]:
        """The ``length`` rows ending at tau as the env's history buffer holds
        them: frames before the episode's first row repeat that first row
        (Isaac's reset fill). Requires tau - length + 1 >= oldest."""
        back = torch.arange(length - 1, -1, -1, device=self.device)  # oldest first
        es_now = self.episode_step[self._slot(tau), env]
        back = torch.minimum(back.unsqueeze(0), es_now.clamp(min=0).unsqueeze(1))
        steps = tau.unsqueeze(1) - back
        slot = self._slot(steps)
        e = env.unsqueeze(1).expand_as(slot)
        return {
            "hist_states": self.state[slot, e],
            "hist_actions": self.prev_action[slot, e],
            "context": self.context[self._slot(tau), env],
            "rank": self.rank[self._slot(tau), env],
            "frame": self.frame[self._slot(tau), env],
        }
