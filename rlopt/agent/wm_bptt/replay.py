# ``set_terminal`` and ``rollout_mask`` are adapted from NDM Gym (https://github.com/Frank-ZY-Dou/ndm-gym @ 05ea9f3:
# ndm_gym/buffer.py), which ports it from BlinkMimic (https://github.com/fuyichen0725/blinkmimic @ d264658), under the
# MIT License:
#
# Copyright (c) 2026 NDM Gym Contributors
# Copyright (c) 2026 ETH Zurich, Chong Zhang
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Per-env ring of simulator rows for the world-model agent.

All envs write one row per vectorized step, so a row is addressed by
(absolute step tau, env). Episode membership is read from the env's own
episode-step counter: rows tau and tau + 1 of one env are a transition iff
``episode_step[tau + 1] == episode_step[tau] + 1``. A terminal transition
(the env resets) therefore never counts, and neither does anything across
the ring's write head.

``terminal_states`` (``wm.fit_terminal_transitions``) keeps what row tau + 1
cannot hold: the true state after an episode's last transition, which the env's
reset overwrote. A window then fits that transition and stops after it.
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
        *,
        terminal_states: bool = False,
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
        # The collecting actor's deterministic output (action minus its noise).
        self.policy_action = torch.zeros(T, N, action_dim, **f)
        # Outcome of the transition out of each row (set after the env step).
        self.env_reward = torch.zeros(T, N, **f)
        self.terminated = torch.zeros(T, N, dtype=torch.bool, **f)
        # ``wm.fit_terminal_transitions``: the true post-step state of a row whose
        # transition ended its episode; ``has_terminal`` says which rows hold one.
        self.terminal_state = torch.zeros(T, N, state_dim, **f) if terminal_states else None
        self.has_terminal = torch.zeros(T, N, dtype=torch.bool, **f) if terminal_states else None
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
        policy_action: Tensor | None = None,
    ) -> None:
        i = self.steps % self.capacity
        self.policy_action[i] = action if policy_action is None else policy_action
        self.state[i] = state
        self.action[i] = action
        self.prev_action[i] = prev_action
        self.context[i] = context
        self.rank[i] = rank
        self.frame[i] = frame
        self.episode_step[i] = episode_step
        self.push[i] = push
        self.env_reward[i] = 0.0
        self.terminated[i] = False
        if self.has_terminal is not None:
            self.has_terminal[i] = False
        self.steps += 1

    def set_outcome(self, env_reward: Tensor, terminated: Tensor) -> None:
        """Record the env's reward and termination (not timeout) of the step
        taken from the newest row."""
        i = (self.steps - 1) % self.capacity
        self.env_reward[i] = env_reward
        self.terminated[i] = terminated

    def set_terminal(self, env_ids: Tensor, state: Tensor, keep: Tensor) -> None:
        """Record the true post-step ``state`` (k, S) of the ``env_ids`` (k,) whose
        episode ended on the step from the newest row. Rows ``keep`` (k,) leaves
        out stay what they are without this: no fit target."""
        i = (self.steps - 1) % self.capacity
        self.terminal_state[i, env_ids] = state
        self.has_terminal[i, env_ids] = keep

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
        """Rows tau .. tau + length - 1, (B, L, ...), with each transition's
        ``next_state`` and its validity ``valid`` (B, L - 1): row j -> j + 1 is one
        episode with no push. With ``terminal_states``, an episode's last
        transition is also valid, ``next_state`` holds its true state and
        ``ended`` (B, L - 1) marks it; see ``rollout_mask``."""
        steps = tau.unsqueeze(1) + torch.arange(length, device=self.device)
        slot = self._slot(steps)
        e = env.unsqueeze(1).expand_as(slot)
        out = {
            "state": self.state[slot, e],
            "action": self.action[slot, e],
            "policy_action": self.policy_action[slot, e],
            "prev_action": self.prev_action[slot, e],
            "context": self.context[slot, e],
            "rank": self.rank[slot, e],
            "frame": self.frame[slot, e],
            "episode_step": self.episode_step[slot, e],
            "push": self.push[slot, e],
            "env_reward": self.env_reward[slot, e],
            "terminated": self.terminated[slot, e],
        }
        es = out["episode_step"]
        out["valid"] = (es[:, 1:] == es[:, :-1] + 1) & ~out["push"][:, 1:]
        out["next_state"] = out["state"][:, 1:]
        if self.terminal_state is not None:
            rows, cols = slot[:, :-1], e[:, :-1]
            ended = self.has_terminal[rows, cols]
            # A row without a stored state keeps the next row, which is finite and masked.
            out["next_state"] = torch.where(ended.unsqueeze(-1), self.terminal_state[rows, cols], out["next_state"])
            out["valid"] = out["valid"] | ended
            out["ended"] = ended
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
            "episode_step": es_now,
        }


def rollout_mask(win: dict[str, Tensor], k: int) -> Tensor:
    """(B, k): which transitions of an open-loop rollout on ``win`` count. The first
    invalid transition stops the rollout, and so does the step after an episode's
    last transition, which itself counts. Past that the rollout would cross a reset."""
    valid = win["valid"][:, :k].float()
    if "ended" not in win:
        return torch.cumprod(valid, dim=1)
    went_on = valid * (1 - win["ended"][:, :k].float())
    return valid * torch.cat((torch.ones_like(valid[:, :1]), torch.cumprod(went_on[:, :-1], dim=1)), 1)
