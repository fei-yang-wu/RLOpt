"""The task-model interface ``WMBPTT`` is generic over.

The agent owns the algorithm (world-model fit, imagination, actor-critic,
replay); a task model owns what is specific to one environment: how the
physical state is read from observations, the frames the world model works
in, the policy and critic features, and the differentiable reward and
termination. Every method must work on simulator rows and on imagined states
alike, so the two never see different features.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class WorldModelTask(Protocol):
    state_dim: int
    action_dim: int
    context_dim: int
    increment_dim: int
    wm_input_dim: int
    policy_obs_dim: int
    critic_obs_dim: int
    history: int
    obs_group: str
    policy_in_keys: tuple[tuple[str, str], ...]
    increment_floor: Tensor

    def read_obs(self, group: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """``state`` (N, S), ``hist_states`` (N, h, S), ``hist_actions``
        (N, h, A), ``rank``/``frame``/``episode_step`` (N,) long, ``context``
        (N, C), ``push`` (N,) bool."""

    def wm_input(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor: ...

    def increment(self, state: Tensor, next_state: Tensor) -> Tensor: ...

    def integrate(self, state: Tensor, inc: Tensor) -> Tensor: ...

    def state_error(self, pred: Tensor, true: Tensor) -> Tensor: ...

    def policy_obs(
        self,
        hist_states: Tensor,
        hist_actions: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
        noise: bool | None = None,
    ) -> Tensor: ...

    def policy_obs_from_env(self, *tensors: Tensor) -> Tensor: ...

    def critic_obs(self, state: Tensor, context: Tensor, rank: Tensor, frame: Tensor) -> Tensor: ...

    def reward(
        self,
        next_state: Tensor,
        action: Tensor,
        prev_action: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]: ...

    def termination(
        self, next_state: Tensor, context: Tensor, rank: Tensor, frame: Tensor, hard: bool = False
    ) -> Tensor: ...

    def reward_and_termination(
        self,
        next_state: Tensor,
        action: Tensor,
        prev_action: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
    ) -> tuple[Tensor, Tensor]: ...

    def proprio(self, states: Tensor, actions: Tensor, context: Tensor, noise: bool) -> Tensor:
        """Per-frame proprio features (..., P) of ``states`` with last ``actions``."""

    def policy_obs_from_parts(
        self, proprio_hist: Tensor, state: Tensor, rank: Tensor, frame: Tensor
    ) -> Tensor:
        """``policy_obs`` from a precomputed proprio history (N, h, P)."""

    def clip_end(self, rank: Tensor, frame: Tensor) -> Tensor: ...

    def describe(self) -> dict[str, Any]: ...

    # Optional: ``refresh()`` re-reads env state the task model mirrors (for
    # example curriculum-moved termination thresholds). The agent calls it
    # once per iteration, outside compiled code, when it exists.
    # Optional: ``final_obs_is_post_step = True`` says the env's batched
    # ``extras["final_obs"]`` holds each ended episode's state after its last
    # step, before the reset; ``wm.fit_terminal_transitions`` and
    # ``wm.norm_from_collection`` need it.
    # Optional: ``goal_binding()`` returns a JSON-able record of what produces
    # the policy's goal input (for example a frozen encoder's weight hash);
    # checkpoints store it and loading refuses a mismatch.


def load_task(spec: str, env: Any, device: torch.device, **kwargs: Any) -> WorldModelTask:
    """Instantiate ``module:Class`` with ``(env, device, **kwargs)``."""
    if ":" not in spec:
        msg = f"task_model must be 'module:Class', got {spec!r}."
        raise ValueError(msg)
    module_name, class_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    task = cls(env, device, **kwargs)
    if not isinstance(task, WorldModelTask):
        msg = f"{spec} does not implement the WorldModelTask interface."
        raise TypeError(msg)
    return task


def task_proprio(
    task: WorldModelTask, state: Tensor, action: Tensor, context: Tensor,
    rank: Tensor, frame: Tensor, noise: bool,
) -> Tensor:
    """Build one observation, including reference-time features when supported.

    The optional ``proprio_at`` capability extends the original state-only
    interface without changing existing tasks or their saved policies.
    """
    at = getattr(task, "proprio_at", None)
    if at is not None:
        return at(state, action, context, rank, frame, noise)
    return task.proprio(state, action, context, noise)


def history_proprio(task: WorldModelTask, rows: Mapping[str, Tensor], noise: bool) -> Tensor:
    """Reconstruct history with each observation's original reference cursor.

    Before an asynchronous episode's first frame, repeat that frame exactly,
    just as replay.history repeats the reset state and previous action.
    """
    if getattr(task, "proprio_at", None) is None:
        return task.proprio(rows["hist_states"], rows["hist_actions"], rows["context"].unsqueeze(1), noise)
    back = torch.arange(task.history - 1, -1, -1, device=rows["frame"].device)
    back = torch.minimum(back.unsqueeze(0), rows["episode_step"].clamp_min(0).unsqueeze(1))
    prop = task_proprio(
        task, rows["hist_states"], rows["hist_actions"], rows["context"].unsqueeze(1),
        rows["rank"].unsqueeze(1), rows["frame"].unsqueeze(1) - back, noise,
    )
    # Reset-filled history repeats one observed frame, including its single
    # noise draw. Only real subsequent frames receive independent noise.
    slots = task.history - 1 - back
    return prop.gather(1, slots.unsqueeze(-1).expand_as(prop))
