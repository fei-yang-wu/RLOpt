# Copyright (c) 2022-2026, The RLOpt developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The one imitation-environment surface RLOpt knows about.

Agents used to duck-type the environment in a dozen places, each walking the
wrapper stack for a differently-named method (``sample_expert_batch``,
``current_expert_macro_transition_batch``, ``set_agent_latent_command``, ...).
That spread the environment's API across the whole package and made every
environment-side rename an RLOpt change.

Instead there is ONE capability object. An imitation environment exposes it as
``env.imitation_interface``; RLOpt resolves it once
(:func:`resolve_imitation_interface`) and everything else is typed attribute
access on :class:`ImitationEnvInterface`. Environments that predate the
attribute are still supported: :class:`LegacyMethodInterface` adapts the
historical method names, so this module is the only place in RLOpt that knows
them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from torch import Tensor
from tensordict import TensorDict

__all__ = [
    "ImitationEnvInterface",
    "LegacyMethodInterface",
    "require_imitation_interface",
    "resolve_imitation_interface",
    "supports",
]

# The wrapper attributes an environment stack is threaded through.
_WRAPPER_ATTRS = ("base_env", "env", "_env", "unwrapped")

# Capability name -> the historical environment method it was called by.
_LEGACY_METHOD_NAMES: dict[str, str] = {
    "sample_expert_batch": "sample_expert_batch",
    "sample_expert_macro_transition_batch": "sample_expert_macro_transition_batch",
    "current_expert_macro_transition_batch": "current_expert_macro_transition_batch",
    "current_achieved_macro_transition_batch": "current_achieved_macro_transition_batch",
    "expert_macro_feature_slices": "expert_macro_feature_slices",
    "expert_trajectory_motion_names": "expert_trajectory_motion_names",
    "offline_dataset_mapper_params": "get_offline_dataset_mapper_params",
    "publish_actor_command": "set_agent_latent_command",
}


@runtime_checkable
class ImitationEnvInterface(Protocol):
    """What an imitation environment offers an RLOpt agent.

    Every member is optional in the sense that a given environment may not
    implement it (a vanilla tracking task has no skill encoder to feed); use
    :func:`supports` to probe, and fail loudly naming the capability when the
    configured algorithm requires one that is absent.
    """

    def sample_expert_batch(
        self, batch_size: int, required_keys: Sequence[Any]
    ) -> TensorDict | None:
        """Expert transitions for the imitation/IRL losses."""
        ...

    def sample_expert_macro_transition_batch(
        self,
        batch_size: int,
        horizon_steps: int,
        split: str | None = None,
        eval_fraction: float = 0.1,
        split_seed: int = 0,
        trajectory_ranks: Sequence[int] | Tensor | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        """Offline expert macro-transitions for skill-encoder training."""
        ...

    def current_expert_macro_transition_batch(
        self,
        horizon_steps: int,
        env_ids: Tensor | Sequence[int] | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        """The live expert macro-transition at each environment's cursor."""
        ...

    def expert_macro_feature_slices(self, horizon_steps: int) -> dict[str, Any]:
        """Feature-name to slice map of one macro-state frame."""
        ...

    def expert_trajectory_motion_names(self) -> list[str]:
        """Motion name per trajectory rank."""
        ...

    def offline_dataset_mapper_params(self) -> dict[str, Any]:
        """Parameters mapping an offline dataset onto this environment's keys."""
        ...

    def publish_actor_command(
        self, command: Tensor, env_ids: Tensor | None = None
    ) -> None:
        """Publish the agent-produced actor command (e.g. a skill latent)."""
        ...


class LegacyMethodInterface:
    """Adapter over an environment that predates ``imitation_interface``.

    Resolves each capability lazily by walking the wrapper stack for its
    historical method name, so an environment only pays for what an agent
    actually asks for and a missing capability is reported by name.
    """

    def __init__(self, env: object) -> None:
        self._env = env
        self._cache: dict[str, Callable | None] = {}

    def __repr__(self) -> str:
        return f"LegacyMethodInterface({type(self._env).__name__})"

    def _method(self, capability: str) -> Callable | None:
        if capability in self._cache:
            return self._cache[capability]
        method_name = _LEGACY_METHOD_NAMES.get(capability)
        method = (
            None if method_name is None else _walk_for_callable(self._env, method_name)
        )
        self._cache[capability] = method
        return method

    def __getattr__(self, name: str) -> Callable:
        method = self._method(name)
        if method is None:
            raise AttributeError(
                f"The environment does not expose the {name!r} capability."
            )
        return method


def _walk_for_callable(env: object, method_name: str) -> Callable | None:
    """Find a callable attribute anywhere in the wrapper stack."""
    stack: list[object] = [env]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        method = getattr(current, method_name, None)
        if callable(method):
            return method
        for attr_name in _WRAPPER_ATTRS:
            try:
                next_obj = getattr(current, attr_name, None)
            except Exception:
                continue
            if next_obj is None:
                continue
            if isinstance(next_obj, list | tuple):
                stack.extend(next_obj)
            else:
                stack.append(next_obj)
    return None


def resolve_imitation_interface(env: object) -> ImitationEnvInterface:
    """The environment's imitation interface -- the single resolution point.

    Walks the wrapper stack once for an ``imitation_interface`` attribute and
    falls back to :class:`LegacyMethodInterface` for environments that expose
    the historical method names instead. Always returns an object; probe it
    with :func:`supports` and fail loudly where a capability is required.
    """
    stack: list[object] = [env]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        interface = getattr(current, "imitation_interface", None)
        if interface is not None:
            return interface
        for attr_name in _WRAPPER_ATTRS:
            try:
                next_obj = getattr(current, attr_name, None)
            except Exception:
                continue
            if next_obj is None:
                continue
            if isinstance(next_obj, list | tuple):
                stack.extend(next_obj)
            else:
                stack.append(next_obj)
    return LegacyMethodInterface(env)


def supports(interface: object, capability: str) -> bool:
    """Whether an interface actually provides a capability."""
    try:
        return callable(getattr(interface, capability, None))
    except AttributeError:
        return False


def require_imitation_interface(
    env: object, capability: str, *, purpose: str
) -> Callable:
    """Resolve one required capability, failing loudly when it is absent."""
    interface = resolve_imitation_interface(env)
    if not supports(interface, capability):
        raise ValueError(
            f"{purpose} the environment does not expose the {capability!r} "
            "capability of its imitation interface."
        )
    return getattr(interface, capability)
