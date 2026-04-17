"""Latent-command rollout plumbing shared by latent-conditioned agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, nn

from rlopt.config_utils import BatchKey, ObsKey, flatten_feature_tensor


class LatentCommandCollectorPolicy(nn.Module):
    """Collector wrapper that stamps the current latent command into rollouts."""

    def __init__(
        self,
        *,
        inject_fn: Callable[[TensorDict], None],
        policy_module: nn.Module,
    ):
        super().__init__()
        self._inject_fn = inject_fn
        self.policy_module = policy_module

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self._inject_fn(tensordict)
        return self.policy_module(tensordict)


class RandomLatentCommandSampler:
    """Stateful random latent sampler used by collector-time latent commands."""

    def __init__(
        self,
        latent_dim: int,
        *,
        latent_steps_min: int,
        latent_steps_max: int,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.latent_steps_min = max(1, int(latent_steps_min))
        self.latent_steps_max = max(self.latent_steps_min, int(latent_steps_max))
        self._latents: Tensor | None = None
        self._latent_steps: Tensor | None = None

    def _sample_latents(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        latents = torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)
        return F.normalize(latents, dim=-1, eps=1.0e-6)

    def _sample_steps(self, batch_size: int, *, device: torch.device) -> Tensor:
        if self.latent_steps_min == self.latent_steps_max:
            return torch.full(
                (batch_size,),
                self.latent_steps_min,
                device=device,
                dtype=torch.long,
            )
        return torch.randint(
            self.latent_steps_min,
            self.latent_steps_max + 1,
            (batch_size,),
            device=device,
        )

    @staticmethod
    def _done_mask(td: TensorDict, *, batch_size: int, device: torch.device) -> Tensor:
        done_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        candidate_keys: list[BatchKey] = [
            cast(BatchKey, "done"),
            cast(BatchKey, "terminated"),
            cast(BatchKey, "truncated"),
            cast(BatchKey, "is_init"),
            cast(BatchKey, ("next", "done")),
            cast(BatchKey, ("next", "terminated")),
            cast(BatchKey, ("next", "truncated")),
            cast(BatchKey, ("next", "is_init")),
        ]
        available_keys = td.keys(True)

        for key in candidate_keys:
            if key not in available_keys:
                continue
            value = cast(Tensor, td.get(key)).reshape(-1).to(device=device).bool()
            if value.numel() == batch_size:
                done_mask |= value
        return done_mask

    def sample_for_step(
        self,
        td: TensorDict,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size = int(td.numel())
        if batch_size <= 0:
            return torch.empty(0, self.latent_dim, device=device, dtype=dtype)

        if (
            self._latents is None
            or self._latent_steps is None
            or self._latents.shape[0] != batch_size
            or self._latents.device != device
            or self._latents.dtype != dtype
        ):
            self._latents = self._sample_latents(
                batch_size,
                device=device,
                dtype=dtype,
            )
            self._latent_steps = self._sample_steps(batch_size, device=device)

        renew_mask = self._done_mask(
            td,
            batch_size=batch_size,
            device=device,
        ) | (self._latent_steps <= 0)
        if bool(renew_mask.any()):
            renew_count = int(renew_mask.sum().item())
            self._latents[renew_mask] = self._sample_latents(
                renew_count,
                device=device,
                dtype=dtype,
            )
            self._latent_steps[renew_mask] = self._sample_steps(
                renew_count,
                device=device,
            )

        latents = self._latents
        self._latent_steps = self._latent_steps - 1
        return latents


class LatentCommandController:
    """Owned latent-command runtime for agents that publish commands to envs."""

    def __init__(
        self,
        *,
        env: Any,
        latent_key: ObsKey,
        latent_dim: int,
        latent_steps_min: int,
        latent_steps_max: int,
        discover_env_method: Callable[[Any, str], Callable[..., Any] | None],
    ) -> None:
        self.latent_key = latent_key
        self.latent_dim = int(latent_dim)
        self._collector_policy_wrapper: LatentCommandCollectorPolicy | None = None
        self._random_latent_sampler = RandomLatentCommandSampler(
            self.latent_dim,
            latent_steps_min=latent_steps_min,
            latent_steps_max=latent_steps_max,
        )

        self._env_latent_setter = discover_env_method(env, "set_agent_latent_command")

        available_keys = set(env.observation_spec.keys(True))
        if self.latent_key not in available_keys:
            msg = (
                "Environment observation spec is missing latent key "
                f"{self.latent_key!r}."
            )
            raise KeyError(msg)

        latent_shape = tuple(
            int(dim) for dim in env.observation_spec[self.latent_key].shape
        )
        batch_prefix = tuple(int(dim) for dim in getattr(env, "batch_size", ()))
        if batch_prefix and latent_shape[: len(batch_prefix)] == batch_prefix:
            latent_shape = latent_shape[len(batch_prefix) :]
        if len(latent_shape) != 1 or int(latent_shape[0]) != self.latent_dim:
            msg = (
                f"Latent observation {self.latent_key!r} has shape {latent_shape}, "
                f"expected ({self.latent_dim},)."
            )
            raise ValueError(msg)

    def collector_policy(
        self,
        *,
        inject_fn: Callable[[TensorDict], None],
        policy_module: nn.Module,
    ) -> nn.Module:
        if self._collector_policy_wrapper is None:
            self._collector_policy_wrapper = LatentCommandCollectorPolicy(
                inject_fn=inject_fn,
                policy_module=policy_module,
            )
        return self._collector_policy_wrapper

    def sample_unit_latents(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        latents = torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)
        return F.normalize(latents, dim=-1, eps=1.0e-6)

    def publish_latents_to_env(self, latents: Tensor) -> None:
        if self._env_latent_setter is None:
            return
        self._env_latent_setter(latents)

    def inject_random_latent_command(
        self,
        td: TensorDict,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if td.numel() <= 0:
            return
        latents = self._random_latent_sampler.sample_for_step(
            td,
            device=device,
            dtype=dtype,
        )
        td.set(
            cast(BatchKey, self.latent_key),
            latents.reshape(*td.batch_size, self.latent_dim),
        )
        self.publish_latents_to_env(latents.reshape(-1, self.latent_dim))

    def prepare_rollout_batch_for_training(self, data: TensorDict) -> None:
        del data

    def latent_condition_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
    ) -> Tensor | None:
        if self.latent_key not in td.keys(True):
            return None
        latent = cast(Tensor, td.get(cast(BatchKey, self.latent_key)))
        latent = flatten_feature_tensor(latent, 1)
        return latent.detach() if detach else latent


class LatentCommandMixin:
    """Backward-compatible shim around ``LatentCommandController``."""

    _latent_command_controller: LatentCommandController

    def _init_latent_commands(
        self,
        env,
        *,
        latent_key: ObsKey,
        latent_dim: int,
        latent_steps_min: int,
        latent_steps_max: int,
    ) -> None:
        self._latent_command_controller = LatentCommandController(
            env=env,
            latent_key=latent_key,
            latent_dim=latent_dim,
            latent_steps_min=latent_steps_min,
            latent_steps_max=latent_steps_max,
            discover_env_method=self._discover_env_method,
        )
        self._latent_key = self._latent_command_controller.latent_key
        self._latent_dim = self._latent_command_controller.latent_dim

    def _init_latent_skills(
        self,
        env,
        *,
        latent_key: ObsKey,
        latent_dim: int,
        latent_steps_min: int,
        latent_steps_max: int,
    ) -> None:
        self._init_latent_commands(
            env,
            latent_key=latent_key,
            latent_dim=latent_dim,
            latent_steps_min=latent_steps_min,
            latent_steps_max=latent_steps_max,
        )

    @property
    def collector_policy(self):
        policy_operator = self.actor_critic.get_policy_operator()
        return self._latent_command_controller.collector_policy(
            inject_fn=self._inject_latent_command,
            policy_module=policy_operator,
        )

    def _sample_unit_latents(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return self._latent_command_controller.sample_unit_latents(
            batch_size,
            device=device,
            dtype=dtype,
        )

    def _publish_latents_to_env(self, latents: Tensor) -> None:
        self._latent_command_controller.publish_latents_to_env(latents)

    def _inject_latent_command(
        self,
        td: TensorDict,
    ) -> None:
        self._latent_command_controller.inject_random_latent_command(
            td,
            device=self.device,
            dtype=torch.float32,
        )

    def _prepare_latent_rollout_batch_for_training(self, data: TensorDict) -> None:
        self._latent_command_controller.prepare_rollout_batch_for_training(data)

    def _latent_condition_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
    ) -> Tensor | None:
        return self._latent_command_controller.latent_condition_from_td(
            td,
            detach=detach,
        )
