"""Shared latent-skill utilities for agent-managed latent commands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, nn

from rlopt.config_utils import BatchKey, ObsKey, flatten_feature_tensor
from rlopt.utils import get_activation_class


class LatentSkillCollectorPolicy(nn.Module):
    """Collector wrapper that stamps the current latent command into rollouts."""

    def __init__(self, agent: LatentSkillMixin, policy_module: nn.Module):
        super().__init__()
        self.agent = agent
        self.policy_module = policy_module

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self.agent._inject_collector_latents(tensordict)
        return self.policy_module(tensordict)


class LatentEncoder(nn.Module):
    """Simple normalized MLP encoder used by latent-conditioned agents."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        activation: str,
    ) -> None:
        super().__init__()
        act_cls = get_activation_class(activation)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(act_cls())
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs_features: Tensor) -> Tensor:
        return F.normalize(self.network(obs_features), dim=-1, eps=1.0e-6)


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE returns for auxiliary reward heads over a rollout batch."""

    time_steps = int(rewards.shape[0])
    rewards_2d = rewards.reshape(time_steps, -1)
    values_2d = values.reshape(time_steps, -1)
    next_values_2d = next_values.reshape(time_steps, -1)
    dones_2d = dones.reshape(time_steps, -1).to(dtype=rewards_2d.dtype)

    advantages = torch.zeros_like(rewards_2d)
    last_advantage = torch.zeros_like(rewards_2d[0])

    for step_idx in range(time_steps - 1, -1, -1):
        not_done = 1.0 - dones_2d[step_idx]
        delta = (
            rewards_2d[step_idx]
            + gamma * next_values_2d[step_idx] * not_done
            - values_2d[step_idx]
        )
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[step_idx] = last_advantage

    returns = advantages + values_2d
    return advantages.reshape_as(rewards), returns.reshape_as(values)


class LatentSkillMixin:
    """Mixin that manages per-env latent commands and rollout stamping."""

    _latent_key: ObsKey
    _latent_dim: int
    _collector_policy_wrapper: LatentSkillCollectorPolicy | None
    _collector_latents: Tensor | None
    _collector_latent_steps: Tensor | None

    def _init_latent_skills(
        self,
        env,
        *,
        latent_key: ObsKey,
        latent_dim: int,
        latent_steps_min: int,
        latent_steps_max: int,
    ) -> None:
        self._latent_key = latent_key
        self._latent_dim = int(latent_dim)
        self._latent_steps_min = max(1, int(latent_steps_min))
        self._latent_steps_max = max(self._latent_steps_min, int(latent_steps_max))

        self._collector_policy_wrapper = None
        self._collector_latents = None
        self._collector_latent_steps = None
        self._env_latent_setter = self._discover_env_method(
            env, "set_agent_latent_command"
        )

        available_keys = set(env.observation_spec.keys(True))
        if self._latent_key not in available_keys:
            msg = (
                "Environment observation spec is missing latent key "
                f"{self._latent_key!r}."
            )
            raise KeyError(msg)

        latent_shape = tuple(
            int(dim) for dim in env.observation_spec[self._latent_key].shape
        )
        batch_prefix = tuple(int(dim) for dim in getattr(env, "batch_size", ()))
        if (
            len(batch_prefix) > 0
            and len(latent_shape) >= len(batch_prefix)
            and latent_shape[: len(batch_prefix)] == batch_prefix
        ):
            latent_shape = latent_shape[len(batch_prefix) :]
        if len(latent_shape) != 1 or int(latent_shape[0]) != self._latent_dim:
            msg = (
                f"Latent observation {self._latent_key!r} has shape {latent_shape}, "
                f"expected ({self._latent_dim},)."
            )
            raise ValueError(msg)

    @property
    def collector_policy(self):
        policy_operator = self.actor_critic.get_policy_operator()
        if self._collector_policy_wrapper is None:
            self._collector_policy_wrapper = LatentSkillCollectorPolicy(
                self, policy_operator
            )
        return self._collector_policy_wrapper

    def _sample_unit_latents(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        latents = torch.randn(batch_size, self._latent_dim, device=device, dtype=dtype)
        return F.normalize(latents, dim=-1, eps=1.0e-6)

    def _sample_collector_latents(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        env_ids: Tensor | None = None,
        td: TensorDict | None = None,
    ) -> Tensor:
        del env_ids, td
        return self._sample_unit_latents(batch_size, device=device, dtype=dtype)

    def _sample_latent_steps(self, batch_size: int, *, device: torch.device) -> Tensor:
        if self._latent_steps_min == self._latent_steps_max:
            return torch.full(
                (batch_size,),
                self._latent_steps_min,
                device=device,
                dtype=torch.long,
            )
        return torch.randint(
            self._latent_steps_min,
            self._latent_steps_max + 1,
            (batch_size,),
            device=device,
        )

    def _done_mask_from_td(self, td: TensorDict, *, batch_size: int) -> Tensor:
        keys = td.keys(True)
        done_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
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

        for key in candidate_keys:
            if key not in keys:
                continue
            value = cast(Tensor, td.get(key)).reshape(-1).to(self.device).bool()
            if value.numel() == batch_size:
                done_mask = done_mask | value
        return done_mask

    def _ensure_collector_latent_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        td: TensorDict | None = None,
    ) -> None:
        if (
            self._collector_latents is None
            or self._collector_latent_steps is None
            or self._collector_latents.shape[0] != batch_size
            or self._collector_latents.device != device
            or self._collector_latents.dtype != dtype
        ):
            self._collector_latents = self._sample_collector_latents(
                batch_size,
                device=device,
                dtype=dtype,
                env_ids=None,
                td=td,
            )
            self._collector_latent_steps = self._sample_latent_steps(
                batch_size,
                device=device,
            )
            self._publish_latents_to_env(self._collector_latents)

    def _publish_latents_to_env(self, latents: Tensor) -> None:
        if self._env_latent_setter is None:
            return
        self._env_latent_setter(latents)

    def _inject_collector_latents(self, td: TensorDict) -> None:
        batch_size = int(td.numel())
        if batch_size <= 0:
            return

        dtype = torch.float32
        self._ensure_collector_latent_state(
            batch_size,
            device=self.device,
            dtype=dtype,
            td=td,
        )
        assert self._collector_latents is not None
        assert self._collector_latent_steps is not None

        done_mask = self._done_mask_from_td(td, batch_size=batch_size)
        renew_mask = done_mask | (self._collector_latent_steps <= 0)
        if bool(renew_mask.any()):
            renew_env_ids = renew_mask.nonzero(as_tuple=False).squeeze(-1)
            renew_count = int(renew_mask.sum().item())
            renew_td = cast(TensorDict, td[renew_env_ids])
            self._collector_latents[renew_mask] = self._sample_collector_latents(
                renew_count,
                device=self.device,
                dtype=dtype,
                env_ids=renew_env_ids,
                td=renew_td,
            )
            self._collector_latent_steps[renew_mask] = self._sample_latent_steps(
                renew_count,
                device=self.device,
            )

        latents = self._collector_latents.reshape(*td.batch_size, self._latent_dim)
        td.set(cast(BatchKey, self._latent_key), latents)
        self._publish_latents_to_env(self._collector_latents)
        self._collector_latent_steps = self._collector_latent_steps - 1

    def _prepare_latent_rollout_batch_for_training(self, data: TensorDict) -> None:
        keys = data.keys(True)
        if self._latent_key not in keys:
            msg = (
                "Collected rollout batch is missing latent key "
                f"{self._latent_key!r}."
            )
            raise KeyError(msg)

        next_latent_key = cast(BatchKey, ("next", self._latent_key))
        if next_latent_key not in keys:
            msg = (
                "Collected rollout batch is missing next latent key "
                f"{next_latent_key!r}."
            )
            raise KeyError(msg)

    def _latent_condition_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
    ) -> Tensor | None:
        if self._latent_key not in td.keys(True):
            return None
        latent = cast(Tensor, td.get(cast(BatchKey, self._latent_key)))
        latent = flatten_feature_tensor(latent, 1)
        return latent.detach() if detach else latent

    def _inject_predict_latents(
        self,
        td_data: dict[BatchKey, Tensor],
        batch_shape: tuple[int, ...],
    ) -> None:
        if cast(BatchKey, self._latent_key) in td_data:
            return
        batch_size = 1
        for dim in batch_shape:
            batch_size *= int(dim)
        latents = self._sample_unit_latents(
            batch_size,
            device=self.device,
            dtype=torch.float32,
        ).reshape(*batch_shape, self._latent_dim)
        td_data[cast(BatchKey, self._latent_key)] = latents
        self._publish_latents_to_env(latents.reshape(batch_size, self._latent_dim))


def infer_batch_shape_from_mapping(
    values: Mapping[BatchKey, Tensor],
    key_feature_ndims: Mapping[ObsKey, int],
) -> tuple[int, ...]:
    """Infer the common batch shape for a dict of observation tensors."""

    batch_shape: tuple[int, ...] | None = None
    for key, value in values.items():
        feature_ndim = int(key_feature_ndims[cast(ObsKey, key)])
        current_batch_shape = tuple(
            value.shape[:-feature_ndim] if feature_ndim > 0 else value.shape
        )
        if batch_shape is None:
            batch_shape = current_batch_shape
        elif current_batch_shape != batch_shape:
            msg = (
                "Observation batch shape mismatch: "
                f"{batch_shape} vs {current_batch_shape}."
            )
            raise ValueError(msg)
    return batch_shape or (1,)
