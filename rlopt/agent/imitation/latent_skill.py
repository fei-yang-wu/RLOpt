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
        self.agent._inject_latent_command(tensordict)
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


class RandomLatentSampler:
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


class LatentSkillMixin:
    """Mixin that manages per-env latent commands and rollout stamping."""

    _latent_key: ObsKey
    _latent_dim: int
    _collector_policy_wrapper: LatentSkillCollectorPolicy | None
    _random_latent_sampler: RandomLatentSampler | None

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

        self._collector_policy_wrapper = None
        self._random_latent_sampler = RandomLatentSampler(
            self._latent_dim,
            latent_steps_min=latent_steps_min,
            latent_steps_max=latent_steps_max,
        )
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

    def _publish_latents_to_env(self, latents: Tensor) -> None:
        self._env_latent_setter(latents)

    def _inject_latent_command(
        self,
        td: TensorDict,
    ) -> None:
        if td.numel() <= 0:
            return

        assert self._random_latent_sampler is not None
        latents = self._random_latent_sampler.sample_for_step(
            td,
            device=self.device,
            dtype=torch.float32,
        )
        td.set(
            cast(BatchKey, self._latent_key),
            latents.reshape(*td.batch_size, self._latent_dim),
        )
        self._publish_latents_to_env(latents.reshape(-1, self._latent_dim))

    def _prepare_latent_rollout_batch_for_training(self, data: TensorDict) -> None:
        del data

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
