"""Pure latent-imitation helpers shared across latent-conditioned agents."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rlopt.utils import get_activation_class


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
