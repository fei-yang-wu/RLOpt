"""Discriminator network for GAIL."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Discriminator(nn.Module):
    """Discriminator network for GAIL.

    The discriminator learns to distinguish between expert and policy
    state-action pairs. It outputs a probability that a given (s,a) pair
    came from the expert demonstration.

    Args:
        observation_dim: Dimension of observation space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        activation: Activation function to use
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # Build network
        layers = []
        input_dim = observation_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            input_dim = hidden_dim

        # Output layer: probability that input is from expert
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        """Compute discriminator output.

        Args:
            observation: Observation tensor [..., obs_dim]
            action: Action tensor [..., action_dim]

        Returns:
            Probability that (observation, action) is from expert [..., 1]
        """
        # Concatenate observation and action
        x = torch.cat([observation, action], dim=-1)
        return self.network(x)

    def compute_reward(self, observation: Tensor, action: Tensor) -> Tensor:
        """Compute GAIL reward from discriminator output.

        The GAIL reward is: r(s,a) = -log(1 - D(s,a))
        where D(s,a) is the discriminator output.

        This encourages the policy to produce state-action pairs
        that the discriminator thinks are from the expert.

        Args:
            observation: Observation tensor [..., obs_dim]
            action: Action tensor [..., action_dim]

        Returns:
            GAIL reward tensor [...]
        """
        d_sa = self.forward(observation, action)
        # r(s,a) = -log(1 - D(s,a))
        # Add small epsilon for numerical stability
        eps = 1e-8
        reward = -torch.log(1 - d_sa + eps).squeeze(-1)
        return reward

    def compute_loss(
        self,
        expert_obs: Tensor,
        expert_action: Tensor,
        policy_obs: Tensor,
        policy_action: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute discriminator loss.

        The discriminator is trained to maximize:
            E_expert[log D(s,a)] + E_policy[log(1 - D(s,a))]

        This is equivalent to minimizing the binary cross-entropy loss.

        Args:
            expert_obs: Expert observations [batch, obs_dim]
            expert_action: Expert actions [batch, action_dim]
            policy_obs: Policy observations [batch, obs_dim]
            policy_action: Policy actions [batch, action_dim]

        Returns:
            Tuple of (loss, info_dict)
        """
        # Discriminator output for expert data (should be close to 1)
        expert_logits = self.forward(expert_obs, expert_action)

        # Discriminator output for policy data (should be close to 0)
        policy_logits = self.forward(policy_obs, policy_action)

        # Binary cross-entropy loss
        # Expert samples should be classified as 1 (expert)
        expert_loss = -torch.log(expert_logits + 1e-8).mean()
        # Policy samples should be classified as 0 (policy)
        policy_loss = -torch.log(1 - policy_logits + 1e-8).mean()

        loss = expert_loss + policy_loss

        # Accuracy metrics
        expert_acc = (expert_logits > 0.5).float().mean()
        policy_acc = (policy_logits < 0.5).float().mean()

        info = {
            "discriminator_loss": loss.detach(),
            "expert_loss": expert_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "expert_accuracy": expert_acc,
            "policy_accuracy": policy_acc,
            "expert_d_mean": expert_logits.mean().detach(),
            "policy_d_mean": policy_logits.mean().detach(),
        }

        return loss, info
