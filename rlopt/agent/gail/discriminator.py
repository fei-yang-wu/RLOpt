"""Discriminator network for GAIL."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

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
                msg = f"Unknown activation: {activation}"
                raise ValueError(msg)
            input_dim = hidden_dim

        # Output layer: logits (probabilities are computed on demand).
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        """Compute discriminator logits."""
        x = torch.cat([observation, action], dim=-1)
        return self.network(x)

    def forward_logits(self, observation: Tensor, action: Tensor) -> Tensor:
        """Alias for :meth:`forward` to make logit access explicit in callers."""
        return self.forward(observation, action)

    def predict_proba(self, observation: Tensor, action: Tensor) -> Tensor:
        """Compute discriminator output.

        Args:
            observation: Observation tensor [..., obs_dim]
            action: Action tensor [..., action_dim]

        Returns:
            Probability that (observation, action) is from expert [..., 1]
        """
        return torch.sigmoid(self.forward(observation, action))

    def get_logit_layer(self) -> nn.Linear | None:
        """Return the final linear logit layer when available."""
        for module in reversed(list(self.network.modules())):
            if isinstance(module, nn.Linear):
                return module
        return None

    def all_weights(self) -> list[Tensor]:
        """Return all learnable 2D weight tensors for explicit L2 regularization."""
        return [param for param in self.parameters() if param.ndim >= 2]

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
        d_sa = self.predict_proba(observation, action)
        one_minus_prob = torch.clamp_min(
            1.0 - d_sa, torch.tensor(1.0e-4, device=d_sa.device, dtype=d_sa.dtype)
        )
        return -torch.log(one_minus_prob).squeeze(-1)

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
        # Discriminator output for expert/policy data (logits).
        expert_logits = self.forward(expert_obs, expert_action)
        policy_logits = self.forward(policy_obs, policy_action)

        expert_targets = torch.ones_like(expert_logits)
        policy_targets = torch.zeros_like(policy_logits)
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, expert_targets)
        policy_loss = F.binary_cross_entropy_with_logits(policy_logits, policy_targets)
        loss = 0.5 * (expert_loss + policy_loss)

        # Accuracy metrics
        expert_probs = torch.sigmoid(expert_logits)
        policy_probs = torch.sigmoid(policy_logits)
        expert_acc = (expert_probs > 0.5).float().mean()
        policy_acc = (policy_probs < 0.5).float().mean()

        info = {
            "discriminator_loss": loss.detach(),
            "expert_loss": expert_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "expert_accuracy": expert_acc,
            "policy_accuracy": policy_acc,
            "expert_d_mean": expert_probs.mean().detach(),
            "policy_d_mean": policy_probs.mean().detach(),
        }

        return loss, info
