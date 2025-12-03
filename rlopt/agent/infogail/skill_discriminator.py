"""Skill-conditioned discriminator for Info-GAIL."""

import torch
import torch.nn as nn
from torch import Tensor


class SkillConditionedDiscriminator(nn.Module):
    """Discriminator conditioned on skill: D(s, a, z)."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        skill_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()
        
        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build discriminator network
        layers = []
        in_dim = observation_dim + action_dim + skill_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                act_fn(),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, observation: Tensor, action: Tensor, skill: Tensor) -> Tensor:
        """Classify state-action-skill tuple.
        
        Args:
            observation: [batch, obs_dim]
            action: [batch, action_dim]
            skill: [batch, skill_dim]
            
        Returns:
            prob: Probability of being expert [batch, 1]
        """
        x = torch.cat([observation, action, skill], dim=-1)
        return self.network(x)
    
    def compute_reward(self, observation: Tensor, action: Tensor, skill: Tensor) -> Tensor:
        """Compute GAIL reward: r = -log(1 - D(s, a, z)).
        
        Args:
            observation: [batch, obs_dim]
            action: [batch, action_dim]
            skill: [batch, skill_dim]
            
        Returns:
            reward: GAIL reward [batch, 1]
        """
        d_sa_z = self.forward(observation, action, skill)
        # Clamp for numerical stability
        d_sa_z = torch.clamp(d_sa_z, 1e-7, 1 - 1e-7)
        reward = -torch.log(1 - d_sa_z)
        return reward

