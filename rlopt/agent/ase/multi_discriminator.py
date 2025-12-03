"""Multi-discriminator for ASE."""

import torch
import torch.nn as nn
from torch import Tensor

from rlopt.agent.imitation.gail.discriminator import Discriminator


class MultiDiscriminator(nn.Module):
    """Multiple discriminators, one per skill."""

    def __init__(
        self,
        num_skills: int,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()
        self.num_skills = num_skills
        
        # Create one discriminator per skill
        self.discriminators = nn.ModuleList([
            Discriminator(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                activation=activation,
            )
            for _ in range(num_skills)
        ])
    
    def forward(self, observation: Tensor, action: Tensor, skill_idx: Tensor) -> Tensor:
        """Classify state-action for specific skill.
        
        Args:
            observation: [batch, obs_dim]
            action: [batch, action_dim]
            skill_idx: [batch] - skill indices (0 to num_skills-1)
            
        Returns:
            prob: Probability of being expert [batch, 1]
        """
        batch_size = observation.shape[0]
        outputs = []
        
        for i in range(batch_size):
            skill_id = skill_idx[i].item()
            disc = self.discriminators[skill_id]
            out = disc(observation[i:i+1], action[i:i+1])
            outputs.append(out)
        
        return torch.cat(outputs, dim=0)
    
    def forward_all(self, observation: Tensor, action: Tensor) -> Tensor:
        """Get discriminator outputs for all skills.
        
        Args:
            observation: [batch, obs_dim]
            action: [batch, action_dim]
            
        Returns:
            probs: [batch, num_skills] - probabilities for each skill
        """
        outputs = []
        for disc in self.discriminators:
            out = disc(observation, action)
            outputs.append(out)
        return torch.cat(outputs, dim=-1)  # [batch, num_skills]
    
    def compute_reward(self, observation: Tensor, action: Tensor, skill_idx: Tensor) -> Tensor:
        """Compute GAIL reward for specific skill.
        
        Args:
            observation: [batch, obs_dim]
            action: [batch, action_dim]
            skill_idx: [batch] - skill indices
            
        Returns:
            reward: GAIL reward [batch, 1]
        """
        d_sa = self.forward(observation, action, skill_idx)
        d_sa = torch.clamp(d_sa, 1e-7, 1 - 1e-7)
        reward = -torch.log(1 - d_sa)
        return reward


class StyleDiscriminator(nn.Module):
    """Classifies which skill a trajectory belongs to."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_skills: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_skills = num_skills
        
        # GRU for processing trajectory
        self.gru = nn.GRU(
            input_size=obs_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Classifier head
        self.fc_out = nn.Linear(hidden_dim, num_skills)
        
    def forward(self, trajectory: Tensor) -> Tensor:
        """Classify trajectory into skills.
        
        Args:
            trajectory: [batch, seq_len, obs_dim + action_dim]
            
        Returns:
            logits: Skill logits [batch, num_skills]
        """
        _, hidden = self.gru(trajectory)
        h = hidden[-1]  # [batch, hidden_dim]
        logits = self.fc_out(h)
        return logits
    
    def get_probs(self, trajectory: Tensor) -> Tensor:
        """Get skill probabilities."""
        logits = self.forward(trajectory)
        return torch.softmax(logits, dim=-1)

