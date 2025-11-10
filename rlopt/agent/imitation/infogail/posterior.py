"""Posterior network for skill inference from trajectories."""

import torch
import torch.nn as nn
from torch import Tensor


class SkillPosterior(nn.Module):
    """Infers skill from trajectory: q(z|τ)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        skill_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.skill_dim = skill_dim
        
        # GRU for processing trajectory
        self.gru = nn.GRU(
            input_size=obs_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Output layer: mean and log_std for Gaussian
        self.fc_out = nn.Linear(hidden_dim, skill_dim * 2)
        
    def forward(self, trajectory: Tensor) -> tuple[Tensor, Tensor]:
        """Infer skill from trajectory.
        
        Args:
            trajectory: [batch, seq_len, obs_dim + action_dim]
            
        Returns:
            mu: Mean of skill distribution [batch, skill_dim]
            log_std: Log standard deviation [batch, skill_dim]
        """
        # Process trajectory with GRU
        _, hidden = self.gru(trajectory)  # hidden: [num_layers, batch, hidden_dim]
        
        # Use last layer's hidden state
        h = hidden[-1]  # [batch, hidden_dim]
        
        # Compute skill distribution
        out = self.fc_out(h)
        mu, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, -10, 2)
        
        return mu, log_std
    
    def sample(self, trajectory: Tensor, deterministic: bool = False) -> Tensor:
        """Sample skill from posterior.
        
        Args:
            trajectory: [batch, seq_len, obs_dim + action_dim]
            deterministic: If True, return mean
            
        Returns:
            skill: Sampled skill [batch, skill_dim]
        """
        mu, log_std = self.forward(trajectory)
        
        if deterministic:
            return mu
        
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        return mu + std * eps
    
    def log_prob(self, trajectory: Tensor, skill: Tensor) -> Tensor:
        """Compute log probability of skill given trajectory.
        
        Args:
            trajectory: [batch, seq_len, obs_dim + action_dim]
            skill: Skills [batch, skill_dim]
            
        Returns:
            log_prob: Log probability [batch]
        """
        mu, log_std = self.forward(trajectory)
        std = torch.exp(log_std)
        
        # Gaussian log probability
        log_prob = -0.5 * (
            ((skill - mu) / std) ** 2 + 
            2 * log_std + 
            torch.log(torch.tensor(2 * 3.14159))
        )
        return log_prob.sum(dim=-1)

