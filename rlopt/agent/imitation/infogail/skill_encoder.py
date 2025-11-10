"""Skill encoder for Info-GAIL."""

import torch
import torch.nn as nn
from torch import Tensor


class SkillEncoder(nn.Module):
    """Encodes observations into skill embeddings."""

    def __init__(
        self,
        obs_dim: int,
        skill_dim: int,
        hidden_dims: list[int] = [128, 128],
        activation: str = "relu",
    ):
        super().__init__()
        self.skill_dim = skill_dim
        
        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build encoder network
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                act_fn(),
            ])
            in_dim = hidden_dim
        
        # Output layer: mean and log_std for Gaussian
        layers.append(nn.Linear(in_dim, skill_dim * 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute skill distribution parameters.
        
        Args:
            obs: Observations [batch, obs_dim]
            
        Returns:
            mu: Mean of skill distribution [batch, skill_dim]
            log_std: Log standard deviation [batch, skill_dim]
        """
        out = self.network(obs)
        mu, log_std = torch.chunk(out, 2, dim=-1)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -10, 2)
        return mu, log_std
    
    def sample(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        """Sample skill from distribution.
        
        Args:
            obs: Observations [batch, obs_dim]
            deterministic: If True, return mean; otherwise sample
            
        Returns:
            skill: Sampled skill [batch, skill_dim]
        """
        mu, log_std = self.forward(obs)
        
        if deterministic:
            return mu
        
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        return mu + std * eps
    
    def log_prob(self, obs: Tensor, skill: Tensor) -> Tensor:
        """Compute log probability of skill given observation.
        
        Args:
            obs: Observations [batch, obs_dim]
            skill: Skills [batch, skill_dim]
            
        Returns:
            log_prob: Log probability [batch]
        """
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        
        # Gaussian log probability
        log_prob = -0.5 * (
            ((skill - mu) / std) ** 2 + 
            2 * log_std + 
            torch.log(torch.tensor(2 * 3.14159))
        )
        return log_prob.sum(dim=-1)

