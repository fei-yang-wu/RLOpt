"""Multi-discriminator helpers for ASE compatibility."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from rlopt.agent.gail.discriminator import Discriminator


class MultiDiscriminator(nn.Module):
    """Multiple discriminators, one per skill."""

    def __init__(
        self,
        num_skills: int,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_skills = num_skills
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Create one discriminator per skill
        self.discriminators = nn.ModuleList(
            [
                Discriminator(
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    hidden_dims=list(hidden_dims),
                    activation=activation,
                )
                for _ in range(num_skills)
            ]
        )

    def forward_logits(self, observation: Tensor, action: Tensor, skill_idx: Tensor) -> Tensor:
        """Classify state-action for specific skill.

        Args:
            observation: [batch, obs_dim]
            action: [batch, action_dim]
            skill_idx: [batch] - skill indices (0 to num_skills-1)

        Returns:
            logits: [batch, 1]
        """
        if skill_idx.ndim > 1:
            skill_idx = skill_idx.squeeze(-1)
        skill_idx = skill_idx.to(dtype=torch.long, device=observation.device)

        logits = torch.zeros(
            observation.shape[0], 1, device=observation.device, dtype=observation.dtype
        )
        for skill_id, disc in enumerate(self.discriminators):
            mask = skill_idx == skill_id
            if not bool(mask.any()):
                continue
            logits[mask] = disc(observation[mask], action[mask])
        return logits

    def forward(self, observation: Tensor, action: Tensor, skill_idx: Tensor) -> Tensor:
        """Return expert probability for each (obs, action, skill_idx)."""
        return torch.sigmoid(self.forward_logits(observation, action, skill_idx))

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
            out = disc.predict_proba(observation, action)
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
