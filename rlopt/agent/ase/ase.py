"""ASE (Adversarial Skill Embeddings) implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict

from rlopt.agent.imitation.infogail.infogail import InfoGAIL, InfoGAILConfig, InfoGAILRLOptConfig
from .multi_discriminator import MultiDiscriminator, StyleDiscriminator


@dataclass
class ASEConfig(InfoGAILConfig):
    """Configuration for ASE-specific parameters."""
    
    # Number of distinct skills
    num_skills: int = 8
    
    # Reward coefficients
    style_coeff: float = 1.0  # Weight for style reward
    diversity_coeff: float = 0.5  # Weight for diversity reward
    task_coeff: float = 0.1  # Weight for task reward (if using environment reward)
    
    # Style discriminator
    style_hidden_dim: int = 128
    style_num_layers: int = 1
    style_lr: float = 3e-4
    
    # Training
    use_multi_discriminator: bool = True
    skill_sampling: str = "uniform"  # uniform, sequential, or curriculum


@dataclass
class ASERLOptConfig(InfoGAILRLOptConfig):
    """Configuration for ASE with RLOpt."""
    
    ase: ASEConfig = field(default_factory=ASEConfig)


class ASE(InfoGAIL):
    """ASE with multiple discriminators and diversity rewards."""

    def __init__(self, env, config: ASERLOptConfig):
        self.config: ASERLOptConfig = config
        self.log = logging.getLogger(__name__)
        
        # Store dimensions
        self._obs_dim = env.observation_spec["observation"].shape[-1]
        self._action_dim = env.action_spec.shape[-1]
        
        # Initialize skill index tracking
        self.current_skill_idx = 0
        
        # Expert replay buffer
        self._expert_buffer = None
        self._expert_iter = None
        self._warned_no_expert = False
        
        # Discriminator placeholder
        self.discriminator = None  # type: ignore
        self.multi_discriminator = None
        
        # Initialize parent (InfoGAIL)
        from rlopt.base_class import BaseAlgorithm
        BaseAlgorithm.__init__(self, env, config)
        
        # Construct target network updater
        self.target_net_updater = self._construct_target_net_updater()
        
        # Initialize style discriminator
        self.style_discriminator = StyleDiscriminator(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            num_skills=config.ase.num_skills,
            hidden_dim=config.ase.style_hidden_dim,
            num_layers=config.ase.style_num_layers,
        ).to(self.device)
        
        # Move models to device
        if hasattr(self, 'skill_encoder'):
            self.skill_encoder = self.skill_encoder.to(self.device)
        if hasattr(self, 'posterior'):
            self.posterior = self.posterior.to(self.device)
        
        self.log.info("ASE initialized")
        self.log.info(f"  Number of skills: {config.ase.num_skills}")
        self.log.info(f"  Multi-discriminator: {config.ase.use_multi_discriminator}")
        if hasattr(self, 'actor_critic'):
            policy_params = sum(p.numel() for p in self.actor_critic.get_policy_operator().parameters())
            self.log.info(f"  Policy: {policy_params} parameters")

    def _set_optimizers(
        self, optimizer_cls: type, optimizer_kwargs: dict
    ) -> list[torch.optim.Optimizer]:
        """Set up optimizers for all components."""
        # Initialize multi-discriminator
        if self.config.ase.use_multi_discriminator:
            self.multi_discriminator = MultiDiscriminator(
                num_skills=self.config.ase.num_skills,
                observation_dim=self._obs_dim,
                action_dim=self._action_dim,
                hidden_dims=self.config.gail.discriminator_hidden_dims,
                activation=self.config.gail.discriminator_activation,
            ).to(self.device)
            self.discriminator = self.multi_discriminator  # For compatibility
        else:
            # Use skill-conditioned discriminator from InfoGAIL
            from rlopt.agent.imitation.infogail.skill_discriminator import SkillConditionedDiscriminator
            self.discriminator = SkillConditionedDiscriminator(
                observation_dim=self._obs_dim,
                action_dim=self._action_dim,
                skill_dim=self.config.infogail.skill_dim,
                hidden_dims=self.config.gail.discriminator_hidden_dims,
                activation=self.config.gail.discriminator_activation,
            ).to(self.device)
        
        # Initialize skill encoder and posterior (from InfoGAIL)
        from rlopt.agent.imitation.infogail.skill_encoder import SkillEncoder
        from rlopt.agent.imitation.infogail.posterior import SkillPosterior
        
        self.skill_encoder = SkillEncoder(
            obs_dim=self._obs_dim,
            skill_dim=self.config.infogail.skill_dim,
            hidden_dims=self.config.infogail.skill_hidden_dims,
        ).to(self.device)
        
        self.posterior = SkillPosterior(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            skill_dim=self.config.infogail.skill_dim,
            hidden_dim=self.config.infogail.posterior_hidden_dim,
            num_layers=self.config.infogail.posterior_num_layers,
        ).to(self.device)
        
        # SAC optimizers
        critic_params = list(
            self.loss_module.qvalue_network_params.flatten_keys().values()  # type: ignore[attr-defined]
        )
        actor_params = list(
            self.loss_module.actor_network_params.flatten_keys().values()  # type: ignore[attr-defined]
        )

        optimizers = [
            optimizer_cls(actor_params, **optimizer_kwargs),
            optimizer_cls(critic_params, **optimizer_kwargs),
        ]

        # Alpha optimizer
        if hasattr(self.loss_module, "log_alpha"):
            param = self.loss_module.log_alpha
            if isinstance(param, torch.nn.Parameter):
                optimizers.append(torch.optim.Adam([param], lr=3.0e-4))

        # Discriminator optimizer
        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.gail.discriminator_lr,
        )
        
        # Style discriminator optimizer
        self.style_optim = torch.optim.Adam(
            self.style_discriminator.parameters(),
            lr=self.config.ase.style_lr,
        )
        
        # Skill encoder optimizer
        self.skill_encoder_optim = torch.optim.Adam(
            self.skill_encoder.parameters(),
            lr=self.config.infogail.skill_encoder_lr,
        )
        
        # Posterior optimizer
        self.posterior_optim = torch.optim.Adam(
            self.posterior.parameters(),
            lr=self.config.infogail.posterior_lr,
        )

        self.log.info("Optimizers configured for ASE")
        
        return optimizers

    def _sample_skill_idx(self, batch_size: int) -> Tensor:
        """Sample skill indices."""
        if self.config.ase.skill_sampling == "uniform":
            return torch.randint(0, self.config.ase.num_skills, (batch_size,), device=self.device)
        elif self.config.ase.skill_sampling == "sequential":
            # Rotate through skills
            idx = torch.full((batch_size,), self.current_skill_idx, device=self.device)
            self.current_skill_idx = (self.current_skill_idx + 1) % self.config.ase.num_skills
            return idx
        else:
            raise ValueError(f"Unknown skill sampling: {self.config.ase.skill_sampling}")

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        """Perform one ASE update step."""
        batch_size = sampled_tensordict.batch_size[0]
        
        # Sample skill indices
        skill_indices = self._sample_skill_idx(batch_size)
        
        # Extract state-action pairs
        obs = sampled_tensordict["observation"]
        action = sampled_tensordict["action"]
        
        # 1. Update multi-discriminator
        for _ in range(self.config.gail.discriminator_steps):
            expert_batch = self._next_expert_batch()
            
            if expert_batch is not None:
                expert_obs = expert_batch["observation"]
                expert_action = expert_batch["action"]
                expert_batch_size = expert_obs.shape[0]
                
                # Sample skill indices for expert
                expert_skill_indices = self._sample_skill_idx(expert_batch_size)
                
                # Discriminator predictions
                if self.config.ase.use_multi_discriminator:
                    d_policy = self.multi_discriminator(obs, action, skill_indices)
                    d_expert = self.multi_discriminator(expert_obs, expert_action, expert_skill_indices)
                else:
                    # Use skill-conditioned discriminator
                    skills = self._sample_skill_prior(batch_size)
                    expert_skills = self._sample_skill_prior(expert_batch_size)
                    d_policy = self.discriminator(obs, action, skills)
                    d_expert = self.discriminator(expert_obs, expert_action, expert_skills)
                
                # Binary cross-entropy loss
                disc_loss = -(
                    torch.log(d_expert + 1e-7).mean() +
                    torch.log(1 - d_policy + 1e-7).mean()
                )
                
                self.discriminator_optim.zero_grad()
                disc_loss.backward()
                self.discriminator_optim.step()
            else:
                disc_loss = torch.tensor(0.0, device=self.device)
        
        # 2. Compute style reward
        with torch.no_grad():
            if self.config.ase.use_multi_discriminator:
                style_reward = self.multi_discriminator.compute_reward(obs, action, skill_indices)
            else:
                skills = self._sample_skill_prior(batch_size)
                style_reward = self.discriminator.compute_reward(obs, action, skills)
        
        # 3. Compute diversity reward
        trajectory = torch.cat([obs.unsqueeze(1), action.unsqueeze(1)], dim=-1)
        style_probs = self.style_discriminator.get_probs(trajectory)
        
        # Diversity = log p(skill|trajectory)
        diversity_reward = torch.log(style_probs[torch.arange(batch_size), skill_indices] + 1e-7).unsqueeze(-1)
        
        # 4. Combined reward
        combined_reward = (
            self.config.ase.style_coeff * style_reward +
            self.config.ase.diversity_coeff * diversity_reward
        )
        
        # Optionally add task reward
        if self.config.ase.task_coeff > 0:
            env_reward = sampled_tensordict["next", "reward"]
            combined_reward = combined_reward + self.config.ase.task_coeff * env_reward
        
        # Replace environment reward
        sampled_tensordict["next", "reward"] = combined_reward
        
        # 5. Update SAC policy
        loss_td = self.loss_module(sampled_tensordict)
        
        # Actor loss
        loss_actor = loss_td["loss_actor"]
        self.optim[0].zero_grad()
        loss_actor.backward()
        self.optim[0].step()
        
        # Critic loss
        loss_qvalue = loss_td["loss_qvalue"]
        self.optim[1].zero_grad()
        loss_qvalue.backward()
        self.optim[1].step()
        
        # Alpha loss
        if len(self.optim) > 2 and "loss_alpha" in loss_td.keys():
            loss_alpha = loss_td["loss_alpha"]
            self.optim[2].zero_grad()
            loss_alpha.backward()
            self.optim[2].step()
        
        # 6. Update style discriminator
        # Create pseudo-labels (skill indices)
        style_logits = self.style_discriminator(trajectory.detach())
        style_loss = nn.functional.cross_entropy(style_logits, skill_indices)
        
        self.style_optim.zero_grad()
        style_loss.backward()
        self.style_optim.step()
        
        # 7. Update target networks
        self.target_net_updater.step()
        
        # Return diagnostics
        return TensorDict(
            {
                "loss_actor": loss_actor.detach(),
                "loss_qvalue": loss_qvalue.detach(),
                "loss_discriminator": disc_loss.detach() if isinstance(disc_loss, Tensor) else disc_loss,
                "loss_style": style_loss.detach(),
                "style_reward_mean": style_reward.mean().detach(),
                "diversity_reward_mean": diversity_reward.mean().detach(),
                "combined_reward_mean": combined_reward.mean().detach(),
            },
            batch_size=[],
        )

    def save(self, path: str | Path) -> None:
        """Save ASE model."""
        path = Path(path)
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "style_discriminator": self.style_discriminator.state_dict(),
                "skill_encoder": self.skill_encoder.state_dict(),
                "posterior": self.posterior.state_dict(),
                "config": self.config,
            },
            path,
        )
        self.log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load ASE model."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.style_discriminator.load_state_dict(checkpoint["style_discriminator"])
        self.skill_encoder.load_state_dict(checkpoint["skill_encoder"])
        self.posterior.load_state_dict(checkpoint["posterior"])
        self.log.info(f"Model loaded from {path}")

