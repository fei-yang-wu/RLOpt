"""Info-GAIL (Information-Theoretic GAIL) implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict

from rlopt.agent.gail.gail import GAIL, GAILConfig, GAILRLOptConfig
from rlopt.configs import NetworkConfig
from .skill_encoder import SkillEncoder
from .posterior import SkillPosterior
from .skill_discriminator import SkillConditionedDiscriminator


@dataclass
class InfoGAILConfig(GAILConfig):
    """Configuration for Info-GAIL-specific parameters."""
    
    # Skill configuration
    skill_dim: int = 8
    skill_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    
    # Posterior configuration
    posterior_hidden_dim: int = 128
    posterior_num_layers: int = 1
    trajectory_length: int = 50
    
    # Training configuration
    mi_coefficient: float = 0.1  # Weight for mutual information term
    skill_encoder_lr: float = 3e-4
    posterior_lr: float = 3e-4
    
    # Skill prior
    skill_prior: str = "gaussian"  # gaussian or uniform


@dataclass
class InfoGAILRLOptConfig(GAILRLOptConfig):
    """Configuration for Info-GAIL with RLOpt."""
    
    infogail: InfoGAILConfig = field(default_factory=InfoGAILConfig)


class InfoGAIL(GAIL):
    """Info-GAIL with skill embeddings for diverse behavior learning."""

    def __init__(self, env, config: InfoGAILRLOptConfig):
        self.config: InfoGAILRLOptConfig = config
        self.log = logging.getLogger(__name__)
        
        # Store dimensions
        self._obs_dim = env.observation_spec["observation"].shape[-1]
        self._action_dim = env.action_spec.shape[-1]
        
        # Initialize skill-related components
        self.skill_encoder = SkillEncoder(
            obs_dim=self._obs_dim,
            skill_dim=config.infogail.skill_dim,
            hidden_dims=config.infogail.skill_hidden_dims,
        )
        
        self.posterior = SkillPosterior(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            skill_dim=config.infogail.skill_dim,
            hidden_dim=config.infogail.posterior_hidden_dim,
            num_layers=config.infogail.posterior_num_layers,
        )
        
        # Trajectory buffer for skill inference
        self.trajectory_buffer = []
        self.max_trajectory_length = config.infogail.trajectory_length
        
        # Expert replay buffer (set later)
        self._expert_buffer = None
        self._expert_iter = None
        self._warned_no_expert = False
        
        # Discriminator placeholder (will be skill-conditioned)
        self.discriminator = None  # type: ignore
        
        # Call parent __init__ (BaseAlgorithm, not GAIL)
        # We override discriminator construction
        from rlopt.base_class import BaseAlgorithm
        BaseAlgorithm.__init__(self, env, config)
        
        # Construct target network updater
        self.target_net_updater = self._construct_target_net_updater()
        
        # Move skill models to device
        self.skill_encoder = self.skill_encoder.to(self.device)
        self.posterior = self.posterior.to(self.device)
        
        self.log.info("Info-GAIL initialized")
        self.log.info(f"  Skill dimension: {config.infogail.skill_dim}")
        self.log.info(f"  Discriminator: {sum(p.numel() for p in self.discriminator.parameters())} parameters")
        if hasattr(self, 'actor_critic'):
            policy_params = sum(p.numel() for p in self.actor_critic.get_policy_operator().parameters())
            self.log.info(f"  Policy: {policy_params} parameters")

    def _set_optimizers(
        self, optimizer_cls: type, optimizer_kwargs: dict
    ) -> list[torch.optim.Optimizer]:
        """Set up optimizers for all components."""
        # Initialize skill-conditioned discriminator
        if self.discriminator is None:
            self.discriminator = SkillConditionedDiscriminator(
                observation_dim=self._obs_dim,
                action_dim=self._action_dim,
                skill_dim=self.config.infogail.skill_dim,
                hidden_dims=self.config.gail.discriminator_hidden_dims,
                activation=self.config.gail.discriminator_activation,
            ).to(self.device)
        
        # SAC optimizers (from BaseAlgorithm)
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

        self.log.info("Optimizers configured:")
        self.log.info(f"  Actor: {optimizer_kwargs.get('lr', 'N/A')}")
        self.log.info(f"  Critic: {optimizer_kwargs.get('lr', 'N/A')}")
        self.log.info(f"  Alpha: 3.0e-4")
        self.log.info(f"  Discriminator: {self.config.gail.discriminator_lr}")
        self.log.info(f"  Skill Encoder: {self.config.infogail.skill_encoder_lr}")
        self.log.info(f"  Posterior: {self.config.infogail.posterior_lr}")
        
        return optimizers

    def _sample_skill_prior(self, batch_size: int) -> Tensor:
        """Sample skills from prior distribution."""
        if self.config.infogail.skill_prior == "gaussian":
            return torch.randn(batch_size, self.config.infogail.skill_dim, device=self.device)
        elif self.config.infogail.skill_prior == "uniform":
            return torch.rand(batch_size, self.config.infogail.skill_dim, device=self.device) * 2 - 1
        else:
            raise ValueError(f"Unknown skill prior: {self.config.infogail.skill_prior}")

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        """Perform one Info-GAIL update step."""
        batch_size = sampled_tensordict.batch_size[0]
        
        # Sample skills from prior
        skills = self._sample_skill_prior(batch_size)
        
        # Extract state-action pairs
        obs = sampled_tensordict["observation"]
        action = sampled_tensordict["action"]
        
        # 1. Update discriminator
        for _ in range(self.config.gail.discriminator_steps):
            # Get expert batch
            expert_batch = self._next_expert_batch()
            
            if expert_batch is not None:
                expert_obs = expert_batch["observation"]
                expert_action = expert_batch["action"]
                expert_batch_size = expert_obs.shape[0]
                
                # Sample skills for expert data
                expert_skills = self._sample_skill_prior(expert_batch_size)
                
                # Discriminator predictions
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
        
        # 2. Compute GAIL rewards with skills
        with torch.no_grad():
            gail_rewards = self.discriminator.compute_reward(obs, action, skills)
        
        # 3. Compute mutual information bonus
        # For simplicity, use current obs as trajectory (can be extended)
        trajectory = torch.cat([obs.unsqueeze(1), action.unsqueeze(1)], dim=-1)  # [batch, 1, obs+action]
        
        posterior_log_prob = self.posterior.log_prob(trajectory, skills)
        
        # Prior log prob (standard Gaussian)
        if self.config.infogail.skill_prior == "gaussian":
            prior_log_prob = -0.5 * (skills ** 2).sum(dim=-1) - 0.5 * self.config.infogail.skill_dim * np.log(2 * np.pi)
        else:
            prior_log_prob = torch.zeros_like(posterior_log_prob)
        
        mi_bonus = (posterior_log_prob - prior_log_prob).unsqueeze(-1)
        
        # 4. Combined reward
        combined_reward = gail_rewards + self.config.infogail.mi_coefficient * mi_bonus
        
        # Replace environment reward with combined reward
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
        
        # Alpha loss (if applicable)
        if len(self.optim) > 2 and "loss_alpha" in loss_td.keys():
            loss_alpha = loss_td["loss_alpha"]
            self.optim[2].zero_grad()
            loss_alpha.backward()
            self.optim[2].step()
        
        # 6. Update posterior network
        # Recompute for gradient
        posterior_log_prob_grad = self.posterior.log_prob(trajectory, skills)
        posterior_loss = -posterior_log_prob_grad.mean()
        
        self.posterior_optim.zero_grad()
        posterior_loss.backward()
        self.posterior_optim.step()
        
        # 7. Update target networks
        self.target_net_updater.step()
        
        # Return diagnostics
        return TensorDict(
            {
                "loss_actor": loss_actor.detach(),
                "loss_qvalue": loss_qvalue.detach(),
                "loss_discriminator": disc_loss.detach() if isinstance(disc_loss, Tensor) else disc_loss,
                "loss_posterior": posterior_loss.detach(),
                "gail_reward_mean": gail_rewards.mean().detach(),
                "mi_bonus_mean": mi_bonus.mean().detach(),
                "combined_reward_mean": combined_reward.mean().detach(),
            },
            batch_size=[],
        )

    def save(self, path: str | Path) -> None:
        """Save Info-GAIL model."""
        path = Path(path)
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "skill_encoder": self.skill_encoder.state_dict(),
                "posterior": self.posterior.state_dict(),
                "config": self.config,
            },
            path,
        )
        self.log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load Info-GAIL model."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.skill_encoder.load_state_dict(checkpoint["skill_encoder"])
        self.posterior.load_state_dict(checkpoint["posterior"])
        self.log.info(f"Model loaded from {path}")
