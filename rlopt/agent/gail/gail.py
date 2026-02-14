"""GAIL (Generative Adversarial Imitation Learning) implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer, ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType
from torchrl.modules import (
    ActorCriticOperator,
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import SACLoss, SoftUpdate

from rlopt.base_class import BaseAlgorithm, RLOptConfig
from rlopt.configs import OptimizerConfig, NetworkConfig
from rlopt.imitation.expert_buffer import ExpertReplayBuffer
from rlopt.utils import get_activation_class
from .discriminator import Discriminator


@dataclass
class GAILConfig:
    """Configuration for GAIL-specific parameters."""

    # Discriminator configuration
    discriminator_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    discriminator_activation: str = "relu"
    discriminator_lr: float = 3e-4

    # Training configuration
    expert_batch_size: int = 256
    discriminator_steps: int = 1  # Number of discriminator updates per policy update
    discriminator_loss_coeff: float = 1.0  # Coefficient for discriminator loss

    # Reward configuration
    use_gail_reward: bool = True  # If False, use environment reward (for debugging)
    gail_reward_coeff: float = 1.0  # Coefficient for GAIL reward


@dataclass
class GAILRLOptConfig(RLOptConfig):
    """Configuration for GAIL with RLOpt."""

    gail: GAILConfig = field(default_factory=GAILConfig)

    # Override default SAC config for imitation learning
    # Use lower learning rates for more stable training
    actor_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=3e-4)
    )
    critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=3e-4)
    )
    alpha_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=3e-4)
    )


class GAIL(BaseAlgorithm):
    """GAIL (Generative Adversarial Imitation Learning).

    GAIL uses adversarial training to learn from expert demonstrations.
    A discriminator is trained to distinguish between expert and policy
    state-action pairs, while the policy is trained to fool the discriminator.

    The discriminator provides rewards: r(s,a) = -log(1 - D(s,a))
    where D(s,a) is the discriminator output.

    Args:
        env: Environment to train on
        config: Configuration for GAIL
    """

    def __init__(self, env: EnvBase, config: GAILRLOptConfig):
        self.config: GAILRLOptConfig = config
        self.log = logging.getLogger(__name__)
        
        # Expert replay buffer
        self._expert_buffer: ExpertReplayBuffer | None = None
        self._expert_iter = None
        self._warned_no_expert = False
        
        # Store discriminator dimensions (will initialize in _set_optimizers)
        self._obs_dim = env.observation_spec["observation"].shape[-1]
        self._action_dim = env.action_spec.shape[-1]
        self.discriminator: Discriminator = None  # type: ignore

        # Call super().__init__() which will build networks, loss, and optimizers
        super().__init__(env, config)
        
        # Construct target network updater for SAC
        self.target_net_updater = self._construct_target_net_updater()

        self.log.info("GAIL initialized")
        self.log.info(
            f"  Discriminator: {sum(p.numel() for p in self.discriminator.parameters())} parameters"
        )
        # Count policy parameters from actor_critic
        if hasattr(self, 'actor_critic'):
            policy_params = sum(p.numel() for p in self.actor_critic.get_policy_operator().parameters())
            self.log.info(f"  Policy: {policy_params} parameters")
    
    # ========================================================================
    # Abstract method implementations (required by BaseAlgorithm)
    # ========================================================================
    
    def _construct_feature_extractor(
        self, feature_extractor_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """GAIL does not require a feature extractor."""
        raise NotImplementedError("GAIL does not require a feature extractor by default")
    
    def _construct_policy(
        self, policy_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Construct SAC policy for GAIL."""
        # Define policy output distribution
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": self.env.action_spec_unbatched.space.low.to(self.device),
            "high": self.env.action_spec_unbatched.space.high.to(self.device),
            "tanh_loc": False,
        }
        
        # Build policy network
        if policy_net is None:
            policy_mlp = MLP(
                in_features=self.config.policy.input_dim,
                activation_class=get_activation_class(self.config.policy.activation_fn),
                out_features=2 * self.env.action_spec_unbatched.shape[-1],
                num_cells=list(self.config.policy.num_cells),
                device=self.device,
            )
        else:
            policy_mlp = policy_net
        
        # Add parameter extractor for Normal distribution
        extractor = NormalParamExtractor(
            scale_mapping="biased_softplus_1.0",
            scale_lb=0.1,
        ).to(self.device)
        net = torch.nn.Sequential(policy_mlp, extractor)
        
        # Wrap in TensorDictModule
        policy_td = TensorDictModule(
            module=net,
            in_keys=self.config.policy.get_input_keys(),
            out_keys=["loc", "scale"],
        )
        
        return ProbabilisticActor(
            policy_td,
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=False,
            default_interaction_type=ExplorationType.RANDOM,
        )
    
    def _construct_q_function(self, q_net: nn.Module | None = None) -> TensorDictModule:
        """Construct Q-function for SAC."""
        if self.config.q_function is None:
            raise ValueError("GAIL requires a Q-function configuration")
        
        # Build Q-network
        if q_net is None:
            q_mlp = MLP(
                in_features=self.config.q_function.input_dim,
                activation_class=get_activation_class(self.config.q_function.activation_fn),
                out_features=1,
                num_cells=list(self.config.q_function.num_cells),
                device=self.device,
            )
        else:
            q_mlp = q_net
        
        # Q-function takes both observation and action
        in_keys = self.config.q_function.get_input_keys()
        if "action" not in in_keys:
            in_keys.append("action")
        
        return ValueOperator(
            module=q_mlp,
            in_keys=in_keys,
            out_keys=["state_action_value"],
        )
    
    def _construct_actor_critic(self) -> TensorDictModule:
        """Construct actor-critic for SAC."""
        assert isinstance(self.q_function, TensorDictModule)
        assert isinstance(self.policy, TensorDictModule)
        
        # Use identity module as common operator (no feature extraction)
        class IdentityModule(torch.nn.Module):
            def forward(self, *x):
                return x[0] if len(x) == 1 else x
        
        common_operator = TensorDictModule(
            module=IdentityModule(),
            in_keys=["observation"],
            out_keys=["observation"],
        )
        
        return ActorCriticOperator(
            common_operator=common_operator,
            policy_operator=self.policy,
            value_operator=self.q_function,
        )
    
    def _construct_loss_module(self) -> nn.Module:
        """Construct SAC loss module."""
        assert isinstance(self.config, GAILRLOptConfig)
        
        # Initialize lazy layers
        fake_tensordict = self.env.fake_tensordict()
        with torch.no_grad():
            _ = self.actor_critic(fake_tensordict)
        
        loss_module = SACLoss(
            actor_network=self.actor_critic.get_policy_operator(),
            qvalue_network=self.actor_critic.get_critic_operator(),
            num_qvalue_nets=2,
            loss_function=self.config.loss.loss_critic_type,
            delay_actor=False,
            delay_qvalue=True,
        )
        loss_module.make_value_estimator(gamma=self.config.loss.gamma)
        
        return loss_module
    
    def _construct_data_buffer(self) -> ReplayBuffer:
        """Construct replay buffer for policy samples."""
        storage = LazyTensorStorage(max_size=self.config.replay_buffer.size)
        return TensorDictReplayBuffer(
            storage=storage,
            batch_size=self.config.loss.mini_batch_size,
        )
    
    def _construct_target_net_updater(self) -> SoftUpdate:
        """Construct target network updater for SAC."""
        eps = self.config.optim.target_update_polyak
        return SoftUpdate(self.loss_module, eps=eps)
    
    def predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        """Predict action from observation."""
        with torch.no_grad():
            td = TensorDict(
                {"observation": observation},
                batch_size=observation.shape[:-1],
            )
            td = self.policy_module(td)
            if deterministic:
                # Use mean of distribution
                return td["loc"]
            else:
                # Sample from distribution
                return td["action"]

    # ========================================================================
    # GAIL-specific methods
    # ========================================================================

    def set_expert_buffer(self, expert_buffer: ExpertReplayBuffer) -> None:
        """Set the expert replay buffer."""
        self._expert_buffer = expert_buffer
        self.log.info(f"Expert buffer attached: {len(expert_buffer)} samples")

    def _next_expert_batch(self) -> TensorDict | None:
        """Sample a batch from expert buffer."""
        if self._expert_buffer is None:
            return None

        try:
            # ExpertReplayBuffer.sample() doesn't take arguments - batch_size is set at init
            return self._expert_buffer.sample().to(self.device)
        except Exception as e:
            self.log.warning(f"Failed to sample expert batch: {e}")
            return None

    def _set_optimizers(
        self, optimizer_cls: type, optimizer_kwargs: dict
    ) -> list[torch.optim.Optimizer]:
        """Set up optimizers for all trainable components."""
        # Initialize discriminator (needs device from parent __init__)
        if self.discriminator is None:
            self.discriminator = Discriminator(
                observation_dim=self._obs_dim,
                action_dim=self._action_dim,
                hidden_dims=self.config.gail.discriminator_hidden_dims,
                activation=self.config.gail.discriminator_activation,
            ).to(self.device)
        
        # Create SAC optimizers (actor, critic, alpha)
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

        # Alpha optimizer for SAC entropy temperature
        if hasattr(self.loss_module, "log_alpha"):
            param = self.loss_module.log_alpha
            if isinstance(param, torch.nn.Parameter):
                optimizers.append(torch.optim.Adam([param], lr=3.0e-4))

        # Discriminator optimizer
        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.gail.discriminator_lr,
        )

        self.log.info("Optimizers configured:")
        self.log.info(f"  Actor: {optimizer_kwargs.get('lr', 'N/A')}")
        self.log.info(f"  Critic: {optimizer_kwargs.get('lr', 'N/A')}")
        self.log.info(f"  Alpha: 3.0e-4")
        self.log.info(f"  Discriminator: {self.config.gail.discriminator_lr}")
        
        return optimizers

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        """Perform one GAIL update step.

        Args:
            sampled_tensordict: Batch of transitions from replay buffer

        Returns:
            TensorDict with loss values and diagnostics
        """
        # Validate input
        if not self._validate_tensordict(
            sampled_tensordict, "update:input", raise_error=False
        ):
            self.log.debug("Discarding batch due to non-finite values")
            return self._dummy_loss_tensordict()

        # Zero all grads
        self.optim.zero_grad(set_to_none=True)
        self.discriminator_optim.zero_grad(set_to_none=True)

        # Extract policy data
        policy_obs = sampled_tensordict["observation"]
        policy_action = sampled_tensordict["action"]

        # Sample expert data
        expert_td = self._next_expert_batch()
        if expert_td is None and not self._warned_no_expert:
            self.log.warning("No expert data available, skipping discriminator update")
            self._warned_no_expert = True

        # ====================================================================
        # 1. Update discriminator
        # ====================================================================
        discriminator_info = {}
        if expert_td is not None:
            expert_obs = expert_td["observation"].to(self.device)
            expert_action = expert_td["action"].to(self.device)

            for _ in range(self.config.gail.discriminator_steps):
                # Compute discriminator loss
                disc_loss, disc_info = self.discriminator.compute_loss(
                    expert_obs=expert_obs,
                    expert_action=expert_action,
                    policy_obs=policy_obs.detach(),  # Don't backprop through policy
                    policy_action=policy_action.detach(),
                )

                # Update discriminator
                disc_loss = disc_loss * self.config.gail.discriminator_loss_coeff
                disc_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    max_norm=1.0,
                )

                self.discriminator_optim.step()
                self.discriminator_optim.zero_grad(set_to_none=True)

                discriminator_info = disc_info

        # ====================================================================
        # 2. Compute GAIL rewards for policy update
        # ====================================================================
        if self.config.gail.use_gail_reward:
            # Use discriminator-based rewards
            with torch.no_grad():
                gail_rewards = self.discriminator.compute_reward(
                    policy_obs, policy_action
                )

            # Replace rewards in tensordict
            sampled_tensordict_for_sac = sampled_tensordict.clone()
            sampled_tensordict_for_sac["reward"] = (
                gail_rewards * self.config.gail.gail_reward_coeff
            ).unsqueeze(-1)
        else:
            # Use environment rewards (for debugging)
            sampled_tensordict_for_sac = sampled_tensordict

        # ====================================================================
        # 3. Update policy with SAC
        # ====================================================================
        loss_td = cast(TensorDict, self.loss_module(sampled_tensordict_for_sac))
        loss_td = self._sanitize_loss_tensordict(loss_td, "update:raw_sac_loss")

        actor_loss = cast(Tensor, loss_td["loss_actor"])
        q_loss = cast(Tensor, loss_td["loss_qvalue"])
        alpha_loss = cast(Tensor, loss_td["loss_alpha"])
        total_sac_loss = (actor_loss + q_loss + alpha_loss).sum()
        total_sac_loss.backward()

        # Validate gradients
        if not self._validate_gradients("update:post_backward", raise_error=False):
            self.log.debug("Non-finite gradients detected, skipping update")
            self.optim.zero_grad(set_to_none=True)
            return self._dummy_loss_tensordict()

        # Step SAC optimizers
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update target networks
        self.target_net_updater.step()

        # ====================================================================
        # 4. Attach diagnostics
        # ====================================================================
        out = cast(TensorDict, loss_td.clone())

        # Add discriminator diagnostics
        for key, value in discriminator_info.items():
            out[key] = value

        # Add GAIL reward diagnostics
        if self.config.gail.use_gail_reward:
            with torch.no_grad():
                out["gail_reward_mean"] = gail_rewards.mean()
                out["gail_reward_std"] = gail_rewards.std()

        return out

    def _dummy_loss_tensordict(self) -> TensorDict:
        """Return dummy loss tensordict for error cases."""
        return TensorDict(
            {
                "loss_actor": torch.tensor(0.0, device=self.device),
                "loss_qvalue": torch.tensor(0.0, device=self.device),
                "loss_alpha": torch.tensor(0.0, device=self.device),
                "discriminator_loss": torch.tensor(0.0, device=self.device),
                "gail_reward_mean": torch.tensor(0.0, device=self.device),
                "gail_reward_std": torch.tensor(0.0, device=self.device),
            },
            batch_size=[],
        )

    def train(self) -> None:
        """Train GAIL on the environment."""
        cfg = self.config
        assert isinstance(cfg, GAILRLOptConfig)
        
        frames_per_batch = cfg.collector.frames_per_batch
        total_frames = cfg.collector.total_frames
        init_random_frames = int(cfg.collector.init_random_frames)
        
        # UTD ratio for SAC updates
        utd_ratio = 1.0  # Standard SAC update ratio
        num_updates = max(1, int(frames_per_batch * utd_ratio))

        # Compilation (if requested)
        compile_mode = None
        if cfg.compile.compile:
            compile_mode = cfg.compile.compile_mode
            if compile_mode in ("", None):
                compile_mode = "reduce-overhead" if not cfg.compile.cudagraphs else "default"
            
            self.log.info(f"Compiling update function with mode: {compile_mode}")
            from torchrl._utils import compile_with_warmup
            self.update = compile_with_warmup(self.update, mode=compile_mode, warmup=1)  # type: ignore[method-assign]
        
        # CUDAGraphs (experimental)
        if cfg.compile.cudagraphs:
            if self.device.type == "cuda":
                import warnings
                from tensordict.nn import CudaGraphModule
                warnings.warn(
                    "CudaGraphModule is experimental and may lead to silently wrong results.",
                    category=UserWarning,
                )
                self.log.warning("Wrapping update with CudaGraphModule (experimental)")
                self.update = CudaGraphModule(
                    self.update, in_keys=[], out_keys=[], warmup=5
                )  # type: ignore[method-assign]
            else:
                self.log.warning(f"CUDAGraphs requested but device is {self.device.type}, not CUDA. Skipping.")

        # Training loop
        collected_frames = 0
        pbar = self.collector

        self.log.info("Starting GAIL training")
        self.log.info(f"  Total frames: {total_frames:,}")
        self.log.info(f"  Frames per batch: {frames_per_batch:,}")
        self.log.info(f"  Updates per batch: {num_updates:,}")

        for i, data in enumerate(pbar):
            # Add episode rewards/lengths if available
            if ("next", "done") in data.keys(True):
                done_indices = data["next", "done"]
                if done_indices.any():
                    episode_rewards = data["next", "episode_reward"][done_indices]
                    episode_length = data["next", "step_count"][done_indices]
                    self.episode_lengths.extend(episode_length.cpu().tolist())
                    self.episode_rewards.extend(episode_rewards.cpu().tolist())

            # Store in replay buffer
            self.data_buffer.extend(data.reshape(-1))  # type: ignore[arg-type]
            collected_frames += data.numel()

            # Training updates
            if collected_frames >= init_random_frames:
                for _ in range(num_updates):
                    # Sample from policy replay buffer
                    sampled_td = self.data_buffer.sample().to(self.device)
                    
                    # Perform GAIL update (discriminator + policy)
                    loss_td = self.update(sampled_td).clone()
                    
                    # Log metrics - convert TensorDict to dict
                    metrics_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                                   for k, v in loss_td.items()}
                    if self.episode_rewards:
                        metrics_dict["episode/reward"] = float(np.mean(list(self.episode_rewards)[-10:]))
                    self.log_metrics(metrics_dict, step=collected_frames)

            # Check if done
            if collected_frames >= total_frames:
                break

        self.log.info("Training complete")
        self.collector.shutdown()

    def save(self, path: str | Path) -> None:
        """Save GAIL model."""
        path = Path(path)
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "config": self.config,
            },
            path,
        )
        self.log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load GAIL model."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.log.info(f"Model loaded from {path}")
