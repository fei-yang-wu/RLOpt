from typing import Any, Dict, Optional, Type, Union, List
from abc import ABC, abstractmethod
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchrl
from torchrl.data import ReplayBuffer
from torchrl.envs import EnvBase

from tensordict import TensorDict
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore


# ----------------------------------------------------
# Logger base class
# ----------------------------------------------------
class Logger:
    def __init__(self, config: DictConfig):
        self.loggers = []
        self.config = config

        if config.logging.use_tb:
            from torch.utils.tensorboard import SummaryWriter

            self.loggers.append(SummaryWriter(log_dir=config.logging.log_dir))

        if config.logging.use_wandb:
            import wandb

            wandb.init(
                project=config.logging.wandb_project,
                config=OmegaConf.to_container(config),
            )

    def log(self, metrics: Dict[str, float], step: int):
        for logger in self.loggers:
            # TensorBoard
            if logger.__class__.__name__ == "SummaryWriter":
                for k, v in metrics.items():
                    logger.add_scalar(k, v, step)
            # WandB
            elif hasattr(logger, "log"):
                logger.log(metrics, step=step)

        if self.config.logging.use_stdout:
            print(f"Step {step}: {metrics}")


# ----------------------------------------------------
# Example Hydra config schema
# ----------------------------------------------------
@dataclass
class AlgorithmConfig:
    # Base configuration
    device: str = "auto"
    seed: int = 42
    total_steps: int = 1_000_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    grad_clip: float = 0.5

    # Whether to compile with torch.compile
    compile: bool = False

    # Whether we are offline (load transitions from buffer/disk)
    offline: bool = False
    num_steps: int = 128  # On-policy rollout length

    # Logging
    logging: DictConfig = DictConfig(
        {
            "use_tb": True,
            "use_wandb": False,
            "wandb_project": "rl-library",
            "log_dir": "./logs",
            "use_stdout": True,
            "log_interval": 1000,
        }
    )

    # Optional reward estimation config for inverse RL
    reward_estimation: Optional[DictConfig] = None

    # Env config placeholder (Hydra overrides will fill these)
    env: DictConfig = DictConfig({})

    # Policy config placeholder
    policy: DictConfig = DictConfig({})

    # (Optional) value networks config placeholders
    # e.g. a config dict for V-network
    value: Optional[DictConfig] = None

    # e.g. a config dict for Q-network
    q: Optional[DictConfig] = None

    # If you want a separate target net update schedule
    target_update_interval: int = 1000
    tau: float = 0.005  # Polyak update coefficient


# Register Hydra config
cs = ConfigStore.instance()
cs.store(name="base_config", node=AlgorithmConfig)


# A small helper to load models from state_dict
def load_model(state_dict: Dict[str, Any]) -> nn.Module:
    # Here you would create the model class and load the state_dict
    # For demonstration, we create a dummy MLP with matching keys
    model = nn.Sequential(nn.Linear(10, 10))  # Simplify to match your architecture
    model.load_state_dict(state_dict)
    return model


# ----------------------------------------------------
# Abstract BaseAlgorithm
# ----------------------------------------------------
class BaseAlgorithm(ABC):
    def __init__(
        self,
        env: EnvBase,
        config: AlgorithmConfig,
        policy: Optional[nn.Module] = None,
        value_net: Optional[nn.Module] = None,
        q_net: Optional[nn.Module] = None,
        reward_estimator: Optional[nn.Module] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        logger: Optional[Logger] = None,
    ):
        super().__init__()
        self.env = env
        self.config = config
        self.logger = logger or Logger(config)
        self.device = self._get_device(config.device)

        # Seed for reproducibility
        self._set_seed(config.seed)

        # (1) Construct or attach networks
        self.policy = policy or self._construct_policy(config.policy)
        self.value_net = value_net or self._construct_value_function(config.value)
        self.q_net = q_net or self._construct_q_function(config.q)
        self.reward_estimator = reward_estimator

        # Move them to device
        if self.policy:
            self.policy.to(self.device)
        if self.value_net:
            self.value_net.to(self.device)
        if self.q_net:
            self.q_net.to(self.device)
        if self.reward_estimator:
            self.reward_estimator.to(self.device)

        # (2) Construct (optional) target networks
        self.target_value_net = None
        self.target_q_net = None
        # If you want a separate target for the value function:
        if self.value_net is not None:
            self.target_value_net = self._construct_target_network(self.value_net)
        # If you want a separate target for the Q function:
        if self.q_net is not None:
            self.target_q_net = self._construct_target_network(self.q_net)

        # (3) Compile if requested
        self._compile_components()

        # (4) Create optimizers
        self.optimizers = self._configure_optimizers()

        # Replay buffer / experience
        self.replay_buffer = replay_buffer
        self.step_count = 0
        self.start_time = time.time()

    # -----------------------------
    # Construction / setup methods
    # -----------------------------
    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _get_device(self, device_str: str) -> torch.device:
        """Decide on CPU or GPU device."""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _construct_policy(self, policy_config: DictConfig) -> nn.Module:
        """Override to build your policy network from config."""
        # Example: simple MLP
        in_features = policy_config.get("in_features", 4)
        out_features = policy_config.get("out_features", 2)
        hidden_size = policy_config.get("hidden_size", 64)
        model = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
        )

        # use torchrl modules for common policy components
        # e.g. CategoricalPolicy, GaussianPolicy, etc.
        

        return model

    def _construct_value_function(
        self, value_config: Optional[DictConfig]
    ) -> Optional[nn.Module]:
        """Override to build your V-network from config."""
        if value_config is None:
            return None
        # Example: simple MLP for state-value
        in_features = value_config.get("in_features", 4)
        hidden_size = value_config.get("hidden_size", 64)
        model = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # V(s)
        )
        return model

    def _construct_q_function(
        self, q_config: Optional[DictConfig]
    ) -> Optional[nn.Module]:
        """Override to build your Q-network from config."""
        if q_config is None:
            return None
        # Example: simple MLP for state-action value
        obs_dim = q_config.get("obs_dim", 4)
        act_dim = q_config.get("act_dim", 2)
        hidden_size = q_config.get("hidden_size", 64)
        model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # Q(s,a)
        )
        return model

    def _construct_target_network(self, net: nn.Module) -> nn.Module:
        """Create a target network as a copy of the provided net."""
        target_net = type(net)()  # same class
        # This assumes your net is a nn.Sequential or a custom class:
        # We copy the state dict so that it starts off identical.
        target_net.load_state_dict(net.state_dict())

        # Freeze parameters if you want them non-trainable:
        for param in target_net.parameters():
            param.requires_grad = False

        return target_net.to(self.device)

    def _compile_components(self):
        """Example: compile performance-critical methods."""
        if self.config.compile:
            self.compute_action = torch.compile(self._compute_action, dynamic=True)
            self.compute_returns = torch.compile(self._compute_returns, dynamic=True)
        else:
            self.compute_action = self._compute_action
            self.compute_returns = self._compute_returns

    def _configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Configure optimizers for different components."""
        # Basic example: one optimizer for each component
        optimizers = {}
        if self.policy:
            optimizers["policy"] = torch.optim.Adam(
                self.policy.parameters(), lr=self.config.learning_rate
            )
        if self.value_net:
            optimizers["value"] = torch.optim.Adam(
                self.value_net.parameters(), lr=self.config.learning_rate
            )
        if self.q_net:
            optimizers["q"] = torch.optim.Adam(
                self.q_net.parameters(), lr=self.config.learning_rate
            )
        if self.reward_estimator and self.config.reward_estimation:
            lr = self.config.reward_estimation.get("learning_rate", 3e-4)
            optimizers["reward"] = torch.optim.Adam(
                self.reward_estimator.parameters(),
                lr=lr,
            )
        return optimizers

    # ------------------------------------
    # Target network updates (optional)
    # ------------------------------------
    def soft_update(
        self, source_net: nn.Module, target_net: nn.Module, tau: float = 0.005
    ):
        """Polyak/soft-update target network parameters."""
        with torch.no_grad():
            for param, target_param in zip(
                source_net.parameters(), target_net.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

    def hard_update(self, source_net: nn.Module, target_net: nn.Module):
        """Hard update for target network (full copy)."""
        target_net.load_state_dict(source_net.state_dict())

    # ------------------------------------
    # Abstract / to-be-implemented methods
    # ------------------------------------
    @abstractmethod
    def _compute_action(self, tensordict: TensorDict) -> TensorDict:
        """Compute the next action given current policy (abstract)."""
        pass

    @abstractmethod
    def _compute_returns(self, rollout: Tensor) -> Tensor:
        """Compute returns and possibly advantages from a rollout."""
        pass

    @abstractmethod
    def _update_policy(self, batch: TensorDict) -> Dict[str, float]:
        """Algorithm-specific policy (and/or value) update logic."""
        pass

    # ------------------------------------
    # RL Loop
    # ------------------------------------
    def collect_experience(self) -> TensorDict:
        """Generic experience collection (for on-policy).
        Override if you do something different (like off-policy sampling).
        """
        if self.config.offline:
            return self._load_offline_data()

        tensordict = self.env.reset()  # Returns a TensorDict
        rollout = []

        for _ in range(self.config.num_steps):
            tensordict = tensordict.to(self.device)
            tensordict = self.compute_action(tensordict)
            tensordict = self.env.step(tensordict)
            rollout.append(tensordict.clone())

            # Prepare next step
            tensordict = tensordict["next"]

        rollout_td = torch.stack(rollout, dim=0)
        returns_td = self.compute_returns(rollout_td)
        return returns_td

    def _load_offline_data(self) -> TensorDict:
        """Load from replay buffer or offline dataset."""
        assert (
            self.replay_buffer is not None
        ), "ReplayBuffer must be provided for offline."
        batch = self.replay_buffer.sample(self.config.batch_size)
        return batch

    def update_reward_estimator(self, batch: TensorDict) -> Dict[str, float]:
        """Update reward function for Inverse RL or learned reward shaping."""
        if not self.reward_estimator:
            return {}

        obs = batch["observation"]
        pred_rewards = self.reward_estimator(obs)
        if "returns" in batch.keys():
            loss = F.mse_loss(pred_rewards, batch["returns"])
        else:
            # Example placeholder
            loss = pred_rewards.mean()

        self.optimizers["reward"].zero_grad()
        loss.backward()
        self.optimizers["reward"].step()
        return {"reward_loss": loss.item()}

    def update_parameters(self, batch: TensorDict) -> Dict[str, float]:
        """Generic parameter update (handles policy, value, reward, etc.)."""
        metrics = {}
        if self.reward_estimator:
            metrics.update(self.update_reward_estimator(batch))

        policy_metrics = self._update_policy(batch)
        metrics.update(policy_metrics)

        return metrics

    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        history = defaultdict(list)

        if not self.config.offline and self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer(storage=[])

        while self.step_count < self.config.total_steps:
            # 1. Collect experience if on-policy
            if not self.config.offline:
                experience_batch = self.collect_experience()
                self.replay_buffer.extend(experience_batch)

            # 2. Sample a batch
            batch = self.replay_buffer.sample(self.config.batch_size)
            batch = batch.to(self.device)

            # 3. Update parameters
            metrics = self.update_parameters(batch)

            # 4. Optional target network updates
            #    e.g. if step_count is multiple of target_update_interval
            if self.step_count % self.config.target_update_interval == 0:
                if self.target_value_net and self.value_net:
                    self.soft_update(
                        self.value_net, self.target_value_net, self.config.tau
                    )
                if self.target_q_net and self.q_net:
                    self.soft_update(self.q_net, self.target_q_net, self.config.tau)

            # 5. Logging
            if self.step_count % self.config.logging.log_interval == 0:
                self.logger.log(metrics, self.step_count)
                for k, v in metrics.items():
                    history[k].append(v)

            self.step_count += self.config.batch_size

        return history

    # ------------------------------------
    # Checkpointing
    # ------------------------------------
    def save_checkpoint(self, path: str):
        """Save complete training state."""
        checkpoint = {
            "policy": self.policy.state_dict() if self.policy else None,
            "value_net": self.value_net.state_dict() if self.value_net else None,
            "q_net": self.q_net.state_dict() if self.q_net else None,
            "target_value_net": (
                self.target_value_net.state_dict() if self.target_value_net else None
            ),
            "target_q_net": (
                self.target_q_net.state_dict() if self.target_q_net else None
            ),
            "reward_estimator": (
                self.reward_estimator.state_dict() if self.reward_estimator else None
            ),
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "step_count": self.step_count,
            "config": OmegaConf.to_container(self.config),
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(cls, path: str, env: EnvBase) -> "BaseAlgorithm":
        """Load complete training state."""
        checkpoint = torch.load(path, map_location="cpu")
        config = OmegaConf.create(checkpoint["config"])

        # Reconstruct networks
        policy_sd = checkpoint["policy"]
        value_sd = checkpoint["value_net"]
        q_sd = checkpoint["q_net"]
        reward_sd = checkpoint["reward_estimator"]

        # You could create them by the same construction logic:
        instance = cls(env=env, config=config)

        if policy_sd is not None:
            instance.policy.load_state_dict(policy_sd)
        if value_sd is not None and instance.value_net is not None:
            instance.value_net.load_state_dict(value_sd)
        if q_sd is not None and instance.q_net is not None:
            instance.q_net.load_state_dict(q_sd)
        if reward_sd is not None and instance.reward_estimator is not None:
            instance.reward_estimator.load_state_dict(reward_sd)

        # If target nets exist
        target_value_sd = checkpoint["target_value_net"]
        if target_value_sd is not None and instance.target_value_net is not None:
            instance.target_value_net.load_state_dict(target_value_sd)
        target_q_sd = checkpoint["target_q_net"]
        if target_q_sd is not None and instance.target_q_net is not None:
            instance.target_q_net.load_state_dict(target_q_sd)

        # Optimizers
        for name, state in checkpoint["optimizers"].items():
            if name in instance.optimizers:
                instance.optimizers[name].load_state_dict(state)

        instance.step_count = checkpoint["step_count"]
        return instance


# ----------------------------------------------------
# Example Implementation or Hydra entry point
# ----------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: AlgorithmConfig) -> None:
    """Simple Hydra entry point."""
    # (1) Create environment
    env = make_env(cfg.env)

    # (2) Instantiate your custom RL algorithm
    #     In practice you’d subclass BaseAlgorithm and implement
    #     _compute_action, _compute_returns, and _update_policy.
    class MyAlgorithm(BaseAlgorithm):
        def _compute_action(self, tensordict: TensorDict) -> TensorDict:
            # E.g., forward pass through policy
            obs = tensordict["observation"]
            with torch.no_grad():
                action = self.policy(obs)
            tensordict.set("action", action)
            return tensordict

        def _compute_returns(self, rollout: TensorDict) -> TensorDict:
            # Simple placeholder
            returns = torch.zeros_like(rollout.get("reward"))
            rollout.set_("returns", returns)
            return rollout

        def _update_policy(self, batch: TensorDict) -> Dict[str, float]:
            # Simple placeholder update
            obs = batch["observation"]
            act = self.policy(obs)
            loss = F.mse_loss(act, torch.zeros_like(act))

            self.optimizers["policy"].zero_grad()
            loss.backward()
            self.optimizers["policy"].step()

            return {"policy_loss": loss.item()}

    algorithm = MyAlgorithm(env=env, config=cfg)
    history = algorithm.train()
    algorithm.save_checkpoint("final_checkpoint.pt")


# ----------------------------------------------------
# Dummy placeholders for demonstration
# ----------------------------------------------------
def make_env(env_config: DictConfig) -> EnvBase:
    """Instantiate or wrap a TorchRL environment from env_config."""
    from torchrl.envs import Compose, TransformedEnv
    from torchrl.envs.libs.gym import GymEnv

    base_env = GymEnv(env_config.get("name", "CartPole-v1"))
    env = TransformedEnv(base_env, Compose())
    return env
