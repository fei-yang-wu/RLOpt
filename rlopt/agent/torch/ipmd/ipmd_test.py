from typing import Any, Dict, List, Optional, Tuple, Type, Union
import copy

import torch as th
from torchrl.envs import GymLikeEnv
from torchrl.modules import MLP, Actor, ContinuousCritic
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import SACLoss
from torchrl.data import ReplayBuffer, ListStorage, TensorDictReplayBuffer, TensorDict
from torchrl.objectives.utils import HardUpdate, SoftUpdate
from torchrl.trainers import Trainer

from gymnasium import make as gym_make
from torchrl.envs import EnvBase, GymWrapper, TransformedEnv

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic




class IPMD:
    
    def make_gym_env(env_id: str):
        env = gym_make(env_id)
        return GymWrapper(env)

    def __init__(
        self,
        env: Union[GymLikeEnv, str],
        policy: Type[MLP],
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        learning_rate: float = 3e-4,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        device: str = "cpu",
        expert_replay_buffer_loc: str = "",
        expert_traj_size: int = 10000,
        student_irl_begin_timesteps: int = int(2e6),
    ):
        # Create the environment
        self.device = device
        #
        self.env = make_gym_env(env) if isinstance(env, str) else env

        # Define actor and critic using MLP architectures
        self.actor = Actor(MLP(self.env.observation_spec, self.env.action_spec)).to(self.device)
        self.critic = ContinuousCritic(MLP(self.env.observation_spec, self.env.action_spec)).to(self.device)
        self.critic_target = ContinuousCritic(MLP(self.env.observation_spec, self.env.action_spec)).to(self.device)
        
        # Sync the critic_target parameters with the critic
        SoftUpdate(self.critic_target, self.critic, tau=1.0)()

        # Define ReplayBuffer
        self.replay_buffer = TensorDictReplayBuffer(
            storage=ListStorage(buffer_size),
            batch_size=batch_size,
        )

        # Load expert data if available
        self.expert_replay_buffer = self._load_expert_buffer(expert_replay_buffer_loc, expert_traj_size)

        # Define SAC Loss for optimization
        self.loss_module = SACLoss(self.actor, self.critic, self.critic_target, entropy_coef=ent_coef).to(self.device)

        # Define optimizers
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.target_update_interval = target_update_interval
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.tau = tau
        self.gamma = gamma
        self.student_irl_begin_timesteps = student_irl_begin_timesteps

    def _load_expert_buffer(self, path: str, traj_size: int):
        if path:
            # Load expert data if the path is provided (dummy implementation here)
            expert_data = th.load(path)
            expert_buffer = TensorDictReplayBuffer(
                storage=ListStorage(traj_size),
                batch_size=self.replay_buffer.batch_size,
            )
            # Populate buffer with expert data
            expert_buffer.extend(expert_data)
            return expert_buffer
        return None

    def train(self, steps: int):
        # Train actor-critic models using replay buffer samples
        for step in range(steps):
            # Sample data from replay buffer
            replay_data = self.replay_buffer.sample()

            # Calculate losses
            loss_actor, loss_critic = self.loss_module(replay_data)

            # Optimize actor
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            # Optimize critic
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            self.critic_optimizer.step()

            # Update target network
            if step % self.target_update_interval == 0:
                SoftUpdate(self.critic_target, self.critic, tau=self.tau)()

    def collect_data(self):
        # Use SyncDataCollector to collect data from the environment
        data_collector = SyncDataCollector(self.env, self.actor, frames_per_batch=self.train_freq, total_frames=10000)
        for batch in data_collector:
            self.replay_buffer.extend(batch)

    def _save(self, path: str):
        # Save model parameters
        th.save(self.actor.state_dict(), f"{path}_actor.pth")
        th.save(self.critic.state_dict(), f"{path}_critic.pth")
