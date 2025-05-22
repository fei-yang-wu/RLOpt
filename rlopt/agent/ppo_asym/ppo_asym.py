import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, Dict, Optional, Type, Union, List, Tuple
import time
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
from rlopt.common.utils import obs_as_tensor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from rlopt.common.buffer import DictRolloutBuffer


class AsymActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms with asymmetric observations.
    The actor uses obs["student"] and the critic uses obs["teacher"].
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        
        # Override the features extractor to handle dictionary observations
        self.features_extractor = None  # We'll handle feature extraction in forward()
        
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in both the actor and the critic

        :param obs: Dictionary containing "student" and "teacher" observations
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Extract features for actor (student) and critic (teacher)
        latent_pi = self.mlp_extractor.forward_actor(obs["student"])
        latent_vf = self.mlp_extractor.forward_critic(obs["teacher"])
        
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        
        # Get action distribution
        actions = self.action_net(latent_pi)
        
        if deterministic:
            actions = self.action_net.get_deterministic_action(latent_pi)
        
        # Get log probs
        log_probs = self.action_net.log_prob(actions)
        
        return actions, values, log_probs
    
    def predict_values(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Dictionary containing "student" and "teacher" observations
        :return: the estimated values.
        """
        # Extract features for critic (teacher)
        latent_vf = self.mlp_extractor.forward_critic(obs["teacher"])
        return self.value_net(latent_vf)


class PPOAsym(PPO):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]] = AsymActorCriticPolicy,
        env: Union[GymEnv, str] = None,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if policy_kwargs is None:
            policy_kwargs = {}
            
        # Ensure the policy is AsymActorCriticPolicy
        if isinstance(policy, str):
            policy = AsymActorCriticPolicy
            
        # Set rollout buffer class to DictRolloutBuffer
        policy_kwargs["rollout_buffer_class"] = DictRolloutBuffer
            
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _get_actor_input(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["student"]

    def _get_critic_input(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["teacher"]

    # We need to override collect_rollouts to handle the dictionary observation space
    # and ensure that the correct parts of the observation are passed to the actor and critic.
    # The train method might also need adjustments depending on how SB3's PPO uses observations.
    # For now, let's assume the default train method will work if collect_rollouts is correctly implemented.

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: DictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``DictRolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.inference_mode():
                # Convert to pytorch tensor if needed
                actor_obs = self._get_actor_input(self._last_obs)
                # Check if actor_obs is a numpy array and convert to tensor
                if isinstance(actor_obs, np.ndarray):
                    actor_obs = obs_as_tensor(actor_obs, self.device)
                
                actions, values, log_probs = self.policy(self._last_obs)

            clipped_actions = actions
            # Rescale and perform action
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = th.clamp(
                        actions,
                        th.as_tensor(self.action_space.low, device=self.device),
                        th.as_tensor(self.action_space.high, device=self.device),
                    )
            
            # Gymnasium VecEnv step return is a tuple ((obs, info), rewards, dones, truncateds)
            # We need to handle the case where obs is a dict
            # Convert actions to numpy for the environment step, if they are tensors
            env_actions = clipped_actions
            if isinstance(env_actions, th.Tensor):
                env_actions = env_actions.cpu().numpy()

            new_obs_dict, rewards, dones, infos = env.step(env_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            
            # Handle timeout mask properly if it exists in infos
            # Bootstrapping on time outs
            # Ensure values is a tensor for broadcasting, and detached
            values_tensor = values.detach()
            if "TimeLimit.truncated" in infos: # Standard gymnasium timeout key
                for i, done in enumerate(dones):
                    if done and infos[i].get("TimeLimit.truncated", False):
                        terminal_obs_critic = self._get_critic_input(infos[i]["terminal_observation"])
                        if isinstance(terminal_obs_critic, np.ndarray):
                             terminal_obs_critic = obs_as_tensor(terminal_obs_critic, self.device)
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs_critic)
                        rewards[i] += self.gamma * terminal_value[0] # Ensure correct indexing for single env value
            elif "time_outs" in infos: # Some envs might use this
                 rewards += self.gamma * values_tensor * infos["time_outs"].unsqueeze(1).to(self.device)

            rollout_buffer.add(
                self._last_obs,  # Store the full dictionary observation
                actions.detach().cpu(), # Actions are derived from student_obs
                rewards,
                self._last_episode_starts,
                values_tensor, # Values are derived from student_obs (actor's view)
                log_probs.detach().cpu(),
            )
            self._last_obs = new_obs_dict
            self._last_episode_starts = dones

        with th.inference_mode():
            # Compute value for the last timestep using critic_input
            critic_obs_final = self._get_critic_input(new_obs_dict)
            if isinstance(critic_obs_final, np.ndarray):
                critic_obs_final = obs_as_tensor(critic_obs_final, self.device)
            values = self.policy.predict_values(critic_obs_final)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True
