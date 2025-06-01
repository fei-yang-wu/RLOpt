from __future__ import annotations

import gc
import time
from collections import deque
from typing import Optional, Union, Generator, NamedTuple, Dict

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common import preprocessing, utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import MaybeCallback, PyTorchObs
from stable_baselines3.common.utils import get_device, get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from rlopt.common import DictRolloutBuffer


class DictRolloutBufferSamplesWithHiddenState(NamedTuple):
    observations: dict[str, th.Tensor]
    actions: th.Tensor
    rewards: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    episode_starts: th.Tensor
    hidden_state: th.Tensor


class DictRolloutBufferWithHiddenState(DictRolloutBuffer):
    """
    Dict Rollout buffer that also stores hidden states for recurrent policies.
    Extends DictRolloutBuffer to add hidden state storage and retrieval.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        hidden_state_shape: tuple[int, ...],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        super().__init__(
            buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs
        )
        self.hidden_states: th.Tensor = th.zeros(
            (self.buffer_size, self.n_envs, *hidden_state_shape),
            dtype=th.float32,
            device=self.device,
        )

    def reset(self) -> None:
        super().reset()
        self.hidden_states = th.zeros(
            (self.buffer_size, self.n_envs, *self.hidden_state_shape),
            dtype=th.float32,
            device=self.device,
        )

    def add(  # type: ignore[override]
        self,
        obs: dict[str, th.Tensor],
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor,
        hidden_state: th.Tensor,
    ) -> None:
        """
        Add data to the buffer including hidden state.
        """
        # Store hidden state
        if hidden_state is not None:
            # hidden_state shape: (1, n_envs, hidden_size)
            # Store as (n_envs, 1, hidden_size) then squeeze to (n_envs, hidden_size)
            self.hidden_states[self.pos] = hidden_state.transpose(0, 1).squeeze(1).detach()
        
        # Call parent add method
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamplesWithHiddenState, None, None]:
        assert self.full, ""
        indices = th.randperm(self.buffer_size * self.n_envs, device=self.device)
        
        # Prepare the data
        if not self.generator_ready:
            # tensor dict handles transpose and reshape
            self.observations = self.swap_and_flatten(
                self.observations
            )  # type : ignore

            _tensor_names = [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "rewards",
                "episode_starts",
                "hidden_states",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: th.Tensor,
        env: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamplesWithHiddenState:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.rewards[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.episode_starts[batch_inds].flatten(),
            self.hidden_states[batch_inds],
        )
        return DictRolloutBufferSamplesWithHiddenState(*data)


class DWLPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            self.share_features_extractor is False
        ), "share_features_extractor must be False"
        print(self.mlp_extractor)
        # Track hidden state for recurrent extractor
        self._hidden_state = None

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Create a custom features extractor for the DWL policy."""
        return DWLExtractor(
            self.observation_space,  # type: ignore[arg-type]
            features_dim=128,  # Based on architecture table
            activation_fn=nn.ELU,
        )

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        episode_starts: th.Tensor | None = None,
        hidden_state: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        preprocessed_obs = preprocessing.preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        # Ensure episode_starts is on the correct device
        if episode_starts is not None:
            episode_starts = episode_starts.to(self.device)
        return self.features_extractor(preprocessed_obs, episode_starts, hidden_state)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).

        # Get privileged state dimension directly from observation space
        privileged_dim = self.observation_space["teacher"].shape[0]  # type: ignore[index]

        self.mlp_extractor = DWLMlpExtractor(  # type: ignore[assignment]
            (
                128,  # latent features from encoder for actor
                privileged_dim,  # privileged state dimension for critic
            ),
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(
        self, obs: th.Tensor | dict, episode_starts: th.Tensor | None = None, deterministic: bool = False, hidden_state: th.Tensor | None = None
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, log probability of the action, new hidden state
        """
        assert isinstance(obs, dict), "obs must be a dictionary"
        # Preprocess the observation if needed
        latent_features, decoded_features, new_hidden = self.extract_features(obs, episode_starts, hidden_state)

        # Actor uses latent features directly
        latent_pi = self.mlp_extractor.forward_actor(latent_features)
        # Critic uses privileged observations directly
        latent_vf = self.mlp_extractor.forward_critic(obs["teacher"])

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob, new_hidden

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        episode_starts: th.Tensor | None = None,
        hidden_state: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None, th.Tensor]:
        assert isinstance(obs, dict), "obs must be a dictionary"
        latent_features, decoded_features, new_hidden = self.extract_features(obs, episode_starts, hidden_state)

        latent_pi = self.mlp_extractor.forward_actor(latent_features)
        # Critic uses privileged observations directly
        latent_vf = self.mlp_extractor.forward_critic(obs["teacher"])

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy, new_hidden

    def predict_values(self, obs: PyTorchObs, episode_starts: th.Tensor | None = None, hidden_state: th.Tensor | None = None) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        assert isinstance(obs, dict), "obs must be a dictionary"
        latent_features, decoded_features, _ = self.extract_features(obs, episode_starts, hidden_state)
        # Critic uses privileged observations directly
        latent_vf = self.mlp_extractor.forward_critic(obs["teacher"])
        return self.value_net(latent_vf)


class DWLPPO(PPO):
    """
    PPO with Denoising World model Learning, using GRU encoder with running memory for observation history.
    """

    def __init__(self, policy, env, **ppo_kwargs):
        ppo_kwargs["rollout_buffer_class"] = DictRolloutBufferWithHiddenState
        super().__init__(policy, env, **ppo_kwargs)
        print(self.policy)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.rollout_buffer_kwargs["device"] = "cuda:0"

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBufferWithHiddenState  # type: ignore[assignment]
            else:
                self.rollout_buffer_class = RolloutBuffer  # type: ignore[assignment]

        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            hidden_state_shape=(self.policy.features_extractor.gru_hidden_size,),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy.to(self.device)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)  # type: ignore[assignment]
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)  # type: ignore[assignment]

    def train(self) -> None:
        self.policy.set_training_mode(True)

        self._update_learning_rate([self.policy.optimizer])

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        clip_range_vf = None  # Initialize to None by default
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        denoise_losses = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.to(self.device)
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten().to(self.device)

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, new_hidden = self.policy.evaluate_actions(
                    {
                        k: v.to(self.device)
                        for k, v in rollout_data.observations.items()  # type: ignore[attr-defined]
                    },
                    actions,  # type: ignore,
                    episode_starts=rollout_data.episode_starts.to(self.device),
                    hidden_state=rollout_data.hidden_state.to(self.device) if rollout_data.hidden_state is not None else None
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages.to(self.device)
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob.to(self.device))

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values.to(self.device) + th.clamp(
                        values - rollout_data.old_values.to(self.device),
                        -clip_range_vf,
                        clip_range_vf,  # type: ignore
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(
                    rollout_data.returns.to(self.device), values_pred
                )
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # Denoising loss - decode from latent to reconstruct privileged observations
                assert isinstance(self.policy, DWLPolicy)

                # Extract observations from rollout data
                obs_dict = {}
                if hasattr(rollout_data.observations, "items"):
                    # If observations is already a dict-like structure
                    obs_dict = {k: v for k, v in rollout_data.observations.items()}  # type: ignore[attr-defined]
                else:
                    # If observations is a tensor, we need to handle it differently
                    # This assumes the rollout buffer stores observations properly
                    obs_dict = rollout_data.observations

                # Get episode_starts from rollout data if available
                episode_starts = rollout_data.episode_starts.to(self.device)

                # Extract features with episode_starts
                latent_features, decoded_features, _ = self.policy.extract_features(
                    obs_dict, episode_starts=episode_starts, hidden_state=rollout_data.hidden_state.to(self.device) if rollout_data.hidden_state is not None else None
                )

                # Get teacher observations for denoising loss
                teacher_obs = (
                    obs_dict["teacher"] if isinstance(obs_dict, dict) else obs_dict
                )

                # Denoising loss: ||s_tilde - s_t||_2 + lambda_r * ||z_t||_1
                denoise_loss = (
                    F.mse_loss(decoded_features, teacher_obs.detach())
                    + 0.01 * th.norm(latent_features, p=1, dim=-1).mean()
                )

                denoise_losses.append(denoise_loss.item())

                # total loss
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + 1.0 * denoise_loss  # Denoising loss coefficient
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob.to(self.device)
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1  # type: ignore[attr-defined]

            if not continue_training:
                break

        # Log statistics
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))  # type: ignore[attr-defined]
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))  # type: ignore[attr-defined]
        self.logger.record("train/clip_range", clip_range)
        self.logger.record(
            "train/learning_rate", self.policy.optimizer.param_groups[0]["lr"]
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/denoise_loss", np.mean(denoise_losses))

    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: DictRolloutBufferWithHiddenState,
        n_rollout_steps: int,
    ) -> bool:
        """
        Special care for policy predict because we only want to take state as input.
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
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

        self._last_obs: Dict[str, th.Tensor]  # type: ignore[assignment]
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        self.ep_infos: list[dict[str, float]] = []  # type: ignore[assignment]

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.inference_mode():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = {k: v.to(self.device) for k, v in self._last_obs.items()}
                actions, values, log_probs, new_hidden = self.policy(obs_tensor, episode_starts=self._last_episode_starts, hidden_state=self._last_hidden_state)

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, gym.spaces.Box):
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

            new_obs, rewards, dones, infos = env.step(
                clipped_actions.to(env.sim_device)  # type: ignore[attr-defined]
            )

            # Ensure tensors are on correct device and avoid redundant transfers
            new_obs = {k: v.to(self.device) for k, v in new_obs.items()}  # type: ignore[attr-defined]
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)  # type: ignore[attr-defined]

            self.num_timesteps += env.num_envs

            infos: dict
            # Record infos
            if "episode" in infos:  # type: ignore[attr-defined]
                self.ep_infos.append(infos["episode"])
            elif "log" in infos:  # type: ignore[attr-defined]
                self.ep_infos.append(infos["log"])

            self.cur_reward_sum += rewards.clone().detach()  # type: ignore[attr-defined]
            self.cur_episode_length += 1.0  # type: ignore[attr-defined]
            new_ids = (dones > 0).nonzero(as_tuple=False)  # type: ignore[attr-defined]

            # record reward and episode length - use detach() to avoid gradient tracking
            if len(new_ids) > 0:
                self.rewbuffer.extend(
                    self.cur_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist()
                )
                self.lenbuffer.extend(
                    self.cur_episode_length[new_ids][:, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                self.cur_reward_sum[new_ids] = 0
                self.cur_episode_length[new_ids] = 0

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Bootstrapping on time outs
            if "time_outs" in infos:  # type: ignore[attr-defined]
                rewards += self.gamma * th.squeeze(
                    values * infos["time_outs"].unsqueeze(1).to(self.device),  # type: ignore[attr-defined]
                    1,
                )

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,  # Already on device
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                new_hidden,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            self._last_hidden_state = new_hidden

        gc.collect()

        with th.inference_mode():
            # Compute value for the last timestep
            values = self.policy.predict_values(new_obs, episode_starts=self._last_episode_starts, hidden_state=self._last_hidden_state)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values.detach().to(rollout_buffer.device),
            dones=dones.detach().to(rollout_buffer.device),
        )

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:  # type: ignore[override]
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()  # type: ignore[attr-defined]

        # store the current number of timesteps
        self.rewbuffer = deque(maxlen=100)  # type: ignore[attr-defined]
        self.lenbuffer = deque(maxlen=100)  # type: ignore[attr-defined]
        self.cur_reward_sum = th.zeros(
            self.env.num_envs, dtype=th.float, device=self.device  # type: ignore[attr-defined]
        )
        self.cur_episode_length = th.zeros(
            self.env.num_envs, dtype=th.float, device=self.device  # type: ignore[attr-defined]
        )

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)  # type: ignore[attr-defined]
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)  # type: ignore[attr-defined]

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = th.ones((self.env.num_envs,), dtype=th.bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()  # type: ignore[attr-defined]
            # Initialize hidden state for recurrent policy
            hidden_size = self.policy.features_extractor.gru_hidden_size
            num_envs = self.env.num_envs
            device = self.device
            self._last_hidden_state = th.zeros(1, num_envs, hidden_size, device=device)

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
            )

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def dump_logs(self, iteration: int) -> None:  # type: ignore[override]
        """
        Write log.

        :param iteration: Current logging iteration
        """

        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = th.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], th.Tensor):
                        ep_info[key] = th.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:  # type: ignore[attr-defined]
                        ep_info[key] = ep_info[key].unsqueeze(0)  # type: ignore[attr-defined]
                    infotensor = th.cat((infotensor, ep_info[key].to(self.device)))  # type: ignore[attr-defined]
                value = th.mean(infotensor).item()
                # log to logger and terminal
                if "/" in key:
                    self.logger.record(key, value)
                else:
                    self.logger.record("Episode/" + key, value)

        if len(self.rewbuffer) > 0:
            self.logger.record(
                "Episode/average_episodic_reward", np.mean(self.rewbuffer)
            )
        if len(self.lenbuffer) > 0:
            self.logger.record(
                "Episode/average_episodic_length", np.mean(self.lenbuffer)
            )

        # Use detach() to avoid memory leaks when converting to scalar
        if len(self.cur_reward_sum) > 0:
            self.logger.record(
                "Episode/episodic_reward", th.max(self.cur_reward_sum).detach().item()
            )
        if len(self.cur_episode_length) > 0:
            self.logger.record(
                "Episode/episodic_length",
                th.max(self.cur_episode_length).detach().item(),
            )
        self.logger.dump(step=self.num_timesteps)


class DWLExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for Denoising World model Learning.
    Uses GRU encoder with running memory for observation history and decoder for privileged state reconstruction.
    Architecture based on the provided table.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 24,
        activation_fn: type[nn.Module] = nn.ELU,
    ):
        if not isinstance(observation_space, spaces.Dict):
            value_error = "DWLExtractor requires a Dict observation space"
            raise ValueError(value_error)

        super().__init__(observation_space, features_dim)

        # Get dimensions from observation space - assuming current observation is 47-dim
        self.current_obs_dim = observation_space["student"].shape[
            0
        ]  # Based on table: GRU(47 → 256)
        self.privileged_dim = observation_space["teacher"].shape[
            0
        ]  # Based on table: decoder output
        self.latent_dim = features_dim  # Based on table: encoder output

        # GRU encoder with running memory - Architecture: GRU(47 → 256)
        self.gru_hidden_size = 512
        self.gru_encoder = nn.GRU(
            input_size=self.current_obs_dim,
            hidden_size=self.gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Encoder post-processing
        self.encoder_post = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 256),
            activation_fn(),
            nn.Linear(256, 256),
            activation_fn(),
            nn.Linear(256, features_dim),
        )

        # Decoder for privileged state reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            activation_fn(),
            nn.Linear(128, 256),
            activation_fn(),
            nn.Linear(256, 512),
            activation_fn(),
            nn.Linear(512, self.privileged_dim),
        )

        # Update features_dim to match the latent dimension
        self._features_dim = features_dim

    def forward(
        self,
        observations: dict[str, th.Tensor],
        episode_starts: th.Tensor | None = None,
        hidden_state: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Extract features using GRU encoder with running memory and decoder for reconstruction.

        :param observations: Dictionary containing current observations
        :param episode_starts: Boolean tensor indicating episode starts for memory reset
        :param hidden_state: Previous GRU hidden state (1, batch, hidden_size)
        :return: Tuple of (latent_features, decoded_privileged_features, new_hidden_state)
        """
        current_obs = observations["student"]  # (batch_size, obs_dim)
        batch_size = current_obs.shape[0]
        device = current_obs.device

        # Initialize hidden state if needed
        if hidden_state is None or hidden_state.shape[1] != batch_size:
            hidden_state = th.zeros(1, batch_size, self.gru_hidden_size, device=device)

        # Reset hidden state for new episodes
        if episode_starts is not None:
            episode_starts = episode_starts.to(device)
            if episode_starts.any():
                # Convert to boolean for indexing
                episode_starts_bool = episode_starts.bool()
                hidden_state[:, episode_starts_bool, :] = 0

        gru_input = current_obs.unsqueeze(1)  # (batch_size, 1, obs_dim)
        gru_output, new_hidden = self.gru_encoder(gru_input, hidden_state)
        gru_out = gru_output.squeeze(1)  # (batch_size, hidden_size)
        latent_features = self.encoder_post(gru_out)  # (batch_size, latent_dim)
        decoded_features = self.decoder(latent_features)  # (batch_size, privileged_dim)
        return latent_features, decoded_features, new_hidden


class DWLMlpExtractor(MlpExtractor):
    """
    MLP extractor for DWL with specific architecture for Actor and Critic networks.
    Actor: Linear(24 → 48), ELU, Linear(48 → 12)
    Critic: Uses privileged state dimension directly, following the table architecture
    """

    def __init__(
        self,
        feature_dims: tuple[int, int],
        net_arch: list[int] | dict[str, list[int]],
        activation_fn: type[nn.Module],
        device: th.device | str = "auto",
    ) -> None:
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        policy_net: list[nn.Module] = []
        value_net: list[nn.Module] = []
        # Use fixed architecture from table instead of net_arch
        last_layer_dim_pi = feature_dims[0]  # student observation dimension
        last_layer_dim_vf = feature_dims[1]  # teacher observation dimension
        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        vf_layers_dims = [512, 256, 128]

        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
