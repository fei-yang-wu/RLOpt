from __future__ import annotations

import gc
import time
from collections import deque
from typing import Optional, Union

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
from stable_baselines3.common.vec_env import VecEnv

from rlopt.common import DictRolloutBuffer


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

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Create a custom features extractor for the DWL policy."""
        return DWLExtractor(
            self.observation_space,  # type: ignore[arg-type]
            features_dim=24,  # Based on architecture table
            activation_fn=nn.ELU,
        )

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        episode_starts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        preprocessed_obs = preprocessing.preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        return self.features_extractor(preprocessed_obs, episode_starts)

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
                24,  # latent features from encoder for actor
                privileged_dim,  # privileged state dimension for critic
            ),
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(
        self, obs: th.Tensor | dict, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        assert isinstance(obs, dict), "obs must be a dictionary"
        # Preprocess the observation if needed
        latent_features, decoded_features = self.extract_features(obs)

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
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        episode_starts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        assert isinstance(obs, dict), "obs must be a dictionary"
        latent_features, decoded_features = self.extract_features(obs, episode_starts)

        latent_pi = self.mlp_extractor.forward_actor(latent_features)
        # Critic uses privileged observations directly
        latent_vf = self.mlp_extractor.forward_critic(obs["teacher"])

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        assert isinstance(obs, dict), "obs must be a dictionary"
        latent_features, decoded_features = self.extract_features(obs)
        # Critic uses privileged observations directly
        latent_vf = self.mlp_extractor.forward_critic(obs["teacher"])
        return self.value_net(latent_vf)


class DWLPPO(PPO):
    """
    PPO with Denoising World model Learning, using GRU encoder with running memory for observation history.
    """

    def __init__(self, policy, env, **ppo_kwargs):
        super().__init__(policy, env, **ppo_kwargs)
        print(self.policy)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.rollout_buffer_kwargs["device"] = "cuda:0"

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer  # type: ignore[assignment]
            else:
                self.rollout_buffer_class = RolloutBuffer  # type: ignore[assignment]

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
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

                values, log_prob, entropy = self.policy.evaluate_actions(
                    {
                        k: v.to(self.device)
                        for k, v in rollout_data.observations.items()  # type: ignore[attr-defined]
                    },
                    actions,  # type: ignore
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

                # Don't pass episode_starts during training evaluation since it's not available
                latent_features, decoded_features = self.policy.extract_features(
                    obs_dict
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
        rollout_buffer: DictRolloutBuffer,
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
                actions, values, log_probs = self.policy(obs_tensor)

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
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        gc.collect()

        with th.inference_mode():
            # Compute value for the last timestep
            values = self.policy.predict_values(new_obs)

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
        self.current_obs_dim = 47  # Based on table: GRU(47 → 256)
        self.privileged_dim = 184  # Based on table: decoder output
        self.latent_dim = 24  # Based on table: encoder output

        # GRU encoder with running memory - Architecture: GRU(47 → 256)
        self.gru_hidden_size = 256
        self.gru_encoder = nn.GRU(
            input_size=self.current_obs_dim,
            hidden_size=self.gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Encoder post-processing - Architecture: Linear(256 → 256), ELU, Linear(256 → 24)
        self.encoder_post = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 256),
            activation_fn(),
            nn.Linear(256, self.latent_dim),
        )

        # Decoder for privileged state reconstruction - Architecture: Linear(24 → 64), ELU, Linear(64 → 184)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            activation_fn(),
            nn.Linear(64, self.privileged_dim),
        )

        # Running GRU hidden state - will be reset on episode starts
        self.register_buffer("gru_hidden", th.zeros(1, 1, self.gru_hidden_size))

        # Update features_dim to match the latent dimension
        self._features_dim = features_dim

    def reset_memory(self, batch_size: int, device: th.device) -> None:
        """Reset the GRU hidden state"""
        self.gru_hidden.data = th.zeros(
            1, batch_size, self.gru_hidden_size, device=device
        )

    def forward(
        self,
        observations: dict[str, th.Tensor],
        episode_starts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Extract features using GRU encoder with running memory and decoder for reconstruction.

        :param observations: Dictionary containing current observations
        :param episode_starts: Boolean tensor indicating episode starts for memory reset
        :return: Tuple of (latent_features, decoded_privileged_features)
        """
        # Get current observation (47-dim based on architecture)
        current_obs = observations[
            "student"
        ]  # Assuming this is the 47-dim current observation
        batch_size = current_obs.shape[0]

        # Reset GRU memory on episode starts
        if episode_starts is not None and episode_starts.any():
            # Reset hidden states for environments that started new episodes
            for i in range(batch_size):
                if episode_starts[i]:
                    self.gru_hidden[:, i, :] = 0

        # Ensure hidden state has correct batch size
        if self.gru_hidden.shape[1] != batch_size:
            self.reset_memory(batch_size, current_obs.device)

        # GRU forward pass - input current observation and previous hidden state
        # Reshape current_obs for GRU: (batch_size, seq_len=1, input_size)
        gru_input = current_obs.unsqueeze(1)  # Shape: (batch_size, 1, 47)

        # GRU forward pass
        gru_output, new_hidden = self.gru_encoder(gru_input, self.gru_hidden)

        # Update hidden state for next timestep
        self.gru_hidden.data = new_hidden.detach()

        # Get the output from the sequence (only one timestep)
        gru_out = gru_output.squeeze(1)  # Shape: (batch_size, 256)

        # Encoder post-processing: Linear(256 → 256), ELU, Linear(256 → 24)
        latent_features = self.encoder_post(gru_out)  # Shape: (batch_size, 24)

        # Decoder: reconstruct privileged state
        decoded_features = self.decoder(latent_features)  # Shape: (batch_size, 184)

        return latent_features, decoded_features


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

        # Use fixed architecture from table instead of net_arch
        latent_dim = feature_dims[0]  # 24 for actor
        privileged_dim = feature_dims[1]  # privileged state dimension for critic

        # Actor network: Linear(24 → 48), ELU, Linear(48 → 12)
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, 48),
            activation_fn(),
            nn.Linear(48, 12),
        ).to(device)

        # Critic network: Takes privileged state directly
        # Architecture: Linear(privileged_dim → 512), ELU, Linear(512 → 512), ELU, Linear(512 → 256), ELU
        # Note: The final Linear(256 → 1) will be handled by the value_net in the policy
        self.value_net = nn.Sequential(
            nn.Linear(privileged_dim, 512),
            activation_fn(),
            nn.Linear(512, 512),
            activation_fn(),
            nn.Linear(512, 256),
            activation_fn(),
        ).to(device)

        # Save dimensions - actor outputs 12, critic outputs 256 (final layer handled by value_net)
        self.latent_dim_pi = 12  # Actor final output dimension
        self.latent_dim_vf = 256  # Critic output dimension before final value layer
