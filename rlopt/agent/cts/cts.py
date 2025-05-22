from __future__ import annotations

import time
from collections import deque
from typing import Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv

from rlopt.common import DictRolloutBuffer


class CTSPolicy(ActorCriticPolicy):
    def __init__(self, *args, latent_dim=32, stack_size=3, teacher_mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        # privileged encoder on obs["teacher"]
        teacher_dim = self.observation_space["teacher"].shape[0]
        self.privileged_encoder = nn.Sequential(
            nn.Linear(teacher_dim, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
            nn.ELU(),
        )
        # proprio encoder on obs["stacked_obs"]
        student_dim = self.observation_space["stacked_obs"].shape[0]
        self.proprio_encoder = nn.Sequential(
            nn.Linear(student_dim, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
            nn.ELU(),
        )
        # force SB3 to rebuild its heads on latent_dim
        self.features_dim = latent_dim
        self._build_mlp_extractor()

        self.stack_size = stack_size

        self.teacher_mask = teacher_mask.to(self.device)

    def forward(self, obs, deterministic=False):
        # Validate input
        assert isinstance(obs, dict), "Observations must be a dictionary"
        assert (
            "teacher" in obs and "student" in obs
        ), "Teacher and student observations must be present"
        assert "stacked_obs" in obs, "Stacked observations must be present"

        # Get teacher mask and encode privileged state
        mask = obs["teacher_mask"].bool()
        z_t = self.privileged_encoder(obs["teacher"]).squeeze(1)
        # Encode proprioceptive state from stacked observations
        z_s = self.proprio_encoder(obs["stacked_obs"]).squeeze(1)
        # Combine encodings based on mask
        z = th.where(mask, z_t, z_s)
        # Get policy and value predictions from combined encoding
        latent_pi, latent_vf = self.mlp_extractor(z)

        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        return self.forward(obs, deterministic)[0]

    def evaluate_actions(self, obs, actions):
        # Get teacher mask and encode privileged state
        mask = obs["teacher_mask"].bool()

        z_t = self.privileged_encoder(obs["teacher"]).squeeze(1)
        # Encode proprioceptive state from stacked observations
        z_s = self.proprio_encoder(obs["stacked_obs"]).squeeze(1)
        # Combine encodings based on mask
        z = th.where(mask, z_t, z_s)
        # Get policy and value predictions from combined encoding
        latent_pi, latent_vf = self.mlp_extractor(z)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def predict_values(self, obs):
        z_t = self.privileged_encoder(obs["teacher"])
        z_s = self.proprio_encoder(obs["stacked_obs"])
        z = th.where(obs["teacher_mask"], z_t, z_s)
        latent_pi, latent_vf = self.mlp_extractor(z)
        return self.value_net(latent_vf)


class CTSPPO(PPO):
    """
    PPO with Concurrent Teacher-Student updates, now with fixed 3:1 teacher:student assignment.
    """

    def __init__(self, policy, env, **ppo_kwargs):

        super().__init__(policy, env, **ppo_kwargs)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

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
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        self.policy.set_training_mode(True)

        self._update_learning_rate([self.policy.optimizer])

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

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
                        for k, v in rollout_data.observations.items()
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

                m = self.policy.teacher_mask
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

                if self.clip_range_vf is None:
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

                with th.no_grad():
                    # reconstruction between encoders
                    z_t = self.policy.privileged_encoder(
                        rollout_data.observations["teacher"].to(self.device)
                    )
                # Use stacked observations for proprio encoder
                z_s = self.policy.proprio_encoder(
                    rollout_data.observations["stacked_obs"].to(self.device)
                )

                loss_rec = F.mse_loss(z_s, z_t)

                # total
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + 1.0 * loss_rec
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

            self._n_updates += 1
            if not continue_training:
                break

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
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

        self._last_obs: Dict[str, th.Tensor]
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        self.ep_infos = []

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
                clipped_actions.to(env.sim_device)
            )

            new_obs = {k: v.to(self.device) for k, v in new_obs.items()}
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            self.num_timesteps += env.num_envs

            infos: dict
            # Record infos
            if "episode" in infos:
                self.ep_infos.append(infos["episode"])
            elif "log" in infos:
                self.ep_infos.append(infos["log"])

            self.cur_reward_sum += rewards.to(self.device)
            self.cur_episode_length += 1.0
            new_ids = (dones > 0).nonzero(as_tuple=False).to(self.device)
            # record reward and episode length
            self.rewbuffer.extend(
                self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
            )
            self.lenbuffer.extend(
                self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
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
            if "time_outs" in infos:
                rewards += self.gamma * th.squeeze(
                    values * infos["time_outs"].unsqueeze(1).to(self.device),
                    1,
                )

            rollout_buffer.add(
                {k: v.to(self.device) for k, v in self._last_obs.items()},  # type: ignore[arg-type]
                actions.to(self.device),
                rewards.to(self.device),
                self._last_episode_starts.to(self.device),  # type: ignore[arg-type]
                values.to(self.device),
                log_probs.to(self.device),
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.inference_mode():
            # Compute value for the last timestep
            values = self.policy.predict_values(
                {k: v.to(self.device) for k, v in new_obs.items()}
            )  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(
            last_values=values.to(rollout_buffer.device),
            dones=dones.to(rollout_buffer.device),
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
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        # store the current number of timesteps
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = th.zeros(
            self.env.num_envs, dtype=th.float, device=self.device
        )
        self.cur_episode_length = th.zeros(
            self.env.num_envs, dtype=th.float, device=self.device
        )

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

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
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
            )

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        :param locs: Local variables
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
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = th.cat((infotensor, ep_info[key].to(self.device)))
                value = th.mean(infotensor).item()
                # log to logger and terminal
                if "/" in key:
                    self.logger.record(key, value)
                else:
                    self.logger.record("Episode/" + key, value)

        self.logger.record("Episode/average_episodic_reward", np.mean(self.rewbuffer))
        self.logger.record("Episode/average_episodic_length", np.mean(self.lenbuffer))
        self.logger.record(
            "Episode/episodic_reward", th.max(self.cur_episode_length).item()
        )
        self.logger.record(
            "Episode/episodic_length", th.max(self.cur_reward_sum).item()
        )
        self.logger.dump(step=self.num_timesteps)
