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
import gc


class CTSPolicy(ActorCriticPolicy):
    def __init__(
        self, *args, latent_dim=128, stack_size=5, teacher_mask=None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Get dimensions
        teacher_dim = self.observation_space["teacher"].shape[0]
        student_dim = self.observation_space["stacked_obs"].shape[0]
        student_obs_dim = self.observation_space["student"].shape[0]

        # Privileged encoder on obs["teacher"]
        self.privileged_encoder = nn.Sequential(
            nn.Linear(teacher_dim, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
            nn.ELU(),
        )

        # Proprio encoder on obs["stacked_obs"]
        self.proprio_encoder = nn.Sequential(
            nn.Linear(student_dim, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
            nn.ELU(),
        )

        # Policy network: takes student observations + encoder latents
        policy_input_dim = student_obs_dim + latent_dim
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # Value network: takes teacher observations + encoder latents
        value_input_dim = teacher_dim + latent_dim
        self.value_net_custom = nn.Sequential(
            nn.Linear(value_input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        # Override features_dim for SB3 compatibility
        self.features_dim = 128
        self._build_mlp_extractor()

        self.stack_size = stack_size
        self.latent_dim = latent_dim

        if teacher_mask is not None:
            self.teacher_mask = teacher_mask.to(self.device)
        else:
            self.teacher_mask = None

    def forward(self, obs, deterministic=False):
        # Validate input
        assert isinstance(obs, dict), "Observations must be a dictionary"
        assert (
            "teacher" in obs and "student" in obs
        ), "Teacher and student observations must be present"
        assert "stacked_obs" in obs, "Stacked observations must be present"

        # Get teacher mask and encode privileged state
        mask = obs["teacher_mask"].bool()
        z_t = self.privileged_encoder(obs["teacher"])
        # Encode proprioceptive state from stacked observations
        # Use no_grad to prevent gradients from policy/value losses
        with th.no_grad():
            z_s = self.proprio_encoder(obs["stacked_obs"])

        # Combine encodings based on mask for policy input
        z = th.where(mask, z_t, z_s)

        # Policy input: student observations + encoder latents
        policy_input = th.cat([obs["student"], z], dim=-1)
        policy_features = self.policy_net(policy_input)

        # Value input: teacher observations + encoder latents
        value_input = th.cat([obs["teacher"], z], dim=-1)
        values = self.value_net_custom(value_input)

        # Use policy features for action distribution
        dist = self._get_action_dist_from_latent(policy_features)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)

        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)[0]

    def evaluate_actions(self, obs, actions):
        # Get teacher mask and encode privileged state
        mask = obs["teacher_mask"].bool()
        z_t = self.privileged_encoder(obs["teacher"])
        # Encode proprioceptive state from stacked observations
        # Use no_grad to prevent gradients from policy/value losses
        with th.no_grad():
            z_s = self.proprio_encoder(obs["stacked_obs"])

        # Combine encodings based on mask
        z = th.where(mask, z_t, z_s)

        # Policy input: student observations + encoder latents
        policy_input = th.cat([obs["student"], z], dim=-1)
        policy_features = self.policy_net(policy_input)

        # Value input: teacher observations + encoder latents
        value_input = th.cat([obs["teacher"], z], dim=-1)
        values = self.value_net_custom(value_input)

        dist = self._get_action_dist_from_latent(policy_features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return values, log_prob, entropy

    def predict_values(self, obs):
        mask = obs["teacher_mask"].bool()
        z_t = self.privileged_encoder(obs["teacher"])
        # Use no_grad to prevent gradients from policy/value losses
        with th.no_grad():
            z_s = self.proprio_encoder(obs["stacked_obs"])

        # Combine encodings based on mask
        z = th.where(mask, z_t, z_s)

        # Value input: teacher observations + encoder latents
        value_input = th.cat([obs["teacher"], z], dim=-1)
        values = self.value_net_custom(value_input)

        return values


class CTSPPO(PPO):
    """
    PPO with Concurrent Teacher-Student updates, now with fixed 3:1 teacher:student assignment.
    """

    def __init__(self, policy, env, **ppo_kwargs):
        super().__init__(policy, env, **ppo_kwargs)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.rollout_buffer_kwargs["device"] = "cuda:0"

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
        clip_range_vf = None  # Initialize to None by default
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

                # Reconstruction loss between encoders - fix memory leak
                # Compute both encodings with gradient tracking for reconstruction loss
                z_t = self.policy.privileged_encoder(
                    rollout_data.observations["teacher"].to(self.device)
                )
                z_s = self.policy.proprio_encoder(
                    rollout_data.observations["stacked_obs"].to(self.device)
                )

                loss_rec = F.mse_loss(
                    z_s, z_t.detach()
                )  # Detach z_t to prevent double backprop

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

        # Log statistics
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_range", clip_range)
        self.logger.record(
            "train/learning_rate", self.policy.optimizer.param_groups[0]["lr"]
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))

    def _analyze_cuda_memory(self, step_name=""):
        """Analyze CUDA memory usage with detailed breakdown"""
        if not th.cuda.is_available():
            return
            
        print(f"\n[CUDA] Memory analysis - {step_name}")
        print(f"[CUDA] Allocated: {th.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"[CUDA] Reserved: {th.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        try:
            snapshot = th.cuda.memory_snapshot()
            allocations = []
            
            for segment in snapshot:
                for block in segment["blocks"]:
                    if block["state"] == "active_alloc":
                        allocations.append({
                            'size': block['size'],
                            'filename': block.get('filename', 'unknown'),
                            'line': block.get('line', 0),
                            'stack': block.get('frames', [])
                        })
            
            # Sort by size (largest first)
            allocations.sort(key=lambda x: x['size'], reverse=True)
            
            print(f"[CUDA] Top 10 persistent allocations:")
            for i, alloc in enumerate(allocations[:10]):
                print(f"  {i+1}. Size: {alloc['size']/1024**2:.2f} MB")
                print(f"     Location: {alloc['filename']}:{alloc['line']}")
                
                # Show stack trace for large allocations
                if alloc['size'] > 1024**2:  # > 1MB
                    print(f"     Stack trace:")
                    for frame in alloc['stack'][:5]:  # Show top 5 frames
                        print(f"       {frame.get('filename', 'unknown')}:{frame.get('line', 0)} in {frame.get('name', 'unknown')}")
                print()
                
            # Group by location
            location_groups = {}
            for alloc in allocations:
                key = f"{alloc['filename']}:{alloc['line']}"
                if key not in location_groups:
                    location_groups[key] = {'count': 0, 'total_size': 0}
                location_groups[key]['count'] += 1
                location_groups[key]['total_size'] += alloc['size']
            
            print(f"[CUDA] Memory by location (top 5):")
            sorted_locations = sorted(location_groups.items(), key=lambda x: x[1]['total_size'], reverse=True)
            for location, info in sorted_locations[:5]:
                print(f"  {location}: {info['total_size']/1024**2:.2f} MB ({info['count']} allocations)")
                
        except Exception as e:
            print(f"[CUDA] memory_snapshot() failed: {e}")
            
        # Find Python objects holding CUDA tensors
        try:
            cuda_objects = []
            for obj in gc.get_objects():
                if hasattr(obj, 'is_cuda') and obj.is_cuda:
                    cuda_objects.append({
                        'type': type(obj).__name__,
                        'size': obj.numel() * obj.element_size() if hasattr(obj, 'numel') else 0,
                        'shape': tuple(obj.shape) if hasattr(obj, 'shape') else 'unknown',
                        'dtype': str(obj.dtype) if hasattr(obj, 'dtype') else 'unknown'
                    })
            
            # Group by type and size
            type_groups = {}
            for obj in cuda_objects:
                obj_type = obj['type']
                if obj_type not in type_groups:
                    type_groups[obj_type] = {'count': 0, 'total_size': 0}
                type_groups[obj_type]['count'] += 1
                type_groups[obj_type]['total_size'] += obj['size']
            
            print(f"[CUDA] Python objects holding CUDA memory:")
            sorted_types = sorted(type_groups.items(), key=lambda x: x[1]['total_size'], reverse=True)
            for obj_type, info in sorted_types[:5]:
                print(f"  {obj_type}: {info['total_size']/1024**2:.2f} MB ({info['count']} objects)")
                
        except Exception as e:
            print(f"[CUDA] Python object analysis failed: {e}")

    def collect_rollouts(
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

            # Ensure tensors are on correct device and avoid redundant transfers
            new_obs = {k: v.to(self.device) for k, v in new_obs.items()}
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            # if th.cuda.is_available() and n_steps % 200 == 0:
            #     self._analyze_cuda_memory(f"after env step - step {n_steps}")

            self.num_timesteps += env.num_envs

            # infos: dict
            # # Record infos
            # if "episode" in infos:
            #     self.ep_infos.append(infos["episode"])
            # elif "log" in infos:
            #     self.ep_infos.append(infos["log"])

            self.cur_reward_sum += rewards.clone().detach()
            self.cur_episode_length += 1.0
            new_ids = (dones > 0).nonzero(as_tuple=False)

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
            if "time_outs" in infos:
                rewards += self.gamma * th.squeeze(
                    values * infos["time_outs"].unsqueeze(1).to(self.device),
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
            values = self.policy.predict_values(
                new_obs
            )  

        rollout_buffer.compute_returns_and_advantage(
            last_values=values.detach().to(rollout_buffer.device),
            dones=dones.detach().to(rollout_buffer.device),
        )

        # callback.update_locals(locals())

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
