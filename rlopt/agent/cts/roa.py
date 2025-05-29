from __future__ import annotations

import gc
import time
from collections import deque
from typing import Optional, Union, Dict

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


class ROAPolicy(ActorCriticPolicy):
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
        """Create a custom features extractor for the CTS policy."""
        return ROAExtractor(
            self.observation_space,  # type: ignore[arg-type]
            features_dim=256,
            activation_fn=nn.ELU,
        )

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        preprocessed_obs = preprocessing.preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )
        return self.features_extractor(preprocessed_obs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = ROAMlpExtractor(  # type: ignore[assignment]
            (
                int(self.features_dim + self.observation_space["student"].shape[0]),  # type: ignore[index]
                int(self.features_dim + self.observation_space["teacher"].shape[0]),  # type: ignore[index]
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
        features = self.extract_features(obs)

        # Use teacher features for policy and value
        teacher_features, stacked_student_features, student_features = features

        pi_features = th.where(
            obs["teacher_mask"].bool(),
            teacher_features,
            stacked_student_features.detach(),
        )

        vf_features = teacher_features
        pi_features = th.cat((pi_features, obs["student"]), dim=-1)
        vf_features = th.cat((vf_features, obs["teacher"]), dim=-1)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        assert isinstance(obs, dict), "obs must be a dictionary"
        features = self.extract_features(obs)
        teacher_features, stacked_student_features, student_features = features
        pi_features = th.where(
            obs["teacher_mask"].bool(),
            teacher_features,
            stacked_student_features.detach(),
        )
        vf_features = teacher_features
        pi_features = th.cat((pi_features, obs["student"]), dim=-1)
        vf_features = th.cat((vf_features, obs["teacher"]), dim=-1)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)

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
        features = self.extract_features(obs)
        teacher_features, _, _ = features
        vf_features = teacher_features
        vf_features = th.cat((vf_features, obs["teacher"]), dim=-1)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return self.value_net(latent_vf)


class ROAPPO(PPO):
    """
    PPO with Regularized Online Adaptation (ROA).
    """

    def __init__(
        self, policy, env, H=10, lambda_latent=1.0, lambda_schedule=None, **ppo_kwargs
    ):
        super().__init__(policy, env, **ppo_kwargs)
        self.H = H  # Update interval for adaptation module
        self.lambda_latent = lambda_latent  # Initial regularization strength
        self.lambda_schedule = lambda_schedule  # Optional schedule function for lambda
        # For curriculum, if not provided, use linear schedule
        if self.lambda_schedule is None:
            self.lambda_schedule = lambda it: self.lambda_latent * min(1.0, it / 10000)
        self.ep_infos = []

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
        # Update lambda according to curriculum
        lambda_latent = self.lambda_schedule(self.update_count)

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
                # --- ROA latent losses ---
                assert isinstance(self.policy, ROAPolicy)
                teacher_features, stacked_student_features, student_features = (
                    self.policy.extract_features(
                        {
                            "teacher": rollout_data.observations["teacher"],
                            "stacked_obs": rollout_data.observations["stacked_obs"],
                            "student": rollout_data.observations["student"],
                        }
                    )
                )
                # z^phi = stacked_student_features, z^mu = teacher_features
                # L1: ||sg[z^mu] - z^phi||_2
                loss_latent_adapt = F.mse_loss(
                    teacher_features.detach(), stacked_student_features
                )
                # L2: ||z^mu - sg[z^phi]||_2
                loss_latent_policy = F.mse_loss(
                    teacher_features, stacked_student_features.detach()
                )
                # --- Alternating update ---
                if self._n_updates % self.H == 0:
                    # Adaptation module update only (Eq. 19 in pseudocode)
                    loss = loss_latent_adapt
                else:
                    # Full RL + latent regularization (Eq. 21 in pseudocode)
                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + lambda_latent * (loss_latent_policy + loss_latent_adapt)
                    )
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
                        allocations.append(
                            {
                                "size": block["size"],
                                "filename": block.get("filename", "unknown"),
                                "line": block.get("line", 0),
                                "stack": block.get("frames", []),
                            }
                        )

            # Sort by size (largest first)
            allocations.sort(key=lambda x: x["size"], reverse=True)

            print(f"[CUDA] Top 10 persistent allocations:")
            for i, alloc in enumerate(allocations[:10]):
                print(f"  {i + 1}. Size: {alloc['size'] / 1024**2:.2f} MB")
                print(f"     Location: {alloc['filename']}:{alloc['line']}")

                # Show stack trace for large allocations
                if alloc["size"] > 1024**2:  # > 1MB
                    print(f"     Stack trace:")
                    for frame in alloc["stack"][:5]:  # Show top 5 frames
                        print(
                            f"       {frame.get('filename', 'unknown')}:{frame.get('line', 0)} in {frame.get('name', 'unknown')}"
                        )
                print()

            # Group by location
            location_groups = {}
            for alloc in allocations:
                key = f"{alloc['filename']}:{alloc['line']}"
                if key not in location_groups:
                    location_groups[key] = {"count": 0, "total_size": 0}
                location_groups[key]["count"] += 1
                location_groups[key]["total_size"] += alloc["size"]

            print(f"[CUDA] Memory by location (top 5):")
            sorted_locations = sorted(
                location_groups.items(), key=lambda x: x[1]["total_size"], reverse=True
            )
            for location, info in sorted_locations[:5]:
                print(
                    f"  {location}: {info['total_size'] / 1024**2:.2f} MB ({info['count']} allocations)"
                )

        except Exception as e:
            print(f"[CUDA] memory_snapshot() failed: {e}")

        # Find Python objects holding CUDA tensors
        try:
            cuda_objects = []
            for obj in gc.get_objects():
                if hasattr(obj, "is_cuda") and obj.is_cuda:
                    cuda_objects.append(
                        {
                            "type": type(obj).__name__,
                            "size": (
                                obj.numel() * obj.element_size()
                                if hasattr(obj, "numel")
                                else 0
                            ),
                            "shape": (
                                tuple(obj.shape) if hasattr(obj, "shape") else "unknown"
                            ),
                            "dtype": (
                                str(obj.dtype) if hasattr(obj, "dtype") else "unknown"
                            ),
                        }
                    )

            # Group by type and size
            type_groups = {}
            for obj in cuda_objects:
                obj_type = obj["type"]
                if obj_type not in type_groups:
                    type_groups[obj_type] = {"count": 0, "total_size": 0}
                type_groups[obj_type]["count"] += 1
                type_groups[obj_type]["total_size"] += obj["size"]

            print(f"[CUDA] Python objects holding CUDA memory:")
            sorted_types = sorted(
                type_groups.items(), key=lambda x: x[1]["total_size"], reverse=True
            )
            for obj_type, info in sorted_types[:5]:
                print(
                    f"  {obj_type}: {info['total_size'] / 1024**2:.2f} MB ({info['count']} objects)"
                )

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

            self.num_timesteps += env.num_envs

            infos: dict
            # Record infos
            if "episode" in infos:
                self.ep_infos.append(infos["episode"])
            elif "log" in infos:
                self.ep_infos.append(infos["log"])

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

    def dump_logs(self, iteration: int) -> None:
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


class ROAExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for Concurrent Teacher-Student policy.
    Handles three observation spaces: teacher, stacked_student, and student.
    Returns a dictionary of features for each observation type.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        activation_fn: type[nn.Module] = nn.ELU,
        latent_dims: tuple[int, int, int] = (512, 512, 512),
    ):
        if not isinstance(observation_space, spaces.Dict):
            value_error = "CTSExtractor requires a Dict observation space"
            raise ValueError(value_error)

        super().__init__(observation_space, features_dim)

        # Get dimensions from observation space
        self.teacher_dim = observation_space["teacher"].shape[0]  # type: ignore[index]
        self.stacked_student_dim = observation_space["stacked_obs"].shape[0]  # type: ignore[index]
        self.student_dim = observation_space["student"].shape[0]  # type: ignore[index]
        self.teacher_latent_dims = latent_dims[0]
        self.stacked_student_latent_dims = latent_dims[1]
        self.student_latent_dims = latent_dims[2]

        # Create MLP extractors for each observation space
        self.teacher_encoder = nn.Sequential(
            nn.Linear(self.teacher_dim, self.teacher_latent_dims),
            activation_fn(),
            nn.Linear(self.teacher_latent_dims, features_dim),
            activation_fn(),
        )

        self.stacked_student_encoder = nn.Sequential(
            nn.Linear(self.stacked_student_dim, self.stacked_student_latent_dims),
            activation_fn(),
            nn.Linear(self.stacked_student_latent_dims, features_dim),
            activation_fn(),
        )

        self.student_encoder = nn.Sequential(
            nn.Linear(self.student_dim, self.student_latent_dims),
            activation_fn(),
            nn.Linear(self.student_latent_dims, features_dim),
            activation_fn(),
        )

        # Update features_dim to match the size of each feature vector
        self._features_dim = features_dim

    def forward(
        self, observations: dict[str, th.Tensor]
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Extract features from each observation space and return them as a dictionary.

        :param observations: Dictionary containing teacher, stacked_obs, and student observations
        :return: Dictionary containing extracted features for each observation type
        """
        # Extract features from each observation space
        teacher_features = self.teacher_encoder(observations["teacher"])
        stacked_student_features = self.stacked_student_encoder(
            observations["stacked_obs"]
        )
        student_features = self.student_encoder(observations["student"])

        return (
            teacher_features,
            stacked_student_features,
            student_features,
        )


class ROAMlpExtractor(MlpExtractor):

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
        last_layer_dim_pi = feature_dims[0]
        last_layer_dim_vf = feature_dims[1]

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
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
