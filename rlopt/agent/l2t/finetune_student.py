from __future__ import annotations

import statistics
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import time
import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    FloatSchedule,
    explained_variance,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv

from rlopt.agent.l2t.policies import (
    AsymmetricLstmPolicy,
    CnnLstmPolicy,
    MlpLstmPolicy,
    MoeLstmPolicy,
    MoeMultiInputLstmPolicy,
    MultiInputLstmPolicy,
    RecurrentActorCriticPolicy,
)
from rlopt.agent.l2t.recurrent_l2t import RecurrentL2T
from rlopt.common.buffer import (
    RecurrentDictRolloutBuffer,
    RecurrentRolloutBuffer,
)

SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")


class RecurrentPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_recurrent_policies`
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
        "MoeLstmPolicy": MoeLstmPolicy,
        "MoeMultiInputLstmPolicy": MoeMultiInputLstmPolicy,
        "AsymmetricLstmPolicy": AsymmetricLstmPolicy,
    }

    def __init__(
        self,
        policy: str | type[RecurrentActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 128,
        batch_size: int | None = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = (
            RecurrentDictRolloutBuffer
            if isinstance(self.observation_space, spaces.Dict)
            else RecurrentRolloutBuffer
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (
            self.n_steps,
            lstm.num_layers,
            self.n_envs,
            lstm.hidden_size,
        )

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert (
                    self.clip_range_vf > 0
                ), "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        self.ep_infos = []

        collection_start = time.time_ns()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = self._last_obs
                episode_starts = (
                    self._last_episode_starts.clone().type(th.float32).to(self.device)
                )
                actions, values, log_probs, lstm_states = self.policy.forward(
                    obs_tensor, lstm_states, episode_starts
                )

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = th.clamp(
                    actions,
                    th.as_tensor(self.action_space.low, device=self.device),
                    th.as_tensor(self.action_space.high, device=self.device),
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            infos: dict
            # Record infos
            if "episode" in infos:
                self.ep_infos.append(infos["episode"])
            elif "log" in infos:
                self.ep_infos.append(infos["log"])

            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Bootstrapping on time outs, but using the last values
            if "time_outs" in infos:
                rewards += self.gamma * th.squeeze(
                    values * infos["time_outs"].unsqueeze(1).to(self.device),
                    1,
                )

            self.cur_reward_sum += rewards
            self.cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

            # record reward and episode length
            self.rewbuffer.extend(
                self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()  # type: ignore[arg-type]
            )
            self.lenbuffer.extend(
                self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
            )
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 1

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = dones.clone().type(th.float32)
            values = self.policy.predict_values(new_obs, lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        collection_end = time.time_ns()
        collection_time = (collection_end - collection_start) / 1e9
        self.logger.record("train/collection_time", collection_time)

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        training_start = time.time_ns()

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (
                        advantages[mask].std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()[mask]
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask])
                        .cpu()
                        .numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        training_end = time.time_ns()
        training_time = (training_end - training_start) / 1e9
        self.logger.record("train/training_time", training_time)

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten().clone().cpu().numpy(),
            self.rollout_buffer.returns.flatten().clone().cpu().numpy(),
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_last_lstm_states"]  # noqa: RUF005

    @classmethod
    def from_recurrent_l2t(
        cls,
        checkpoint_path: str | Path,
        env: GymEnv,
        *,
        device: th.device | str = "auto",
        ppo_kwargs: dict[str, Any] | None = None,
        policy_kwargs_override: dict[str, Any] | None = None,
        load_kwargs: dict[str, Any] | None = None,
        policy_class_override: str | type[RecurrentActorCriticPolicy] | None = None,
        strict_student_load: bool = True,
    ) -> SelfRecurrentPPO:
        """Instantiate a ``RecurrentPPO`` policy seeded with a trained L2T student.

        Args:
            checkpoint_path: Path to a saved ``RecurrentL2T`` checkpoint (file or directory).
            env: Target environment for fine-tuning.
            device: Torch device for both loading the L2T policy and initializing PPO.
            ppo_kwargs: Optional overrides for PPO hyperparameters.
            policy_kwargs_override: Optional overrides for the student policy kwargs.
            load_kwargs: Extra arguments forwarded to ``RecurrentL2T.load``.
            policy_class_override: Optional policy spec (alias or type) to override the saved class.
            strict_student_load: Whether to require an exact parameter match when copying weights.

        Returns:
            A ``RecurrentPPO`` instance whose policy parameters match the loaded student.

        Raises:
            ValueError: If the saved checkpoint does not contain a student policy
                or if the environment action space is incompatible.
        """
        path = Path(checkpoint_path)
        if path.is_dir():
            candidate = path / "recurrent_l2t_model.zip"
            path = candidate if candidate.exists() else path

        load_kwargs = {} if load_kwargs is None else load_kwargs
        l2t_model = RecurrentL2T.load(
            path, env=None, device=device, print_system_info=False, **load_kwargs
        )

        if not hasattr(l2t_model, "student_policy"):
            msg = "Checkpoint does not contain a student policy."
            raise ValueError(msg)

        student_policy = l2t_model.student_policy  # type: ignore[attr-defined]
        if policy_class_override is not None:
            if isinstance(policy_class_override, str):
                if policy_class_override not in cls.policy_aliases:
                    msg = f"Unknown policy alias '{policy_class_override}'."
                    raise KeyError(msg)
                policy_cls = cls.policy_aliases[policy_class_override]
            else:
                policy_cls = policy_class_override
        else:
            policy_cls = type(student_policy)
        student_policy_kwargs = deepcopy(
            getattr(l2t_model, "student_policy_kwargs", {})  # type: ignore[attr-defined]
        )
        if policy_kwargs_override:
            student_policy_kwargs.update(policy_kwargs_override)

        default_kwargs = cls._ppo_kwargs_from_l2t(l2t_model)
        if ppo_kwargs:
            default_kwargs.update(ppo_kwargs)
        default_kwargs["device"] = device

        agent = cls(
            policy=policy_cls,
            env=env,
            policy_kwargs=student_policy_kwargs,
            **default_kwargs,
        )

        l2t_action_space = getattr(l2t_model, "action_space", None)
        env_action_space = getattr(env, "action_space", None)
        if (
            l2t_action_space is not None
            and env_action_space is not None
            and isinstance(l2t_action_space, type(agent.action_space))
        ):
            if isinstance(agent.action_space, spaces.Box):
                if l2t_action_space.shape != agent.action_space.shape:
                    msg = "Action space shape mismatch between checkpoint and environment."
                    raise ValueError(msg)
            elif isinstance(agent.action_space, spaces.Discrete):
                if l2t_action_space.n != agent.action_space.n:
                    msg = "Action space cardinality mismatch."
                    raise ValueError(msg)

        student_state_dict = {
            key: value.to(agent.policy.device)
            if isinstance(value, th.Tensor)
            else value
            for key, value in student_policy.state_dict().items()
        }
        missing, unexpected = agent.policy.load_state_dict(
            student_state_dict, strict=strict_student_load
        )
        if strict_student_load and (missing or unexpected):
            msg = (
                "Failed to load student weights cleanly: "
                f"missing={missing}, unexpected={unexpected}"
            )
            raise ValueError(msg)

        partial_space = getattr(l2t_model, "partial_observation_space", None)
        env_space = getattr(env, "observation_space", None)
        if (
            partial_space is not None
            and env_space is not None
            and isinstance(partial_space, spaces.Box)
            and isinstance(env_space, spaces.Box)
            and partial_space.shape != env_space.shape
        ):
            msg = (
                "Observation space mismatch between checkpointed student "
                f"{partial_space.shape} and environment {env_space.shape}."
            )
            raise ValueError(msg)

        del l2t_model

        return agent

    @staticmethod
    def _ppo_kwargs_from_l2t(l2t_model: RecurrentL2T) -> dict[str, Any]:
        clip_range = deepcopy(getattr(l2t_model, "clip_range", 0.2))
        clip_range_vf = deepcopy(getattr(l2t_model, "clip_range_vf", None))
        ent_coef = getattr(l2t_model, "ent_coef", 0.0)
        vf_coef = getattr(l2t_model, "vf_coef", 0.5)
        if isinstance(ent_coef, th.Tensor):
            ent_coef = float(ent_coef.item())
        if isinstance(vf_coef, th.Tensor):
            vf_coef = float(vf_coef.item())

        stats_window_size = getattr(l2t_model, "_stats_window_size", 100)

        return {
            "learning_rate": getattr(l2t_model, "learning_rate", 3e-4),
            "n_steps": getattr(l2t_model, "n_steps", 128),
            "batch_size": getattr(l2t_model, "batch_size", None),
            "n_epochs": getattr(l2t_model, "n_epochs", 10),
            "gamma": getattr(l2t_model, "gamma", 0.99),
            "gae_lambda": getattr(l2t_model, "gae_lambda", 0.95),
            "clip_range": clip_range,
            "clip_range_vf": clip_range_vf,
            "normalize_advantage": getattr(l2t_model, "normalize_advantage", True),
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": getattr(l2t_model, "max_grad_norm", 0.5),
            "use_sde": getattr(l2t_model, "use_sde", False),
            "sde_sample_freq": getattr(l2t_model, "sde_sample_freq", -1),
            "target_kl": getattr(l2t_model, "target_kl", None),
            "stats_window_size": stats_window_size,
            "tensorboard_log": None,
            "verbose": getattr(l2t_model, "verbose", 0),
            "seed": getattr(l2t_model, "seed", None),
        }

    def dump_logs(self, iteration: int = 0) -> None:
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

        if len(self.rewbuffer) > 1:
            self.logger.record(
                "Episode/average_episodic_reward", statistics.mean(self.rewbuffer)
            )
            self.logger.record(
                "Episode/average_episodic_length", statistics.mean(self.lenbuffer)
            )
            self.logger.record(
                "Episode/max_episodic_length", th.max(self.cur_episode_length).item()
            )
            self.logger.record(
                "Episode/max_episodic_reward",
                th.max(self.cur_reward_sum).item(),  # type: ignore
            )
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:
        total_timesteps, callback = super()._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )
        self._last_episode_starts = th.ones((self.env.num_envs,), dtype=th.bool)
        # store the current number of timesteps
        self.rewbuffer = deque(maxlen=self._stats_window_size)
        self.lenbuffer = deque(maxlen=self._stats_window_size)
        self.cur_reward_sum = th.zeros(
            self.env.num_envs,  # type: ignore
            dtype=th.float,
            device=self.device,  # type: ignore
        )
        self.cur_episode_length = th.zeros(
            self.env.num_envs,  # type: ignore
            dtype=th.float,
            device=self.device,  # type: ignore
        )
        return total_timesteps, callback
