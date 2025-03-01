import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Tuple, List
from collections import deque
import time
import statistics
import pathlib
import io

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from copy import deepcopy

from stable_baselines3.common.buffers import RolloutBuffer, BaseBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)


from sb3_contrib.common.recurrent.type_aliases import RNNStates  # type: ignore
from stable_baselines3.common.save_util import (
    load_from_zip_file,
    recursive_getattr,
    recursive_setattr,
    save_to_zip_file,
)
from stable_baselines3.common.utils import get_system_info
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    TensorDict,
)
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate
from stable_baselines3.common import utils
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import (
    get_device,
)
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecNormalize,
    unwrap_vec_normalize,
)

from rlopt.common.buffer import RLOptDictRecurrentReplayBuffer
from rlopt.common.utils import (
    obs_as_tensor,
    unpad_trajectories,
    swap_and_flatten,
    export_to_onnx,
)
from rlopt.agent.l2t.policies import (
    MlpLstmPolicy,
    CnnLstmPolicy,
    MultiInputLstmPolicy,
    RecurrentActorCriticPolicy,
)


SelfRecurrentStudent = TypeVar("SelfRecurrentStudent", bound="RecurrentStudent")


class RslExpertRecurrentStudent(OnPolicyAlgorithm):
    """
    L2T (Learn to Teach) is a reinforcement learning algorithm that learns to teach a student agent.
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param whole_squences: Whether to use the whole sequence or not
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
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }
    student_policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        student_policy: Union[
            str, Type[RecurrentActorCriticPolicy]
        ] = RecurrentActorCriticPolicy,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        whole_sequences: bool = True,
        n_epochs: int = 3,
        n_batches: int = 5,
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
        rollout_buffer_class: Optional[
            type[RLOptDictRecurrentReplayBuffer]
        ] = RLOptDictRecurrentReplayBuffer,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        mixture_coeff: float = 0.0,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        student_policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        teacher_policy: Optional[th.nn.Module] = None,
    ):
        # if isinstance(policy, str):
        #     self.policy_class = self._get_policy_from_name(policy)
        # else:
        #     self.policy_class = policy

        self.device = get_device(device)
        if verbose >= 1:
            print(f"Using {self.device} device")

        self.verbose = verbose

        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.action_noise: Optional[ActionNoise] = None
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self._last_obs = (  # type: ignore
            None
        )  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs = (
            None
        )  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._episode_num = 0
        # Used for gSDE only
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        # Define number of epoch and batch size for batch generator
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        # Buffers for logging
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # Whether the user passed a custom logger or not
        self._custom_logger = False
        self.env: Optional[VecEnv] = None
        self._vec_normalize_env: Optional[VecNormalize] = None
        supported_action_spaces = (
            spaces.Box,
            spaces.Discrete,
            spaces.MultiDiscrete,
            spaces.MultiBinary,
        )
        support_multi_env = True
        self.whole_sequences = whole_sequences
        # Create and wrap the env if needed
        if env is not None:
            env = maybe_make_env(env, self.verbose)
            env = self._wrap_env(env, self.verbose, True)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            # get VecNormalize object if needed
            self._vec_normalize_env = unwrap_vec_normalize(env)

            if supported_action_spaces is not None:
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires "
                    "a single vectorized environment."
                )

            if self.use_sde and not isinstance(self.action_space, spaces.Box):
                raise ValueError(
                    "generalized State-Dependent Exploration (gSDE) can only be used with continuous actions."
                )

            if isinstance(self.action_space, spaces.Box):
                assert np.all(
                    np.isfinite(
                        np.array([self.action_space.low, self.action_space.high])
                    )
                ), "Continuous action space must have a finite lower and upper bound"

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = th.tensor(ent_coef, device=self.device)
        self.vf_coef = th.tensor(vf_coef, device=self.device)
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.student_policy = student_policy  # type: ignore
        self.mixture_coeff = mixture_coeff

        # self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.student_policy_kwargs = (
            {} if student_policy_kwargs is None else student_policy_kwargs
        )

        self._last_lstm_states = None

        if _init_setup_model:
            self._setup_model()
        self.teacher = teacher_policy
        self.policy = teacher_policy
        self.compiled_policy = teacher_policy

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer_class = RLOptDictRecurrentReplayBuffer

        # self.policy = self.policy_class(  # type: ignore[assignment]
        #     self.observation_space["teacher"],  # type: ignore
        #     self.action_space,
        #     self.lr_schedule,
        #     use_sde=self.use_sde,
        #     **self.policy_kwargs,
        # )
        # self.policy = self.policy.to(self.device)

        self._init_student_policy(self.student_policy, self.student_policy_kwargs)
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        self.compiled_student_policy = th.compile(self.student_policy)  # type: ignore

    def _init_student_policy(
        self,
        student_policy: Union[
            str, Type[ActorCriticPolicy], ActorCriticPolicy, BasePolicy
        ] = ActorCriticPolicy,
        student_policy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(student_policy, str):
            self.student_policy_class = self._get_policy_from_name(student_policy)
        else:
            self.student_policy_class = student_policy

        print("student policy kwargs", student_policy_kwargs)
        # partial_obversevation_space is from Environment's partial_observation_space
        self.partial_observation_space = self.observation_space["student"]  # type: ignore
        self.student_policy = self.student_policy_class(  # pytype:disable=not-instantiable
            self.partial_observation_space,
            self.action_space,
            self.lr_schedule,
            **student_policy_kwargs,  # pytype:disable=not-instantiable # type: ignore
        )
        self.student_policy: RecurrentActorCriticPolicy
        self.student_policy = self.student_policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.student_policy.lstm_actor

        # if not isinstance(self.student_policy, RecurrentActorCriticPolicy):
        #     raise ValueError("Student policy must subclass RecurrentActorCriticPolicy")

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

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            hidden_state_buffer_shape,
            device=self.device,  # type: ignore[arg-type]
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.rollout_buffer: RLOptDictRecurrentReplayBuffer

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
        self.compiled_policy.eval()
        self.compiled_student_policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        self.ep_infos = []
        # lstm_states = deepcopy(self._last_lstm_states)
        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.compiled_student_policy.reset_noise(env.num_envs)

            # flag the student agent to be used
            student_predicted = False
            with th.inference_mode():
                # prepare for the student agent
                self._last_episode_starts: th.Tensor
                episode_starts = self._last_episode_starts.type(th.float32).to(
                    self.device
                )

                obs_tensor = self._last_obs
                if (
                    th.rand(1)[0]
                    <= self.mixture_coeff  # * (1 - self._current_progress_remaining)
                    and self.num_timesteps > 0
                ):
                    actions, values, log_probs, lstm_states = (
                        self.compiled_student_policy.forward(
                            obs_tensor["student"],
                            self._last_lstm_states,  # type: ignore[arg-type]
                            episode_starts,
                        )
                    )
                    student_predicted = True
                else:
                    actions = self.compiled_policy.act(obs_tensor["teacher"])
                    values = self.compiled_policy.evaluate(obs_tensor["teacher"])
                    log_probs = self.compiled_policy.get_actions_log_prob(actions)

                # get the hidden state of the current student state
                if not student_predicted:
                    lstm_states = self.compiled_student_policy.forward_lstm(
                        obs_tensor["student"],
                        self._last_lstm_states,  # type: ignore[arg-type]
                        episode_starts,
                    )

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if False:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.compiled_policy.unscale_action(
                        clipped_actions
                    )
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = th.clamp(
                        actions,
                        th.as_tensor(self.action_space.low, device=self.device),
                        th.as_tensor(self.action_space.high, device=self.device),
                    )
            time_now = time.time_ns()
            new_obs, rewards, dones, infos = env.step(clipped_actions)  # type: ignore[arg-type]
            self.logger.record("time/step", (time.time_ns() - time_now) / 1e9)

            self.num_timesteps += env.num_envs

            infos: dict
            # Record infos
            if "episode" in infos:
                self.ep_infos.append(infos["episode"])
            elif "log" in infos:
                self.ep_infos.append(infos["log"])

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Bootstrapping on time outs
            if "time_outs" in infos:
                rewards += self.gamma * th.squeeze(
                    values * infos["time_outs"].unsqueeze(1).to(self.device),
                    1,
                )

            self.cur_reward_sum += rewards
            self.cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)  # type: ignore[arg-type]

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                lstm_states=self._last_lstm_states,  # type: ignore[arg-type]
                dones=dones,  # type: ignore[arg-type]
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones  # type: ignore[arg-type]
            self._last_lstm_states = lstm_states

            # record reward and episode length
            self.rewbuffer.extend(
                self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
            )
            self.lenbuffer.extend(
                self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
            )
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 1

        with th.inference_mode():
            # Compute value for the last timestep
            values = self.compiled_policy.evaluate(obs_as_tensor(new_obs["teacher"], self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _update_learning_rate(
        self,
        optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer],
        lr: Optional[float] = None,
    ) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record(
            "train/learning_rate", self.lr_schedule(self._current_progress_remaining)
        )

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if lr is not None:
            for optimizer in optimizers:
                update_learning_rate(optimizer, lr)
        else:
            for optimizer in optimizers:
                update_learning_rate(
                    optimizer, self.lr_schedule(self._current_progress_remaining)
                )

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.compiled_student_policy: RecurrentActorCriticPolicy
        self.compiled_policy.eval()
        self.compiled_student_policy.set_training_mode(True)
        # # Update optimizer learning rate
        # self._update_learning_rate(
        #     [self.policy.optimizer, self.student_policy.optimizer]
        # )
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range = th.tensor(clip_range).to(self.device)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
            clip_range_vf = th.tensor(clip_range_vf).to(self.device)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        student_losses = []
        # student_policy_losses = []

        continue_training = True

        generator = self.rollout_buffer.get_generator(
            num_mini_batches=self.n_batches, num_epochs=self.n_epochs
        )
        # train for n_epochs epochs
        for (  # type: ignore[arg-type]
            obs_batch,
            actions_batch,
            values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            masks_batch,
            hidden_batch,
        ) in generator:
            # # Do a complete pass on the rollout buffer
            # for rollout_data in self.rollout_buffer.get(self.batch_size):

            actions = actions_batch
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions_batch.long().flatten()

            # Convert mask from float to bool
            mask = masks_batch > 1e-8  # type: ignore[operator]

            if self.whole_sequences:
                actions = actions["teacher"]  # type: ignore[arg-type]
                observations = obs_batch["teacher"]

            # # teacher is mlp so no funny business
            # values, log_prob, entropy = self.compiled_policy.evaluate_actions(
            #     observations, actions  # type: ignore
            # )

            # print(actions.shape)
            values = self.compiled_policy.evaluate(observations)
            # log_prob = self.compiled_policy.get_actions_log_prob(actions)

            values = values.flatten()

            # Normalize advantage
            advantages = advantages_batch
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            # ratio between old and new policy, should be one at the first iteration
            # ratio = th.exp(log_prob - old_actions_log_prob_batch)

            # # Logging
            # pg_losses.append(policy_loss.item())
            # clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(0.0)

            student_observations = obs_batch["student"]
            # student obs shape is (T, B, feature dim)
            (
                _,
                student_action,
                student_entropy,
                student_log_prob,
                mu,
                sigma,
            ) = self.compiled_student_policy.predict_whole_sequence(
                obs=student_observations,
                deterministic=False,
                lstm_states=hidden_batch,
            )

            # align dim with the teacher action
            student_action = BaseBuffer.swap_and_flatten(
                unpad_trajectories(student_action, mask)
            )

            student_log_prob = BaseBuffer.swap_and_flatten(
                unpad_trajectories(student_log_prob, mask)
            )

            student_ratio = th.exp(student_log_prob - old_actions_log_prob_batch)

            # clipped asym loss
            student_asym_loss_1 = advantages * student_ratio
            student_asym_loss_2 = advantages * th.clamp(
                student_ratio, 1 - clip_range, 1 + clip_range
            )
            student_asym_loss = -th.min(student_asym_loss_1, student_asym_loss_2).mean()
            student_asym_loss = -th.mean(
                advantages * th.clamp(student_ratio, 1 - clip_range, 1 + clip_range)
            ).mean()
            teacher_action = actions.detach()

            student_loss = (
                F.mse_loss(student_action, teacher_action)  # type: ignore
                # + student_asym_loss
            )

            student_losses.append(student_loss.item())

            # Update student agent
            self.compiled_student_policy.optimizer.zero_grad()
            student_loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.compiled_student_policy.parameters(), self.max_grad_norm
            )
            self.compiled_student_policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

            self._update_learning_rate([self.student_policy.optimizer])

        # explained_var = explained_variance(
        #     self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        # )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        # self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.compiled_policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.compiled_policy.log_std).mean().item()
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range.item())
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/student_loss", np.mean(student_losses))
        # self.logger.record(
        #     "train/student_policy_loss", statistics.mean(student_policy_losses)
        # )

    def learn(
        self: SelfRecurrentStudent,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentStudent:
        # return super().learn(
        #     total_timesteps=total_timesteps,
        #     callback=callback,
        #     log_interval=log_interval,
        #     tb_log_name=tb_log_name,
        #     reset_num_timesteps=reset_num_timesteps,
        #     progress_bar=progress_bar,
        # )
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            collection_start = time.time_ns()
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps  # type: ignore
            )
            collection_end = time.time_ns()
            collection_time = (collection_end - collection_start) / 1e9

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            training_start = time.time_ns()
            self.train()
            training_end = time.time_ns()
            training_time = (training_end - training_start) / 1e9

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration, locals())

        callback.on_training_end()

        return self

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
        ]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = [
            "policy",
            "student_policy",
        ]

        return state_dicts, []

    def student_predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.compiled_student_policy.predict(  # type: ignore
            observation["student"], state, episode_start, deterministic  # type: ignore
        )

    def student_predict_and_return_tensor(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.compiled_student_policy.predict_and_return_tensor(  # type: ignore
            observation["student"], state, episode_start, deterministic  # type: ignore
        )

    def teacher_predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.compiled_policy.predict(
            observation["teacher"], state, episode_start, deterministic
        )

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.compiled_policy.predict(
            observation["teacher"], state, episode_start, deterministic
        )

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
        self.rewbuffer = deque(maxlen=self._stats_window_size)
        self.lenbuffer = deque(maxlen=self._stats_window_size)
        self.cur_reward_sum = th.zeros(
            self.env.num_envs, dtype=th.float, device=self.device  # type: ignore
        )
        self.cur_episode_length = th.zeros(
            self.env.num_envs, dtype=th.float, device=self.device  # type: ignore
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

        # print(self.policy)

        print(self.student_policy)

        return total_timesteps, callback

    def _dump_logs(self, iteration: int, locs: dict) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        :param locs: Local variables
        """
        iteration_time = locs["training_end"] - locs["collection_start"]

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
        fps = int(
            self.n_steps
            * self.env.num_envs  # type: ignore
            / (locs["collection_time"] + locs["training_time"])
        )
        self.logger.record("time/fps", fps)
        self.logger.record("time/iteration_time (s)", iteration_time / 1e9)
        self.logger.record(
            "time/collection time per step (s)", locs["collection_time"] / self.n_steps
        )
        self.logger.record("time/training_time (s)", locs["training_time"])
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
                "Episode/max_episodic_reward", th.max(self.cur_reward_sum).item()
            )
        self.logger.dump(step=self.num_timesteps)

    def inference(self):
        # optimize the model for inference
        self.compiled_student_policy = th.compile(self.student_policy)  # type: ignore

    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfRecurrentStudent],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfRecurrentStudent:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )
        # print("data", data)

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if (
                "net_arch" in data["policy_kwargs"]
                and len(data["policy_kwargs"]["net_arch"]) > 0
            ):
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(
                    saved_net_arch[0], dict
                ):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if (
            "policy_kwargs" in kwargs
            and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify new environments"
            )

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            # check_for_correct_spaces(
            #     env, data["observation_space"], data["action_space"]
            # )
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy="MlpPolicy",
            env=env,  # type: ignore
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=False, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(
                e
            ) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            if name == "policy":
                continue
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])  # type: ignore[arg-type]
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )


# @th.jit.script
# def compute_ppo_loss(
#     advantages,
#     values_pred,
#     log_prob,
#     old_log_prob,
#     clip_range,
#     returns,
#     ent_coef,
#     vf_coef,
# ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

#     # ratio between old and new policy, should be one at the first iteration
#     ratio = th.exp(log_prob - old_log_prob)

#     # clipped surrogate loss
#     policy_loss_1 = advantages * ratio
#     policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
#     policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

#     # Logging
#     clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float())

#     # Value loss using the TD(gae_lambda) target
#     value_loss = F.mse_loss(returns, values_pred)

#     # Entropy loss favor exploration
#     # Approximate entropy when no analytical form
#     entropy_loss = -th.mean(-log_prob)

#     loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

#     # return {
#     #     "policy_loss": policy_loss,
#     #     "entropy_loss": entropy_loss,
#     #     "value_loss": value_loss,
#     #     "clip_fraction": clip_fraction,
#     #     "total_loss": loss,
#     # }
#     return policy_loss, entropy_loss, value_loss, clip_fraction, loss
