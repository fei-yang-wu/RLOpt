# The implementation is borrowed from Stable Baselines 3 [https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py]
import warnings
from functools import partial
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from sb3_contrib.common.recurrent.type_aliases import (
    RecurrentDictRolloutBufferSamples,
    RecurrentRolloutBufferSamples,
    RNNStates,
)

from rlopt.common.type_aliases import (
    RecurrentRolloutBufferSequenceSamples,
    RecurrentDictRolloutBufferSequenceSamples,
)

from rlopt.common.utils import split_and_pad_trajectories, unpad_trajectories

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from .type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(
        arr: Union[th.Tensor, TensorDict]
    ) -> Union[th.Tensor, TensorDict]:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.transpose(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = th.randint(0, upper_bound, size=(batch_size,), device=self.device)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: th.Tensor, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return array.clone().detach()
        return array.detach()

    @staticmethod
    def _normalize_obs(
        obs: Union[th.Tensor, Dict[str, th.Tensor]],
        env: Optional[VecNormalize] = None,
    ) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        if env is not None:
            return env.normalize_obs(obs)  # type: ignore
        return obs

    @staticmethod
    def _normalize_reward(
        reward: th.Tensor, env: Optional[VecNormalize] = None
    ) -> th.Tensor:
        if env is not None:
            return env.normalize_reward(reward).astype(th.float32)  # type: ignore
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: th.Tensor
    next_observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    timeouts: th.Tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = th.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=th.float32,
            device=self.device,
        )  # type: ignore

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = th.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape),
                dtype=th.float32,
                device=self.device,
            )  # type: ignore

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )  # type: ignore

        self.rewards = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.dones = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: th.Tensor,
        next_obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        done: th.Tensor,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = th.Tensor(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = th.Tensor(next_obs)
        else:
            self.next_observations[self.pos] = th.Tensor(next_obs)

        self.actions[self.pos] = th.Tensor(action)
        self.rewards[self.pos] = th.Tensor(reward)
        self.dones[self.pos] = th.Tensor(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = th.Tensor(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)  # type: ignore
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                th.randint(1, self.buffer_size, size=(batch_size,))
                .add(self.pos)
                .fmod(self.buffer_size)
            )
        else:
            batch_inds = th.randint(0, self.pos, size=(batch_size,))
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = th.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))  # type: ignore

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:  # type: ignore
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.
        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    episode_starts: th.Tensor
    log_probs: th.Tensor
    values: th.Tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = th.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=th.float32,
            device=self.device,
        )
        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )
        self.rewards = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.returns = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.episode_starts = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.values = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.log_probs = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.advantages = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: th.Tensor
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.
        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.
        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.
        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.detach().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones  # .type(th.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        self.observations[self.pos] = obs.detach()
        self.actions[self.pos] = action.detach()
        self.rewards[self.pos] = reward.detach()
        self.episode_starts[self.pos] = episode_start.detach()
        self.values[self.pos] = value.flatten().detach()
        self.log_probs[self.pos] = log_prob.detach()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:

        assert self.full, ""
        indices = th.randperm(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "rewards",
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

    def _get_samples(
        self,
        batch_inds: th.Tensor,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:

        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.rewards[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*data)


class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: Dict[str, th.Tensor]  # type: ignore[assignment]
    next_observations: Dict[str, th.Tensor]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert isinstance(
            self.obs_shape, dict
        ), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert (
            not optimize_memory_usage
        ), "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: th.zeros(
                (self.buffer_size, self.n_envs, *_obs_shape),
                dtype=th.float32,
                device=self.device,
            )
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: th.zeros(
                (self.buffer_size, self.n_envs, *_obs_shape),
                dtype=th.float32,
                device=self.device,
            )
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )
        self.rewards = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )
        self.dones = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.element_size() * obs.numel()

            total_memory_usage: float = (
                obs_nbytes
                + self.actions.element_size() * self.actions.numel()
                + self.rewards.element_size() * self.rewards.numel()
                + self.dones.element_size() * self.dones.numel()
            )
            if not optimize_memory_usage:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.element_size() * obs.numel()
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(  # type: ignore[override]
        self,
        obs: Dict[str, th.Tensor],
        next_obs: Dict[str, th.Tensor],
        action: th.Tensor,
        reward: th.Tensor,
        done: th.Tensor,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs[key].clone().detach().to(self.device)

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape(
                    (self.n_envs,) + self.obs_shape[key]
                )
            self.next_observations[key][self.pos] = (
                next_obs[key].clone().detach().to(self.device)
            )

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = action.clone().detach().to(self.device)
        self.rewards[self.pos] = reward.clone().detach().to(self.device)
        self.dones[self.pos] = done.clone().detach().to(self.device)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = th.tensor(
                [info.get("TimeLimit.truncated", False) for info in infos],
                device=self.device,
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)  # type: ignore

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: th.Tensor,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = th.randint(
            0, high=self.n_envs, size=(len(batch_inds),), device=self.device
        )

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs(
            {
                key: obs[batch_inds, env_indices, :]
                for key, obs in self.observations.items()
            },
            env,
        )
        next_obs_ = self._normalize_obs(
            {
                key: obs[batch_inds, env_indices, :]
                for key, obs in self.next_observations.items()
            },
            env,
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {
            key: obs.clone().detach().to(self.device) for key, obs in obs_.items()
        }
        next_observations = {
            key: obs.clone().detach().to(self.device) for key, obs in next_obs_.items()
        }

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.actions[batch_inds, env_indices]
            .clone()
            .detach()
            .to(self.device),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=(
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            )
            .reshape(-1, 1)
            .clone()
            .detach()
            .to(self.device),
            rewards=self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            )
            .clone()
            .detach()
            .to(self.device),
        )


class DictRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: TensorDict  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert isinstance(
            self.obs_shape, dict
        ), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        # self.observations_ = dict()
        # for key, obs_input_shape in self.obs_shape.items():
        #     self.observations_[key] = th.zeros(
        #         (self.buffer_size, self.n_envs, *obs_input_shape), dtype=th.float32
        #     )

        self.observations = TensorDict(
            {
                key: th.zeros(
                    (self.buffer_size, self.n_envs, *obs_input_shape),
                    dtype=th.float32,
                    device=self.device,
                )
                for key, obs_input_shape in self.obs_shape.items()
            },
            batch_size=[self.buffer_size, self.n_envs],
        )

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )

        self.rewards = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.returns = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.episode_starts = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.values = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.log_probs = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.advantages = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
        self,
        obs: Dict[str, th.Tensor],
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = obs[key].detach()
            # Reshape needed when using multiple envs with discrete observations
            # as torch cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = action.detach()
        self.rewards[self.pos] = reward.detach()
        self.episode_starts[self.pos] = episode_start.detach()
        self.values[self.pos] = value.detach().flatten()
        self.log_probs[self.pos] = log_prob.detach()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = th.randperm(self.buffer_size * self.n_envs)
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
    ) -> DictRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.rewards[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return DictRolloutBufferSamples(*data)


def pad(
    seq_start_indices: Union[np.ndarray, th.Tensor],
    seq_end_indices: Union[np.ndarray, th.Tensor],
    device: th.device,
    tensor: Union[np.ndarray, th.Tensor],
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Chunk sequences and pad them to have constant dimensions.

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device
    :param tensor: Tensor of shape (batch_size, *tensor_shape)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq, max_length, *tensor_shape)
    """
    if isinstance(tensor, th.Tensor):
        # Convert seq_start_indices and seq_end_indices to numpy arrays
        seq_start_indices = seq_start_indices
        seq_end_indices = seq_end_indices
        seq_len = seq_start_indices.shape[0]
        # Create sequences given start and end
        seq = [
            tensor[seq_start_indices[i].item() : seq_end_indices[i].item() + 1].detach()
            for i in range(seq_len)
        ]
        return th.nn.utils.rnn.pad_sequence(
            seq, batch_first=True, padding_value=padding_value
        )
    elif isinstance(tensor, np.ndarray):
        # Convert seq_start_indices and seq_end_indices to numpy arrays
        seq_start_indices = seq_start_indices.cpu().numpy()
        seq_end_indices = seq_end_indices.cpu().numpy()

        # Create sequences given start and end
        seq = [
            th.tensor(tensor[start : end + 1], device=device)
            for start, end in zip(seq_start_indices, seq_end_indices)
        ]
        return th.nn.utils.rnn.pad_sequence(
            seq, batch_first=True, padding_value=padding_value
        )


def pad_and_flatten(
    seq_start_indices: Union[np.ndarray, th.Tensor],
    seq_end_indices: Union[np.ndarray, th.Tensor],
    device: th.device,
    tensor: Union[np.ndarray, th.Tensor],
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Pad and flatten the sequences of scalar values,
    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device (cpu, gpu, ...)
    :param tensor: Tensor of shape (max_length, n_seq, 1)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad(
        seq_start_indices, seq_end_indices, device, tensor, padding_value
    ).flatten()


def create_sequencers(
    episode_starts: Union[np.ndarray, th.Tensor],
    env_change: Union[np.ndarray, th.Tensor],
    device: th.device,
) -> Tuple[np.ndarray, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :param device: PyTorch device
    :return: Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = th.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    if isinstance(episode_starts, np.ndarray):
        # Retrieve indices of sequence starts
        seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
        # End of sequence are just before sequence starts
        # Last index is also always end of a sequence
        seq_end_indices = np.concatenate(
            [(seq_start_indices - 1)[1:], np.array([len(episode_starts)])]
        )
    elif isinstance(episode_starts, th.Tensor):
        # Retrieve indices of sequence starts
        seq_start_indices = th.where(seq_start == True)[0]  # noqa: E712
        # End of sequence are just before sequence starts
        # Last index is also always end of a sequence
        seq_end_indices = th.cat(
            [
                (seq_start_indices - 1)[1:],
                th.Tensor(np.array([len(episode_starts)])).to(device),
            ]
        ).type(th.int)
    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(
        pad_and_flatten, seq_start_indices, seq_end_indices, device
    )
    return seq_start_indices, local_pad, local_pad_and_flatten


class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
        (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

    def reset(self):
        super().reset()
        self.hidden_states_pi = th.zeros(
            self.hidden_state_shape, dtype=np.float32, device=self.device
        )
        self.cell_states_pi = th.zeros(
            self.hidden_state_shape, dtype=np.float32, device=self.device
        )
        self.hidden_states_vf = th.zeros(
            self.hidden_state_shape, dtype=np.float32, device=self.device
        )
        self.cell_states_vf = th.zeros(
            self.hidden_state_shape, dtype=np.float32, device=self.device
        )

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = lstm_states.pi[0].detach()
        self.cell_states_pi[self.pos] = lstm_states.pi[1].detach()
        self.hidden_states_vf[self.pos] = lstm_states.vf[0].detach()
        self.cell_states_vf[self.pos] = lstm_states.vf[1].detach()

        super().add(*args, **kwargs)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in [
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
            ]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = th.randint(self.buffer_size * self.n_envs, device=self.device)
        indices = th.arange(self.buffer_size * self.n_envs, device=self.device)
        indices = th.cat((indices[split_index:], indices[:split_index]))

        env_change = th.zeros(
            self.buffer_size * self.n_envs, device=self.device
        ).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: th.Tensor,
        env_change: th.Tensor,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states_pi = (
            # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (
            lstm_states_pi[0].contiguous(),
            lstm_states_pi[1].contiguous(),
        )
        lstm_states_vf = (
            lstm_states_vf[0].contiguous(),
            lstm_states_vf[1].contiguous(),
        )

        return RecurrentRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.observations[batch_inds]).reshape(
                (padded_batch_size, *self.obs_shape)
            ),
            actions=self.pad(self.actions[batch_inds]).reshape(
                (padded_batch_size,) + self.actions.shape[1:]
            ),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(th.ones_like(self.returns[batch_inds])),
        )


class RecurrentDictRolloutBuffer(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RecurrentRolloutBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs=n_envs,
        )

    def reset(self):
        super().reset()
        self.hidden_states_pi = th.zeros(
            self.hidden_state_shape, dtype=th.float32, device=self.device
        )
        self.cell_states_pi = th.zeros(
            self.hidden_state_shape, dtype=th.float32, device=self.device
        )
        self.hidden_states_vf = th.zeros(
            self.hidden_state_shape, dtype=th.float32, device=self.device
        )
        self.cell_states_vf = th.zeros(
            self.hidden_state_shape, dtype=th.float32, device=self.device
        )

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = lstm_states.pi[0].detach()
        self.cell_states_pi[self.pos] = lstm_states.pi[1].detach()
        self.hidden_states_vf[self.pos] = lstm_states.vf[0].detach()
        self.cell_states_vf[self.pos] = lstm_states.vf[1].detach()

        super().add(*args, **kwargs)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RecurrentDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in [
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
            ]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            self.observations = self.swap_and_flatten(self.observations)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = th.zeros(
            self.buffer_size * self.n_envs, device=self.device
        ).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: th.Tensor,
        env_change: th.Tensor,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentDictRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states_pi = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (
            lstm_states_pi[0].contiguous(),
            lstm_states_pi[1].contiguous(),
        )
        lstm_states_vf = (
            lstm_states_vf[0].contiguous(),
            lstm_states_vf[1].contiguous(),
        )
        self.observations: TensorDict

        observations = {
            key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()
        }

        observations = {
            key: obs.reshape((padded_batch_size,) + self.obs_shape[key])
            for (key, obs) in observations.items()
        }

        return RecurrentDictRolloutBufferSamples(
            observations=observations,
            actions=self.pad(self.actions[batch_inds]).reshape(
                (padded_batch_size,) + self.actions.shape[1:]
            ),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(th.ones_like(self.returns[batch_inds])),
        )


# utility function for creating RecurrentSequenceRolloutBuffer
def create_sequence_slicer(
    episode_start_indices: np.ndarray, device: Union[th.device, str]
) -> Callable[[Union[np.ndarray, th.Tensor], List[int]], th.Tensor]:
    def create_sequence_minibatch(
        tensor: np.ndarray | th.Tensor, seq_indices: List[int]
    ) -> th.Tensor:
        """
        Create minibatch of whole sequence.

        :param tensor: Tensor that will be sliced (e.g. observations, rewards)
        :param seq_indices: Sequences to be used.
        :return: (max_sequence_length, batch_size=n_seq, features_size)
        """
        if isinstance(tensor, np.ndarray):
            tensor = th.tensor(tensor, device=device)

        return pad_sequence(
            [
                tensor[episode_start_indices[i] : episode_start_indices[i + 1]].to(
                    device=device
                )
                for i in seq_indices
            ]
        )

    return create_sequence_minibatch


class RecurrentSequenceRolloutBuffer(RecurrentRolloutBuffer):
    """
    Sequence Rollout buffer used in on-policy algorithms like A2C/PPO.
    Overrides the RecurrentRolloutBuffer to yield 3d batches of whole sequences

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            hidden_state_shape,
            device,
            gae_lambda,
            gamma,
            n_envs=n_envs,
        )

    def get(
        self, batch_size: int
    ) -> Generator[RecurrentRolloutBufferSequenceSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        # Prepare the data
        if not self.generator_ready:
            self.episode_starts[0, :] = 1
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

            self.episode_start_indices = np.where(self.episode_starts == 1)[0]
            self.generator_ready = True

        random_indices = SubsetRandomSampler(range(len(self.episode_start_indices)))
        # Do not drop last batch so we are sure we sample at least one sequence
        # TODO: allow to change that parameter
        batch_sampler = BatchSampler(random_indices, batch_size, drop_last=False)
        # add a dummy index to make the code below simpler
        episode_start_indices = np.concatenate(
            [self.episode_start_indices, np.array([len(self.episode_starts)])]
        )

        create_minibatch = create_sequence_slicer(episode_start_indices, self.device)

        # yields batches of whole sequences, shape: (max_sequence_length, batch_size=n_seq, features_size))
        for indices in batch_sampler:
            returns_batch = create_minibatch(self.returns, indices)
            masks_batch = pad_sequence(
                [th.ones_like(returns) for returns in th.swapaxes(returns_batch, 0, 1)]
            )

            yield RecurrentRolloutBufferSequenceSamples(
                observations=create_minibatch(self.observations, indices),
                actions=create_minibatch(self.actions, indices),
                old_values=create_minibatch(self.values, indices),
                old_log_prob=create_minibatch(self.log_probs, indices),
                advantages=create_minibatch(self.advantages, indices),
                returns=returns_batch,
                mask=masks_batch,
            )


class RecurrentSequenceDictRolloutBuffer(RecurrentDictRolloutBuffer):
    """
    Sequence Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Overrides the DictRecurrentRolloutBuffer to yield 3d batches of whole sequences

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            hidden_state_shape,
            device,
            gae_lambda,
            gamma,
            n_envs=n_envs,
        )

    def get(
        self, batch_size: int
    ) -> Generator[RecurrentDictRolloutBufferSequenceSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        # Prepare the data
        if not self.generator_ready:
            self.episode_starts[0, :] = 1
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

            self.episode_start_indices = np.where(self.episode_starts == 1)[0]
            self.generator_ready = True

        random_indices = SubsetRandomSampler(range(len(self.episode_start_indices)))
        # drop last batch to prevent extremely small batches causing spurious updates
        batch_sampler = BatchSampler(random_indices, batch_size, drop_last=True)
        # add a dummy index to make the code below simpler
        episode_start_indices = np.concatenate(
            [self.episode_start_indices, np.array([len(self.episode_starts)])]
        )

        create_minibatch = create_sequence_slicer(episode_start_indices, self.device)

        # yields batches of whole sequences, shape: (sequence_length, batch_size=n_seq, features_size)
        for indices in batch_sampler:
            obs_batch = {}
            for key in self.observations:
                obs_batch[key] = create_minibatch(self.observations[key], indices)
            returns_batch = create_minibatch(self.returns, indices)
            masks_batch = pad_sequence(
                [th.ones_like(returns) for returns in th.swapaxes(returns_batch, 0, 1)]
            )

            yield RecurrentDictRolloutBufferSequenceSamples(
                observations=obs_batch,
                actions=create_minibatch(self.actions, indices),
                old_values=create_minibatch(self.values, indices),
                old_log_prob=create_minibatch(self.log_probs, indices),
                advantages=create_minibatch(self.advantages, indices),
                returns=returns_batch,
                mask=masks_batch,
            )


class RLOptDictRecurrentReplayBuffer(ABC):
    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: TensorDict  # type: ignore[assignment]
    actions: th.Tensor
    rewards: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    episode_starts: th.Tensor
    log_probs: th.Tensor
    values: th.Tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ) -> None:
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs
        assert isinstance(
            self.obs_shape, dict
        ), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False

        self.observations = TensorDict(
            {
                key: th.zeros(
                    (self.buffer_size, self.n_envs, *obs_input_shape),
                    dtype=th.float32,
                    device=self.device,
                )
                for key, obs_input_shape in self.obs_shape.items()
            },
            batch_size=[self.buffer_size, self.n_envs],
        )

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=th.float32,
            device=self.device,
        )

        self.rewards = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.returns = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.episode_starts = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.values = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.log_probs = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.advantages = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.float32,
            device=self.device,
        )
        self.dones = th.zeros(
            (self.buffer_size, self.n_envs), dtype=th.float32, device=self.device
        )

        self.hidden_states_pi = th.zeros(
            self.hidden_state_shape, dtype=th.float32, device=self.device
        )
        self.cell_states_pi = th.zeros(
            self.hidden_state_shape, dtype=th.float32, device=self.device
        )
        self.hidden_states_vf = th.zeros(
            self.hidden_state_shape, dtype=th.float32, device=self.device
        )
        self.cell_states_vf = th.zeros(
            self.hidden_state_shape, dtype=th.float32, device=self.device
        )

        self.reset()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    @staticmethod
    def _normalize_obs(
        obs: Union[th.Tensor, Dict[str, th.Tensor]],
        env: Optional[VecNormalize] = None,
    ) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        if env is not None:
            return env.normalize_obs(obs)  # type: ignore
        return obs

    @staticmethod
    def _normalize_reward(
        reward: th.Tensor, env: Optional[VecNormalize] = None
    ) -> th.Tensor:
        if env is not None:
            return env.normalize_reward(reward).astype(th.float32)  # type: ignore
        return reward

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: th.Tensor
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.
        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.
        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.
        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.detach().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones  # .type(th.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def reset(self):

        self.generator_ready = False
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: Dict[str, th.Tensor],
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor,
        lstm_states: RNNStates,
        dones: th.Tensor,
    ) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = lstm_states.pi[0].detach()
        self.cell_states_pi[self.pos] = lstm_states.pi[1].detach()
        self.hidden_states_vf[self.pos] = lstm_states.vf[0].detach()
        self.cell_states_vf[self.pos] = lstm_states.vf[1].detach()

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = obs[key].detach()
            # Reshape needed when using multiple envs with discrete observations
            # as torch cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = action.detach()
        self.rewards[self.pos] = reward.detach()
        self.episode_starts[self.pos] = episode_start.detach()
        self.values[self.pos] = value.detach().flatten()
        self.log_probs[self.pos] = log_prob.detach()
        self.dones[self.pos] = dones.detach()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    @th.compile
    def get_generator(
        self, num_mini_batches: int, num_epochs: int = 5
    ) -> Generator[RecurrentDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        padded_student_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
            self.observations["student"], self.dones.unsqueeze(-1)
        )

        # print("padded_student_obs_trajectories", padded_student_obs_trajectories.shape)

        padded_obs_trajectories = {
            "student": padded_student_obs_trajectories,
            "teacher": self.observations["teacher"],
        }

        padded_action, _ = split_and_pad_trajectories(
            self.actions, self.dones.unsqueeze(-1)
        )

        mini_batch_size = self.n_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones
                last_was_done = th.zeros_like(dones, dtype=th.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = th.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]

                obs_batch = {
                    "teacher": BaseBuffer.swap_and_flatten(
                        padded_obs_trajectories["teacher"][:, start:stop]
                    ),
                    "student": padded_obs_trajectories["student"][
                        :, first_traj:last_traj
                    ],
                }
                actions_batch = self.actions[:, start:stop]
                actions_batch = {
                    "teacher": BaseBuffer.swap_and_flatten(actions_batch),
                    "student": padded_action[:, first_traj:last_traj],
                }

                returns_batch = BaseBuffer.swap_and_flatten(
                    self.returns[:, start:stop]
                ).squeeze(-1)
                advantages_batch = BaseBuffer.swap_and_flatten(
                    self.advantages[:, start:stop]
                ).squeeze(-1)
                values_batch = BaseBuffer.swap_and_flatten(
                    self.values[:, start:stop]
                ).squeeze(-1)
                old_actions_log_prob_batch = BaseBuffer.swap_and_flatten(
                    self.log_probs[:, start:stop]
                ).squeeze(-1)

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_batch_hidden_state_pi = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                        first_traj:last_traj
                    ]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in [
                        self.hidden_states_pi,
                        self.cell_states_pi,
                    ]
                ]

                hid_batch_hidden_state_vf = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                        first_traj:last_traj
                    ]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in [
                        self.hidden_states_vf,
                        self.cell_states_vf,
                    ]
                ]

                hid_batch_hidden_state_pi = (
                    hid_batch_hidden_state_pi[0]
                    if len(hid_batch_hidden_state_pi) == 1
                    else hid_batch_hidden_state_pi
                )

                hid_batch_hidden_state_vf = (
                    hid_batch_hidden_state_vf[0]
                    if len(hid_batch_hidden_state_vf) == 1
                    else hid_batch_hidden_state_vf
                )

                hid_batch = RNNStates(
                    pi=(
                        hid_batch_hidden_state_pi[0],
                        hid_batch_hidden_state_pi[1],
                    ),
                    vf=(
                        hid_batch_hidden_state_vf[0],
                        hid_batch_hidden_state_vf[1],
                    ),
                )

                yield obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, masks_batch, hid_batch

                first_traj = last_traj
