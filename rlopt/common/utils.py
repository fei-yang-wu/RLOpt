from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EventCallback,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecEnvWrapper,
    VecMonitor,
    is_vecenv_wrapped,
    sync_envs_normalization,
)
from torch import nn


def obs_as_tensor(
    obs: Union[th.Tensor, np.ndarray, Dict[str, np.ndarray], Any], device: th.device
) -> Union[th.Tensor, TensorDict]:
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray) or isinstance(obs, th.Tensor):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


# From stable baselines
def explained_variance(
    y_pred: Union[np.ndarray, th.Tensor], y_true: Union[np.ndarray, th.Tensor]
) -> Union[float, np.ndarray, th.Tensor]:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    if isinstance(y_pred, np.ndarray):
        var_y = np.var(y_true)
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y  # type: ignore
    elif isinstance(y_pred, th.Tensor) and isinstance(y_true, th.Tensor):
        var_y = th.var(y_true).item()
        return np.nan if var_y == 0 else 1 - th.var(y_true - y_pred).item() / var_y
    else:
        raise ValueError(
            "y_pred and y_true must be of the same type (np.ndarray or th.Tensor)"
        )


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


# from rsl_rl
# @th.compile
def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    """

    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = th.cat(
        (flat_dones.new_tensor([-1], dtype=th.int64), flat_dones.nonzero()[:, 0])
    )
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = th.split(
        tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list
    )
    # add at least one full length trajectory
    trajectories = trajectories + (
        th.zeros(tensor.shape[0], tensor.shape[-1], device=tensor.device),
    )
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = th.nn.utils.rnn.pad_sequence(trajectories)  # type: ignore
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > th.arange(
        0, tensor.shape[0], device=tensor.device
    ).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def swap_and_flatten(arr: th.Tensor) -> th.Tensor:
    """
    Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
    to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
    to [n_steps * n_envs, ...] (which maintain the order)

    :param arr:
    :return:
    """
    shape = arr.shape
    return arr.reshape(shape[0] * shape[1], -1)


class ParallelEnvFlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space: The observation space of the environment
    """

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # return self.flatten(observations)
        return observations


class OnnxableOnPolicy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)


class OnnxableOffPolicy(th.nn.Module):
    def __init__(self, actor: th.nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, observation: th.Tensor) -> th.Tensor:
        # NOTE: You may have to postprocess (unnormalize) actions
        # to the correct bounds (see commented code below)
        return self.actor(observation, deterministic=True)


def export_to_onnx(
    model: Union[BasePolicy, th.nn.Module],
    onnx_filename: str,
    input_shape: Optional[Tuple[int, ...]] = None,
    export_params: bool = True,
    verbose: int = 0,
) -> None:
    """
    Export a model to ONNX format.

    :param model: The model to export
    :param env: The environment to use to get the observation space
    :param onnx_filename: The name of the output onnx file
    :param use_new_zipfile_serialization: Whether to use the new zipfile serialization or not
    :param input_shape: The shape of the input tensor
    :param input_tensor: The input tensor to use
    :param export_params: Whether to export the parameters of the model
    :param verbose: The verbosity level
    """
    if isinstance(model, BasePolicy):
        model = OnnxableOnPolicy(model)
    elif isinstance(model, th.nn.Module):
        model = OnnxableOffPolicy(model)
    else:
        raise ValueError("model must be an instance of BasePolicy or th.nn.Module")

    if input_shape is None:
        raise ValueError("input_shape must be provided")
    input_tensor = th.zeros(1, *input_shape, dtype=th.float32)  # type: ignore

    # Export the model
    th.onnx.export(
        model,
        (input_tensor,),
        f=onnx_filename,
        export_params=export_params,
        verbose=None,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    if verbose > 0:
        print(f"Model exported in {onnx_filename}")


class OnnxCheckpointCallback(CheckpointCallback):
    """Overwrite CheckpointCallback to export the model to ONNX format.

    Args:
        CheckpointCallback (_type_): _description_
    """

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)

            # Export the model to ONNX
            onnx_filename = self._checkpoint_path(extension="onnx")
            self.model.export_onnx_policy(path=onnx_filename)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if (
                self.save_replay_buffer
                and hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
            ):
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path(
                    "replay_buffer_", extension="pkl"
                )
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(
                        f"Saving model replay buffer checkpoint to {replay_buffer_path}"
                    )

            if (
                self.save_vecnormalize
                and self.model.get_vec_normalize_env() is not None
            ):
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path(
                    "vecnormalize_", extension="pkl"
                )
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True


def _orthogonal_init_(module: nn.Module, gain: float = 1.0) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
