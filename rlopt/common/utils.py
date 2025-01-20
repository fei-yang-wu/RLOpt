from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Callable
import os
import warnings

import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import (
    VecMonitor,
    VecEnvWrapper,
    SubprocVecEnv,
    DummyVecEnv,
    VecEnv,
    sync_envs_normalization,
    is_vecenv_wrapped,
)
import wandb

from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
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
    padded_trajectories = th.nn.utils.rnn.pad_sequence(trajectories)
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
    env: VecEnv,
    onnx_filename: str,
    use_new_zipfile_serialization: bool = False,
    input_shape: Optional[Tuple[int, ...]] = None,
    input_tensor: Optional[th.Tensor] = None,
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
        if input_tensor is not None:
            input_shape = input_tensor.shape
        else:
            input_shape = env.observation_space.shape
    if input_tensor is None:
        input_tensor = th.zeros((1,) + input_shape, dtype=th.float32)

    # Export the model
    th.onnx.export(
        model,
        input_tensor,
        onnx_filename,
        export_params=export_params,
        verbose=verbose,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    if verbose > 0:
        print(f"Model exported in {onnx_filename}")
