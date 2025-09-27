from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from torch import Tensor, nn


def obs_as_tensor(
    obs: th.Tensor | np.ndarray | dict[str, np.ndarray] | Any, device: th.device
) -> th.Tensor | TensorDict:
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray) or isinstance(obs, th.Tensor):
        return th.as_tensor(obs, device=device)
    if isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    raise Exception(f"Unrecognized type of observation {type(obs)}")


# From stable baselines
def explained_variance(
    y_pred: np.ndarray | th.Tensor, y_true: np.ndarray | th.Tensor
) -> float | np.ndarray | th.Tensor:
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
    if isinstance(y_pred, th.Tensor) and isinstance(y_true, th.Tensor):
        var_y = th.var(y_true).item()
        return np.nan if var_y == 0 else 1 - th.var(y_true - y_pred).item() / var_y
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

    def forward(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
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
    model: BasePolicy | th.nn.Module,
    onnx_filename: str,
    input_shape: tuple[int, ...] | None = None,
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
        logger.info(f"Model exported in {onnx_filename}")


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
                logger.info(f"Saving model checkpoint to {model_path}")

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
                    logger.info(
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
                    logger.info(f"Saving model VecNormalize to {vec_normalize_path}")

        return True


def log_info(log_info_dict: dict, metrics_to_log: dict):
    # log all the keys
    for key, value in log_info_dict.items():
        if "/" in key:
            metrics_to_log.update(
                {key: (value.item() if isinstance(value, Tensor) else value)}
            )
        else:
            metrics_to_log.update(
                {
                    "Episode/" + key: (
                        value.item() if isinstance(value, Tensor) else value
                    )
                }
            )


logger = logging.getLogger(__name__)


# -------------------------------
# Model/Agent overview utilities
# -------------------------------
try:
    import torch
    from tensordict.nn import TensorDictModule
    from torch import optim
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    TensorDictModule = nn.Module  # type: ignore[assignment]


def _describe_object(obj: Any) -> str:
    """Compact per-type description for logging."""
    # Tensor
    if torch is not None and isinstance(obj, torch.Tensor):
        return f"Tensor shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}"
    if isinstance(obj, np.ndarray):
        return f"ndarray shape={obj.shape} dtype={obj.dtype}"

    # Gym space
    if hasattr(gym, "Space") and isinstance(obj, gym.Space):
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", None)
        low = getattr(obj, "low", None)
        high = getattr(obj, "high", None)
        rng = ""
        try:
            if low is not None and high is not None and np is not None:
                lo = float(np.min(low))
                hi = float(np.max(high))
                rng = f" range=[{lo:g}, {hi:g}]"
        except Exception:
            pass
        return f"{type(obj).__name__} shape={shape} dtype={dtype}{rng}"

    # Torch/TensorDict modules
    if torch is not None and isinstance(obj, nn.Module):
        total = 0
        trainable = 0
        device: str | None = None
        try:
            for p in obj.parameters():
                n = int(p.numel())
                total += n
                if getattr(p, "requires_grad", False):
                    trainable += n
                if device is None:
                    device = str(getattr(p, "device", None))
        except Exception:
            pass
        extras = []
        if total:
            extras.append(f"params={total:,}")
        if trainable:
            extras.append(f"trainable={trainable:,}")
        if device:
            extras.append(f"device={device}")
        return f"{type(obj).__name__}" + (
            " (" + ", ".join(extras) + ")" if extras else ""
        )

    # Optimizer
    if optim is not None and isinstance(obj, optim.Optimizer):
        try:
            lrs = {pg.get("lr") for pg in obj.param_groups if "lr" in pg}
            return (
                f"{type(obj).__name__} groups={len(obj.param_groups)} lrs={sorted(lrs)}"
            )
        except Exception:
            return type(obj).__name__

    # Mapping / list summarization
    if isinstance(obj, Mapping):
        return f"Mapping[{len(obj)}]"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}[{len(obj)}]"

    # Fallback to short str or type
    try:
        s = str(obj)
        if len(s) <= 120:
            return s
    except Exception:
        pass
    return type(obj).__name__


def log_model_overview(
    components: Mapping[str, Any] | Sequence[tuple[str, Any]],
    title: str | None = None,
    logger: logging.Logger | None = None,
    *,
    max_depth: int = 0,
    indent: int = 0,
) -> None:
    """Log a clean, tree-style overview of model components.

    - components: mapping or sequence of (name, obj)
    - title: optional title header
    - logger: optional logger (defaults to print)
    - max_depth: recurse into nested mappings up to this depth
    - indent: initial left padding in spaces
    """
    if isinstance(components, Mapping):
        items: Sequence[tuple[str, Any]] = list(components.items())
    else:
        items = list(components)

    emit = logger.info if logger is not None else print
    pad = " " * indent

    if title:
        bar = "─" * len(title)
        emit(f"{pad}{title}")
        emit(f"{pad}{bar}")

    def _as_items(obj: Any) -> Sequence[tuple[str, Any]] | None:
        if isinstance(obj, Mapping):
            return list(obj.items())
        return None

    def _print(items: Sequence[tuple[str, Any]], depth: int, base_prefix: str) -> None:
        n = len(items)
        for i, (name, obj) in enumerate(items):
            last = i == n - 1
            branch = "└─" if last else "├─"
            cont = "  " if last else "│ "
            prefix = base_prefix + branch + " "
            emit(f"{pad}{prefix}{name}: {_describe_object(obj)}")
            if depth < max_depth:
                nested = _as_items(obj)
                if nested:
                    _print(nested, depth + 1, base_prefix + cont)

    _print(items, depth=0, base_prefix="")


def log_agent_overview(
    agent: Any,
    title: str | None = None,
    logger: logging.Logger | None = None,
    *,
    include: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    max_depth: int = 0,
    indent: int = 0,
) -> None:
    """Gather common RL components on an agent and log them.

    Auto-detects attributes like feature extractors, policies, values/Qs,
    actor-critics, and L2T student components. Additional attributes can
    be included via `include` or `extra`.
    """
    components: dict[str, Any] = {}

    # Detect teacher-student pattern and group if present
    has_student = any(
        hasattr(agent, nm)
        for nm in (
            "student_feature_extractor",
            "student_policy",
            "student_value_function",
            "student_actor_critic",
            "student",
        )
    )

    if has_student:
        teacher_map: dict[str, Any] = {}
        student_map: dict[str, Any] = {}

        # Teacher components (BaseAlgorithm standard names)
        for name, label in (
            ("feature_extractor", "Feature Extractor"),
            ("policy", "Policy"),
            ("value_function", "Value Function"),
            ("q_function", "Q Function"),
            ("actor_critic", "Actor-Critic"),
        ):
            if hasattr(agent, name):
                try:
                    teacher_map[label] = getattr(agent, name)
                except Exception:
                    pass

        # Student components
        for name, label in (
            ("student_feature_extractor", "Feature Extractor"),
            ("student_policy", "Policy"),
            ("student_value_function", "Value Function"),
            ("student_actor_critic", "Actor-Critic"),
        ):
            if hasattr(agent, name):
                try:
                    student_map[label] = getattr(agent, name)
                except Exception:
                    pass

        # Explicit teacher/student attributes if present
        if hasattr(agent, "teacher"):
            teacher_map.setdefault("Teacher", agent.teacher)
        if hasattr(agent, "student"):
            student_map.setdefault("Student", agent.student)

        components["Teacher"] = teacher_map
        components["Student"] = student_map

    else:
        # Flat components for standard agents
        for name, label in (
            ("feature_extractor", "Feature Extractor"),
            ("policy", "Policy"),
            ("value_function", "Value Function"),
            ("q_function", "Q Function"),
            ("actor_critic", "Actor-Critic"),
        ):
            if hasattr(agent, name):
                try:
                    components[label] = getattr(agent, name)
                except Exception:
                    pass

    # User-included attributes
    if include:
        for name in include:
            if hasattr(agent, name) and name not in components:
                try:
                    components[name] = getattr(agent, name)
                except Exception:
                    pass

    # Extra components mapping
    if extra:
        components.update(extra)

    if not title:
        title = f"{agent.__class__.__name__} Overview"

    log_model_overview(
        components,
        title=title,
        logger=logger or logging.getLogger(getattr(agent, "__module__", __name__)),
        max_depth=max_depth,
        indent=indent,
    )
