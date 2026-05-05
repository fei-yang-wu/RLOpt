"""Pure IPMD helpers for config normalization and reward-model bookkeeping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from rlopt.config_utils import (
    BatchKey,
    ObsKey,
    dedupe_keys,
    epic_distance,
    next_obs_key,
)

IPMD_COMMAND_SOURCES = frozenset({"random", "posterior"})
IPMD_REWARD_MODEL_TYPES = frozenset({"mlp", "grouped"})
IPMD_REWARD_INPUT_TYPES = frozenset({"s", "s'", "sa", "sas"})
IPMD_REWARD_OUTPUT_ACTIVATIONS = frozenset({"none", "tanh", "sigmoid"})
IPMD_REWARD_BLOCK_ORDER: dict[str, tuple[str, ...]] = {
    "s": ("obs",),
    "s'": ("next_obs",),
    "sa": ("obs", "action"),
    "sas": ("obs", "action", "next_obs"),
}


def normalize_ipmd_command_source(command_source: str) -> str:
    normalized = str(command_source).strip().lower()
    if normalized not in IPMD_COMMAND_SOURCES:
        msg = (
            f"Unsupported IPMD command_source {command_source!r}; "
            f"expected one of {sorted(IPMD_COMMAND_SOURCES)}."
        )
        raise ValueError(msg)
    return normalized


def normalize_ipmd_reward_model_type(reward_model_type: str) -> str:
    normalized = str(reward_model_type).strip().lower()
    if normalized not in IPMD_REWARD_MODEL_TYPES:
        msg = (
            f"Unsupported IPMD reward_model_type {reward_model_type!r}; "
            f"expected one of {sorted(IPMD_REWARD_MODEL_TYPES)}."
        )
        raise ValueError(msg)
    return normalized


def normalize_ipmd_reward_input_type(reward_input_type: str) -> str:
    normalized = str(reward_input_type).strip().lower()
    if normalized not in IPMD_REWARD_INPUT_TYPES:
        msg = (
            f"Unsupported IPMD reward_input_type {reward_input_type!r}; "
            f"expected one of {sorted(IPMD_REWARD_INPUT_TYPES)}."
        )
        raise ValueError(msg)
    return normalized


def normalize_ipmd_reward_output_activation(reward_output_activation: str) -> str:
    normalized = str(reward_output_activation).strip().lower()
    if normalized not in IPMD_REWARD_OUTPUT_ACTIVATIONS:
        msg = (
            "Unsupported IPMD reward_output_activation "
            f"{reward_output_activation!r}; expected one of "
            f"{sorted(IPMD_REWARD_OUTPUT_ACTIVATIONS)}."
        )
        raise ValueError(msg)
    return normalized


def require_positive_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def require_non_negative(name: str, value: float) -> float:
    normalized = float(value)
    if normalized < 0.0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


@dataclass(frozen=True)
class RewardInputBlock:
    """One construction-time reward-model input block."""

    kind: str
    dim: int
    obs_keys: tuple[ObsKey, ...] = ()


def build_reward_input_blocks(
    *,
    reward_input_type: str,
    reward_obs_keys: Sequence[ObsKey],
    obs_feature_dims: Mapping[ObsKey, int],
    action_feature_dim: int,
) -> tuple[RewardInputBlock, ...]:
    """Construct one canonical reward-input specification."""
    obs_dim = sum(int(obs_feature_dims[key]) for key in reward_obs_keys)

    blocks: list[RewardInputBlock] = []
    for kind in IPMD_REWARD_BLOCK_ORDER[reward_input_type]:
        if kind in {"obs", "next_obs"}:
            if len(reward_obs_keys) == 0:
                msg = (
                    "IPMD reward_input_type requires observation features, but "
                    "ipmd.reward_input_keys resolved to an empty list."
                )
                raise ValueError(msg)
            blocks.append(
                RewardInputBlock(
                    kind=kind,
                    dim=obs_dim,
                    obs_keys=tuple(reward_obs_keys),
                )
            )
            continue
        if kind == "action":
            blocks.append(RewardInputBlock(kind="action", dim=int(action_feature_dim)))
            continue
        msg = f"Unhandled reward input block kind {kind!r}."
        raise RuntimeError(msg)

    return tuple(blocks)


def required_batch_keys_from_reward_blocks(
    blocks: Sequence[RewardInputBlock],
) -> list[BatchKey]:
    """Return the batch keys needed to materialize reward-model inputs."""
    required: list[BatchKey] = []
    for block in blocks:
        if block.kind == "obs":
            required.extend(block.obs_keys)
            continue
        if block.kind == "next_obs":
            required.extend(next_obs_key(key) for key in block.obs_keys)
            continue
        if block.kind == "action":
            required.append("action")
            continue
        msg = f"Unhandled reward input block kind {block.kind!r}."
        raise RuntimeError(msg)
    return dedupe_keys(required)


def reward_grad_penalty_from_input(reward: Tensor, reward_input: Tensor) -> Tensor:
    """Squared gradient norm of reward with respect to its input features."""
    reward_grad = torch.autograd.grad(
        outputs=reward.sum(),
        inputs=reward_input,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return reward_grad.pow(2).sum(dim=-1).mean()


def reward_alignment_metrics(
    prefix: str, reward_pred: Tensor, reward_true: Tensor
) -> dict[str, float]:
    pred = reward_pred.detach().float().flatten()
    true = reward_true.detach().float().flatten()
    diff = pred - true

    pred_mean = pred.mean()
    true_mean = true.mean()
    pred_centered = pred - pred_mean
    true_centered = true - true_mean

    pred_var = pred_centered.pow(2).mean()
    true_var = true_centered.pow(2).mean()
    cov = (pred_centered * true_centered).mean()

    pearson_corr, corr_distance = epic_distance(pred, true)

    eps = 1e-8
    if pred_var <= eps:
        affine_scale = torch.zeros((), device=pred.device, dtype=pred.dtype)
        affine_bias = true_mean
        fitted = torch.full_like(true, true_mean)
    else:
        affine_scale = cov / pred_var
        affine_bias = true_mean - affine_scale * pred_mean
        fitted = affine_scale * pred + affine_bias

    resid = true - fitted
    ss_res = resid.pow(2).sum()
    ss_tot = true_centered.pow(2).sum()
    if ss_tot <= eps:
        affine_r2 = torch.ones((), device=pred.device, dtype=pred.dtype)
    else:
        affine_r2 = 1.0 - ss_res / ss_tot

    return {
        f"{prefix}/pearson_corr": pearson_corr.item(),
        f"{prefix}/corr_distance": corr_distance.item(),
        f"{prefix}/mae": diff.abs().mean().item(),
        f"{prefix}/rmse": diff.pow(2).mean().sqrt().item(),
        f"{prefix}/affine_scale": affine_scale.item(),
        f"{prefix}/affine_bias": affine_bias.item(),
        f"{prefix}/affine_r2": affine_r2.item(),
        f"{prefix}/target_std": true_var.sqrt().item(),
        f"{prefix}/pred_std": pred_var.sqrt().item(),
    }
