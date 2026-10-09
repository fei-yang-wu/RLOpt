"""Pure IPMD helpers for config normalization and reward-model bookkeeping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rlopt.config_utils import (
    BatchKey,
    ObsKey,
    dedupe_keys,
    epic_distance,
    next_obs_key,
)

IPMD_COMMAND_SOURCES = frozenset(
    {
        "random",
        "posterior",
        "hl_skill",
        "skill_commander",
        "token_planner",
        "continuous_planner",
    }
)
IPMD_REWARD_MODEL_TYPES = frozenset({"mlp", "grouped"})
IPMD_REWARD_INPUT_TYPES = frozenset({"s", "s'", "sa", "sas"})
IPMD_REWARD_OUTPUT_ACTIVATIONS = frozenset({"none", "tanh", "sigmoid"})
IPMD_REWARD_OBJECTIVES = frozenset(
    {"ipmd_diff", "logistic_ratio", "chi2_lsgan", "linear_dual"}
)
"""Reward-estimator objectives (plan 2026-10-07, section 5.4 B2).

- ``"ipmd_diff"``      - the parked campaign objective: ``mean r_pi - mean r_exp``
  through the configured output activation, plus the L2 / gradient / logit
  regularizers. Bitwise the historical path.
- ``"logistic_ratio"`` - label-smoothed BCE classifier; reward = logit (a
  finite log density ratio where supports overlap).
- ``"chi2_lsgan"``     - AMP least-squares discriminator with the R1 penalty on
  expert samples only; reward = D(x), population optimum (p-q)/(p+q).
- ``"linear_dual"``    - IPMD linear dual over whitened features with weight
  decay (fixed point ``w* = Delta / lambda``); no MLP, no optimizer.
"""
IPMD_REWARD_PPO_SQUASHES = frozenset(
    {"auto", "none", "sigmoid", "amp", "affine_frozen"}
)
IPMD_REWARD_PPO_SQUASH_AUTO: dict[str, str] = {
    "ipmd_diff": "none",
    "logistic_ratio": "sigmoid",
    "chi2_lsgan": "amp",
    "linear_dual": "affine_frozen",
}
"""Default PPO squash per objective when ``reward_ppo_squash == "auto"``."""
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


def normalize_ipmd_reward_objective(reward_objective: str) -> str:
    normalized = str(reward_objective).strip().lower()
    if normalized not in IPMD_REWARD_OBJECTIVES:
        msg = (
            f"Unsupported IPMD reward_objective {reward_objective!r}; "
            f"expected one of {sorted(IPMD_REWARD_OBJECTIVES)}."
        )
        raise ValueError(msg)
    return normalized


def normalize_ipmd_reward_ppo_squash(reward_ppo_squash: str) -> str:
    normalized = str(reward_ppo_squash).strip().lower()
    if normalized not in IPMD_REWARD_PPO_SQUASHES:
        msg = (
            f"Unsupported IPMD reward_ppo_squash {reward_ppo_squash!r}; "
            f"expected one of {sorted(IPMD_REWARD_PPO_SQUASHES)}."
        )
        raise ValueError(msg)
    return normalized


def resolve_ipmd_reward_ppo_squash(
    reward_objective: str, reward_ppo_squash: str
) -> str:
    """Return the concrete squash: ``"auto"`` maps through the objective table."""
    objective = normalize_ipmd_reward_objective(reward_objective)
    squash = normalize_ipmd_reward_ppo_squash(reward_ppo_squash)
    if squash == "auto":
        return IPMD_REWARD_PPO_SQUASH_AUTO[objective]
    return squash


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


# --------------------------------------------------------------------------
# Reward-estimator diagnostics shared by every objective (device tensors).
# --------------------------------------------------------------------------


def reward_separation_auc(reward_expert: Tensor, reward_policy: Tensor) -> Tensor:
    """Rank-based AUC ``P(r_exp > r_pi)`` on one update batch, as a 0-d tensor.

    Mann-Whitney through a double ``argsort`` on the pooled rewards; ties are
    broken by sort order (no half credit), which is immaterial for continuous
    estimator outputs. Nothing here touches the host; callers ``.item()`` at
    logging time like every other reward metric.
    """
    r_exp = reward_expert.detach().reshape(-1).float()
    r_pi = reward_policy.detach().reshape(-1).float()
    n_exp = r_exp.numel()
    n_pi = r_pi.numel()
    if n_exp == 0 or n_pi == 0:
        return torch.full((), 0.5, device=r_exp.device)
    pooled = torch.cat([r_exp, r_pi], dim=0)
    order = torch.argsort(pooled)
    ranks = torch.empty_like(pooled)
    ranks[order] = torch.arange(
        1, pooled.numel() + 1, device=pooled.device, dtype=pooled.dtype
    )
    rank_sum_exp = ranks[:n_exp].sum()
    return (rank_sum_exp - 0.5 * n_exp * (n_exp + 1)) / float(n_exp * n_pi)


def reward_saturation_fraction(reward: Tensor, edge: float = 0.05) -> Tensor:
    """Share of rewards within ``edge`` of either end of the batch's own range.

    The saturation diagnostic of the offline screen: a bounded objective that
    rails has most of its mass at the two range edges. Returns a 0-d tensor;
    a constant batch (zero range) counts as fully saturated.
    """
    values = reward.detach().reshape(-1).float()
    if values.numel() == 0:
        return torch.zeros((), device=values.device)
    lo = values.min()
    hi = values.max()
    span = hi - lo
    margin = float(edge) * span
    at_edge = ((values - lo) <= margin) | ((hi - values) <= margin)
    return at_edge.float().mean()


def rollout_done_mask(rollout: Any, like: Tensor) -> Tensor | None:
    """Boolean ``("next", "done")`` mask broadcast to ``like``'s shape, or None.

    ``done`` covers both terminated and truncated rows, which is the mask the
    reward mixing wants: no estimated reward may be paid on a row whose
    successor is a reset.
    """
    done_key: BatchKey = ("next", "done")
    if done_key not in rollout.keys(True):
        return None
    done = rollout[done_key].to(device=like.device, dtype=torch.bool)
    while done.ndim < like.ndim:
        done = done.unsqueeze(-1)
    return done


def squash_reward_for_ppo(
    reward: Tensor,
    squash: str,
    *,
    affine_center: Tensor | None = None,
    affine_scale: Tensor | None = None,
) -> Tensor:
    """Map a raw estimator output to ``r_hat`` per ``reward_ppo_squash``.

    - ``"none"``          - identity (the historical path, returned as-is).
    - ``"sigmoid"``       - ``sigmoid(r)``; the logistic logit becomes the
      classifier probability, in (0, 1).
    - ``"amp"``           - ``max(0, 1 - 0.25 (r - 1)^2)`` (Peng et al. 2021),
      in [0, 1], equal to 1 at ``D = 1`` and 0 at ``D <= -1``.
    - ``"affine_frozen"`` - ``clip(0.5 + (r - m0) / (6 s0), 0, 1)`` with the
      frozen centre and scale supplied by the caller.
    """
    if squash == "none":
        return reward
    if squash == "sigmoid":
        return torch.sigmoid(reward)
    if squash == "amp":
        return (1.0 - 0.25 * (reward - 1.0).pow(2)).clamp(min=0.0)
    if squash == "affine_frozen":
        if affine_center is None or affine_scale is None:
            msg = "affine_frozen squash needs frozen centre and scale tensors."
            raise RuntimeError(msg)
        return (0.5 + (reward - affine_center) / (6.0 * affine_scale)).clamp(0.0, 1.0)
    msg = f"Unhandled reward_ppo_squash {squash!r}."
    raise RuntimeError(msg)


# --------------------------------------------------------------------------
# The IPMD linear dual as a parameter-free module (plan 5.4 B2, linear_dual).
# --------------------------------------------------------------------------


class LinearDualRewardHead(nn.Module):
    """Linear reward ``r(x) = w^T W (x - mu)`` driven by the IPMD dual step.

    No parameters, only buffers, so it has no optimizer and its whole state
    rides in ``reward_estimator_state_dict``. ``mu`` and the second moment
    are running estimates over the pooled policy+expert rows of every reward
    step (EMA with ``momentum``, or the exact cumulative mean when
    ``momentum <= 0``); ``W = (C + ridge I)^(-1/2)`` is recomputed from them
    at every step (symmetric inverse square root through ``eigh`` in float64;
    ``D`` is 71 on the own-heading map, so this costs nothing). In the
    whitened coordinates ``phi_w = W (x - mu)`` the dual step is

        ``w <- w + alpha (Delta - lambda w)``,  ``Delta = E_E[phi_w] - E_pi[phi_w]``

    whose fixed point ``w* = Delta / lambda`` is finite, scales with the
    Mahalanobis gap of the two means, and vanishes at feature matching (see
    the offline ``linear_objectives`` module for why a unit-ball projection
    has none of these). ``closed_form=True`` jumps to ``w*`` directly each
    step. ``forward`` folds the map into one matvec against cached
    ``raw_weight`` / ``bias`` buffers and returns ``[..., 1]`` like the MLP
    estimator, so the rest of the agent treats both the same.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        ridge: float = 1e-3,
        weight_decay: float = 1.0,
        momentum: float = 0.05,
        closed_form: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        dim = int(feature_dim)
        if dim <= 0:
            msg = f"LinearDualRewardHead needs feature_dim > 0, got {feature_dim!r}."
            raise ValueError(msg)
        self.feature_dim = dim
        self.ridge = float(ridge)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.closed_form = bool(closed_form)
        kwargs = {"device": device, "dtype": torch.float32}
        self.register_buffer("feature_mean", torch.zeros(dim, **kwargs))
        self.register_buffer("feature_second_moment", torch.zeros(dim, dim, **kwargs))
        self.register_buffer("whitening", torch.eye(dim, **kwargs))
        self.register_buffer("w", torch.zeros(dim, **kwargs))
        self.register_buffer("raw_weight", torch.zeros(dim, **kwargs))
        self.register_buffer("bias", torch.zeros((), **kwargs))
        self.register_buffer("step", torch.zeros((), dtype=torch.int64, device=device))
        self.register_buffer("whitening_condition", torch.ones((), **kwargs))

    # -- running moments -------------------------------------------------

    @torch.no_grad()
    def update_moments(self, rows: Tensor) -> None:
        """Fold one pooled batch ``[N, D]`` into the running mean / second moment."""
        rows = rows.detach().reshape(-1, self.feature_dim).to(self.feature_mean.dtype)
        batch_mean = rows.mean(dim=0)
        batch_second = rows.T @ rows / float(rows.shape[0])
        self.step += 1
        step = self.step.to(batch_mean.dtype)
        if self.momentum > 0.0:
            weight = torch.full_like(step, self.momentum)
        else:
            weight = 1.0 / step
        # The first batch initializes the moments outright; no host sync.
        weight = torch.where(self.step == 1, torch.ones_like(weight), weight)
        self.feature_mean.lerp_(batch_mean, weight)
        self.feature_second_moment.lerp_(batch_second, weight)

    @torch.no_grad()
    def refresh_whitening(self) -> None:
        """Recompute ``W = (C + ridge I)^(-1/2)`` from the running moments."""
        mean = self.feature_mean.to(torch.float64)
        covariance = self.feature_second_moment.to(torch.float64) - torch.outer(
            mean, mean
        )
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.clamp_min(0.0)
        inverse_root = (eigenvalues + self.ridge).rsqrt()
        transform = (eigenvectors * inverse_root) @ eigenvectors.T
        self.whitening.copy_(transform.to(self.whitening.dtype))
        self.whitening_condition.copy_(
            (eigenvalues[-1] / eigenvalues[0].clamp_min(1e-300)).to(
                self.whitening_condition.dtype
            )
        )

    def whiten(self, rows: Tensor) -> Tensor:
        """``phi_w(x) = (x - mu) W`` (``W`` symmetric)."""
        return (rows.to(self.feature_mean.dtype) - self.feature_mean) @ self.whitening

    # -- the dual step ---------------------------------------------------

    @torch.no_grad()
    def dual_step(
        self,
        policy_rows: Tensor,
        expert_rows: Tensor,
        *,
        alpha: float,
    ) -> dict[str, Tensor]:
        """One reward step: moments, whitening, ``Delta``, and the ``w`` update.

        Returns device tensors: ``delta_norm`` (the Mahalanobis gap),
        ``w_norm``, ``dual_value = w^T Delta`` (``E_E[r] - E_pi[r]`` under the
        updated ``w``; ``||Delta||^2 / lambda`` at the fixed point) and the
        per-side rewards for the caller's metrics.
        """
        policy_rows = policy_rows.detach().reshape(-1, self.feature_dim)
        expert_rows = expert_rows.detach().reshape(-1, self.feature_dim)
        self.update_moments(torch.cat([policy_rows, expert_rows], dim=0))
        self.refresh_whitening()
        phi_pi = self.whiten(policy_rows)
        phi_exp = self.whiten(expert_rows)
        delta = phi_exp.mean(dim=0) - phi_pi.mean(dim=0)
        if self.closed_form:
            self.w.copy_(delta / self.weight_decay)
        else:
            self.w.add_(float(alpha) * (delta - self.weight_decay * self.w))
        self._refresh_fold()
        r_pi = phi_pi @ self.w
        r_exp = phi_exp @ self.w
        return {
            "delta_norm": delta.norm(),
            "w_norm": self.w.norm(),
            "dual_value": self.w @ delta,
            "reward_policy": r_pi,
            "reward_expert": r_exp,
        }

    @torch.no_grad()
    def _refresh_fold(self) -> None:
        raw = self.whitening @ self.w
        self.raw_weight.copy_(raw)
        self.bias.copy_(-(self.feature_mean @ raw))

    def forward(self, features: Tensor) -> Tensor:
        """``[..., D] -> [..., 1]``: one matvec against the folded weight."""
        reward = features.to(self.raw_weight.dtype) @ self.raw_weight + self.bias
        return reward.unsqueeze(-1)
