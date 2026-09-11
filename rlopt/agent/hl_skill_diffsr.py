from __future__ import annotations

import copy
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torch import Tensor, nn

from rlopt.agent.hl_skill_encoder import (
    LATENT_MODES as _LATENT_MODES,
)
from rlopt.agent.hl_skill_encoder import (
    SkillLatentSpec,
    build_skill_encoder,
)
from rlopt.agent.ipmd.module import BilinearSR, build_bilinear_sr


def _require_positive_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_non_negative_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized < 0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_positive_float(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_non_negative_float(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_fraction(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        msg = f"{name} must be in (0, 1), got {value!r}."
        raise ValueError(msg)
    return normalized


def _normalize_split_value(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "":
        return None
    if normalized not in {"all", "train", "eval"}:
        msg = f"{name} must be one of 'all', 'train', or 'eval', got {value!r}."
        raise ValueError(msg)
    return normalized


def _normalize_encoder_window_mode(name: str, value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"full", "intermediate"}:
        return normalized
    if _encoder_window_suffix_steps(normalized) is not None:
        return normalized
    msg = (
        f"{name} must be 'full', 'intermediate', or 'suffix<N>' (N >= 1), "
        f"got {value!r}."
    )
    raise ValueError(msg)


def _encoder_window_suffix_steps(mode: str) -> int | None:
    """Return N for a ``suffix<N>`` window mode, else None."""
    match = re.fullmatch(r"suffix([1-9]\d*)", str(mode))
    if match is None:
        return None
    return int(match.group(1))


def _normalize_transition_objective(name: str, value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "occupancy": "state_occupancy",
        "chain": "semimarkov_chain",
        "delta": "endpoint_delta",
        "jepa": "jepa_ntp",
    }
    normalized = aliases.get(normalized, normalized)
    choices = {
        "endpoint",
        "state_occupancy",
        "semimarkov_chain",
        "endpoint_delta",
        "jepa_ntp",
        "reconstruction",
    }
    if normalized not in choices:
        msg = f"{name} must be one of {sorted(choices)}, got {value!r}."
        raise ValueError(msg)
    return normalized


def _normalize_command_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {"fz": "phi", "z_fz": "z_phi"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"z", "phi", "z_phi"}:
        msg = (
            "command_mode must be one of 'z', 'phi', or 'z_phi' "
            f"(aliases: 'fz', 'z_fz'), got {value!r}."
        )
        raise ValueError(msg)
    return normalized


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _resolve_device(device: torch.device | str | None, env: object) -> torch.device:
    if device is not None and str(device).strip().lower() != "auto":
        return torch.device(device)
    env_device = getattr(env, "device", None)
    if env_device is not None:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encoder_window_steps(config: HighLevelSkillDiffSRConfig) -> int:
    if config.encoder_window_mode == "intermediate":
        return int(config.horizon_steps) - 1
    suffix = _encoder_window_suffix_steps(config.encoder_window_mode)
    if suffix is not None:
        return suffix
    return int(config.horizon_steps)


class _WindowReconstructionDecoder(nn.Module):
    """Decode a skill code into the exact window visible to the encoder."""

    def __init__(
        self,
        *,
        z_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
    ) -> None:
        super().__init__()
        activation_types: dict[str, type[nn.Module]] = {
            "elu": nn.ELU,
            "mish": nn.Mish,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        activation_type = activation_types[str(activation)]
        layers: list[nn.Module] = []
        previous_dim = int(z_dim)
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, int(hidden_dim)), activation_type()))
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.network(z)


_ROOT_QPOS_FRAME_DIM = 38  # qpos(29) + anchor_pos_b(3) + anchor_ori_b(6)


def _rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """6D rotation (two raw columns) -> 3x3 matrix, Gram-Schmidt."""
    a1, a2 = rot6d[..., 0:3], rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(
        a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


_FULL_BODY_FRAME_DIM = 67


def _reanchor_heading_frames(frames: Tensor, anchor: Tensor) -> Tensor:
    """Re-express ``root_qpos`` frames in ``anchor``'s heading frame.

    Both are already expressed in some common heading frame (the sampled
    window's own slot-0 heading anchor). The next-chunk target must instead be
    what the encoder would see with the NEXT chunk's state as its anchor:
    yaw-only rotation, xy-only origin, exactly the ``expert_heading`` /
    ``robot_heading`` pretrain convention. Joint angles are frame-invariant;
    only the anchor pos/ori block transforms. Heading composition is exact
    here because the outer frame is itself yaw-only, so yaw components add.
    """
    width = int(frames.shape[-1])
    if width not in (_ROOT_QPOS_FRAME_DIM, _FULL_BODY_FRAME_DIM):
        msg = (
            "heading re-anchoring needs the 38-wide root_qpos or 67-wide "
            f"full_body frame, got {width}."
        )
        raise ValueError(msg)
    # Invariant prefix: joint qpos (root_qpos) or joint qpos+qvel (full_body).
    # Joint velocities are frame-invariant under a yaw re-anchor; only the
    # trailing anchor pos(3) + rot6d(6) block transforms.
    inv = width - 9
    rotation = _rot6d_to_matrix(anchor[..., inv + 3 : inv + 9])
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    zeros = torch.zeros_like(cos)
    ones = torch.ones_like(cos)
    # R_yaw^T, applied from the left to positions and rotations.
    yaw_t = torch.stack(
        [
            torch.stack([cos, sin, zeros], dim=-1),
            torch.stack([-sin, cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    origin = anchor[..., inv : inv + 3].clone()
    origin[..., 2] = 0.0  # xy-only origin: absolute height survives
    while yaw_t.dim() < frames.dim() + 1:
        yaw_t = yaw_t.unsqueeze(-3)
        origin = origin.unsqueeze(-2)
    position = torch.einsum(
        "...ij,...j->...i", yaw_t, frames[..., inv : inv + 3] - origin
    )
    rotations = _rot6d_to_matrix(frames[..., inv + 3 : inv + 9])
    rotated = yaw_t @ rotations
    ori6d = torch.cat([rotated[..., :, 0], rotated[..., :, 1]], dim=-1)
    return torch.cat([frames[..., :inv], position, ori6d], dim=-1)


def _sigreg_epps_pulley(z: Tensor, num_sketches: int) -> Tensor:
    """Sketched isotropic-Gaussian regularizer (LeJEPA's SIGReg).

    Project the batch onto random unit directions and penalize the squared
    deviation of each projection's empirical characteristic function from the
    standard Gaussian's, integrated under a N(0,1) weight (the Epps-Pulley
    statistic, evaluated by trapezoid quadrature). Zero iff every 1-D
    projection is standard normal, which forces the embedding distribution
    toward isotropic Gaussian — collapse (a point mass) and scale explosion
    both score badly, so no negatives are needed.
    """
    directions = torch.randn(
        int(num_sketches), z.shape[-1], device=z.device, dtype=z.dtype
    )
    directions = torch.nn.functional.normalize(directions, dim=-1)
    projections = z @ directions.T  # (batch, sketches)
    t = torch.linspace(-5.0, 5.0, 17, device=z.device, dtype=z.dtype)
    angles = projections.unsqueeze(-1) * t  # (batch, sketches, t)
    emp_cos = angles.cos().mean(dim=0)
    emp_sin = angles.sin().mean(dim=0)
    gauss_cf = torch.exp(-0.5 * t.square())
    weight = torch.exp(-0.5 * t.square()) / math.sqrt(2.0 * math.pi)
    integrand = ((emp_cos - gauss_cf).square() + emp_sin.square()) * weight
    return torch.trapezoid(integrand, t, dim=-1).mean()


def _encoder_input_window(
    config: HighLevelSkillDiffSRConfig, future_window: Tensor
) -> Tensor:
    if config.encoder_window_mode == "intermediate":
        return future_window[:, :-1, :]
    suffix = _encoder_window_suffix_steps(config.encoder_window_mode)
    if suffix is not None:
        # Last ``suffix`` slots of the INTERMEDIATE window: the endpoint stays
        # hidden, exactly as in 'intermediate', and only the near-boundary
        # frames remain visible. suffix N < H-1 ablates the early/mid window;
        # suffix N = H-1 equals 'intermediate' by construction.
        return future_window[:, -(suffix + 1) : -1, :]
    return future_window


def _source_window_steps(config: HighLevelSkillDiffSRConfig) -> int:
    """Frames phi conditions on: the current state plus its past chunk."""
    return int(config.source_history_steps) + 1


def _build_diffsr(
    config: HighLevelSkillDiffSRConfig,
    state_dim: int,
    device: torch.device,
    *,
    next_obs_dim: int | None = None,
) -> BilinearSR:
    # phi's source widens with the past chunk; obs_norm normalizes the TARGET
    # (`next_obs_dim`), so a wider source does not disturb it.
    return build_bilinear_sr(
        "diffsr",
        obs_dim=state_dim * _source_window_steps(config),
        next_obs_dim=int(next_obs_dim) if next_obs_dim is not None else state_dim,
        action_dim=config.z_dim,
        feature_dim=config.diffsr_feature_dim,
        embed_dim=config.diffsr_embed_dim,
        g_hidden_dims=config.diffsr_g_hidden_dims,
        f_hidden_dims=config.diffsr_f_hidden_dims,
        phi_parameterization=config.diffsr_phi_parameterization,
        mu_conditioning=config.diffsr_mu_conditioning,
        mu_hidden_dims=config.diffsr_mu_hidden_dims,
        num_noises=config.diffsr_num_noises,
        use_ema_for_policy=False,
        x_min=config.diffsr_x_min,
        x_max=config.diffsr_x_max,
        device=device,
    )


def _jepa_ntp_target_dim(config: HighLevelSkillDiffSRConfig, state_dim: int) -> int:
    """Return the denoising target width for a JEPA DiffSR head."""
    head = str(config.jepa_ntp_head)
    if head == "diff_token":
        return int(config.z_dim)
    if head == "diff_pair":
        return int(state_dim) + int(config.z_dim)
    if head == "diff_chunk":
        span_steps = int(config.horizon_steps)
        if str(config.jepa_ntp_chunk_span) == "boundary_next":
            span_steps += 1
        return span_steps * int(state_dim)
    msg = f"JEPA head {head!r} is not a DiffSR head."
    raise ValueError(msg)


def _macro_batch_state_history(
    batch: TensorDictBase,
    *,
    batch_size: int,
    history_steps: int,
    state_dim: int,
    device: torch.device,
) -> Tensor:
    """Materialize ``hl/state_history`` as ``[B, history_steps + 1, state_dim]``.

    The sampler emits this only when it is asked for a nonzero
    ``state_history_steps``, and its last slot IS ``s_t``. Frames are already in
    ``s_t``'s heading frame, so the past reads as negative displacement.
    """
    state_history = batch.get(("hl", "state_history"))
    if state_history is None:
        msg = (
            "Expert macro batch is missing hl/state_history. The sampler emits "
            "it only when called with state_history_steps > 0."
        )
        raise ValueError(msg)
    state_history = cast(Tensor, state_history).to(device=device, dtype=torch.float32)
    expected = (int(batch_size), int(history_steps) + 1, int(state_dim))
    if tuple(state_history.shape) != expected:
        msg = (
            f"hl/state_history shape mismatch: expected {expected}, "
            f"got {tuple(state_history.shape)}."
        )
        raise ValueError(msg)
    return state_history


def _validate_macro_batch(
    batch: TensorDictBase,
    *,
    batch_size: int,
    horizon_steps: int,
    device: torch.device,
    state_dim: int | None = None,
    source: str = "Expert",
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate and materialize ``hl/{state,future_window,target}`` from a macro batch.

    ``state_dim=None`` infers the state width from the batch (offline trainer /
    preflight); a provided ``state_dim`` is asserted (online sampler). ``source``
    only prefixes the missing-keys message.
    """
    state = batch.get(("hl", "state"))
    future_window = batch.get(("hl", "future_window"))
    target = batch.get(("hl", "target"))
    missing = [
        name
        for name, value in (
            ("hl/state", state),
            ("hl/future_window", future_window),
            ("hl/target", target),
        )
        if value is None
    ]
    if missing:
        msg = f"{source} macro batch is missing keys: {missing}."
        raise ValueError(msg)
    state = cast(Tensor, state).to(device=device, dtype=torch.float32)
    future_window = cast(Tensor, future_window).to(device=device, dtype=torch.float32)
    target = cast(Tensor, target).to(device=device, dtype=torch.float32)
    if state.ndim != 2:
        msg = f"hl/state must have shape [B, D], got {tuple(state.shape)}."
        raise ValueError(msg)
    resolved_dim = int(state.shape[-1]) if state_dim is None else int(state_dim)
    expected_state = (int(batch_size), resolved_dim)
    expected_window = (int(batch_size), int(horizon_steps), resolved_dim)
    if tuple(state.shape) != expected_state:
        msg = (
            f"hl/state shape mismatch: expected {expected_state}, "
            f"got {tuple(state.shape)}."
        )
        raise ValueError(msg)
    if tuple(future_window.shape) != expected_window:
        msg = (
            f"hl/future_window shape mismatch: expected {expected_window}, "
            f"got {tuple(future_window.shape)}."
        )
        raise ValueError(msg)
    if tuple(target.shape) != expected_state:
        msg = (
            f"hl/target shape mismatch: expected {expected_state}, "
            f"got {tuple(target.shape)}."
        )
        raise ValueError(msg)
    return state, future_window, target


def _effective_rank(z: Tensor) -> Tensor:
    """Participation-ratio effective rank of ``z`` as a 0-dim tensor."""
    if int(z.shape[0]) < 2:
        return z.new_zeros(())
    centered = z - z.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    total = singular_values.sum()
    if bool((total <= 1.0e-12).item()):
        return z.new_zeros(())
    probs = singular_values / total
    entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
    return torch.exp(entropy)


@dataclass
class HighLevelSkillDiffSRConfig:
    """Configuration for offline high-level skill DiffSR training."""

    horizon_steps: int = 25
    z_dim: int = 256
    diffsr_feature_dim: int = 128
    diffsr_embed_dim: int = 512
    diffsr_phi_parameterization: str = "concat"
    """phi(s, z) parameterization forwarded to the SR module.

    "concat" is the simple-concat path; "bilinear" restores the legacy matrix
    form g(z)^T F(s); "affine" is the same matrix form with g(z) = A z + b a
    single linear layer, which makes phi -- and therefore the diffusion score
    field of p(next | s, z) -- affine in z. All three are implemented by
    `BilinearSR.forward_phi`; this field is what lets the pretrain entrypoint
    select between them, and its default matches `BilinearSR`'s so omitting it
    changes nothing. The value is stored in the checkpoint config, so a loaded
    encoder rebuilds the same parameterization. "identity" sets phi(s, z) = z
    and needs `diffsr_feature_dim == z_dim`; see `diffsr_mu_conditioning`."""
    diffsr_mu_conditioning: str = "next"
    """What the DiffSR denoiser mu sees: "next" is mu(s', t), the default;
    "pair" is mu(s, s', t), the transition pair. With
    `diffsr_phi_parameterization="identity"` the pair form makes the noise
    prediction `<z, E(s, s', t)>` with `E = mu`: the product-of-experts
    reparameterization, linear in z with no bias. Stored in the checkpoint
    config like the phi parameterization. Applies to every DiffSR head this
    config builds (the endpoint head and the diff_* NTP heads)."""
    batch_size: int = 8192
    num_updates: int = 2000
    log_interval: int = 100
    eval_batches: int = 4
    eval_batch_size: int | None = None
    train_split: str | None = "train"
    eval_split: str | None = "eval"
    eval_trajectory_fraction: float = 0.1
    trajectory_split_seed: int = 0
    preflight_batch_size: int = 8
    encoder_window_mode: str = "full"
    source_history_steps: int = 0
    """Past frames added to phi's conditioning, on top of the current state.

    ``0`` reproduces every existing arm: phi conditions on ``s_t`` alone. The
    macro state is ``root_qpos`` and carries no velocities, so a single frame
    cannot express which way the motion is already going; a past chunk supplies
    that. phi's source width becomes ``(source_history_steps + 1) * state_dim``.

    phi, g and mu are pretrain-only heads, so this does not change the deployed
    command: the tracker still receives ``z`` from the encoder.
    """
    source_anchor: str = "current"
    """Heading frame of the whole sampled window when a past chunk is used.

    ``current``: anchor on ``s_t``. The past reads as NEGATIVE displacement --
    where the motion came from, expressed in where it is now. The encoder's
    state and future window are byte-identical to a ``source_history_steps=0``
    run, so the encoder stays comparable with existing checkpoints.

    ``past_start``: anchor on the OLDEST past frame ``s[t-history]``, making
    ``s[t-history] .. s[t+H]`` one macro window anchored at its first slot.
    Displacement is positive throughout and ``s_t`` is no longer canonical, so
    the ENCODER's input distribution changes and its ``z`` is not comparable
    with encoders trained under ``current``. A tracker cannot bind such an
    encoder until the frozen command sampler can anchor on the robot's heading
    ``history`` steps ago; today it anchors on the live heading only.

    Read only when ``source_history_steps > 0``.
    """
    transition_objective: str = "endpoint"
    """DiffSR factorization used to train the skill code.

    ``endpoint`` preserves the original ``p(s[t+H] | s[t], z)`` loss.
    ``state_occupancy`` samples one configured checkpoint ``h_k`` per row and
    trains ``p(s[t+h_k] | s[t], z)`` without exposing ``h_k`` to the decoder,
    making the learned density an option-conditioned state-occupancy mixture.
    ``semimarkov_chain`` samples one adjacent checkpoint pair and trains
    ``p(s[t+h_k] | s[t+h_{k-1}], z)`` with one code held across the chain.
    ``endpoint_delta`` predicts ``s[t+H] - s[t]`` instead of absolute endpoint.
    ``reconstruction`` decodes the exact state window visible to the encoder
    from ``z``. It uses the same offline pretrain and frozen rollout path as
    the DiffSR objectives, so it is a controlled objective ablation.
    """
    reconstruction_target: str = "input_window"
    """What the ``reconstruction`` decoder reproduces from ``z``.

    ``input_window``: the exact state window visible to the encoder (state plus
    the encoder-mode slice of the future window). Purely local — nothing ties
    the code to where the motion ends up.
    ``endpoint``: the macro state one horizon later (``hl/target``) only. A
    deterministic MSE endpoint regression, the non-denoising partner of the
    ``endpoint`` DiffSR objective.
    ``full_window``: the state plus every future-window slot including the
    endpoint, regardless of ``encoder_window_mode`` — the encoder still never
    sees the endpoint, so the decoder must predict it."""
    jepa_loss: str = "sigreg"
    """Anti-collapse mechanism for ``jepa_ntp``.

    ``sigreg``: LeJEPA-style prediction MSE plus sketched isotropic-Gaussian
    regularization, no negatives.
    ``sigreg_ebm``: the repo's own chunk-token recipe — the DiffSR bilinear
    spectral factorization ``phi(s, z) = g(z)^T F(s)`` stays in the loss as
    the energy-based grounding of the token, the predictor does chunk-wise
    NTP toward the EMA target token, and SIGReg (not negatives) prevents
    latent collapse.
    ``infonce``: bilinear spectral energy over in-batch negatives."""
    jepa_sigreg_coeff: float = 1.0
    """Weight of the SIGReg term against the prediction MSE."""
    jepa_ntp_head: str = "mlp"
    """Estimator for the next-chunk prediction term of ``sigreg_ebm``.

    ``mlp``: the deterministic 3-layer predictor with an MSE toward the target
    token — the conditional MEAN of the next token (the measured hub recipe).
    ``diff_token``: a second DiffSR diffusion head models the next token
    GENERATIVELY, p(z_next | s_t, z_t); it replaces the MSE at the same
    ``jepa_ntp_coeff`` slot. The target still comes from the (EMA/stopgrad)
    target encoder.
    ``diff_chunk``: the diffusion head models the next chunk's H raw frames
    p(x_{t+H+1:t+2H} | s_t, z_t), flattened, in the EXECUTED chunk's heading
    frame so the cross-chunk displacement (the drift information) stays in
    the target. Every target is data — with this head the objective carries
    no self-target and therefore no EMA dependence in the loss.
    ``diff_pair``: the diffusion head models the next chunk's anchor state
    and its token JOINTLY, p(s_{t+H}, z_next | s_t, z_t) over the
    concatenated (state_dim + z_dim) target — the joint captures the
    correlation between where the robot lands and what the next command is,
    which separate heads cannot.
    Every diffusion head is pretrain-only and requires
    ``jepa_loss='sigreg_ebm'`` and ``jepa_context_chunks=0``."""
    jepa_ntp_chunk_anchor: str = "executed"
    """Heading frame of the ``diff_chunk`` target.

    ``executed``: the next chunk's frames stay in the EXECUTED chunk's slot-0
    frame, so the cross-chunk displacement (drift) is part of the target.
    ``next``: the frames are re-anchored onto s_{t+H}'s own heading frame —
    exactly what the encoder would see one publication later — which erases
    the displacement and asks only "what does the next chunk look like in its
    own frame". Only read by ``jepa_ntp_head='diff_chunk'``."""
    jepa_ntp_chunk_span: str = "next"
    """Frames the ``diff_chunk`` head denoises.

    ``next``: the next chunk's H frames, ``s[t+H+1 .. t+2H]`` (the measured
    round-4 recipe).
    ``boundary_next``: the executed chunk's boundary plus the next chunk,
    ``s[t+H .. t+2H]`` (H+1 frames) — one head covering both diffusion
    targets, for the merged-head cell where ``jepa_endpoint_coeff=0`` drops
    the separate endpoint term. Requires ``jepa_ntp_chunk_anchor='executed'``
    (the boundary frame is defined in the executed chunk's own frame; a
    re-anchored boundary would be the identity slot). Only read by
    ``jepa_ntp_head='diff_chunk'``."""
    jepa_endpoint_coeff: float = 1.0
    """Weight of the endpoint DiffSR term inside ``sigreg_ebm``.

    ``0`` drops ``p(s[t+H] | s[t], z)`` from the loss. The endpoint head is
    still built and serialized, so the checkpoint contract is unchanged; its
    weights simply receive no gradient. Meant for the merged-head cell
    (``jepa_ntp_chunk_span='boundary_next'``), where the boundary frame lives
    inside the chunk head's target instead."""
    jepa_ntp_coeff: float = 1.0
    """Weight of the chunk-NTP prediction MSE inside ``sigreg_ebm``.

    ``0`` drops the predictor from the loss entirely: the objective becomes
    DiffSR endpoint grounding + SIGReg on the chunk token — the chunk-wise
    non-JEPA cell of the mechanism ablation. The predictor still exists and
    its copy-gate metrics still log; it just receives no gradient."""
    jepa_token_pred_coeff: float = 0.0
    """Weight of the mlp token-prediction MSE ``||P(z1) - z2||^2`` ADDED
    alongside a diffusion NTP head inside ``sigreg_ebm``.

    ``0`` (default) reproduces every existing arm exactly. Positive requires a
    diffusion ``jepa_ntp_head`` — under the mlp head that MSE already IS the
    NTP term (``jepa_ntp_coeff``), and a second copy would double-count it.
    The additive cell asks whether the EMA-lagged latent-dynamics term (owns
    global drift in the mechanism square) STACKS with chunk generation (whose
    G win comes from drift kept in the data target), or substitutes for it."""
    jepa_sigreg_sketches: int = 64
    """Random 1-D projection directions per step for SIGReg."""
    jepa_tau: float = 0.1
    """InfoNCE temperature for the ``jepa_ntp`` objective."""
    jepa_ema_momentum: float = 0.996
    """EMA momentum of the ``jepa_ntp`` target encoder."""
    jepa_energy_dim: int = 256
    """Width of the bilinear (spectral) energy factorization g(.)^T f(.)."""
    jepa_context_chunks: int = 0
    """Preceding chunks the predictor conditions on (``jepa_ntp`` only).

    ``0``: the chunk PAIR — predict chunk t+1's token from chunk t's alone.
    ``1``: the chunk TRIPLET (wiki/skill-encoder-jepa-plan.md phase 2): encode
    the preceding chunk too, and predict chunk t+1's token from
    ``cat(z[t-1], z[t])``. The predictor becomes a dynamics model in the code
    space, with the option-view state (the previous code) made explicit."""
    jepa_target_encoder_mode: str = "ema"
    """Where the next chunk's target token comes from (``jepa_ntp`` only).

    ``ema``: a momentum copy of the encoder (standard JEPA machinery).
    ``online``: the ONE online encoder on both sides, no EMA copy and no
    stop-gradient — the LeJEPA shape, where SIGReg alone carries the
    anti-collapse burden.
    ``stopgrad``: the ONE online encoder on both sides with a stop-gradient
    on the target branch, no EMA copy — the SimSiam-style asymmetry-only
    cell. Separates the EMA trick's asymmetry (which blocks the
    co-adaptation shortcut) from its lag. Offline pretraining only: online
    finetuning of a non-``ema`` checkpoint is refused."""
    transition_offsets: tuple[int, ...] = ()
    """Strictly increasing checkpoint offsets in ``[1, horizon_steps]``.

    Empty means ``(horizon_steps,)`` for endpoint objectives and every step
    ``1..horizon_steps`` for occupancy/chain objectives.
    """
    macro_frame_stride: int = 1
    """Reference frames between consecutive macro-window slots at pretrain time.

    Provenance, not a knob the trainer acts on: the environment owns the macro
    cadence (``env.expert_macro_frame_stride``), and the pretrain entrypoint
    copies its value here so the checkpoint carries it. 1 is the historical
    consecutive-frame window; 5 is SONIC's released tokenizer cadence, where 10
    slots span 0.9 s instead of 0.18 s and the endpoint target is ``s[t+50]``,
    one stride past the last slot the encoder reads.
    The macro state's width is identical either way, so a low-level run that
    loads this encoder under a different stride is silently off-distribution
    unless it compares this field -- which
    :class:`FrozenHighLevelSkillCommandSampler` does.
    """
    macro_anchor_mode: str = "robot"
    """Frame convention of the macro window at pretrain time.

    Provenance, like ``macro_frame_stride``: the environment owns the
    convention (``env.expert_macro_anchor_mode``) and the pretrain entrypoint
    copies its value here so the checkpoint carries it. "robot" is the
    historical split (expert-anchored pretrain, robot-anchored rollout);
    "expert_heading" expresses both in the expert's slot-0 heading frame, so
    pretrain and rollout encoder inputs match by construction;
    "robot_heading" is SONIC v1.1's convention -- rollout anchors at the LIVE
    robot's heading (yaw-only) frame, pretrain keeps the expert slot-0 heading
    frame because offline data has no robot. The width is
    identical in every mode, so a low level that loads this encoder under a
    different mode is silently off-distribution unless it compares this
    field -- which :class:`FrozenHighLevelSkillCommandSampler` does. A
    checkpoint written before the field existed reads back as "robot", which
    is what it was.
    """
    latent_mode: str = "deterministic"
    reg_coeff: float = 1.0e-3
    categorical_groups: int = 8
    categorical_categories: int = 32
    gaussian_logstd_min: float = -5.0
    gaussian_logstd_max: float = 2.0
    gumbel_codebook_size: int = 512
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    gumbel_tau_anneal_iters: int = 2000
    gumbel_hard: bool = True
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    # SONIC-matched token space: 64 dims x 32 levels ~= 320 bits per command,
    # i.e. gear_sonic's tokens of shape (2, 32) at num_fsq_levels=32. Used only
    # by latent_mode="sonic_fsq", which publishes the quantizer output directly
    # and therefore requires z_dim == len(sonic_fsq_levels).
    sonic_fsq_levels: tuple[int, ...] = (32,) * 64
    vq_codebook_size: int = 512
    vq_ema_decay: float = 0.99
    vq_dead_code_reset_iters: int = 0
    encoder_hidden_dims: tuple[int, ...] = (1024, 512, 512)
    encoder_activation: str = "mish"
    encoder_layer_norm: bool = True
    diffsr_f_hidden_dims: tuple[int, ...] = (512, 512)
    diffsr_g_hidden_dims: tuple[int, ...] = (512,)
    diffsr_mu_hidden_dims: tuple[int, ...] = (512,)
    diffsr_num_noises: int = 8
    diffsr_x_min: float = -10.0
    diffsr_x_max: float = 10.0
    encoder_lr: float = 3.0e-4
    diffsr_lr: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    reconstruction_norm_eps: float = 1.0e-6
    device: str = "auto"
    diffsr_state_output_init_std: float = 1.0e-3
    # Optional co-trained skill commander (System-1 planner). When enabled, a
    # SkillCommander is BC'd to the encoder's z (detached) from the current
    # state + language goal, jointly with the encoder/DiffSR pretraining.
    cotrain_commander: bool = False
    commander_language_embeddings_path: str = ""
    commander_hidden_dims: tuple[int, ...] = (1024, 512, 512)
    commander_lr: float = 3.0e-4
    commander_cosine_loss_coeff: float = 1.0
    commander_z_norm_coeff: float = 1.0e-4
    commander_state_noise_std: float = 0.0

    def validate(self) -> None:
        self.horizon_steps = _require_positive_int("horizon_steps", self.horizon_steps)
        self.macro_frame_stride = _require_positive_int(
            "macro_frame_stride", self.macro_frame_stride
        )
        if self.macro_anchor_mode not in (
            "robot",
            "expert_heading",
            "robot_heading",
        ):
            msg = (
                "macro_anchor_mode must be 'robot', 'expert_heading' or "
                f"'robot_heading', got {self.macro_anchor_mode!r}."
            )
            raise ValueError(msg)
        self.z_dim = _require_positive_int("z_dim", self.z_dim)
        self.diffsr_feature_dim = _require_positive_int(
            "diffsr_feature_dim", self.diffsr_feature_dim
        )
        self.diffsr_embed_dim = _require_positive_int(
            "diffsr_embed_dim", self.diffsr_embed_dim
        )
        self.batch_size = _require_positive_int("batch_size", self.batch_size)
        self.num_updates = _require_positive_int("num_updates", self.num_updates)
        self.log_interval = _require_positive_int("log_interval", self.log_interval)
        self.eval_batches = _require_positive_int("eval_batches", self.eval_batches)
        if self.eval_batch_size is not None:
            self.eval_batch_size = _require_positive_int(
                "eval_batch_size", self.eval_batch_size
            )
        self.train_split = _normalize_split_value("train_split", self.train_split)
        self.eval_split = _normalize_split_value("eval_split", self.eval_split)
        self.eval_trajectory_fraction = _require_fraction(
            "eval_trajectory_fraction", self.eval_trajectory_fraction
        )
        self.trajectory_split_seed = int(self.trajectory_split_seed)
        self.preflight_batch_size = _require_positive_int(
            "preflight_batch_size", self.preflight_batch_size
        )
        self.encoder_window_mode = _normalize_encoder_window_mode(
            "encoder_window_mode", self.encoder_window_mode
        )
        if self.encoder_window_mode == "intermediate" and self.horizon_steps <= 1:
            msg = "encoder_window_mode='intermediate' requires horizon_steps > 1."
            raise ValueError(msg)
        self.source_history_steps = _require_non_negative_int(
            "source_history_steps", self.source_history_steps
        )
        self.source_anchor = str(self.source_anchor).strip().lower()
        if self.source_anchor not in {"current", "past_start"}:
            msg = (
                "source_anchor must be 'current' or 'past_start', got "
                f"{self.source_anchor!r}."
            )
            raise ValueError(msg)
        if self.source_anchor == "past_start" and self.source_history_steps < 1:
            msg = (
                "source_anchor='past_start' requires source_history_steps >= 1: "
                "with no past chunk there is no earlier frame to anchor on."
            )
            raise ValueError(msg)
        if self.source_history_steps > 0 and self.transition_objective != "jepa_ntp":
            msg = (
                "source_history_steps > 0 is implemented for "
                "transition_objective='jepa_ntp' only, got "
                f"{self.transition_objective!r}."
            )
            raise ValueError(msg)
        if self.source_history_steps > 0 and int(self.jepa_context_chunks) != 0:
            msg = (
                "source_history_steps > 0 requires jepa_context_chunks=0: the "
                "past chunk and the triplet's preceding chunk are two different "
                "ways to give the model history, and their interaction is "
                "untested."
            )
            raise ValueError(msg)
        window_suffix = _encoder_window_suffix_steps(self.encoder_window_mode)
        if window_suffix is not None and window_suffix > self.horizon_steps - 1:
            msg = (
                f"encoder_window_mode='suffix{window_suffix}' needs "
                f"suffix <= horizon_steps - 1 (= {self.horizon_steps - 1}): the "
                "suffix is taken from the intermediate window, which excludes "
                "the endpoint."
            )
            raise ValueError(msg)
        self.transition_objective = _normalize_transition_objective(
            "transition_objective", self.transition_objective
        )
        self.transition_offsets = tuple(
            _require_positive_int("transition_offsets", offset)
            for offset in self.transition_offsets
        )
        if not self.transition_offsets:
            if self.transition_objective in {
                "endpoint",
                "endpoint_delta",
                "reconstruction",
            }:
                self.transition_offsets = (self.horizon_steps,)
            else:
                self.transition_offsets = tuple(range(1, self.horizon_steps + 1))
        if tuple(sorted(set(self.transition_offsets))) != self.transition_offsets:
            msg = (
                "transition_offsets must be unique and strictly increasing, got "
                f"{self.transition_offsets!r}."
            )
            raise ValueError(msg)
        if self.transition_offsets[-1] > self.horizon_steps:
            msg = (
                "transition_offsets cannot exceed horizon_steps: "
                f"{self.transition_offsets!r} vs {self.horizon_steps}."
            )
            raise ValueError(msg)
        if self.transition_offsets[-1] != self.horizon_steps:
            msg = (
                "transition_offsets must include horizon_steps as the final "
                f"checkpoint, got {self.transition_offsets!r} with "
                f"horizon_steps={self.horizon_steps}."
            )
            raise ValueError(msg)
        if self.transition_objective in {
            "endpoint",
            "endpoint_delta",
            "reconstruction",
        } and self.transition_offsets != (self.horizon_steps,):
            msg = (
                f"transition_objective={self.transition_objective!r} requires "
                f"transition_offsets=({self.horizon_steps},), got "
                f"{self.transition_offsets!r}."
            )
            raise ValueError(msg)
        if self.latent_mode not in _LATENT_MODES:
            msg = (
                f"latent_mode must be one of {_LATENT_MODES}, got {self.latent_mode!r}."
            )
            raise ValueError(msg)
        self.reg_coeff = _require_non_negative_float("reg_coeff", self.reg_coeff)
        self.categorical_groups = _require_positive_int(
            "categorical_groups", self.categorical_groups
        )
        self.categorical_categories = _require_positive_int(
            "categorical_categories", self.categorical_categories
        )
        if (
            self.latent_mode in ("categorical", "gumbel_multicat")
            and self.z_dim % self.categorical_groups != 0
        ):
            msg = (
                f"latent_mode={self.latent_mode!r} requires z_dim divisible by "
                f"categorical_groups (per-group code dim = z_dim // groups): "
                f"z_dim={self.z_dim}, categorical_groups={self.categorical_groups}."
            )
            raise ValueError(msg)
        self.gaussian_logstd_min = float(self.gaussian_logstd_min)
        self.gaussian_logstd_max = float(self.gaussian_logstd_max)
        if self.gaussian_logstd_max <= self.gaussian_logstd_min:
            msg = (
                "gaussian_logstd_max must be > gaussian_logstd_min, got "
                f"{self.gaussian_logstd_max} <= {self.gaussian_logstd_min}."
            )
            raise ValueError(msg)
        self.gumbel_codebook_size = _require_positive_int(
            "gumbel_codebook_size", self.gumbel_codebook_size
        )
        self.gumbel_hard = bool(self.gumbel_hard)
        self.fsq_levels = tuple(
            _require_positive_int("fsq_levels", level) for level in self.fsq_levels
        )
        if self.latent_mode == "fsq" and any(level < 2 for level in self.fsq_levels):
            msg = f"fsq_levels must each be >= 2, got {self.fsq_levels!r}."
            raise ValueError(msg)
        self.sonic_fsq_levels = tuple(
            _require_positive_int("sonic_fsq_levels", level)
            for level in self.sonic_fsq_levels
        )
        if self.latent_mode == "sonic_fsq":
            if any(level < 2 for level in self.sonic_fsq_levels):
                msg = (
                    "sonic_fsq_levels must each be >= 2, got "
                    f"{self.sonic_fsq_levels!r}."
                )
                raise ValueError(msg)
            if self.z_dim != len(self.sonic_fsq_levels):
                msg = (
                    "latent_mode='sonic_fsq' publishes the quantizer output as the "
                    "command, so z_dim must equal len(sonic_fsq_levels): "
                    f"z_dim={self.z_dim}, "
                    f"len(sonic_fsq_levels)={len(self.sonic_fsq_levels)}."
                )
                raise ValueError(msg)
        self.vq_codebook_size = _require_positive_int(
            "vq_codebook_size", self.vq_codebook_size
        )
        self.vq_dead_code_reset_iters = int(self.vq_dead_code_reset_iters)
        self.encoder_hidden_dims = tuple(
            _require_positive_int("encoder_hidden_dims", dim)
            for dim in self.encoder_hidden_dims
        )
        self.encoder_activation = str(self.encoder_activation).strip().lower()
        if self.encoder_activation not in {"elu", "mish", "relu", "silu"}:
            msg = (
                "encoder_activation must be one of ['elu', 'mish', 'relu', "
                f"'silu'], got {self.encoder_activation!r}."
            )
            raise ValueError(msg)
        self.encoder_layer_norm = bool(self.encoder_layer_norm)
        self.diffsr_f_hidden_dims = tuple(
            _require_positive_int("diffsr_f_hidden_dims", dim)
            for dim in self.diffsr_f_hidden_dims
        )
        self.diffsr_g_hidden_dims = tuple(
            _require_positive_int("diffsr_g_hidden_dims", dim)
            for dim in self.diffsr_g_hidden_dims
        )
        self.diffsr_mu_hidden_dims = tuple(
            _require_positive_int("diffsr_mu_hidden_dims", dim)
            for dim in self.diffsr_mu_hidden_dims
        )
        self.diffsr_num_noises = _require_positive_int(
            "diffsr_num_noises", self.diffsr_num_noises
        )
        self.encoder_lr = _require_positive_float("encoder_lr", self.encoder_lr)
        self.diffsr_lr = _require_positive_float("diffsr_lr", self.diffsr_lr)
        self.weight_decay = _require_non_negative_float(
            "weight_decay", self.weight_decay
        )
        if self.grad_clip_norm is not None:
            self.grad_clip_norm = _require_positive_float(
                "grad_clip_norm", self.grad_clip_norm
            )
        self.reconstruction_norm_eps = _require_positive_float(
            "reconstruction_norm_eps", self.reconstruction_norm_eps
        )
        self.reconstruction_target = str(self.reconstruction_target).strip().lower()
        if self.reconstruction_target not in {
            "input_window",
            "endpoint",
            "full_window",
        }:
            msg = (
                "reconstruction_target must be 'input_window', 'endpoint', or "
                f"'full_window', got {self.reconstruction_target!r}."
            )
            raise ValueError(msg)
        self.jepa_context_chunks = int(self.jepa_context_chunks)
        if self.jepa_context_chunks not in (0, 1):
            msg = (
                "jepa_context_chunks must be 0 (chunk pair) or 1 (chunk "
                f"triplet), got {self.jepa_context_chunks}."
            )
            raise ValueError(msg)
        self.jepa_ntp_coeff = _require_non_negative_float(
            "jepa_ntp_coeff", self.jepa_ntp_coeff
        )
        self.jepa_ntp_head = str(self.jepa_ntp_head).strip().lower()
        self.jepa_ntp_chunk_anchor = str(self.jepa_ntp_chunk_anchor).strip().lower()
        if self.jepa_ntp_chunk_anchor not in {"executed", "next"}:
            msg = (
                "jepa_ntp_chunk_anchor must be 'executed' or 'next', got "
                f"{self.jepa_ntp_chunk_anchor!r}."
            )
            raise ValueError(msg)
        self.jepa_ntp_chunk_span = str(self.jepa_ntp_chunk_span).strip().lower()
        if self.jepa_ntp_chunk_span not in {"next", "boundary_next"}:
            msg = (
                "jepa_ntp_chunk_span must be 'next' or 'boundary_next', got "
                f"{self.jepa_ntp_chunk_span!r}."
            )
            raise ValueError(msg)
        if (
            self.jepa_ntp_chunk_span == "boundary_next"
            and self.jepa_ntp_chunk_anchor != "executed"
        ):
            msg = (
                "jepa_ntp_chunk_span='boundary_next' requires "
                "jepa_ntp_chunk_anchor='executed': the boundary frame is "
                "defined in the executed chunk's own frame."
            )
            raise ValueError(msg)
        self.jepa_endpoint_coeff = _require_non_negative_float(
            "jepa_endpoint_coeff", self.jepa_endpoint_coeff
        )
        self.jepa_token_pred_coeff = _require_non_negative_float(
            "jepa_token_pred_coeff", self.jepa_token_pred_coeff
        )
        if self.jepa_token_pred_coeff > 0 and self.jepa_ntp_head == "mlp":
            msg = (
                "jepa_token_pred_coeff > 0 requires a diffusion jepa_ntp_head: "
                "under the mlp head the token-prediction MSE already is the "
                "NTP term (jepa_ntp_coeff), and a second copy would "
                "double-count it."
            )
            raise ValueError(msg)
        if self.jepa_ntp_head not in {"mlp", "diff_token", "diff_chunk", "diff_pair"}:
            msg = (
                "jepa_ntp_head must be 'mlp', 'diff_token', 'diff_chunk' or "
                f"'diff_pair', got {self.jepa_ntp_head!r}."
            )
            raise ValueError(msg)
        if self.jepa_ntp_head != "mlp":
            if self.jepa_loss != "sigreg_ebm":
                msg = (
                    "jepa_ntp_head="
                    f"{self.jepa_ntp_head!r} requires jepa_loss='sigreg_ebm'; "
                    f"got {self.jepa_loss!r}."
                )
                raise ValueError(msg)
            if int(self.jepa_context_chunks) != 0:
                msg = (
                    "The diffusion NTP heads support only the chunk pair "
                    f"(jepa_context_chunks=0), got {self.jepa_context_chunks}."
                )
                raise ValueError(msg)
        self.jepa_target_encoder_mode = (
            str(self.jepa_target_encoder_mode).strip().lower()
        )
        if self.jepa_target_encoder_mode not in {"ema", "online", "stopgrad"}:
            msg = (
                "jepa_target_encoder_mode must be 'ema', 'online' or "
                f"'stopgrad', got {self.jepa_target_encoder_mode!r}."
            )
            raise ValueError(msg)
        self.diffsr_state_output_init_std = _require_non_negative_float(
            "diffsr_state_output_init_std", self.diffsr_state_output_init_std
        )
        self.cotrain_commander = bool(self.cotrain_commander)
        self.commander_hidden_dims = tuple(
            _require_positive_int("commander_hidden_dims", dim)
            for dim in self.commander_hidden_dims
        )
        self.commander_lr = _require_positive_float("commander_lr", self.commander_lr)
        self.commander_cosine_loss_coeff = _require_non_negative_float(
            "commander_cosine_loss_coeff", self.commander_cosine_loss_coeff
        )
        self.commander_z_norm_coeff = _require_non_negative_float(
            "commander_z_norm_coeff", self.commander_z_norm_coeff
        )
        self.commander_state_noise_std = _require_non_negative_float(
            "commander_state_noise_std", self.commander_state_noise_std
        )
        self.commander_language_embeddings_path = str(
            self.commander_language_embeddings_path
        ).strip()
        if self.cotrain_commander and not self.commander_language_embeddings_path:
            msg = (
                "commander_language_embeddings_path is required when "
                "cotrain_commander is enabled."
            )
            raise ValueError(msg)
        self.device = str(self.device)

    def latent_spec(self) -> SkillLatentSpec:
        """Project the latent-method fields into the encoder factory's spec."""
        return SkillLatentSpec(
            latent_mode=self.latent_mode,
            gaussian_logstd_min=self.gaussian_logstd_min,
            gaussian_logstd_max=self.gaussian_logstd_max,
            categorical_groups=self.categorical_groups,
            categorical_categories=self.categorical_categories,
            gumbel_codebook_size=self.gumbel_codebook_size,
            gumbel_tau_start=self.gumbel_tau_start,
            gumbel_tau_end=self.gumbel_tau_end,
            gumbel_tau_anneal_iters=self.gumbel_tau_anneal_iters,
            gumbel_hard=self.gumbel_hard,
            fsq_levels=tuple(self.fsq_levels),
            sonic_fsq_levels=tuple(self.sonic_fsq_levels),
            vq_codebook_size=self.vq_codebook_size,
            vq_ema_decay=self.vq_ema_decay,
            vq_dead_code_reset_iters=self.vq_dead_code_reset_iters,
        )

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _jsonable(asdict(self)))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> HighLevelSkillDiffSRConfig:
        known_fields = {item.name for item in fields(cls)}
        kwargs = {key: values[key] for key in known_fields if key in values}
        tuple_fields = {
            "encoder_hidden_dims",
            "diffsr_f_hidden_dims",
            "diffsr_g_hidden_dims",
            "diffsr_mu_hidden_dims",
            "fsq_levels",
            "sonic_fsq_levels",
            "commander_hidden_dims",
            "transition_offsets",
        }
        for key in tuple_fields:
            if key in kwargs:
                kwargs[key] = tuple(int(item) for item in kwargs[key])
        config = cls(**kwargs)
        config.validate()
        return config


class FrozenHighLevelSkillCommandSampler:
    """High-level encoder used as an online latent-command source.

    The default mode is frozen and preserves the original rollout behavior.  When
    online finetuning is enabled, the sampler also caches one macro input per
    renewed command so PPO minibatches can recompute skill commands with gradient
    flow into the skill encoder.
    """

    def __init__(
        self,
        *,
        env: object,
        checkpoint_path: str | Path,
        latent_dim: int,
        latent_steps_min: int,
        latent_steps_max: int,
        discover_env_method: Callable[[object, str], Callable[..., Any] | None],
        horizon_steps: int | None = None,
        command_phase_mode: str = "none",
        code_latent_dim: int | None = None,
        phase_period: int | None = None,
        phase_source: str = "hold",
        command_mode: str = "z",
        device: torch.device | str | None = None,
        finetune_enabled: bool = False,
        achieved_coeff: float = 0.0,
        pg_coeff: float = 0.05,
        offline_diffsr_coeff: float = 1.0,
        anchor_coeff: float = 0.01,
        z_norm_coeff: float | None = None,
        lr: float = 3.0e-5,
        grad_clip_norm: float | None = 1.0,
        offline_batch_size: int = 8192,
        update_interval: int = 1,
        train_diffsr: bool = False,
    ) -> None:
        self.latent_dim = _require_positive_int("latent_dim", latent_dim)
        self.latent_steps_min = max(1, int(latent_steps_min))
        self.latent_steps_max = max(self.latent_steps_min, int(latent_steps_max))
        self.finetune_enabled = bool(finetune_enabled)
        self.achieved_coeff = float(achieved_coeff)
        self.pg_coeff = _require_non_negative_float("pg_coeff", pg_coeff)
        self.offline_diffsr_coeff = _require_non_negative_float(
            "offline_diffsr_coeff", offline_diffsr_coeff
        )
        self.anchor_coeff = _require_non_negative_float("anchor_coeff", anchor_coeff)
        self.lr = _require_positive_float("lr", lr)
        self.grad_clip_norm = (
            None
            if grad_clip_norm is None
            else _require_positive_float("grad_clip_norm", grad_clip_norm)
        )
        self.offline_batch_size = _require_positive_int(
            "offline_batch_size", offline_batch_size
        )
        self.update_interval = _require_positive_int("update_interval", update_interval)
        self.train_diffsr = bool(train_diffsr)
        self.command_mode = _normalize_command_mode(command_mode)
        self.command_phase_mode = str(command_phase_mode).strip().lower()
        if self.command_phase_mode not in {"none", "sin_cos"}:
            msg = (
                "command_phase_mode must be 'none' or 'sin_cos', got "
                f"{command_phase_mode!r}."
            )
            raise ValueError(msg)
        self.phase_dim = 2 if self.command_phase_mode == "sin_cos" else 0
        self.phase_period = (
            _require_positive_int("phase_period", int(phase_period))
            if phase_period is not None
            else self.latent_steps_max
        )
        self.phase_source = str(phase_source).strip().lower()
        if self.phase_source not in {"hold", "episode"}:
            msg = f"phase_source must be 'hold' or 'episode', got {phase_source!r}."
            raise ValueError(msg)
        if self.phase_source == "episode" and self.latent_steps_max > 1:
            # The posterior update detects code renewal by the phase value, and
            # an episode clock no longer marks renewals. Live phase exists for
            # hold 1, where the hold clock is constant and renewal is every step.
            msg = (
                "phase_source='episode' requires latent_steps_max == 1, got "
                f"{self.latent_steps_max}."
            )
            raise ValueError(msg)
        self.device = _resolve_device(device, env)
        from rlopt.env_interface import require_imitation_interface, supports
        from rlopt.env_interface import resolve_imitation_interface

        self._env_interface = resolve_imitation_interface(env)
        self._current_macro_sampler = require_imitation_interface(
            env,
            "current_expert_macro_transition_batch",
            purpose="command_source='hl_skill' requires it but",
        )
        self._offline_macro_sampler = (
            self._env_interface.sample_expert_macro_transition_batch
            if supports(self._env_interface, "sample_expert_macro_transition_batch")
            else None
        )
        if self.finetune_enabled and self._offline_macro_sampler is None:
            msg = (
                "Online high-level skill finetuning requires the environment to "
                "expose sample_expert_macro_transition_batch(...)."
            )
            raise ValueError(msg)

        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=self.device,
            weights_only=False,
        )
        self.config = HighLevelSkillDiffSRConfig.from_dict(checkpoint["config"])
        if self.config.source_anchor != "current":
            raise ValueError(
                "Live skill commands require source_anchor='current'; "
                "a past_start checkpoint needs a delayed heading anchor."
            )
        if self.finetune_enabled and self.config.source_history_steps > 0:
            raise ValueError(
                "Online finetuning with source_history_steps > 0 is not supported."
            )
        if self.finetune_enabled and self.config.transition_objective not in (
            "endpoint",
            "jepa_ntp",
        ):
            msg = (
                "Online skill-encoder finetuning supports the endpoint DiffSR "
                "and jepa_ntp objectives; freeze this encoder or add the "
                f"matching {self.config.transition_objective!r} online loss."
            )
            raise ValueError(msg)
        if (
            horizon_steps is not None
            and int(horizon_steps) != self.config.horizon_steps
        ):
            msg = (
                "Configured hl_skill_horizon_steps does not match checkpoint "
                f"horizon_steps: {int(horizon_steps)} != {self.config.horizon_steps}."
            )
            raise ValueError(msg)
        self._require_matching_macro_frame_stride(env)
        self._require_matching_macro_anchor_mode(env)
        self.skill_z_dim = int(self.config.z_dim)
        self.command_code_dim = self._command_code_dim_for_mode()
        if (
            code_latent_dim is not None
            and int(code_latent_dim) > 0
            and int(code_latent_dim) != self.command_code_dim
        ):
            msg = (
                "Frozen high-level skill code_latent_dim must match command-mode "
                f"pre-phase width: {int(code_latent_dim)} != "
                f"{self.command_code_dim} for command_mode={self.command_mode!r}."
            )
            raise ValueError(msg)
        expected_latent_dim = self.command_code_dim + self.phase_dim
        if expected_latent_dim != self.latent_dim:
            msg = (
                "Frozen high-level skill command width must match ipmd.latent_dim: "
                f"command_mode={self.command_mode!r} pre-phase width "
                f"{self.command_code_dim} + phase_dim {self.phase_dim} "
                f"!= {self.latent_dim}."
            )
            raise ValueError(msg)

        state_dict = checkpoint["skill_encoder_state_dict"]
        self.encoder_window_steps = _encoder_window_steps(self.config)
        self.state_dim = self._state_dim_from_encoder_state(
            state_dict,
            window_steps=self.encoder_window_steps,
        )
        self.skill_encoder = build_skill_encoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.encoder_hidden_dims,
            spec=self.config.latent_spec(),
            activation=self.config.encoder_activation,
            layer_norm=self.config.encoder_layer_norm,
        ).to(self.device)
        self.skill_encoder.load_state_dict(state_dict)

        self.initial_skill_encoder = build_skill_encoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.encoder_hidden_dims,
            spec=self.config.latent_spec(),
            activation=self.config.encoder_activation,
            layer_norm=self.config.encoder_layer_norm,
        ).to(self.device)
        self.initial_skill_encoder.load_state_dict(state_dict)
        self.initial_skill_encoder.eval()
        self.initial_skill_encoder.requires_grad_(False)

        self.diffsr = _build_diffsr(self.config, self.state_dim, self.device).to(
            self.device
        )
        diffsr_state = checkpoint.get("diffsr_state_dict")
        needs_diffsr_checkpoint = self.finetune_enabled or self.command_mode != "z"
        if diffsr_state is None and needs_diffsr_checkpoint:
            msg = (
                "Online high-level skill finetuning and non-z command modes "
                "require checkpoints with diffsr_state_dict."
            )
            raise ValueError(msg)
        if diffsr_state is not None and needs_diffsr_checkpoint:
            self.diffsr.load_state_dict(diffsr_state)
        feature_norm_state = checkpoint.get("feature_normalization_state_dict")
        obs_norm = getattr(self.diffsr, "obs_norm", None)
        if (
            needs_diffsr_checkpoint
            and isinstance(obs_norm, nn.Module)
            and feature_norm_state
        ):
            obs_norm.load_state_dict(feature_norm_state)

        # A merged JEPA checkpoint trains phi in its next-token prediction
        # DiffSR head. Its endpoint DiffSR can be present but receive zero
        # objective weight. Non-z command modes must therefore restore the
        # trained head instead of silently publishing the endpoint head.
        self.command_diffsr = self.diffsr
        if (
            self.command_mode != "z"
            and self.config.transition_objective == "jepa_ntp"
            and float(self.config.jepa_endpoint_coeff) == 0.0
            and str(self.config.jepa_ntp_head) != "mlp"
        ):
            if (
                self.config.jepa_loss != "sigreg_ebm"
                or float(self.config.jepa_ntp_coeff) <= 0.0
            ):
                raise ValueError("The checkpoint has no trained DiffSR command head.")
            jepa_state = checkpoint.get("jepa_state_dict")
            ntp_state = jepa_state.get("ntp_diffsr") if jepa_state else None
            if ntp_state is None:
                msg = (
                    "Non-z commands from a JEPA DiffSR checkpoint require "
                    "jepa_state_dict['ntp_diffsr']; the trained phi head is missing."
                )
                raise ValueError(msg)
            self.command_diffsr = _build_diffsr(
                self.config,
                self.state_dim,
                self.device,
                next_obs_dim=_jepa_ntp_target_dim(self.config, self.state_dim),
            ).to(self.device)
            self.command_diffsr.load_state_dict(ntp_state)
            self.command_diffsr.eval()
            self.command_diffsr.requires_grad_(False)

        self.z_norm_coeff = (
            _require_non_negative_float("z_norm_coeff", z_norm_coeff)
            if z_norm_coeff is not None
            else float(self.config.reg_coeff)
        )

        self.skill_encoder.train(self.finetune_enabled)
        self.skill_encoder.requires_grad_(self.finetune_enabled)
        self.diffsr.train(self.finetune_enabled and self.train_diffsr)
        self.diffsr.requires_grad_(self.finetune_enabled and self.train_diffsr)

        # jepa_ntp keeps its objective heads OUTSIDE the encoder: an EMA target
        # encoder that supplies the next chunk's token, a predictor, and the
        # bilinear energy pair. The pretrain checkpoint carries all four
        # (`jepa_state_dict`), so online finetuning continues the same
        # objective rather than swapping in a different one.
        self.jepa_target_encoder: nn.Module | None = None
        self.jepa_predictor: nn.Module | None = None
        self.jepa_g: nn.Module | None = None
        self.jepa_f: nn.Module | None = None
        if self.finetune_enabled and self.config.transition_objective == "jepa_ntp":
            self._restore_jepa_modules(checkpoint)

        self.optimizer: torch.optim.Optimizer | None = None
        if self.finetune_enabled:
            params: list[nn.Parameter] = list(self.skill_encoder.parameters())
            if self.train_diffsr:
                params.extend(list(self.diffsr.parameters()))
            params.extend(self._jepa_head_parameters())
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.finetune_updates = 0

        self._codes: Tensor | None = None
        self._latent_steps: Tensor | None = None
        self._episode_steps: Tensor | None = None
        self._active_macro_ids: Tensor | None = None
        self._cache_state_chunks: list[Tensor] = []
        self._current_command_source: Tensor | None = None
        self._cache_future_window_chunks: list[Tensor] = []
        self._cache_target_chunks: list[Tensor] = []
        self._cache_initial_z_chunks: list[Tensor] = []
        self._next_macro_id = 0

    @staticmethod
    def _state_dim_from_encoder_state(
        state_dict: Mapping[str, Tensor],
        *,
        window_steps: int,
    ) -> int:
        first_weight = state_dict.get("net.0.weight")
        if first_weight is None or first_weight.ndim != 2:
            msg = (
                "Checkpoint skill encoder is missing first linear weight net.0.weight."
            )
            raise ValueError(msg)
        input_dim = int(first_weight.shape[1])
        divisor = int(window_steps) + 1
        if input_dim % divisor != 0:
            msg = (
                "Checkpoint skill encoder input width is incompatible with "
                f"window_steps={window_steps}: input_dim={input_dim}."
            )
            raise ValueError(msg)
        return input_dim // divisor

    def _require_matching_macro_frame_stride(self, env: object) -> None:
        """Refuse an encoder pretrained on a different macro-window cadence.

        The macro state is the same width at every stride, so pairing a
        stride-1 encoder with a stride-5 environment (or the reverse) produces
        no shape error and no warning -- only a silently off-distribution
        command. An environment that does not publish its stride is a pre-
        stride surface, which can only be serving 1.
        """
        from rlopt.env_interface import resolve_imitation_interface, supports

        interface = resolve_imitation_interface(env)
        if not supports(interface, "expert_macro_frame_stride"):
            env_stride = 1
        else:
            env_stride = int(interface.expert_macro_frame_stride())
        checkpoint_stride = int(self.config.macro_frame_stride)
        if env_stride != checkpoint_stride:
            msg = (
                "Skill encoder macro-window stride does not match the "
                f"environment: checkpoint was pretrained at stride "
                f"{checkpoint_stride}, env.expert_macro_frame_stride is "
                f"{env_stride}. The macro state is the same width at both, so "
                "this cannot be detected downstream -- set the environment to "
                "the encoder's stride or pretrain a new encoder."
            )
            raise ValueError(msg)

    def _require_matching_macro_anchor_mode(self, env: object) -> None:
        """Refuse an encoder pretrained under a different macro frame convention.

        Same detection problem as the stride: the macro state has the same
        width in the "robot", "expert_heading" and "robot_heading"
        conventions, so pairing a
        mismatched encoder produces no shape error -- only a silently
        off-distribution command. An environment that does not publish its
        mode is a pre-mode surface, which can only be serving "robot".
        """
        from rlopt.env_interface import resolve_imitation_interface, supports

        interface = resolve_imitation_interface(env)
        if not supports(interface, "expert_macro_anchor_mode"):
            env_mode = "robot"
        else:
            env_mode = str(interface.expert_macro_anchor_mode())
        checkpoint_mode = str(self.config.macro_anchor_mode)
        if env_mode != checkpoint_mode:
            msg = (
                "Skill encoder macro-window anchor mode does not match the "
                f"environment: checkpoint was pretrained under "
                f"{checkpoint_mode!r}, env.expert_macro_anchor_mode is "
                f"{env_mode!r}. The macro state is the same width in both, so "
                "this cannot be detected downstream -- set the environment to "
                "the encoder's mode or pretrain a new encoder."
            )
            raise ValueError(msg)

    def _command_code_dim_for_mode(self) -> int:
        if self.command_mode == "z":
            return int(self.config.z_dim)
        if self.command_mode == "phi":
            return int(self.config.diffsr_feature_dim)
        if self.command_mode == "z_phi":
            return int(self.config.z_dim) + int(self.config.diffsr_feature_dim)
        msg = f"Unsupported high-level skill command mode: {self.command_mode!r}."
        raise ValueError(msg)

    @staticmethod
    def _done_mask(
        td: TensorDictBase, *, batch_size: int, device: torch.device
    ) -> Tensor:
        done_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        candidate_keys: list[str | tuple[str, ...]] = [
            "done",
            "terminated",
            "truncated",
            "is_init",
            ("next", "done"),
            ("next", "terminated"),
            ("next", "truncated"),
            ("next", "is_init"),
        ]
        available_keys = td.keys(True)
        for key in candidate_keys:
            if key not in available_keys:
                continue
            value = cast(Tensor, td.get(key)).reshape(-1).to(device=device).bool()
            if value.numel() == batch_size:
                done_mask |= value
        return done_mask

    def _sample_steps(self, count: int, *, device: torch.device) -> Tensor:
        if self.latent_steps_min == self.latent_steps_max:
            return torch.full(
                (count,),
                self.latent_steps_min,
                device=device,
                dtype=torch.long,
            )
        return torch.randint(
            self.latent_steps_min,
            self.latent_steps_max + 1,
            (count,),
            device=device,
        )

    def _sample_offline_macro_batch(
        self,
        batch_size: int,
        *,
        horizon_steps: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One expert macro batch; `horizon_steps` overrides for chunk pairs."""
        if self._offline_macro_sampler is None:
            msg = "sample_expert_macro_transition_batch(...) is unavailable."
            raise RuntimeError(msg)
        horizon = int(horizon_steps or self.config.horizon_steps)
        batch = self._offline_macro_sampler(
            batch_size=int(batch_size),
            horizon_steps=horizon,
            split=self.config.train_split,
            eval_fraction=float(self.config.eval_trajectory_fraction),
            split_seed=int(self.config.trajectory_split_seed),
        )
        return _validate_macro_batch(
            batch,
            batch_size=int(batch_size),
            horizon_steps=horizon,
            device=self.device,
            state_dim=self.state_dim,
            source="Offline expert",
        )

    def _command_code_from_state_z(self, source: Tensor, z: Tensor) -> Tensor:
        if self.command_mode == "z":
            return z
        phi = self.command_diffsr.forward_phi(source, z)
        if self.command_mode == "phi":
            return phi
        if self.command_mode == "z_phi":
            return torch.cat((z, phi), dim=-1)
        msg = f"Unsupported high-level skill command mode: {self.command_mode!r}."
        raise ValueError(msg)

    def _append_command_phase(self, code_latents: Tensor, phase: Tensor) -> Tensor:
        if self.phase_dim == 0:
            return code_latents
        angle = phase.to(device=code_latents.device, dtype=code_latents.dtype)
        angle = angle.reshape(-1) * (2.0 * math.pi)
        phase_features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        return torch.cat((code_latents, phase_features), dim=-1)

    @torch.no_grad()
    def _encode_current_macro_batch(
        self,
        env_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Keep the five-tensor return contract used by planner adapters. Only
        # phi modes need the past source; z deployment retains its old gather.
        history_steps = (
            int(self.config.source_history_steps) if self.command_mode != "z" else 0
        )
        batch = self._current_macro_sampler(
            horizon_steps=int(self.config.horizon_steps),
            env_ids=env_ids,
            **({"state_history_steps": history_steps} if history_steps else {}),
        )
        batch_size = int(env_ids.numel())
        state, future_window, target = _validate_macro_batch(
            batch,
            batch_size=batch_size,
            horizon_steps=int(self.config.horizon_steps),
            device=self.device,
            state_dim=self.state_dim,
            source="Current expert",
        )
        z = self.skill_encoder(state, _encoder_input_window(self.config, future_window))
        initial_z = self.initial_skill_encoder(
            state,
            _encoder_input_window(self.config, future_window),
        )
        if history_steps > 0:
            state_history = _macro_batch_state_history(
                batch,
                batch_size=batch_size,
                history_steps=history_steps,
                state_dim=self.state_dim,
                device=self.device,
            )
            self._current_command_source = state_history.reshape(batch_size, -1)
        else:
            self._current_command_source = state
        return z, state, future_window, target, initial_z

    def start_rollout_cache(self) -> None:
        """Reset one-rollout macro cache used by online skill finetuning."""
        if not self.finetune_enabled:
            return
        self._cache_state_chunks.clear()
        self._cache_future_window_chunks.clear()
        self._cache_target_chunks.clear()
        self._cache_initial_z_chunks.clear()
        self._next_macro_id = 0
        if self._latent_steps is not None:
            self._latent_steps.zero_()
        if self._episode_steps is not None:
            self._episode_steps.zero_()
        if self._active_macro_ids is not None:
            self._active_macro_ids.fill_(-1)

    def _append_rollout_cache(
        self,
        *,
        state: Tensor,
        future_window: Tensor,
        target: Tensor,
        initial_z: Tensor,
    ) -> Tensor:
        count = int(state.shape[0])
        macro_ids = torch.arange(
            self._next_macro_id,
            self._next_macro_id + count,
            device=self.device,
            dtype=torch.long,
        )
        self._next_macro_id += count
        self._cache_state_chunks.append(state.detach().clone())
        self._cache_future_window_chunks.append(future_window.detach().clone())
        self._cache_target_chunks.append(target.detach().clone())
        self._cache_initial_z_chunks.append(initial_z.detach().clone())
        return macro_ids

    def _cached_rollout_tensors(self) -> dict[str, Tensor]:
        if not self._cache_state_chunks:
            msg = "No high-level skill macro cache is available for this rollout."
            raise RuntimeError(msg)
        return {
            "state": torch.cat(self._cache_state_chunks, dim=0),
            "future_window": torch.cat(self._cache_future_window_chunks, dim=0),
            "target": torch.cat(self._cache_target_chunks, dim=0),
            "initial_z": torch.cat(self._cache_initial_z_chunks, dim=0),
        }

    def _z_diagnostics_tensors(self, z: Tensor, *, prefix: str) -> dict[str, Tensor]:
        if int(z.shape[0]) < 2:
            rank = torch.zeros((), device=z.device, dtype=z.dtype)
        else:
            centered = z - z.mean(dim=0, keepdim=True)
            singular_values = torch.linalg.svdvals(centered)
            total = singular_values.sum()
            if bool((total <= 1.0e-12).item()):
                rank = torch.zeros((), device=z.device, dtype=z.dtype)
            else:
                probs = singular_values / total
                entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
                rank = torch.exp(entropy)
        z_std = z.std(dim=0, unbiased=False)
        return {
            f"{prefix}_z_abs_mean": z.abs().mean().detach(),
            f"{prefix}_z_rms": z.pow(2).mean().sqrt().detach(),
            f"{prefix}_z_dim_std_mean": z_std.mean().detach(),
            f"{prefix}_z_effective_rank": rank.detach(),
        }

    def latent_commands_from_rollout_batch(
        self,
        batch: TensorDictBase,
        *,
        detach: bool,
    ) -> Tensor:
        macro_id = cast(Tensor, batch.get(("hl_skill", "macro_id"))).reshape(-1)
        phase = cast(Tensor, batch.get(("hl_skill", "phase"))).reshape(-1)
        if macro_id.numel() == 0:
            return torch.empty(0, self.latent_dim, device=self.device)
        macro_id = macro_id.to(device=self.device, dtype=torch.long)
        cached = self._cached_rollout_tensors()
        if bool((macro_id < 0).any()) or bool(
            (macro_id >= cached["state"].shape[0]).any()
        ):
            msg = "Rollout contains high-level skill macro IDs outside the cache."
            raise RuntimeError(msg)
        state = cached["state"].index_select(0, macro_id)
        future_window = cached["future_window"].index_select(0, macro_id)
        if detach:
            with torch.no_grad():
                z = self.skill_encoder(
                    state,
                    _encoder_input_window(self.config, future_window),
                )
                command_code = self._command_code_from_state_z(state, z)
                command = self._append_command_phase(command_code, phase)
        else:
            z = self.skill_encoder(
                state, _encoder_input_window(self.config, future_window)
            )
            command_code = self._command_code_from_state_z(state, z)
            command = self._append_command_phase(command_code, phase)
        return command.reshape(*batch.batch_size, self.latent_dim)

    def _restore_jepa_modules(self, checkpoint: dict[str, Any]) -> None:
        """Rebuild the pretrain objective's heads from the checkpoint.

        Refusing here rather than re-initializing is deliberate: freshly
        random predictor and energy heads would make the first online updates
        a random-direction pull on an encoder that took 50k pretrain updates
        to shape.
        """
        state = checkpoint.get("jepa_state_dict")
        if not state:
            msg = (
                "Online jepa_ntp finetuning needs 'jepa_state_dict' in the "
                "skill checkpoint (target encoder, predictor, and energy "
                "heads). This checkpoint predates it or was written by a "
                "non-jepa run."
            )
            raise ValueError(msg)
        if (
            str(self.config.jepa_target_encoder_mode) != "ema"
            or int(self.config.jepa_context_chunks) != 0
            or str(getattr(self.config, "jepa_ntp_head", "mlp")) != "mlp"
        ):
            msg = (
                "Online jepa_ntp finetuning supports only the chunk-pair EMA "
                "recipe; this checkpoint was pretrained with "
                f"jepa_target_encoder_mode="
                f"{self.config.jepa_target_encoder_mode!r}, "
                f"jepa_context_chunks={self.config.jepa_context_chunks}."
            )
            raise ValueError(msg)
        z_dim = int(self.config.z_dim)
        energy_dim = int(self.config.jepa_energy_dim)
        self.jepa_target_encoder = copy.deepcopy(self.skill_encoder)
        self.jepa_target_encoder.load_state_dict(state["target_encoder"])
        self.jepa_target_encoder.eval()
        self.jepa_target_encoder.requires_grad_(False)
        self.jepa_predictor = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, z_dim),
        ).to(self.device)
        self.jepa_predictor.load_state_dict(state["predictor"])
        self.jepa_g = nn.Sequential(
            nn.Linear(z_dim, 512), nn.SiLU(), nn.Linear(512, energy_dim)
        ).to(self.device)
        self.jepa_g.load_state_dict(state["g"])
        self.jepa_f = nn.Sequential(
            nn.Linear(z_dim, 512), nn.SiLU(), nn.Linear(512, energy_dim)
        ).to(self.device)
        self.jepa_f.load_state_dict(state["f"])

    def _jepa_head_parameters(self) -> list[nn.Parameter]:
        """Trainable JEPA heads; the EMA target encoder is never among them."""
        heads = [self.jepa_predictor, self.jepa_g, self.jepa_f]
        return [p for head in heads if head is not None for p in head.parameters()]

    def on_after_finetune_step(self) -> None:
        """EMA the target encoder toward the online one, after the step.

        The momentum is per FINETUNE UPDATE, exactly as in pretraining. Online
        those updates are rare (one per ``update_interval`` RL updates against
        50k in pretraining), so with the pretrain momentum the target stays
        close to the pretrained encoder: the objective reads as "predict what
        the PRETRAINED token geometry says the next achieved chunk is", while
        the online encoder adapts underneath it. That is consistent with the
        anchor loss, which already pulls z toward the pretrained encoder.
        Momentum 1.0 freezes the target outright.
        """
        if self.jepa_target_encoder is None:
            return
        momentum = float(self.config.jepa_ema_momentum)
        with torch.no_grad():
            for target, online in zip(
                self.jepa_target_encoder.parameters(),
                self.skill_encoder.parameters(),
                strict=True,
            ):
                target.mul_(momentum).add_(online, alpha=1.0 - momentum)

    def should_update_online(self, update_idx: int) -> bool:
        return self.finetune_enabled and int(update_idx) % self.update_interval == 0

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = list(self.skill_encoder.parameters())
        if self.train_diffsr:
            params.extend(list(self.diffsr.parameters()))
        params.extend(self._jepa_head_parameters())
        return params

    def _jepa_chunk_pair(
        self, state: Tensor, window: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Split one 2H window into the chunk pair the NTP objective reads.

        Identical construction to `_jepa_train_step`: chunk 1 is the sampled
        state plus frames 1..H; chunk 2 is frame H-1 (its own state) plus
        frames H+1..2H, re-anchored onto its own heading frame, which is what
        the encoder would see one publication later.
        """
        horizon = int(self.config.horizon_steps)
        chunk1_window = window[:, :horizon]
        chunk2_anchor = window[:, horizon - 1]
        chunk2_state = _reanchor_heading_frames(chunk2_anchor, chunk2_anchor)
        chunk2_window = _reanchor_heading_frames(
            window[:, horizon : 2 * horizon], chunk2_anchor
        )
        z1 = self.skill_encoder(
            state, _encoder_input_window(self.config, chunk1_window)
        )
        with torch.no_grad():
            if str(self.config.jepa_target_encoder_mode) == "ema":
                assert self.jepa_target_encoder is not None
                self.jepa_target_encoder.eval()
                target_encoder = self.jepa_target_encoder
            else:
                # online / stopgrad: the ONE online encoder is the target.
                target_encoder = self.skill_encoder
            z2, _, _ = target_encoder.encode(
                chunk2_state,
                _encoder_input_window(self.config, chunk2_window),
                deterministic=True,
            )
        return z1, z2, chunk1_window

    def _jepa_objective(
        self, state: Tensor, window: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """The pretrain NTP objective, evaluated on one 2H batch."""
        assert self.jepa_predictor is not None
        z1, z2, chunk1_window = self._jepa_chunk_pair(state, window)
        prediction = self.jepa_predictor(z1)
        horizon = int(self.config.horizon_steps)
        jepa_loss = str(self.config.jepa_loss)
        metrics: dict[str, Tensor] = {}
        if jepa_loss == "sigreg_ebm":
            endpoint = window[:, horizon - 1]
            if self.train_diffsr:
                self.diffsr.update_obs_norm(endpoint.detach())
            zero_reward = torch.zeros(state.shape[0], 1, device=self.device)
            _, diffsr_loss, _ = self.diffsr.compute_loss(
                state, z1, endpoint, zero_reward
            )
            ntp = F.mse_loss(prediction, z2)
            sigreg = _sigreg_epps_pulley(
                z1, num_sketches=int(self.config.jepa_sigreg_sketches)
            )
            objective = (
                diffsr_loss
                + float(self.config.jepa_ntp_coeff) * ntp
                + float(self.config.jepa_sigreg_coeff) * sigreg
            )
            metrics["hl_skill_jepa_ntp"] = ntp.detach()
            metrics["hl_skill_jepa_sigreg"] = sigreg.detach()
            metrics["hl_skill_jepa_diffsr"] = diffsr_loss.detach()
        elif jepa_loss == "sigreg":
            ntp = F.mse_loss(prediction, z2)
            sigreg = _sigreg_epps_pulley(
                z1, num_sketches=int(self.config.jepa_sigreg_sketches)
            )
            objective = ntp + float(self.config.jepa_sigreg_coeff) * sigreg
            metrics["hl_skill_jepa_ntp"] = ntp.detach()
            metrics["hl_skill_jepa_sigreg"] = sigreg.detach()
        else:
            assert self.jepa_g is not None
            assert self.jepa_f is not None
            logits = (self.jepa_g(prediction) @ self.jepa_f(z2).T) / float(
                self.config.jepa_tau
            )
            labels = torch.arange(logits.shape[0], device=logits.device)
            objective = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )
            metrics["hl_skill_jepa_infonce"] = objective.detach()
        del chunk1_window
        return objective, metrics

    def _jepa_online_finetune_loss(
        self, pg_loss: Tensor, offline_size: int
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Online finetune under the SAME objective the encoder was pretrained on.

        The endpoint path cannot be reused here: a jepa_ntp encoder's token is
        shaped by next-chunk prediction against an EMA target, and finetuning
        it with the DiffSR endpoint head would silently optimize a different
        objective than the one that produced the checkpoint.
        """
        horizon = int(self.config.horizon_steps)
        # Two adjacent chunks per row, hence 2H frames.
        state, window, _ = self._sample_offline_macro_batch(
            offline_size, horizon_steps=2 * horizon
        )
        offline_objective, metrics = self._jepa_objective(state, window)

        with torch.no_grad():
            initial_z = self.initial_skill_encoder(
                state,
                _encoder_input_window(self.config, window[:, :horizon]),
            )
        z_anchor = self.skill_encoder(
            state, _encoder_input_window(self.config, window[:, :horizon])
        )
        anchor_loss = F.mse_loss(z_anchor, initial_z)
        z_norm_loss = z_anchor.pow(2).mean()

        # Same objective on windows the robot ACTUALLY produced. The ring
        # returns None until it fills, which contributes zero rather than
        # stalling the update.
        achieved_loss = torch.zeros((), device=self.device)
        if self.achieved_coeff > 0.0:
            from rlopt.env_interface import supports

            interface = self._env_interface
            if not supports(interface, "sample_achieved_chunk_windows"):
                msg = (
                    "hl_skill_achieved_coeff > 0 needs the environment to "
                    "expose sample_achieved_chunk_windows (set "
                    "env.achieved_ring_capacity > 0)."
                )
                raise RuntimeError(msg)
            achieved = interface.sample_achieved_chunk_windows(
                offline_size, 2 * horizon
            )
            if achieved is not None:
                a_state = achieved.get(("hl", "state")).to(
                    device=self.device, dtype=torch.float32
                )
                a_window = achieved.get(("hl", "future_window")).to(
                    device=self.device, dtype=torch.float32
                )
                achieved_loss, achieved_metrics = self._jepa_objective(
                    a_state, a_window
                )
                metrics.update(
                    {f"achieved_{k}": v for k, v in achieved_metrics.items()}
                )

        total_loss = (
            self.pg_coeff * pg_loss
            + self.offline_diffsr_coeff * offline_objective
            + self.achieved_coeff * achieved_loss
            + self.anchor_coeff * anchor_loss
            + self.z_norm_coeff * z_norm_loss
        )
        metrics.update(
            {
                "hl_skill_total_loss": total_loss.detach(),
                "hl_skill_pg_loss": pg_loss.detach(),
                "hl_skill_offline_loss": offline_objective.detach(),
                "hl_skill_achieved_loss": achieved_loss.detach(),
                "hl_skill_anchor_loss": anchor_loss.detach(),
                "hl_skill_z_norm": z_norm_loss.detach(),
            }
        )
        return total_loss, metrics

    def compute_online_finetune_loss(
        self,
        batch: TensorDictBase,
        *,
        latent_key: str | tuple[str, ...],
        actor_loss_fn: Callable[[TensorDictBase], Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.finetune_enabled:
            zero = torch.zeros((), device=self.device)
            return zero, {}

        pg_loss = torch.zeros((), device=self.device)
        if self.pg_coeff > 0.0:
            command = self.latent_commands_from_rollout_batch(batch, detach=False)
            pg_batch = batch.clone(False)
            pg_batch.set(latent_key, command)
            pg_loss = actor_loss_fn(pg_batch)

        offline_size = int(self.offline_batch_size)
        if self.config.transition_objective == "jepa_ntp":
            return self._jepa_online_finetune_loss(pg_loss, offline_size)

        state, future_window, target = self._sample_offline_macro_batch(offline_size)
        z = self.skill_encoder(state, _encoder_input_window(self.config, future_window))
        zero_reward = torch.zeros(state.shape[0], 1, device=self.device)
        _, diffsr_loss, _ = self.diffsr.compute_loss(state, z, target, zero_reward)
        with torch.no_grad():
            initial_z = self.initial_skill_encoder(
                state,
                _encoder_input_window(self.config, future_window),
            )
        anchor_loss = F.mse_loss(z, initial_z)
        z_norm_loss = z.pow(2).mean()

        # Online-dynamics term, no policy gradient: the same DiffSR endpoint
        # objective, computed on windows of motion the robot ACTUALLY produced,
        # served by the environment's raw-pose ring in the same heading-anchor
        # convention as the expert windows. None (ring not yet filled, or mass
        # resets) contributes zero rather than stalling the update.
        achieved_loss = torch.zeros((), device=self.device)
        if self.achieved_coeff > 0.0:
            from rlopt.env_interface import supports

            # The sampler resolved its interface at construction; it has no
            # raw `env` attribute (that name belongs to the offline trainer).
            interface = self._env_interface
            if not supports(interface, "sample_achieved_chunk_windows"):
                msg = (
                    "hl_skill_achieved_coeff > 0 needs the environment to "
                    "expose sample_achieved_chunk_windows (set "
                    "env.achieved_ring_capacity > 0)."
                )
                raise RuntimeError(msg)
            achieved = interface.sample_achieved_chunk_windows(
                offline_size, int(self.config.horizon_steps)
            )
            if achieved is not None:
                a_state = achieved.get(("hl", "state")).to(
                    device=self.device, dtype=torch.float32
                )
                a_window = achieved.get(("hl", "future_window")).to(
                    device=self.device, dtype=torch.float32
                )
                a_target = achieved.get(("hl", "target")).to(
                    device=self.device, dtype=torch.float32
                )
                a_z = self.skill_encoder(
                    a_state, _encoder_input_window(self.config, a_window)
                )
                a_zero_reward = torch.zeros(a_state.shape[0], 1, device=self.device)
                _, achieved_loss, _ = self.diffsr.compute_loss(
                    a_state, a_z, a_target, a_zero_reward
                )

        total_loss = (
            self.pg_coeff * pg_loss
            + self.offline_diffsr_coeff * diffsr_loss
            + self.achieved_coeff * achieved_loss
            + self.anchor_coeff * anchor_loss
            + self.z_norm_coeff * z_norm_loss
        )
        metrics = {
            "hl_skill_total_loss": total_loss.detach(),
            "hl_skill_pg_loss": pg_loss.detach(),
            "hl_skill_diffsr_loss": diffsr_loss.detach(),
            "hl_skill_achieved_loss": achieved_loss.detach(),
            "hl_skill_anchor_loss": anchor_loss.detach(),
            "hl_skill_z_norm_loss": z_norm_loss.detach(),
        }
        metrics.update(self._z_diagnostics_tensors(z.detach(), prefix="hl_skill"))
        return total_loss, metrics

    def checkpoint_state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "skill_encoder_state_dict": self.skill_encoder.state_dict(),
            "finetune_updates": int(self.finetune_updates),
        }
        if self.train_diffsr:
            state["diffsr_state_dict"] = self.diffsr.state_dict()
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        return state

    def load_checkpoint_state_dict(self, state: Mapping[str, Any]) -> None:
        if "skill_encoder_state_dict" in state:
            self.skill_encoder.load_state_dict(state["skill_encoder_state_dict"])
        if self.train_diffsr and "diffsr_state_dict" in state:
            self.diffsr.load_state_dict(state["diffsr_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.finetune_updates = int(state.get("finetune_updates", 0))

    @torch.no_grad()
    def sample_for_step(
        self,
        td: TensorDictBase,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size = int(td.numel())
        if batch_size <= 0:
            return torch.empty(0, self.latent_dim, device=device, dtype=dtype)
        if (
            self._codes is None
            or self._latent_steps is None
            or self._codes.shape[0] != batch_size
            or self._codes.shape[1] != self.command_code_dim
            or self._codes.device != device
            or self._codes.dtype != dtype
            or self._active_macro_ids is None
            or self._active_macro_ids.shape[0] != batch_size
            or self._active_macro_ids.device != device
        ):
            self._codes = torch.zeros(
                batch_size,
                self.command_code_dim,
                device=device,
                dtype=dtype,
            )
            self._latent_steps = torch.zeros(
                batch_size, device=device, dtype=torch.long
            )
            self._episode_steps = torch.zeros(
                batch_size, device=device, dtype=torch.long
            )
            self._active_macro_ids = torch.full(
                (batch_size,),
                -1,
                device=device,
                dtype=torch.long,
            )

        assert self._codes is not None
        assert self._latent_steps is not None
        assert self._active_macro_ids is not None
        # Hold-1 fast path: every environment renews every step, so the
        # data-dependent mask is always all-true. Skipping it avoids two
        # host synchronizations (`bool(mask.any())` and `nonzero`) inside
        # every policy forward -- collection and learn alike. Values are
        # identical to the masked path.
        always_renew = self.latent_steps_min == 1 and self.latent_steps_max == 1
        if always_renew:
            renew = True
            env_ids = torch.arange(batch_size, device=device)
        else:
            renew_mask = self._done_mask(
                td,
                batch_size=batch_size,
                device=device,
            ) | (self._latent_steps <= 0)
            renew = bool(renew_mask.any())
            env_ids = (
                torch.nonzero(renew_mask, as_tuple=False).reshape(-1) if renew else None
            )
        if renew:
            z, state, future_window, target, initial_z = (
                self._encode_current_macro_batch(env_ids.to(self.device))
            )
            source = state
            if self.command_mode != "z" and self.config.source_history_steps > 0:
                assert self._current_command_source is not None
                source = self._current_command_source
            command_codes = self._command_code_from_state_z(source, z)
            command_codes = command_codes.to(
                device=device,
                dtype=dtype,
            )
            self._codes.index_copy_(0, env_ids.to(device=device), command_codes)
            self._latent_steps.index_copy_(
                0,
                env_ids.to(device=device),
                self._sample_steps(int(env_ids.numel()), device=device),
            )
            if self.finetune_enabled:
                macro_ids = self._append_rollout_cache(
                    state=state,
                    future_window=future_window,
                    target=target,
                    initial_z=initial_z,
                ).to(device=device)
                self._active_macro_ids.index_copy_(
                    0,
                    env_ids.to(device=device),
                    macro_ids,
                )

        if self.phase_source == "episode":
            # Live clock for hold 1, where the hold-derived phase is constant:
            # steps since reset modulo phase_period. Tensor ops only, so the
            # hold-1 fast path above keeps its no-host-sync property.
            assert self._episode_steps is not None
            done_mask = self._done_mask(td, batch_size=batch_size, device=device)
            self._episode_steps = torch.where(
                done_mask,
                torch.zeros_like(self._episode_steps),
                self._episode_steps,
            )
            phase = torch.remainder(self._episode_steps, self.phase_period).to(
                torch.float32
            ) / float(self.phase_period)
            self._episode_steps = self._episode_steps + 1
        else:
            phase = (
                (self.phase_period - self._latent_steps).clamp(min=0).to(torch.float32)
            )
            phase = phase / float(self.phase_period)
        latents = self._append_command_phase(self._codes, phase)
        if self.finetune_enabled:
            td.set(
                ("hl_skill", "macro_id"),
                self._active_macro_ids.reshape(*td.batch_size),
            )
            td.set(
                ("hl_skill", "phase"),
                phase.to(device=device, dtype=torch.float32).reshape(*td.batch_size),
            )
        self._latent_steps = self._latent_steps - 1
        return latents


@dataclass
class HighLevelSkillDiffSRTrainState:
    update: int = 0
    elapsed_seconds: float = 0.0
    last_metrics: dict[str, float] = field(default_factory=dict)


class HighLevelSkillDiffSRTrainer:
    """Offline trainer for high-level skill encoders and controlled objectives."""

    def __init__(self, config: HighLevelSkillDiffSRConfig, env: object) -> None:
        self.config = config
        self.config.validate()
        self.env = env
        self.device = _resolve_device(self.config.device, self.env)
        preflight_size = min(self.config.batch_size, self.config.preflight_batch_size)
        state, future_window, target = self._sample_and_validate_macro_batch(
            preflight_size,
            split=self.config.train_split,
        )
        del future_window, target
        self.state_dim = int(state.shape[-1])
        self.encoder_window_steps = _encoder_window_steps(self.config)
        self.feature_slices = self._resolve_feature_slices()

        self.skill_encoder = build_skill_encoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.encoder_hidden_dims,
            spec=self.config.latent_spec(),
            activation=self.config.encoder_activation,
            layer_norm=self.config.encoder_layer_norm,
        ).to(self.device)
        self.diffsr = _build_diffsr(self.config, self.state_dim, self.device).to(
            self.device
        )
        self._initialize_diffsr_state_output()
        self.reconstruction_decoder: nn.Module | None = None
        optimizer_groups: list[dict[str, Any]] = [
            {
                "params": self.skill_encoder.parameters(),
                "lr": self.config.encoder_lr,
            }
        ]
        if self.config.transition_objective == "reconstruction":
            if self.config.reconstruction_target == "endpoint":
                reconstruction_output_dim = self.state_dim
            elif self.config.reconstruction_target == "full_window":
                reconstruction_output_dim = self.state_dim * (
                    int(self.config.horizon_steps) + 1
                )
            else:
                reconstruction_output_dim = self.state_dim * (
                    self.encoder_window_steps + 1
                )
            self.reconstruction_decoder = _WindowReconstructionDecoder(
                z_dim=self.config.z_dim,
                output_dim=reconstruction_output_dim,
                hidden_dims=tuple(reversed(self.config.encoder_hidden_dims)),
                activation=self.config.encoder_activation,
            ).to(self.device)
            optimizer_groups.append(
                {
                    "params": self.reconstruction_decoder.parameters(),
                    "lr": self.config.encoder_lr,
                }
            )
        else:
            optimizer_groups.append(
                {"params": self.diffsr.parameters(), "lr": self.config.diffsr_lr}
            )
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=self.config.weight_decay,
        )
        self.update = 0

        # Chunk-wise next-token prediction with a JEPA-style EMA target and a
        # bilinear (spectral) energy head. The encoder's chunk token is the
        # "token"; the objective predicts the NEXT chunk's target-encoder token
        # from the current one and scores the pair with the low-rank energy
        # E(z, z') = g(pred(z))^T f(z'), trained as symmetric InfoNCE over
        # in-batch negatives. The DiffSR heads stay constructed but unused.
        self.jepa_target_encoder: nn.Module | None = None
        self.jepa_predictor: nn.Module | None = None
        self.jepa_ntp_diffsr: BilinearSR | None = None
        self.jepa_g: nn.Module | None = None
        self.jepa_f: nn.Module | None = None
        if self.config.transition_objective == "jepa_ntp":
            if str(self.config.jepa_loss) not in ("sigreg", "sigreg_ebm", "infonce"):
                msg = (
                    "jepa_loss must be 'sigreg', 'sigreg_ebm' or 'infonce', "
                    f"got {self.config.jepa_loss!r}."
                )
                raise ValueError(msg)
            if int(self.state_dim) not in (
                _ROOT_QPOS_FRAME_DIM,
                _FULL_BODY_FRAME_DIM,
            ):
                msg = (
                    "jepa_ntp re-anchors the next chunk in heading frame and "
                    "needs the 38-wide root_qpos or 67-wide full_body macro "
                    f"state, got {int(self.state_dim)}."
                )
                raise ValueError(msg)
            if str(self.config.jepa_target_encoder_mode) == "ema":
                self.jepa_target_encoder = copy.deepcopy(self.skill_encoder)
                for parameter in self.jepa_target_encoder.parameters():
                    parameter.requires_grad_(False)
            # online mode: the ONE encoder serves both sides; no copy exists.
            z_dim = int(self.config.z_dim)
            energy_dim = int(self.config.jepa_energy_dim)
            predictor_in = z_dim * (1 + int(self.config.jepa_context_chunks))
            self.jepa_predictor = nn.Sequential(
                nn.Linear(predictor_in, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, z_dim),
            ).to(self.device)
            self.jepa_g = nn.Sequential(
                nn.Linear(z_dim, 512), nn.SiLU(), nn.Linear(512, energy_dim)
            ).to(self.device)
            self.jepa_f = nn.Sequential(
                nn.Linear(z_dim, 512), nn.SiLU(), nn.Linear(512, energy_dim)
            ).to(self.device)
            if str(self.config.jepa_ntp_head) != "mlp":
                # Generative next-chunk head: a second DiffSR whose denoising
                # target is the next TOKEN (diff_token, z_dim wide) or the
                # next chunk's H raw frames (diff_chunk, H*state_dim wide).
                self.jepa_ntp_diffsr = _build_diffsr(
                    self.config,
                    self.state_dim,
                    self.device,
                    next_obs_dim=_jepa_ntp_target_dim(self.config, self.state_dim),
                ).to(self.device)
            self.optimizer.add_param_group(
                {
                    "params": [
                        *self.jepa_predictor.parameters(),
                        *self.jepa_g.parameters(),
                        *self.jepa_f.parameters(),
                        *(
                            self.jepa_ntp_diffsr.parameters()
                            if self.jepa_ntp_diffsr is not None
                            else []
                        ),
                    ],
                    "lr": self.config.encoder_lr,
                }
            )

        # Optional co-trained skill commander (System-1 planner). BC'd to the
        # encoder's z (detached) from current state + language goal.
        self.commander: nn.Module | None = None
        self.commander_optimizer: torch.optim.Optimizer | None = None
        self.commander_rank_embeddings: Tensor | None = None
        self.commander_lang_embed_dim = 0
        if self.config.cotrain_commander:
            self._init_commander()

    def _encode_skill(
        self,
        state: Tensor,
        future_window: Tensor,
        *,
        deterministic: bool = False,
        step: int | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return self.skill_encoder.encode(
            state,
            _encoder_input_window(self.config, future_window),
            deterministic=deterministic,
            step=step,
        )

    def _reconstruction_target(
        self,
        state: Tensor,
        future_window: Tensor,
        endpoint: Tensor,
    ) -> Tensor:
        if self.config.reconstruction_target == "endpoint":
            return endpoint
        if self.config.reconstruction_target == "full_window":
            return torch.cat((state, future_window.reshape(state.shape[0], -1)), dim=-1)
        encoder_window = _encoder_input_window(self.config, future_window)
        return torch.cat((state, encoder_window.reshape(state.shape[0], -1)), dim=-1)

    def _reconstruction_eval_metrics(
        self,
        state: Tensor,
        future_window: Tensor,
        endpoint: Tensor,
        z: Tensor,
        *,
        prefix: str,
    ) -> dict[str, float]:
        assert self.reconstruction_decoder is not None
        target = self._reconstruction_target(state, future_window, endpoint)
        prediction = self.reconstruction_decoder(z)
        zero_prediction = self.reconstruction_decoder(torch.zeros_like(z))
        shuffled_z = (
            z[torch.randperm(z.shape[0], device=z.device)] if int(z.shape[0]) > 1 else z
        )
        shuffled_prediction = self.reconstruction_decoder(shuffled_z)
        error = prediction - target
        return {
            f"{prefix}/reconstruction_loss_eval": float(error.pow(2).mean().item()),
            f"{prefix}/reconstruction_mae_eval": float(error.abs().mean().item()),
            f"{prefix}/reconstruction_loss_zero_z_eval": float(
                F.mse_loss(zero_prediction, target).item()
            ),
            f"{prefix}/reconstruction_loss_shuffled_z_eval": float(
                F.mse_loss(shuffled_prediction, target).item()
            ),
        }

    def _initialize_diffsr_state_output(self) -> None:
        init_std = float(self.config.diffsr_state_output_init_std)
        if init_std <= 0.0:
            return
        state_net = getattr(self.diffsr, "state_net", None)
        fc2 = getattr(state_net, "fc2", None)
        if isinstance(fc2, nn.Linear):
            nn.init.normal_(fc2.weight, mean=0.0, std=init_std)
            nn.init.zeros_(fc2.bias)

    def _sample_macro_batch(
        self,
        batch_size: int,
        *,
        split: str | None,
    ) -> TensorDictBase:
        from rlopt.env_interface import require_imitation_interface

        sampler = require_imitation_interface(
            self.env,
            "sample_expert_macro_transition_batch",
            purpose="Offline skill-encoder training requires it but",
        )
        return sampler(
            batch_size=int(batch_size),
            horizon_steps=int(self.config.horizon_steps),
            split=split,
            eval_fraction=float(self.config.eval_trajectory_fraction),
            split_seed=int(self.config.trajectory_split_seed),
        )

    def _sample_and_validate_macro_batch(
        self,
        batch_size: int,
        *,
        split: str | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = self._sample_macro_batch(batch_size, split=split)
        return _validate_macro_batch(
            batch,
            batch_size=batch_size,
            horizon_steps=int(self.config.horizon_steps),
            device=self.device,
            source="Expert",
        )

    def _resolve_feature_slices(self) -> dict[str, tuple[int, int]]:
        from rlopt.env_interface import resolve_imitation_interface, supports

        interface = resolve_imitation_interface(self.env)
        if not supports(interface, "expert_macro_feature_slices"):
            return {}
        provider = interface.expert_macro_feature_slices
        raw_slices = provider(horizon_steps=int(self.config.horizon_steps))
        if raw_slices is None:
            return {}
        feature_slices: dict[str, tuple[int, int]] = {}
        used_ranges: list[tuple[int, int, str]] = []
        for raw_name, raw_bounds in dict(raw_slices).items():
            name = str(raw_name)
            start, end = raw_bounds
            start = int(start)
            end = int(end)
            if start < 0 or end <= start or end > self.state_dim:
                msg = (
                    "Invalid expert macro feature slice for "
                    f"{name!r}: {(start, end)} with state_dim={self.state_dim}."
                )
                raise ValueError(msg)
            for other_start, other_end, other_name in used_ranges:
                if start < other_end and other_start < end:
                    msg = (
                        "Overlapping expert macro feature slices: "
                        f"{name!r}={(start, end)} overlaps "
                        f"{other_name!r}={(other_start, other_end)}."
                    )
                    raise ValueError(msg)
            used_ranges.append((start, end, name))
            feature_slices[name] = (start, end)
        return feature_slices

    @staticmethod
    def _z_diagnostics(z: Tensor, *, prefix: str) -> dict[str, float]:
        z_std = z.std(dim=0, unbiased=False)
        return {
            f"{prefix}/z_abs_mean": float(z.abs().mean().item()),
            f"{prefix}/z_rms": float(z.pow(2).mean().sqrt().item()),
            f"{prefix}/z_dim_std_mean": float(z_std.mean().item()),
            f"{prefix}/z_dim_std_min": float(z_std.min().item()),
            f"{prefix}/z_effective_rank": float(_effective_rank(z).item()),
        }

    def _diffsr_loss_for_z(self, state: Tensor, z: Tensor, target: Tensor) -> Tensor:
        zero_reward = torch.zeros(state.shape[0], 1, device=self.device)
        _, loss, _ = self.diffsr.compute_loss(state, z, target, zero_reward)
        return loss

    def _ntp_diffsr_loss(self, state: Tensor, z: Tensor, target: Tensor) -> Tensor:
        """Denoising loss of the generative next-chunk head (never the mean MSE)."""
        assert self.jepa_ntp_diffsr is not None
        zero_reward = torch.zeros(state.shape[0], 1, device=self.device)
        _, loss, _ = self.jepa_ntp_diffsr.compute_loss(state, z, target, zero_reward)
        return loss

    def _objective_transition_at(
        self,
        state: Tensor,
        future_window: Tensor,
        offset_index: int,
    ) -> tuple[Tensor, Tensor, int]:
        """Build one transition factor for the configured skill objective."""
        offsets = self.config.transition_offsets
        index = int(offset_index)
        if index < 0 or index >= len(offsets):
            msg = f"offset_index must be in [0, {len(offsets)}), got {index}."
            raise IndexError(msg)
        offset = int(offsets[index])
        target = future_window[:, offset - 1, :]
        objective = self.config.transition_objective
        if objective == "semimarkov_chain" and index > 0:
            source = future_window[:, int(offsets[index - 1]) - 1, :]
        else:
            source = state
        if objective == "endpoint_delta":
            target = target - state
        return source, target, offset

    def _sample_objective_transition(
        self,
        state: Tensor,
        future_window: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample one unbiased objective factor independently for each row."""
        num_offsets = len(self.config.transition_offsets)
        batch_size = int(state.shape[0])
        if num_offsets == 1:
            source, target, offset = self._objective_transition_at(
                state, future_window, 0
            )
            selected_offsets = torch.full(
                (batch_size,), offset, device=state.device, dtype=torch.long
            )
            return source, target, selected_offsets

        sources: list[Tensor] = []
        targets: list[Tensor] = []
        for offset_index in range(num_offsets):
            source, target, _ = self._objective_transition_at(
                state, future_window, offset_index
            )
            sources.append(source)
            targets.append(target)
        source_stack = torch.stack(sources, dim=1)
        target_stack = torch.stack(targets, dim=1)
        selected_indices = torch.randint(
            0, num_offsets, (batch_size,), device=state.device
        )
        row_indices = torch.arange(batch_size, device=state.device)
        offset_values = torch.tensor(
            self.config.transition_offsets, device=state.device, dtype=torch.long
        )
        return (
            source_stack[row_indices, selected_indices],
            target_stack[row_indices, selected_indices],
            offset_values[selected_indices],
        )

    def _objective_eval_loss_metrics(
        self,
        state: Tensor,
        future_window: Tensor,
        z: Tensor,
        zero_z: Tensor,
        shuffled_z: Tensor,
        *,
        prefix: str,
        source_override: Tensor | None = None,
    ) -> dict[str, float]:
        """Evaluate every checkpoint factor, then report their uniform mean.

        ``source_override`` replaces phi's conditioning with the flattened past
        chunk when the arm was trained with ``source_history_steps > 0``; phi's
        input width is derived from that setting, so the single-frame source
        would not even be the right shape.
        """
        real_losses: list[float] = []
        zero_losses: list[float] = []
        shuffled_losses: list[float] = []
        metrics: dict[str, float] = {}
        for offset_index in range(len(self.config.transition_offsets)):
            source, target, offset = self._objective_transition_at(
                state, future_window, offset_index
            )
            if source_override is not None:
                source = source_override
            real = float(self._diffsr_loss_for_z(source, z, target).item())
            zero = float(self._diffsr_loss_for_z(source, zero_z, target).item())
            shuffled = float(self._diffsr_loss_for_z(source, shuffled_z, target).item())
            real_losses.append(real)
            zero_losses.append(zero)
            shuffled_losses.append(shuffled)
            metrics.update(
                {
                    f"{prefix}/loss_real_z_eval_h{offset}": real,
                    f"{prefix}/loss_zero_z_eval_h{offset}": zero,
                    f"{prefix}/loss_shuffled_z_eval_h{offset}": shuffled,
                }
            )
        metrics.update(
            {
                f"{prefix}/loss_real_z_eval": sum(real_losses) / len(real_losses),
                f"{prefix}/loss_zero_z_eval": sum(zero_losses) / len(zero_losses),
                f"{prefix}/loss_shuffled_z_eval": sum(shuffled_losses)
                / len(shuffled_losses),
            }
        )
        return metrics

    def _objective_reconstruction_metrics(
        self,
        state: Tensor,
        future_window: Tensor,
        z: Tensor,
        *,
        prefix: str,
    ) -> dict[str, float]:
        """Sample each configured factor and keep old aggregate metric names."""
        per_offset: list[dict[str, float]] = []
        metrics: dict[str, float] = {}
        for offset_index in range(len(self.config.transition_offsets)):
            source, target, offset = self._objective_transition_at(
                state, future_window, offset_index
            )
            current = self._sample_reconstruction_metrics(
                source, z, target, prefix=prefix
            )
            per_offset.append(current)
            for key, value in current.items():
                suffix = key.removeprefix(f"{prefix}/")
                metrics[f"{prefix}/{suffix}_h{offset}"] = value
        if per_offset:
            for key in per_offset[0]:
                metrics[key] = sum(item[key] for item in per_offset) / len(per_offset)
        return metrics

    @staticmethod
    def _metric_safe_feature_name(name: str) -> str:
        return name.replace("/", "_").replace(" ", "_")

    def _state_error_metrics(
        self,
        error: Tensor,
        *,
        prefix: str,
        stem: str,
    ) -> dict[str, float]:
        metrics = {
            f"{prefix}/{stem}_l1": float(error.abs().sum(dim=-1).mean().item()),
            f"{prefix}/{stem}_mse": float(error.pow(2).sum(dim=-1).mean().item()),
            f"{prefix}/{stem}_dim_mse": float(error.pow(2).mean().item()),
        }
        for name, (start, end) in self.feature_slices.items():
            group_error = error[:, start:end]
            metric_name = self._metric_safe_feature_name(name)
            metrics.update(
                {
                    f"{prefix}/{stem}_{metric_name}_l1": float(
                        group_error.abs().sum(dim=-1).mean().item()
                    ),
                    f"{prefix}/{stem}_{metric_name}_mse": float(
                        group_error.pow(2).sum(dim=-1).mean().item()
                    ),
                    f"{prefix}/{stem}_{metric_name}_dim_mse": float(
                        group_error.pow(2).mean().item()
                    ),
                }
            )
        return metrics

    def _state_normalized_error_metrics(
        self,
        error: Tensor,
        target: Tensor,
        *,
        prefix: str,
        stem: str,
    ) -> dict[str, float]:
        variance = target.var(dim=0, unbiased=False).clamp_min(
            float(self.config.reconstruction_norm_eps)
        )
        normalized_sq = error.pow(2) / variance.unsqueeze(0)
        metrics = {
            f"{prefix}/norm_{stem}_mse": float(normalized_sq.sum(dim=-1).mean().item()),
            f"{prefix}/norm_{stem}_dim_mse": float(normalized_sq.mean().item()),
        }
        for name, (start, end) in self.feature_slices.items():
            group_normalized_sq = normalized_sq[:, start:end]
            metric_name = self._metric_safe_feature_name(name)
            metrics.update(
                {
                    f"{prefix}/norm_{stem}_{metric_name}_mse": float(
                        group_normalized_sq.sum(dim=-1).mean().item()
                    ),
                    f"{prefix}/norm_{stem}_{metric_name}_dim_mse": float(
                        group_normalized_sq.mean().item()
                    ),
                }
            )
        return metrics

    def _sample_reconstruction_metrics(
        self,
        state: Tensor,
        z: Tensor,
        target: Tensor,
        *,
        prefix: str,
    ) -> dict[str, float]:
        if not self.diffsr.supports_sampling:
            return {}
        sampled_target, _ = self.diffsr.sample(
            s=state,
            a=z,
            preserve_history=False,
        )
        error = sampled_target - target
        metrics = self._state_error_metrics(error, prefix=prefix, stem="sample_recon")
        metrics.update(
            self._state_normalized_error_metrics(
                error,
                target,
                prefix=prefix,
                stem="sample_recon",
            )
        )
        return metrics

    @staticmethod
    def _linear_probe_design(features: Tensor) -> Tensor:
        ones = torch.ones(
            features.shape[0],
            1,
            device=features.device,
            dtype=features.dtype,
        )
        return torch.cat([features, ones], dim=-1)

    @staticmethod
    def _solve_linear_probe(
        xtx: Tensor,
        xty: Tensor,
        *,
        ridge: float,
    ) -> Tensor:
        regularizer = torch.eye(
            xtx.shape[0],
            device=xtx.device,
            dtype=xtx.dtype,
        )
        regularizer[-1, -1] = 0.0
        return torch.linalg.solve(xtx + float(ridge) * regularizer, xty)

    def _window_probe_error_metrics(
        self,
        error_flat: Tensor,
        variance_flat: Tensor,
        *,
        prefix: str,
        stem: str,
    ) -> dict[str, float]:
        batch_size = int(error_flat.shape[0])
        error_window = error_flat.reshape(
            batch_size,
            self.config.horizon_steps,
            self.state_dim,
        )
        mid_step = int(self.config.horizon_steps // 2)
        normalized_sq = error_flat.pow(2) / variance_flat.unsqueeze(0)
        return {
            f"{prefix}/{stem}_mse": float(error_flat.pow(2).sum(dim=-1).mean().item()),
            f"{prefix}/{stem}_dim_mse": float(error_flat.pow(2).mean().item()),
            f"{prefix}/{stem}_norm_dim_mse": float(normalized_sq.mean().item()),
            f"{prefix}/{stem}_step_first_dim_mse": float(
                error_window[:, 0, :].pow(2).mean().item()
            ),
            f"{prefix}/{stem}_step_mid_dim_mse": float(
                error_window[:, mid_step, :].pow(2).mean().item()
            ),
            f"{prefix}/{stem}_step_final_dim_mse": float(
                error_window[:, -1, :].pow(2).mean().item()
            ),
        }

    @torch.no_grad()
    def evaluate_window_probe(
        self,
        *,
        train_batches: int = 4,
        eval_batches: int | None = None,
        batch_size: int | None = None,
        prefix: str = "train",
        train_split: str | None = None,
        eval_split: str | None = None,
        ridge: float = 1.0e-3,
    ) -> dict[str, float]:
        train_batches = _require_positive_int("train_batches", train_batches)
        eval_batches = (
            self.config.eval_batches if eval_batches is None else eval_batches
        )
        eval_batches = _require_positive_int("eval_batches", eval_batches)
        batch_size = (
            self.config.eval_batch_size or self.config.batch_size
            if batch_size is None
            else batch_size
        )
        batch_size = _require_positive_int("batch_size", batch_size)
        ridge = _require_non_negative_float("ridge", ridge)
        train_split = self.config.train_split if train_split is None else train_split
        eval_split = self.config.eval_split if eval_split is None else eval_split

        encoder_was_training = self.skill_encoder.training
        self.skill_encoder.eval()

        z_xtx: Tensor | None = None
        z_xty: Tensor | None = None
        state_xtx: Tensor | None = None
        state_xty: Tensor | None = None
        target_sum: Tensor | None = None
        target_sq_sum: Tensor | None = None
        train_samples = 0

        for _ in range(train_batches):
            state, future_window, _ = self._sample_and_validate_macro_batch(
                batch_size,
                split=train_split,
            )
            z, *_ = self._encode_skill(state, future_window, deterministic=True)
            target_flat = future_window.reshape(batch_size, -1)
            z_design = self._linear_probe_design(z).to(torch.float64)
            state_design = self._linear_probe_design(state).to(torch.float64)
            target64 = target_flat.to(torch.float64)

            batch_z_xtx = z_design.T @ z_design
            batch_z_xty = z_design.T @ target64
            batch_state_xtx = state_design.T @ state_design
            batch_state_xty = state_design.T @ target64
            z_xtx = batch_z_xtx if z_xtx is None else z_xtx + batch_z_xtx
            z_xty = batch_z_xty if z_xty is None else z_xty + batch_z_xty
            state_xtx = (
                batch_state_xtx if state_xtx is None else state_xtx + batch_state_xtx
            )
            state_xty = (
                batch_state_xty if state_xty is None else state_xty + batch_state_xty
            )
            batch_target_sum = target64.sum(dim=0)
            batch_target_sq_sum = target64.pow(2).sum(dim=0)
            target_sum = (
                batch_target_sum
                if target_sum is None
                else target_sum + batch_target_sum
            )
            target_sq_sum = (
                batch_target_sq_sum
                if target_sq_sum is None
                else target_sq_sum + batch_target_sq_sum
            )
            train_samples += int(batch_size)

        assert z_xtx is not None
        assert z_xty is not None
        assert state_xtx is not None
        assert state_xty is not None
        assert target_sum is not None
        assert target_sq_sum is not None
        z_weights = self._solve_linear_probe(z_xtx, z_xty, ridge=ridge)
        state_weights = self._solve_linear_probe(state_xtx, state_xty, ridge=ridge)
        mean_flat = target_sum / float(train_samples)
        variance_flat = (
            target_sq_sum / float(train_samples) - mean_flat.pow(2)
        ).clamp_min(float(self.config.reconstruction_norm_eps))

        accum: dict[str, float] = {}
        eval_samples = 0
        for _ in range(eval_batches):
            state, future_window, _ = self._sample_and_validate_macro_batch(
                batch_size,
                split=eval_split,
            )
            z, *_ = self._encode_skill(state, future_window, deterministic=True)
            if int(z.shape[0]) > 1:
                shuffled_z = z[torch.randperm(z.shape[0], device=z.device)]
            else:
                shuffled_z = z.clone()
            target_flat = future_window.reshape(batch_size, -1).to(torch.float64)
            z_design = self._linear_probe_design(z).to(torch.float64)
            shuffled_z_design = self._linear_probe_design(shuffled_z).to(torch.float64)
            state_design = self._linear_probe_design(state).to(torch.float64)

            prediction_errors = {
                "window_probe_z": z_design @ z_weights - target_flat,
                "window_probe_z_shuffled": shuffled_z_design @ z_weights - target_flat,
                "window_probe_state": state_design @ state_weights - target_flat,
                "window_probe_mean": mean_flat.unsqueeze(0) - target_flat,
            }
            batch_metrics: dict[str, float] = {}
            for stem, error_flat in prediction_errors.items():
                batch_metrics.update(
                    self._window_probe_error_metrics(
                        error_flat,
                        variance_flat,
                        prefix=prefix,
                        stem=stem,
                    )
                )
            for key, value in batch_metrics.items():
                accum[key] = accum.get(key, 0.0) + float(value) * float(batch_size)
            eval_samples += int(batch_size)

        for key in accum:
            accum[key] /= float(eval_samples)
        accum[f"{prefix}/window_probe_train_samples"] = float(train_samples)
        accum[f"{prefix}/window_probe_eval_samples"] = float(eval_samples)
        accum[f"{prefix}/window_probe_ridge"] = float(ridge)

        if encoder_was_training:
            self.skill_encoder.train()
        return accum

    def _init_commander(self) -> None:
        from rlopt.agent.skill_commander import (  # noqa: PLC0415
            SkillCommander,
            build_rank_embedding_lookup,
            load_language_embedding_table,
        )

        names_provider = getattr(self.env, "expert_trajectory_motion_names", None)
        if not callable(names_provider):
            msg = "cotrain_commander requires env.expert_trajectory_motion_names()."
            raise ValueError(msg)
        table = load_language_embedding_table(
            self.config.commander_language_embeddings_path
        )
        self.commander_lang_embed_dim = int(table["embed_dim"])
        self.commander_rank_embeddings = build_rank_embedding_lookup(
            table, [str(name) for name in names_provider()], self.device
        )
        self.commander = SkillCommander(
            state_dim=self.state_dim,
            lang_embed_dim=self.commander_lang_embed_dim,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.commander_hidden_dims,
        ).to(self.device)
        self.commander_optimizer = torch.optim.AdamW(
            self.commander.parameters(), lr=self.config.commander_lr
        )

    def _commander_lang_for_batch(self, batch: TensorDictBase) -> Tensor:
        traj_rank = batch.get(("hl", "traj_rank"))
        if traj_rank is None:
            msg = "cotrain_commander requires 'traj_rank' in the macro batch."
            raise ValueError(msg)
        traj_rank = traj_rank.reshape(-1).to(device=self.device, dtype=torch.long)
        assert self.commander_rank_embeddings is not None
        return self.commander_rank_embeddings.index_select(0, traj_rank)

    def _commander_train_step(
        self, batch: TensorDictBase, state: Tensor, z: Tensor
    ) -> dict[str, float]:
        assert self.commander is not None
        assert self.commander_optimizer is not None
        self.commander.train()
        lang = self._commander_lang_for_batch(batch)
        z_target = z.detach()
        cmd_state = state
        std = float(self.config.commander_state_noise_std)
        if std > 0.0 and int(state.shape[0]) > 1:
            per_dim = state.std(dim=0, keepdim=True)
            cmd_state = state + std * per_dim * torch.randn_like(state)
        z_hat = self.commander(cmd_state, lang)
        mse = F.mse_loss(z_hat, z_target)
        cosine = F.cosine_similarity(z_hat, z_target, dim=-1).mean()
        loss = (
            mse
            + self.config.commander_cosine_loss_coeff * (1.0 - cosine)
            + self.config.commander_z_norm_coeff * z_hat.pow(2).mean()
        )
        self.commander_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.commander.parameters(),
                max_norm=float(self.config.grad_clip_norm),
            )
        self.commander_optimizer.step()
        return {
            "train/commander_mse": float(mse.detach().item()),
            "train/commander_cosine": float(cosine.detach().item()),
            "train/commander_loss": float(loss.detach().item()),
        }

    @torch.no_grad()
    def _commander_eval_metrics(
        self, batch: TensorDictBase, state: Tensor, z: Tensor, *, prefix: str
    ) -> dict[str, float]:
        assert self.commander is not None
        was_training = self.commander.training
        self.commander.eval()
        lang = self._commander_lang_for_batch(batch)
        z_hat = self.commander(state, lang)
        metrics = {
            f"{prefix}/commander_cosine": float(
                F.cosine_similarity(z_hat, z, dim=-1).mean().item()
            ),
            f"{prefix}/commander_mse": float(F.mse_loss(z_hat, z).item()),
        }
        if int(lang.shape[0]) > 1:
            shuffled = lang[torch.randperm(lang.shape[0], device=lang.device)]
            z_hat_shuffled = self.commander(state, shuffled)
            metrics[f"{prefix}/commander_cosine_shuffled_lang"] = float(
                F.cosine_similarity(z_hat_shuffled, z, dim=-1).mean().item()
            )
        if was_training:
            self.commander.train()
        return metrics

    def _sample_jepa_window(
        self,
        sampler: Callable[..., TensorDictBase],
        *,
        batch_size: int,
        chunk_steps: int,
        split: str | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Draw one jepa window and phi's conditioning source.

        Returns ``(state, window, source)`` where ``state`` is ``s_t``,
        ``window`` is ``s[t+1 .. t+chunk_steps]`` and ``source`` is what phi
        conditions on, flattened to ``[batch, (history + 1) * state_dim]``.
        With ``source_history_steps=0`` the source IS the state, which
        reproduces every existing arm exactly.

        Neither history mode re-anchors anything. ``current`` takes the data
        plane's own past chunk, which is already expressed in ``s_t``'s heading
        frame because the sampler anchors at ``center_index=past_steps``.
        ``past_start`` instead draws ONE longer window and relabels the sampled
        cursor as ``t - history``, so slot 0 is the oldest past frame and the
        whole span shares that anchor by construction.
        """
        history = int(self.config.source_history_steps)
        if history > 0 and str(self.config.source_anchor) == "past_start":
            span = history + int(chunk_steps)
            batch = sampler(
                batch_size=batch_size,
                horizon_steps=span,
                split=split,
                eval_fraction=float(self.config.eval_trajectory_fraction),
                split_seed=int(self.config.trajectory_split_seed),
            )
            anchor_state, anchor_window, _ = _validate_macro_batch(
                batch,
                batch_size=batch_size,
                horizon_steps=span,
                device=self.device,
            )
            sequence = torch.cat([anchor_state.unsqueeze(1), anchor_window], dim=1)
            source = sequence[:, : history + 1].reshape(batch_size, -1)
            return sequence[:, history], sequence[:, history + 1 :], source

        batch = sampler(
            batch_size=batch_size,
            horizon_steps=int(chunk_steps),
            split=split,
            eval_fraction=float(self.config.eval_trajectory_fraction),
            split_seed=int(self.config.trajectory_split_seed),
            **({"state_history_steps": history} if history > 0 else {}),
        )
        state, window, _ = _validate_macro_batch(
            batch,
            batch_size=batch_size,
            horizon_steps=int(chunk_steps),
            device=self.device,
        )
        if history == 0:
            return state, window, state
        state_history = _macro_batch_state_history(
            batch,
            batch_size=batch_size,
            history_steps=history,
            state_dim=int(state.shape[-1]),
            device=self.device,
        )
        return state, window, state_history.reshape(batch_size, -1)

    def _jepa_train_step(self) -> dict[str, float]:
        assert self.jepa_predictor is not None
        assert self.jepa_g is not None and self.jepa_f is not None
        ema_mode = str(self.config.jepa_target_encoder_mode) == "ema"
        if ema_mode:
            assert self.jepa_target_encoder is not None
        context = int(self.config.jepa_context_chunks)
        self.skill_encoder.train()
        horizon = int(self.config.horizon_steps)
        from rlopt.env_interface import require_imitation_interface

        sampler = require_imitation_interface(
            self.env,
            "sample_expert_macro_transition_batch",
            purpose="Offline skill-encoder training requires it but",
        )
        # One draw of (2 + context) adjacent TILED chunks. At context=0 the
        # window spans 2H frames: the executed chunk is the sampled state plus
        # frames 1..H, the target chunk is frame H-1 (its state, = s_{t+H})
        # plus frames H+1..2H, re-anchored onto its own state -- exactly what
        # the encoder sees one publication later. At context=1 the window
        # spans 3H frames, the SAME construction yields chunks 0/1/2, and the
        # predictor reads cat(z0, z1): the option-view dynamics model of
        # wiki/skill-encoder-jepa-plan.md phase 2.
        total_chunks = 2 + context
        state, window, source = self._sample_jepa_window(
            sampler,
            batch_size=int(self.config.batch_size),
            chunk_steps=total_chunks * horizon,
            split=self.config.train_split,
        )

        def chunk_at(index: int) -> tuple[Tensor, Tensor]:
            """Chunk ``index`` (state, window), anchored on its own state."""
            if index == 0:
                return state, window[:, :horizon]
            anchor = window[:, index * horizon - 1]
            return (
                _reanchor_heading_frames(anchor, anchor),
                _reanchor_heading_frames(
                    window[:, index * horizon : (index + 1) * horizon], anchor
                ),
            )

        exec_state, exec_window = chunk_at(context)
        target_state, target_window = chunk_at(context + 1)
        # phi's conditioning. Identical to exec_state unless a past chunk was
        # requested; validate() pins context=0 in that case, so `source` always
        # belongs to the executed chunk.
        exec_source = (
            source if int(self.config.source_history_steps) > 0 else exec_state
        )
        z1, reg_loss, info = self.skill_encoder.encode(
            exec_state,
            _encoder_input_window(self.config, exec_window),
            step=self.update,
        )
        if ema_mode:
            with torch.no_grad():
                self.jepa_target_encoder.eval()  # type: ignore[union-attr]
                z2, _, _ = self.jepa_target_encoder.encode(  # type: ignore[union-attr]
                    target_state,
                    _encoder_input_window(self.config, target_window),
                    deterministic=True,
                )
        elif str(self.config.jepa_target_encoder_mode) == "stopgrad":
            # SimSiam-style asymmetry without lag: same online encoder, target
            # branch detached. Prediction cannot co-adapt the target, but the
            # target tracks the online weights with zero delay.
            with torch.no_grad():
                z2, _, _ = self.skill_encoder.encode(
                    target_state,
                    _encoder_input_window(self.config, target_window),
                    deterministic=True,
                )
        else:
            # LeJEPA shape: the ONE online encoder on both sides, gradients
            # flowing through both; SIGReg carries the anti-collapse burden.
            z2, _, _ = self.skill_encoder.encode(
                target_state,
                _encoder_input_window(self.config, target_window),
                step=self.update,
            )
        if context > 0:
            context_state, context_window = chunk_at(0)
            z_context, _, _ = self.skill_encoder.encode(
                context_state,
                _encoder_input_window(self.config, context_window),
                step=self.update,
            )
            predictor_input = torch.cat((z_context, z1), dim=-1)
        else:
            predictor_input = z1
        prediction = self.jepa_predictor(predictor_input)
        # Gate metric for the triplet program: the predicted next token must
        # beat copy-the-previous-token on these TILED chunks, else the
        # dynamics model is vacuous. Logged in every mode.
        with torch.no_grad():
            copy_mse = F.mse_loss(z1.detach(), z2.detach())
            pred_mse = F.mse_loss(prediction.detach(), z2.detach())
        if self.config.jepa_loss == "sigreg_ebm":
            # Ours: chunk-wise NTP grounded by the DiffSR spectral EBM. The
            # bilinear factorization phi(s, z) = g(z)^T F(s) keeps the token
            # predictive OF THE STATE TRANSITION (the energy grounding), the
            # predictor keeps it predictable ONE CHUNK AHEAD, and SIGReg holds
            # the token distribution isotropic-Gaussian so neither term can
            # collapse it. Endpoint target = the executed chunk's boundary
            # s_{t+H}, in the executed chunk's own anchor -- exactly the
            # standard endpoint objective on that chunk.
            boundary = window[:, (context + 1) * horizon - 1]
            if context > 0:
                endpoint = _reanchor_heading_frames(
                    boundary, window[:, context * horizon - 1]
                )
            else:
                endpoint = boundary
            self.diffsr.update_obs_norm(endpoint.detach())
            diffsr_loss = self._diffsr_loss_for_z(exec_source, z1, endpoint)
            head = str(self.config.jepa_ntp_head)
            if head == "diff_token":
                ntp_target = z2.detach()
                self.jepa_ntp_diffsr.update_obs_norm(ntp_target)  # type: ignore[union-attr]
                ntp = self._ntp_diffsr_loss(exec_source, z1, ntp_target)
            elif head == "diff_chunk":
                if str(self.config.jepa_ntp_chunk_anchor) == "next":
                    # Re-anchored onto s_{t+H}'s own heading frame — what the
                    # encoder sees one publication later; displacement erased.
                    # target_window was built by chunk_at(context + 1) above.
                    ntp_target = target_window.reshape(z1.shape[0], -1)
                else:
                    # EXECUTED chunk's slot-0 frame: the cross-chunk
                    # displacement (drift) stays in the target. context==0 is
                    # enforced by validate(). 'boundary_next' widens the slice
                    # by one slot to include the executed chunk's boundary
                    # s[t+H] — the merged-head cell, where the separate
                    # endpoint term is dropped via jepa_endpoint_coeff=0.
                    span_start = (
                        horizon - 1
                        if str(self.config.jepa_ntp_chunk_span) == "boundary_next"
                        else horizon
                    )
                    ntp_target = window[:, span_start : 2 * horizon].reshape(
                        z1.shape[0], -1
                    )
                self.jepa_ntp_diffsr.update_obs_norm(ntp_target.detach())  # type: ignore[union-attr]
                ntp = self._ntp_diffsr_loss(exec_source, z1, ntp_target.detach())
            elif head == "diff_pair":
                # Joint next (state, token): the endpoint in the executed
                # chunk's frame concatenated with the target encoder's token.
                ntp_target = torch.cat([endpoint, z2], dim=-1).detach()
                self.jepa_ntp_diffsr.update_obs_norm(ntp_target)  # type: ignore[union-attr]
                ntp = self._ntp_diffsr_loss(exec_source, z1, ntp_target)
            else:
                ntp = F.mse_loss(prediction, z2)
            sigreg = _sigreg_epps_pulley(
                z1, num_sketches=int(self.config.jepa_sigreg_sketches)
            )
            objective = (
                float(self.config.jepa_endpoint_coeff) * diffsr_loss
                + float(self.config.jepa_ntp_coeff) * ntp
            )
            if float(self.config.jepa_token_pred_coeff) > 0:
                # Additive EMA-trick term next to a diffusion head: the mlp
                # predictor's MSE toward the target encoder's token. z2 is
                # gradient-free here (EMA/stopgrad guard in validate is the
                # mlp double-count check; the target mode is unrestricted).
                objective = objective + float(
                    self.config.jepa_token_pred_coeff
                ) * F.mse_loss(prediction, z2)
            loss = (
                objective
                + float(self.config.jepa_sigreg_coeff) * sigreg
                + self.config.reg_coeff * reg_loss
            )
            # The endpoint and NTP terms merged into jepa_objective are the two
            # axes of the window-usage question; log them separately too.
            term_metrics = {
                "train/jepa_endpoint_loss": float(diffsr_loss.detach().item()),
                "train/jepa_ntp_loss": float(ntp.detach().item()),
            }
            with torch.no_grad():
                logits = -torch.cdist(prediction, z2)
        elif self.config.jepa_loss == "sigreg":
            # LeJEPA: plain prediction MSE toward the EMA target token, with
            # SIGReg carrying the whole anti-collapse burden. Applied to the
            # ONLINE embeddings (z1): the target branch inherits the geometry
            # through the EMA. No negatives, no energy heads.
            objective = F.mse_loss(prediction, z2)
            sigreg = _sigreg_epps_pulley(
                z1, num_sketches=int(self.config.jepa_sigreg_sketches)
            )
            loss = (
                objective
                + float(self.config.jepa_sigreg_coeff) * sigreg
                + self.config.reg_coeff * reg_loss
            )
            term_metrics = {}
            with torch.no_grad():
                # Batch NTP retrieval accuracy as a scale-free diagnostic even
                # though nothing is trained contrastively.
                logits = -torch.cdist(prediction, z2)
        else:
            logits = (self.jepa_g(prediction) @ self.jepa_f(z2).T) / float(
                self.config.jepa_tau
            )
            labels_ce = torch.arange(logits.shape[0], device=logits.device)
            objective = 0.5 * (
                F.cross_entropy(logits, labels_ce)
                + F.cross_entropy(logits.T, labels_ce)
            )
            sigreg = torch.zeros((), device=z1.device)
            loss = objective + self.config.reg_coeff * reg_loss
            term_metrics = {}

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics = self._z_diagnostics(z1.detach(), prefix="train")
        if self.config.grad_clip_norm is not None:
            params = [
                *self.skill_encoder.parameters(),
                *self.jepa_predictor.parameters(),
                *self.jepa_g.parameters(),
                *self.jepa_f.parameters(),
                # sigreg_ebm trains the DiffSR heads too; harmless otherwise.
                *self.diffsr.parameters(),
                *(
                    self.jepa_ntp_diffsr.parameters()
                    if self.jepa_ntp_diffsr is not None
                    else []
                ),
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, max_norm=float(self.config.grad_clip_norm)
            )
            metrics["train/grad_norm"] = float(grad_norm.item())
        self.optimizer.step()
        if ema_mode:
            momentum = float(self.config.jepa_ema_momentum)
            with torch.no_grad():
                for target, online in zip(
                    self.jepa_target_encoder.parameters(),  # type: ignore[union-attr]
                    self.skill_encoder.parameters(),
                    strict=True,
                ):
                    target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        self.skill_encoder.on_after_train_step(self.update)
        self.update += 1
        with torch.no_grad():
            labels = torch.arange(logits.shape[0], device=logits.device)
            accuracy = (logits.argmax(dim=1) == labels).float().mean()
        metrics.update(
            {
                "train/loss": float(loss.detach().item()),
                "train/jepa_objective": float(objective.detach().item()),
                "train/jepa_sigreg": float(sigreg.detach().item()),
                "train/jepa_ntp_accuracy": float(accuracy.item()),
                # The phase-2 gate pair: prediction must beat copying the
                # previous token. ratio < 1 passes.
                "train/jepa_copy_mse": float(copy_mse.item()),
                "train/jepa_pred_mse": float(pred_mse.item()),
                "train/jepa_pred_over_copy": float(
                    (pred_mse / copy_mse.clamp_min(1.0e-12)).item()
                ),
                "train/reg_loss": float(reg_loss.detach().item()),
                **term_metrics,
                **{f"train/{k}": float(v.item()) for k, v in info.items()},
            }
        )
        return metrics

    def train_step(self) -> dict[str, float]:
        if self.config.transition_objective == "jepa_ntp":
            return self._jepa_train_step()
        if self.config.transition_objective == "reconstruction":
            return self._reconstruction_train_step()
        self.skill_encoder.train()
        self.diffsr.train()
        batch = self._sample_macro_batch(
            self.config.batch_size, split=self.config.train_split
        )
        state, future_window, target = _validate_macro_batch(
            batch,
            batch_size=self.config.batch_size,
            horizon_steps=int(self.config.horizon_steps),
            device=self.device,
        )
        del target
        # reg_loss is the per-method latent regularizer (L2 / KL / commitment / 0),
        # weighted uniformly by reg_coeff; info carries method-specific diagnostics.
        z, reg_loss, info = self._encode_skill(state, future_window, step=self.update)
        objective_state, objective_target, selected_offsets = (
            self._sample_objective_transition(state, future_window)
        )
        self.diffsr.update_obs_norm(objective_target.detach())
        diffsr_loss = self._diffsr_loss_for_z(objective_state, z, objective_target)
        z_norm_loss = z.pow(2).mean()
        loss = diffsr_loss + self.config.reg_coeff * reg_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics = self._z_diagnostics(z.detach(), prefix="train")
        if self.config.grad_clip_norm is not None:
            params = [*self.skill_encoder.parameters(), *self.diffsr.parameters()]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=float(self.config.grad_clip_norm),
            )
            metrics["train/grad_norm"] = float(grad_norm.item())
        self.optimizer.step()
        self.skill_encoder.on_after_train_step(self.update)
        self.update += 1
        metrics.update(
            {
                "train/loss": float(loss.detach().item()),
                "train/diffsr_loss": float(diffsr_loss.detach().item()),
                "train/z_norm_loss": float(z_norm_loss.detach().item()),
                "train/reg_loss": float(reg_loss.detach().item()),
                "train/transition_offset_mean": float(
                    selected_offsets.to(torch.float32).mean().item()
                ),
                "train/transition_offset_min": float(selected_offsets.min().item()),
                "train/transition_offset_max": float(selected_offsets.max().item()),
                **{f"train/{k}": float(v.item()) for k, v in info.items()},
            }
        )
        if self.commander is not None:
            metrics.update(self._commander_train_step(batch, state, z))
        return metrics

    def _reconstruction_train_step(self) -> dict[str, float]:
        assert self.reconstruction_decoder is not None
        self.skill_encoder.train()
        self.reconstruction_decoder.train()
        batch = self._sample_macro_batch(
            self.config.batch_size, split=self.config.train_split
        )
        state, future_window, endpoint = _validate_macro_batch(
            batch,
            batch_size=self.config.batch_size,
            horizon_steps=int(self.config.horizon_steps),
            device=self.device,
        )
        z, reg_loss, info = self._encode_skill(state, future_window, step=self.update)
        target = self._reconstruction_target(state, future_window, endpoint)
        prediction = self.reconstruction_decoder(z)
        reconstruction_loss = F.mse_loss(prediction, target)
        loss = reconstruction_loss + self.config.reg_coeff * reg_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics = self._z_diagnostics(z.detach(), prefix="train")
        if self.config.grad_clip_norm is not None:
            parameters = [
                *self.skill_encoder.parameters(),
                *self.reconstruction_decoder.parameters(),
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(self.config.grad_clip_norm),
            )
            metrics["train/grad_norm"] = float(grad_norm.item())
        self.optimizer.step()
        self.skill_encoder.on_after_train_step(self.update)
        self.update += 1
        error = prediction.detach() - target
        metrics.update(
            {
                "train/loss": float(loss.detach().item()),
                "train/reconstruction_loss": float(reconstruction_loss.detach().item()),
                "train/reconstruction_mae": float(error.abs().mean().item()),
                "train/reconstruction_max_abs": float(error.abs().max().item()),
                "train/reg_loss": float(reg_loss.detach().item()),
                **{f"train/{key}": float(value.item()) for key, value in info.items()},
            }
        )
        if self.commander is not None:
            metrics.update(self._commander_train_step(batch, state, z))
        return metrics

    @torch.no_grad()
    def _jepa_eval_term_metrics(
        self, *, batch_size: int, split: str | None, prefix: str
    ) -> dict[str, float]:
        """Endpoint and NTP losses of the sigreg_ebm objective on ``split``.

        Mirrors the ``_jepa_train_step`` target construction (context 0 only)
        without gradient or normalizer updates, so pretrain arms that differ
        only in ``encoder_window_mode`` compare on one matched eval metric.
        Returns {} for configurations it does not cover.

        Each term is also evaluated with the batch-shuffled code, and the pair
        is reported as ``*_z_explained``: ``1 - real / shuffled``, the fraction
        of the head's loss that knowing THIS window's code removes. That ratio
        is dimensionless, so it stays comparable when ``horizon_steps`` changes
        the width of the chunk target and moves the raw losses wholesale.
        """
        if (
            self.config.transition_objective != "jepa_ntp"
            or str(self.config.jepa_loss) != "sigreg_ebm"
            or int(self.config.jepa_context_chunks) != 0
        ):
            return {}
        horizon = int(self.config.horizon_steps)
        from rlopt.env_interface import require_imitation_interface

        sampler = require_imitation_interface(
            self.env,
            "sample_expert_macro_transition_batch",
            purpose="jepa eval term metrics require it but",
        )
        state, window, source = self._sample_jepa_window(
            sampler,
            batch_size=int(batch_size),
            chunk_steps=2 * horizon,
            split=split,
        )
        z1, *_ = self.skill_encoder.encode(
            state,
            _encoder_input_window(self.config, window[:, :horizon]),
            deterministic=True,
        )
        # Control code: the same codes, paired with the wrong windows. It keeps
        # the code distribution identical and destroys only the pairing.
        if int(z1.shape[0]) > 1:
            z_shuffled = z1[torch.randperm(z1.shape[0], device=z1.device)]
        else:
            z_shuffled = z1.clone()
        endpoint = window[:, horizon - 1]
        endpoint_loss = self._diffsr_loss_for_z(source, z1, endpoint)
        endpoint_shuffled = self._diffsr_loss_for_z(source, z_shuffled, endpoint)

        def _target_token() -> Tensor:
            anchor = window[:, horizon - 1]
            target_state = _reanchor_heading_frames(anchor, anchor)
            target_window = _reanchor_heading_frames(
                window[:, horizon : 2 * horizon], anchor
            )
            encoder = (
                self.jepa_target_encoder
                if str(self.config.jepa_target_encoder_mode) == "ema"
                and self.jepa_target_encoder is not None
                else self.skill_encoder
            )
            was_training = encoder.training
            encoder.eval()
            token, *_ = encoder.encode(
                target_state,
                _encoder_input_window(self.config, target_window),
                deterministic=True,
            )
            if was_training:
                encoder.train()
            return token

        head = str(self.config.jepa_ntp_head)
        if head == "diff_chunk":
            if str(self.config.jepa_ntp_chunk_anchor) == "next":
                anchor = window[:, horizon - 1]
                ntp_target = _reanchor_heading_frames(
                    window[:, horizon : 2 * horizon], anchor
                ).reshape(z1.shape[0], -1)
            else:
                span_start = (
                    horizon - 1
                    if str(self.config.jepa_ntp_chunk_span) == "boundary_next"
                    else horizon
                )
                ntp_target = window[:, span_start : 2 * horizon].reshape(
                    z1.shape[0], -1
                )
            ntp_loss = self._ntp_diffsr_loss(source, z1, ntp_target)
            ntp_shuffled = self._ntp_diffsr_loss(source, z_shuffled, ntp_target)
        elif head == "diff_token":
            token = _target_token()
            ntp_loss = self._ntp_diffsr_loss(source, z1, token)
            ntp_shuffled = self._ntp_diffsr_loss(source, z_shuffled, token)
        elif head == "diff_pair":
            pair = torch.cat([endpoint, _target_token()], dim=-1)
            ntp_loss = self._ntp_diffsr_loss(source, z1, pair)
            ntp_shuffled = self._ntp_diffsr_loss(source, z_shuffled, pair)
        else:
            assert self.jepa_predictor is not None
            token = _target_token()
            ntp_loss = F.mse_loss(self.jepa_predictor(z1), token)
            ntp_shuffled = F.mse_loss(self.jepa_predictor(z_shuffled), token)

        def _explained(real: Tensor, shuffled: Tensor) -> float:
            """Fraction of the control loss that the true code removes."""
            control = float(shuffled.item())
            if abs(control) < 1e-12:
                return 0.0
            return 1.0 - float(real.item()) / control

        return {
            f"{prefix}/jepa_endpoint_loss_eval": float(endpoint_loss.item()),
            f"{prefix}/jepa_endpoint_loss_shuffled_eval": float(
                endpoint_shuffled.item()
            ),
            f"{prefix}/jepa_endpoint_z_explained": _explained(
                endpoint_loss, endpoint_shuffled
            ),
            f"{prefix}/jepa_ntp_loss_eval": float(ntp_loss.item()),
            f"{prefix}/jepa_ntp_loss_shuffled_eval": float(ntp_shuffled.item()),
            f"{prefix}/jepa_ntp_z_explained": _explained(ntp_loss, ntp_shuffled),
        }

    def evaluate(
        self,
        *,
        num_batches: int | None = None,
        batch_size: int | None = None,
        prefix: str = "train",
        include_reconstruction: bool = False,
        split: str | None = None,
    ) -> dict[str, float]:
        num_batches = self.config.eval_batches if num_batches is None else num_batches
        batch_size = (
            self.config.eval_batch_size or self.config.batch_size
            if batch_size is None
            else batch_size
        )
        num_batches = _require_positive_int("num_batches", num_batches)
        batch_size = _require_positive_int("batch_size", batch_size)
        split = self.config.eval_split if split is None else split

        encoder_was_training = self.skill_encoder.training
        diffsr_was_training = self.diffsr.training
        reconstruction_was_training = (
            self.reconstruction_decoder.training
            if self.reconstruction_decoder is not None
            else False
        )
        self.skill_encoder.eval()
        self.diffsr.eval()
        if self.reconstruction_decoder is not None:
            self.reconstruction_decoder.eval()
        history = int(self.config.source_history_steps)
        accum: dict[str, float] = {}
        for _ in range(num_batches):
            batch: TensorDictBase | None
            eval_source: Tensor | None
            if history > 0:
                # phi conditions on a past chunk, and under
                # source_anchor='past_start' the encoder's own input is anchored
                # on the oldest past frame. Sampling through the shared helper
                # keeps this evaluation on the SAME geometry the arm trains on;
                # the plain macro batch would score the encoder off-distribution.
                from rlopt.env_interface import require_imitation_interface

                sampler = require_imitation_interface(
                    self.env,
                    "sample_expert_macro_transition_batch",
                    purpose="Offline skill-encoder evaluation requires it but",
                )
                state, future_window, eval_source = self._sample_jepa_window(
                    sampler,
                    batch_size=batch_size,
                    chunk_steps=int(self.config.horizon_steps),
                    split=split,
                )
                batch, _target = None, future_window[:, -1]
            else:
                batch = self._sample_macro_batch(batch_size, split=split)
                state, future_window, _target = _validate_macro_batch(
                    batch,
                    batch_size=batch_size,
                    horizon_steps=int(self.config.horizon_steps),
                    device=self.device,
                )
                eval_source = None
            z, *_ = self._encode_skill(state, future_window, deterministic=True)
            zero_z = torch.zeros_like(z)
            if int(z.shape[0]) > 1:
                shuffled_z = z[torch.randperm(z.shape[0], device=z.device)]
            else:
                shuffled_z = z.clone()
            batch_metrics = self._z_diagnostics(z, prefix=prefix)
            if self.reconstruction_decoder is not None:
                batch_metrics.update(
                    self._reconstruction_eval_metrics(
                        state,
                        future_window,
                        _target,
                        z,
                        prefix=prefix,
                    )
                )
            else:
                batch_metrics.update(
                    self._objective_eval_loss_metrics(
                        state,
                        future_window,
                        z,
                        zero_z,
                        shuffled_z,
                        prefix=prefix,
                        source_override=eval_source,
                    )
                )
            batch_metrics.update(
                self._jepa_eval_term_metrics(
                    batch_size=batch_size, split=split, prefix=prefix
                )
            )
            # Per-method diversity / collapse diagnostics.
            diversity = self.skill_encoder.diversity_metrics(
                state, _encoder_input_window(self.config, future_window)
            )
            batch_metrics.update(
                {
                    f"{prefix}/diversity/{key}": float(value.item())
                    for key, value in diversity.items()
                }
            )
            if self.commander is not None and batch is not None:
                batch_metrics.update(
                    self._commander_eval_metrics(batch, state, z, prefix=prefix)
                )
            if include_reconstruction and self.reconstruction_decoder is None:
                batch_metrics.update(
                    self._objective_reconstruction_metrics(
                        state,
                        future_window,
                        z,
                        prefix=prefix,
                    )
                )
            for key, value in batch_metrics.items():
                accum[key] = accum.get(key, 0.0) + float(value)
        for key in accum:
            accum[key] /= float(num_batches)
        if encoder_was_training:
            self.skill_encoder.train()
        if diffsr_was_training:
            self.diffsr.train()
        if reconstruction_was_training and self.reconstruction_decoder is not None:
            self.reconstruction_decoder.train()
        return accum

    def train(
        self,
        *,
        log_callback: Callable[[dict[str, float | int]], None] | None = None,
        checkpoint_path: str | Path | None = None,
        reconstruction_eval: bool = False,
    ) -> HighLevelSkillDiffSRTrainState:
        start_time = time.perf_counter()
        state = HighLevelSkillDiffSRTrainState()
        best_eval = float("inf")
        best_path = (
            Path(checkpoint_path).with_name("best.pt")
            if checkpoint_path is not None
            else None
        )
        for _ in range(self.config.num_updates):
            metrics = self.train_step()
            should_log = (
                self.update in (1, self.config.num_updates)
                or self.update % self.config.log_interval == 0
            )
            if should_log:
                metrics.update(
                    self.evaluate(
                        prefix="train",
                        include_reconstruction=reconstruction_eval,
                    )
                )
                eval_metric = (
                    "train/reconstruction_loss_eval"
                    if self.reconstruction_decoder is not None
                    else "train/loss_real_z_eval"
                )
                eval_loss = metrics.get(eval_metric)
                if (
                    best_path is not None
                    and eval_loss is not None
                    and eval_loss < best_eval
                ):
                    best_eval = float(eval_loss)
                    self.save_checkpoint(best_path)
                elapsed = time.perf_counter() - start_time
                row: dict[str, float | int] = {
                    "update": int(self.update),
                    "elapsed_seconds": float(elapsed),
                    **metrics,
                }
                if log_callback is not None:
                    log_callback(row)
                state.last_metrics = {
                    key: float(value) for key, value in metrics.items()
                }
                state.elapsed_seconds = float(elapsed)
                state.update = int(self.update)
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
        return state

    def checkpoint_state_dict(self) -> dict[str, Any]:
        obs_norm = getattr(self.diffsr, "obs_norm", None)
        feature_norm_state = (
            obs_norm.state_dict() if isinstance(obs_norm, nn.Module) else {}
        )
        checkpoint = {
            "skill_encoder_state_dict": self.skill_encoder.state_dict(),
            "diffsr_state_dict": self.diffsr.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "update": int(self.update),
            "feature_normalization_state_dict": feature_norm_state,
        }
        if self.jepa_predictor is not None:
            # Deployment reads only the encoder; these are for resume and for
            # post-hoc inspection of the energy landscape. online-mode runs
            # have no target encoder and their checkpoints refuse online
            # finetuning, so the missing key is safe.
            jepa_state: dict[str, Any] = {
                "predictor": self.jepa_predictor.state_dict(),
                "g": self.jepa_g.state_dict(),  # type: ignore[union-attr]
                "f": self.jepa_f.state_dict(),  # type: ignore[union-attr]
            }
            if self.jepa_target_encoder is not None:
                jepa_state["target_encoder"] = self.jepa_target_encoder.state_dict()
            if self.jepa_ntp_diffsr is not None:
                jepa_state["ntp_diffsr"] = self.jepa_ntp_diffsr.state_dict()
            checkpoint["jepa_state_dict"] = jepa_state
        if self.reconstruction_decoder is not None:
            checkpoint["reconstruction_decoder_state_dict"] = (
                self.reconstruction_decoder.state_dict()
            )
        return checkpoint

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Stage through node-local disk. torch's zip writer fails on some
        # network filesystems with "[enforce fail at inline_container.cc] .
        # unexpected pos N vs N-48" / "basic_ios::clear: iostream error" when
        # writing a multi-GB checkpoint directly to shared scratch (ICE job
        # 5577507, 2026-08-15); the same directory accepts ordinary writes, so
        # the failure is in the streaming zip write, not permissions or quota.
        # Writing locally and moving is also atomic from a reader's view.
        import os  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        staging_root = os.environ.get("TMPDIR") or "/tmp"
        handle, staged = tempfile.mkstemp(
            prefix=f"{target.stem}.", suffix=".pt", dir=staging_root
        )
        os.close(handle)
        try:
            torch.save(self.checkpoint_state_dict(), staged)
            shutil.move(staged, target)
        finally:
            if os.path.exists(staged):
                os.unlink(staged)
        Path(f"{target}.json").write_text(
            json.dumps({"update": int(self.update)}), encoding="utf-8"
        )

    def commander_checkpoint_state_dict(
        self, *, skill_checkpoint_path: str = ""
    ) -> dict[str, Any]:
        """SkillCommander-format checkpoint for the co-trained commander."""
        if self.commander is None:
            msg = "No co-trained commander to checkpoint."
            raise RuntimeError(msg)
        return {
            "generator_state_dict": self.commander.state_dict(),
            "config": {
                "generator_hidden_dims": list(self.config.commander_hidden_dims)
            },
            "skill_config": self.config.to_dict(),
            "skill_checkpoint_path": str(skill_checkpoint_path),
            "state_dim": int(self.state_dim),
            "lang_embed_dim": int(self.commander_lang_embed_dim),
            "z_dim": int(self.config.z_dim),
            "update": int(self.update),
        }

    def save_commander_checkpoint(
        self, path: str | Path, *, skill_checkpoint_path: str = ""
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.commander_checkpoint_state_dict(
                skill_checkpoint_path=skill_checkpoint_path
            ),
            target,
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        loaded_config = HighLevelSkillDiffSRConfig.from_dict(checkpoint["config"])
        if loaded_config.horizon_steps != self.config.horizon_steps:
            msg = (
                "Checkpoint horizon_steps does not match trainer construction: "
                f"{loaded_config.horizon_steps} != {self.config.horizon_steps}."
            )
            raise ValueError(msg)
        if loaded_config.z_dim != self.config.z_dim:
            msg = (
                "Checkpoint z_dim does not match trainer construction: "
                f"{loaded_config.z_dim} != {self.config.z_dim}."
            )
            raise ValueError(msg)
        if loaded_config.encoder_window_mode != self.config.encoder_window_mode:
            msg = (
                "Checkpoint encoder_window_mode does not match trainer construction: "
                f"{loaded_config.encoder_window_mode!r} != {self.config.encoder_window_mode!r}."
            )
            raise ValueError(msg)
        if loaded_config.source_history_steps != self.config.source_history_steps:
            msg = (
                "Checkpoint source_history_steps does not match trainer "
                f"construction: {loaded_config.source_history_steps} != "
                f"{self.config.source_history_steps}. phi's input width is "
                "derived from it, so the restore would be silently wrong."
            )
            raise ValueError(msg)
        if loaded_config.source_anchor != self.config.source_anchor:
            msg = (
                "Checkpoint source_anchor does not match trainer construction: "
                f"{loaded_config.source_anchor!r} != {self.config.source_anchor!r}."
            )
            raise ValueError(msg)
        if loaded_config.transition_objective != self.config.transition_objective:
            msg = (
                "Checkpoint transition_objective does not match trainer "
                f"construction: {loaded_config.transition_objective!r} != "
                f"{self.config.transition_objective!r}."
            )
            raise ValueError(msg)
        if loaded_config.transition_offsets != self.config.transition_offsets:
            msg = (
                "Checkpoint transition_offsets do not match trainer construction: "
                f"{loaded_config.transition_offsets!r} != "
                f"{self.config.transition_offsets!r}."
            )
            raise ValueError(msg)
        self.config = loaded_config
        self.encoder_window_steps = _encoder_window_steps(self.config)
        self.skill_encoder.load_state_dict(checkpoint["skill_encoder_state_dict"])
        self.diffsr.load_state_dict(checkpoint["diffsr_state_dict"])
        if self.reconstruction_decoder is not None:
            decoder_state = checkpoint.get("reconstruction_decoder_state_dict")
            if decoder_state is None:
                msg = "Reconstruction checkpoint has no reconstruction decoder state."
                raise ValueError(msg)
            self.reconstruction_decoder.load_state_dict(decoder_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        feature_norm_state = checkpoint.get("feature_normalization_state_dict")
        obs_norm = getattr(self.diffsr, "obs_norm", None)
        if isinstance(obs_norm, nn.Module) and feature_norm_state:
            obs_norm.load_state_dict(feature_norm_state)
        self.update = int(checkpoint.get("update", 0))
        return cast(dict[str, Any], checkpoint)
