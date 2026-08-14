"""Invariants of the jepa_ntp chunk re-anchoring and objective plumbing."""

from __future__ import annotations

import math

import torch

from rlopt.agent.hl_skill_diffsr import (
    _normalize_transition_objective,
    _reanchor_heading_frames,
    _rot6d_to_matrix,
)


def _frame(qpos: torch.Tensor, pos, yaw: float, pitch: float = 0.0) -> torch.Tensor:
    """One 38-wide root_qpos frame with rotation yaw @ pitch (both about fixed axes)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    yaw_m = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    pitch_m = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotation = yaw_m @ pitch_m
    ori6d = torch.cat([rotation[:, 0], rotation[:, 1]])
    return torch.cat([qpos, torch.tensor(pos, dtype=torch.float32), ori6d])


def test_reanchor_zeroes_own_xy_and_yaw_but_keeps_height_and_pitch() -> None:
    qpos = torch.randn(29)
    anchor = _frame(qpos, [1.5, -2.0, 0.83], yaw=0.7, pitch=0.2)
    out = _reanchor_heading_frames(anchor, anchor)
    # xy and yaw cancel; absolute height and pitch (tilt vs gravity) survive.
    assert torch.allclose(out[:29], qpos)
    assert torch.allclose(out[29:31], torch.zeros(2), atol=1e-5)
    assert torch.allclose(out[31], torch.tensor(0.83), atol=1e-5)
    rotation = _rot6d_to_matrix(out[32:38])
    assert abs(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))) < 1e-5
    assert abs(float(-torch.asin(rotation[2, 0].clamp(-1, 1))) - 0.2) < 1e-4


def test_reanchor_batched_window_is_rigid() -> None:
    torch.manual_seed(0)
    qpos = torch.randn(29)
    anchor = _frame(qpos, [0.4, 0.9, 0.8], yaw=-1.1)
    frame_a = _frame(qpos, [0.6, 1.1, 0.85], yaw=-0.9)
    frame_b = _frame(qpos, [0.9, 1.2, 0.78], yaw=-0.6)
    window = torch.stack([frame_a, frame_b]).unsqueeze(0)  # (1, 2, 38)
    out = _reanchor_heading_frames(window, anchor.unsqueeze(0))
    # Yaw-only re-anchoring is rigid in the horizontal plane: pairwise
    # distances between frames are unchanged.
    before = torch.linalg.vector_norm(frame_a[29:32] - frame_b[29:32])
    after = torch.linalg.vector_norm(out[0, 0, 29:32] - out[0, 1, 29:32])
    assert torch.allclose(before, after, atol=1e-5)
    # Relative yaw between the two frames is also unchanged.
    def yaw_of(frame: torch.Tensor) -> float:
        rotation = _rot6d_to_matrix(frame[32:38])
        return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

    assert abs(
        (yaw_of(frame_a) - yaw_of(frame_b))
        - (yaw_of(out[0, 0]) - yaw_of(out[0, 1]))
    ) < 1e-5


def test_jepa_objective_is_registered_with_alias() -> None:
    assert _normalize_transition_objective("transition_objective", "jepa") == "jepa_ntp"
    assert (
        _normalize_transition_objective("transition_objective", "jepa_ntp")
        == "jepa_ntp"
    )
