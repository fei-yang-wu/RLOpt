"""Invariants of the jepa_ntp chunk re-anchoring and objective plumbing."""

from __future__ import annotations

import math

import pytest
import torch

from rlopt.agent.hl_skill_diffsr import (
    HighLevelSkillDiffSRConfig,
    _encoder_input_window,
    _encoder_window_steps,
    _normalize_encoder_window_mode,
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

    assert (
        abs(
            (yaw_of(frame_a) - yaw_of(frame_b))
            - (yaw_of(out[0, 0]) - yaw_of(out[0, 1]))
        )
        < 1e-5
    )


def test_jepa_objective_is_registered_with_alias() -> None:
    assert _normalize_transition_objective("transition_objective", "jepa") == "jepa_ntp"
    assert (
        _normalize_transition_objective("transition_objective", "jepa_ntp")
        == "jepa_ntp"
    )


def test_sigreg_discriminates_gaussian_from_collapse_and_blowup() -> None:
    from rlopt.agent.hl_skill_diffsr import _sigreg_epps_pulley

    torch.manual_seed(0)
    gaussian = torch.randn(4096, 64)
    collapsed = torch.zeros(4096, 64)
    blown_up = torch.randn(4096, 64) * 40.0
    good = float(_sigreg_epps_pulley(gaussian, 64))
    bad_collapse = float(_sigreg_epps_pulley(collapsed, 64))
    bad_scale = float(_sigreg_epps_pulley(blown_up, 64))
    assert good < 1e-3
    assert bad_collapse > 10 * good
    assert bad_scale > 10 * good


def test_encoder_window_mode_accepts_suffix() -> None:
    assert _normalize_encoder_window_mode("m", "suffix3") == "suffix3"
    assert _normalize_encoder_window_mode("m", " SUFFIX2 ") == "suffix2"
    for bad in ("suffix0", "suffix", "suffix-1", "prefix2"):
        with pytest.raises(ValueError, match="must be"):
            _normalize_encoder_window_mode("m", bad)


def test_suffix_window_slices_intermediate_tail() -> None:
    config = HighLevelSkillDiffSRConfig(horizon_steps=10, encoder_window_mode="suffix2")
    config.validate()
    assert _encoder_window_steps(config) == 2
    future_window = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    # Intermediate window is slots 0..8 (s_{t+1..t+9}); suffix2 keeps its last
    # two slots (s_{t+8}, s_{t+9}) and never the endpoint slot 9 (s_{t+10}).
    out = _encoder_input_window(config, future_window)
    assert out.shape == (1, 2, 1)
    assert out.flatten().tolist() == [7.0, 8.0]

    intermediate = HighLevelSkillDiffSRConfig(
        horizon_steps=10, encoder_window_mode="intermediate"
    )
    intermediate.validate()
    suffix9 = HighLevelSkillDiffSRConfig(
        horizon_steps=10, encoder_window_mode="suffix9"
    )
    suffix9.validate()
    assert torch.equal(
        _encoder_input_window(suffix9, future_window),
        _encoder_input_window(intermediate, future_window),
    )


def test_suffix_window_rejects_endpoint_reach() -> None:
    config = HighLevelSkillDiffSRConfig(
        horizon_steps=10, encoder_window_mode="suffix10"
    )
    with pytest.raises(ValueError, match="suffix"):
        config.validate()


def test_sigreg_is_differentiable() -> None:
    from rlopt.agent.hl_skill_diffsr import _sigreg_epps_pulley

    z = torch.randn(256, 64, requires_grad=True)
    loss = _sigreg_epps_pulley(z * 3.0, 32)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def _stub_trainer(config):
    """Minimal carrier for the sampling helper (no env, no Isaac)."""
    from rlopt.agent.hl_skill_diffsr import HighLevelSkillDiffSRTrainer

    class _Stub:
        _sample_jepa_window = HighLevelSkillDiffSRTrainer._sample_jepa_window

        def __init__(self, cfg):
            self.config = cfg
            self.device = torch.device("cpu")

    return _Stub(config)


def _index_sampler(cursor=100, width=3, record=None):
    """Fake macro sampler whose every feature value IS the frame's index.

    Mirrors the data plane: with ``state_history_steps=k`` the sequence spans
    ``[cursor-k, cursor+horizon]``, ``state`` is the cursor, and the history
    block ends AT the cursor.
    """
    from tensordict import TensorDict

    def sampler(
        *,
        batch_size,
        horizon_steps,
        split,
        eval_fraction,
        split_seed,
        state_history_steps=0,
    ):
        if record is not None:
            record.append(
                {
                    "horizon_steps": horizon_steps,
                    "state_history_steps": state_history_steps,
                }
            )
        idx = torch.arange(cursor - state_history_steps, cursor + horizon_steps + 1)
        seq = idx.to(torch.float32)[None, :, None].expand(batch_size, -1, width)
        payload = {
            "state": seq[:, state_history_steps],
            "future_window": seq[:, state_history_steps + 1 :],
            "target": seq[:, -1],
        }
        if state_history_steps > 0:
            payload["state_history"] = seq[:, : state_history_steps + 1]
        return TensorDict(
            {"hl": TensorDict(payload, batch_size=[batch_size])},
            batch_size=[batch_size],
        )

    return sampler


def _cfg(**kw):
    base = dict(
        horizon_steps=10,
        encoder_window_mode="intermediate",
        transition_objective="jepa_ntp",
        jepa_loss="sigreg_ebm",
        jepa_ntp_head="diff_chunk",
        z_dim=8,
    )
    base.update(kw)
    config = HighLevelSkillDiffSRConfig(**base)
    config.validate()
    return config


def test_history_zero_leaves_the_source_as_the_state() -> None:
    trainer = _stub_trainer(_cfg())
    calls = []
    state, window, source = trainer._sample_jepa_window(
        _index_sampler(record=calls), batch_size=2, chunk_steps=20, split="train"
    )
    # Unchanged sampling contract: no history requested, 2H window.
    assert calls == [{"horizon_steps": 20, "state_history_steps": 0}]
    assert torch.equal(source, state)
    assert state[0, 0].item() == 100.0
    assert window.shape[1] == 20 and window[0, 0, 0].item() == 101.0


def test_current_anchor_sources_the_past_and_leaves_the_window_alone() -> None:
    trainer = _stub_trainer(_cfg(source_history_steps=10, source_anchor="current"))
    calls = []
    state, window, source = trainer._sample_jepa_window(
        _index_sampler(record=calls), batch_size=2, chunk_steps=20, split="train"
    )
    assert calls == [{"horizon_steps": 20, "state_history_steps": 10}]
    # The encoder's view is identical to the history-free run.
    assert state[0, 0].item() == 100.0
    assert window.shape[1] == 20 and window[0, 0, 0].item() == 101.0
    # phi's source is s[t-10..t]: 11 frames ending AT the current state.
    assert source.shape == (2, 11 * 3)
    frames = source.reshape(2, 11, 3)[0, :, 0]
    assert frames.tolist() == [float(v) for v in range(90, 101)]


def test_past_start_anchor_shifts_the_cursor_and_never_reanchors() -> None:
    trainer = _stub_trainer(_cfg(source_history_steps=10, source_anchor="past_start"))
    calls = []
    state, window, source = trainer._sample_jepa_window(
        _index_sampler(record=calls), batch_size=2, chunk_steps=20, split="train"
    )
    # ONE longer window, no history request: slot 0 is already the oldest past
    # frame, so the anchor is where we want it and nothing is re-anchored.
    assert calls == [{"horizon_steps": 30, "state_history_steps": 0}]
    # The sampled cursor is relabeled as t-10, so s_t sits 10 frames later.
    assert state[0, 0].item() == 110.0
    assert window.shape[1] == 20 and window[0, 0, 0].item() == 111.0
    frames = source.reshape(2, 11, 3)[0, :, 0]
    assert frames.tolist() == [float(v) for v in range(100, 111)]
    # The source ends at s_t in both anchor modes; only the frame differs.
    assert frames[-1].item() == state[0, 0].item()
