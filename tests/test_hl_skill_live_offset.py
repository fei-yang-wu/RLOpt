"""Planar-offset augmentation and the live-offset online objective (2026-10-10).

Under ``robot_heading`` a live encoder window is its pretraining window moved
by one planar rigid offset: the live robot's heading frame relative to the
reference's own slot-0 heading frame. These tests pin that transform, the
pretrain augmentation that samples it, and the online objective that replays
the policy's measured offsets through the pretraining next-chunk loss.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from test_hl_skill_online_jepa import _config, _discover, _FakeEnv

from rlopt.agent.hl_skill_diffsr import (
    FrozenHighLevelSkillCommandSampler,
    HighLevelSkillDiffSRTrainer,
    _apply_planar_offset,
    _macro_slot0_offset,
    _sample_lognormal_planar_offsets,
)

HORIZON = 2
Z_DIM = 4

DIFF_CHUNK = {
    "jepa_ntp_head": "diff_chunk",
    "jepa_ntp_chunk_anchor": "executed",
    "jepa_ntp_chunk_span": "boundary_next",
    "jepa_endpoint_coeff": 0.0,
    "jepa_sigreg_coeff": 0.0,
    "reg_coeff": 0.0,
    "source_history_steps": 2,
    "source_anchor": "current",
}


def _interleaved(rot: torch.Tensor) -> torch.Tensor:
    return rot[..., :2].reshape(*rot.shape[:-2], 6)


def _rotation(interleaved: torch.Tensor) -> torch.Tensor:
    cols = interleaved.reshape(*interleaved.shape[:-1], 3, 2)
    a1 = torch.nn.functional.normalize(cols[..., 0], dim=-1)
    a2 = torch.nn.functional.normalize(cols[..., 1] - (a1 * cols[..., 1]).sum(-1, keepdim=True) * a1, dim=-1)
    return torch.stack([a1, a2, torch.cross(a1, a2, dim=-1)], dim=-1)


def _yaw(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack(
        [torch.stack([c, -s, zero], -1), torch.stack([s, c, zero], -1), torch.stack([zero, zero, one], -1)], -2
    )


def _canonical_window(batch: int, frames: int) -> torch.Tensor:
    """Random root_qpos frames whose slot 0 sits at the heading origin."""
    q = torch.nn.functional.normalize(torch.randn(batch, frames, 4), dim=-1)
    w, x, y, z = q.unbind(-1)
    rot = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
    # Remove slot 0's yaw and xy from the whole window.
    yaw0 = torch.atan2(rot[:, 0, 1, 0], rot[:, 0, 0, 0])
    undo = _yaw(-yaw0)[:, None]
    rot = undo @ rot
    pos = torch.randn(batch, frames, 3)
    pos = pos - torch.cat([pos[:, :1, :2], torch.zeros(batch, 1, 1)], dim=-1)
    pos = torch.einsum("btij,btj->bti", undo.expand(-1, frames, -1, -1), pos)
    return torch.cat([torch.randn(batch, frames, 29) * 0.3, pos, _interleaved(rot)], dim=-1)


def test_planar_offset_is_the_rigid_move_from_slot0_to_the_live_frame() -> None:
    torch.manual_seed(0)
    frames = _canonical_window(5, 6)
    xy0, yaw0 = _macro_slot0_offset(frames[:, 0])
    assert torch.allclose(xy0, torch.zeros_like(xy0), atol=1e-5)
    assert torch.allclose(yaw0, torch.zeros_like(yaw0), atol=1e-5)
    xy = torch.randn(5, 2) * 0.3
    yaw = (torch.rand(5) - 0.5) * 2.0
    moved = _apply_planar_offset(frames, xy, yaw)
    # Slot 0 lands exactly at the offset.
    got_xy, got_yaw = _macro_slot0_offset(moved[:, 0])
    assert torch.allclose(got_xy, xy, atol=1e-5) and torch.allclose(got_yaw, yaw, atol=1e-5)
    # Same as the matrix route: p' = Rz p + t, R' = Rz R; joints and heights untouched.
    rz = _yaw(yaw)[:, None]
    pos = torch.einsum("btij,btj->bti", rz.expand(-1, 6, -1, -1), frames[..., 29:32])
    pos = pos + torch.cat([xy, torch.zeros(5, 1)], -1)[:, None]
    assert torch.allclose(moved[..., 29:32], pos, atol=1e-5)
    assert torch.allclose(_rotation(moved[..., 32:38]), rz @ _rotation(frames[..., 32:38]), atol=1e-5)
    assert torch.equal(moved[..., :29], frames[..., :29])
    # The inverse move restores the window.
    back = _apply_planar_offset(moved, torch.zeros_like(xy), -yaw)
    back = back - torch.cat(
        [torch.zeros(5, 29), torch.einsum("bij,bj->bi", _yaw(-yaw), torch.cat([xy, torch.zeros(5, 1)], -1)), torch.zeros(5, 6)], -1
    )[:, None]
    assert torch.allclose(back, frames, atol=1e-5)
    with pytest.raises(ValueError, match="38-wide"):
        _apply_planar_offset(torch.zeros(2, 3, 67), xy[:2], yaw[:2])


def test_lognormal_offsets_match_the_configured_medians() -> None:
    torch.manual_seed(1)
    config = _config(macro_offset_aug="lognormal", **DIFF_CHUNK)
    config.validate()
    xy, yaw = _sample_lognormal_planar_offsets(config, 200_000, torch.device("cpu"))
    assert math.isclose(float(xy.norm(dim=-1).median()), 0.05, rel_tol=0.03)
    assert math.isclose(float(torch.rad2deg(yaw.abs()).median()), 1.1, rel_tol=0.03)
    assert abs(float(yaw.mean())) < 0.01  # both signs


def test_pretrain_with_offset_augmentation_trains(tmp_path: Path) -> None:
    torch.manual_seed(2)
    trainer = HighLevelSkillDiffSRTrainer(_config(macro_offset_aug="lognormal", **DIFF_CHUNK), _FakeEnv())
    metrics = trainer.train_step()
    assert math.isfinite(float(metrics["train/loss"]))
    with pytest.raises(ValueError, match="macro_offset_aug"):
        _config(macro_offset_aug="gaussian", **DIFF_CHUNK).validate()


def _checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_config(**DIFF_CHUNK), _FakeEnv())
    trainer.train_step()
    path = tmp_path / "skill.pt"
    trainer.save_checkpoint(path)
    return path


def _live_sampler(checkpoint: Path, **overrides: object) -> FrozenHighLevelSkillCommandSampler:
    kwargs = {
        "env": _FakeEnv(),
        "checkpoint_path": checkpoint,
        "latent_dim": Z_DIM,
        "latent_steps_min": 1,
        "latent_steps_max": 1,
        "discover_env_method": _discover,
        "horizon_steps": HORIZON,
        "finetune_enabled": True,
        "pg_coeff": 0.0,
        "offline_batch_size": 8,
        "online_objective": "ntp_live_offset",
        "live_offset_ring": 16,
    }
    kwargs.update(overrides)
    return FrozenHighLevelSkillCommandSampler(**kwargs)  # type: ignore[arg-type]


def test_live_offset_objective_records_offsets_and_trains_encoder_and_head(tmp_path: Path) -> None:
    sampler = _live_sampler(_checkpoint(tmp_path))
    assert sampler.jepa_ntp_diffsr is not None and sampler.jepa_predictor is None
    head_params = {id(p) for p in sampler.jepa_ntp_diffsr.parameters()}
    assert head_params <= {id(p) for group in sampler.optimizer.param_groups for p in group["params"]}
    td = TensorDict({}, batch_size=[4])
    for _ in range(5):  # 20 renewals into a 16-slot ring: wraps
        sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
    assert sampler._offset_ring_count == 16
    loss, metrics = sampler.compute_online_finetune_loss(td, latent_key="latent", actor_loss_fn=lambda _: None)
    assert torch.isfinite(loss) and float(metrics["hl_skill_live_offset_ring_count"]) == 16.0
    loss.backward()
    encoder_grad = sum(float(p.grad.abs().sum()) for p in sampler.skill_encoder.parameters() if p.grad is not None)
    head_grad = sum(float(p.grad.abs().sum()) for p in sampler.jepa_ntp_diffsr.parameters() if p.grad is not None)
    assert encoder_grad > 0 and head_grad > 0
    # No rollout cache: PPO batches are untouched.
    assert "hl_skill" not in td.keys()


def test_live_offset_objective_refuses_policy_gradient_and_other_heads(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="pg_coeff"):
        _live_sampler(checkpoint, pg_coeff=0.05)
    torch.manual_seed(0)
    mlp = HighLevelSkillDiffSRTrainer(_config(), _FakeEnv())
    mlp.train_step()
    mlp_path = tmp_path / "mlp.pt"
    mlp.save_checkpoint(mlp_path)
    with pytest.raises(ValueError, match="ntp_live_offset"):
        _live_sampler(mlp_path)


def test_live_offset_head_round_trips_through_the_sampler_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    trained = _live_sampler(checkpoint)
    with torch.no_grad():
        for p in trained.jepa_ntp_diffsr.parameters():
            p.add_(0.5)
    fresh = _live_sampler(checkpoint)
    fresh.load_checkpoint_state_dict(trained.checkpoint_state_dict())
    for a, b in zip(fresh.jepa_ntp_diffsr.parameters(), trained.jepa_ntp_diffsr.parameters(), strict=True):
        assert torch.equal(a, b)
