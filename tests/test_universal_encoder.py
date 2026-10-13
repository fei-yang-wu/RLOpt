from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

warnings.filterwarnings(
    "ignore",
    message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
    category=DeprecationWarning,
    append=False,
)

from test_jepa_recon_regression_planner import _jepa_config  # noqa: E402
from test_language_tilt_planner import _AnyHorizonEnv  # noqa: E402

from rlopt.agent.hl_skill_diffsr import HighLevelSkillDiffSRTrainer  # noqa: E402
from rlopt.agent.universal_encoder import (  # noqa: E402
    UniversalEncoderConfig,
    UniversalEncoderDistiller,
    UniversalSkillEncoder,
)


def _views(frames: torch.Tensor) -> dict[str, torch.Tensor]:
    # "full" is the teacher frame; "part" a lossy device view of it.
    return {"full": frames, "part": frames[..., 29:38]}


def _config(**overrides) -> UniversalEncoderConfig:
    base = {
        "z_dim": 5,
        "frames": 3,  # horizon 3, intermediate window: s_t + 2 slots
        "view_widths": {"full": 38, "part": 9},
        "modes": {"robot": ["full"], "device": ["part"]},
        "hidden_dims": (32, 16),
        "batch_size": 8,
        "num_updates": 2,
        "eval_batches": 1,
        "eval_batch_size": 8,
    }
    base.update(overrides)
    return UniversalEncoderConfig(**base)


def _teacher() -> HighLevelSkillDiffSRTrainer:
    torch.manual_seed(0)
    env = _AnyHorizonEnv(deterministic=False, horizon_steps=3, state_dim=38)
    return HighLevelSkillDiffSRTrainer(_jepa_config(env), env)


def test_masked_views_are_ignored_and_presence_is_visible() -> None:
    torch.manual_seed(1)
    encoder = UniversalSkillEncoder(_config())
    frames = torch.randn(4, 3, 38)
    views = _views(frames)
    z_device = encoder.encode_mode(views, "device")
    # The device mode never sees the full frame: perturbing it changes nothing.
    perturbed = {"full": torch.randn_like(frames), "part": views["part"]}
    assert torch.allclose(encoder.encode_mode(perturbed, "device"), z_device)
    # Absent views may be omitted entirely.
    assert torch.allclose(encoder.encode_mode({"full": None, "part": views["part"]}, "device"), z_device)
    with pytest.raises(ValueError, match="marked present"):
        encoder.encode_mode({"full": None, "part": views["part"]}, "robot")


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="unknown views"):
        _config(modes={"bad": ["nope"]}).validate()
    with pytest.raises(ValueError, match="unknown modes"):
        _config(mode_probs={"ghost": 1.0}).validate()
    weights = _config(mode_probs={"robot": 3.0, "device": 1.0}).mode_weights()
    assert torch.allclose(weights, torch.tensor([0.75, 0.25]))
    again = UniversalEncoderConfig.from_dict(_config().to_dict())
    assert again.modes == {"robot": ["full"], "device": ["part"]}


def test_distiller_trains_student_only_and_round_trips(tmp_path: Path) -> None:
    teacher = _teacher()
    teacher_before = {k: v.clone() for k, v in teacher.skill_encoder.state_dict().items()}
    ntp_before = {k: v.clone() for k, v in teacher.jepa_ntp_diffsr.state_dict().items()}  # type: ignore[union-attr]
    distiller = UniversalEncoderDistiller(_config(ntp_coeff=0.5), teacher=teacher, view_fn=_views)
    student_before = {k: v.clone() for k, v in distiller.student.state_dict().items()}
    metrics = distiller.train_step()
    for key in ("train/loss", "train/align", "train/ntp_loss"):
        assert key in metrics, key
    assert metrics["train/ntp_loss"] > 0.0
    assert any(not torch.equal(student_before[k], v) for k, v in distiller.student.state_dict().items())
    assert all(torch.equal(teacher_before[k], v) for k, v in teacher.skill_encoder.state_dict().items())
    assert all(
        torch.equal(ntp_before[k], v)
        for k, v in teacher.jepa_ntp_diffsr.state_dict().items()  # type: ignore[union-attr]
    )
    eval_metrics = distiller.evaluate(prefix="eval")
    for mode in ("robot", "device"):
        for stat in ("align", "cosine", "r2", "z_rms", "ntp_loss"):
            assert f"eval/{mode}/{stat}" in eval_metrics
    assert "eval/teacher/ntp_loss" in eval_metrics
    path = tmp_path / "universal.pt"
    distiller.save_checkpoint(path, teacher_checkpoint_path="teacher.pt", view_spec={"kind": "test"})
    state = torch.load(path, weights_only=False)
    restored = UniversalSkillEncoder.from_checkpoint_state_dict(state)
    views = _views(torch.randn(2, 3, 38))
    restored.eval()
    distiller.student.eval()
    assert torch.allclose(restored.encode_mode(views, "robot"), distiller.student.encode_mode(views, "robot"))
    assert state["view_spec"] == {"kind": "test"}


def test_distiller_rejects_frame_mismatch() -> None:
    teacher = _teacher()
    with pytest.raises(ValueError, match="frames"):
        UniversalEncoderDistiller(_config(frames=4), teacher=teacher, view_fn=_views)


def test_distiller_augmentation_pass_aligns_to_the_teacher_on_perturbed_windows() -> None:
    teacher = _teacher()
    calls = []

    def augment(frames: torch.Tensor) -> torch.Tensor:
        calls.append(frames.shape)
        return frames + 0.01 * torch.randn_like(frames)

    distiller = UniversalEncoderDistiller(_config(), teacher=teacher, view_fn=_views, augment_fn=augment)
    metrics = distiller.train_step()
    assert metrics["train/align_aug"] > 0.0
    assert len(calls) == 1
    eval_metrics = distiller.evaluate(prefix="eval")
    assert "eval_aug/robot/cosine" in eval_metrics
    assert "eval_aug/device/align" in eval_metrics


def test_distiller_live_windows_pass() -> None:
    teacher = _teacher()
    live = torch.randn(20, 3, 38).to(torch.float16)
    distiller = UniversalEncoderDistiller(
        _config(), teacher=teacher, view_fn=_views, live_frames=live, live_batch_size=6
    )
    metrics = distiller.train_step()
    assert metrics["train/align_live"] > 0.0
    with pytest.raises(ValueError, match="live_frames"):
        UniversalEncoderDistiller(
            _config(), teacher=teacher, view_fn=_views, live_frames=torch.randn(4, 5, 38), live_batch_size=2
        )
