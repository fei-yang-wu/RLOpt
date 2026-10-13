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

from test_language_tilt_planner import _AnyHorizonEnv  # noqa: E402
from test_skill_commander import _make_language_table  # noqa: E402

from rlopt.agent.hl_skill_diffsr import (  # noqa: E402
    HighLevelSkillDiffSRConfig,
    HighLevelSkillDiffSRTrainer,
    _NormalizedReconstructionHead,
)
from rlopt.agent.latent_regression_planner import LatentRegressionPlanner  # noqa: E402


def _jepa_config(env: _AnyHorizonEnv, **overrides) -> HighLevelSkillDiffSRConfig:
    """The production jepa recipe (sigreg_ebm / diff_chunk / boundary_next, past chunk)."""
    base = {
        "horizon_steps": env.horizon_steps,
        "z_dim": 5,
        "diffsr_feature_dim": 4,
        "diffsr_embed_dim": 8,
        "batch_size": 8,
        "num_updates": 3,
        "log_interval": 1,
        "eval_batches": 1,
        "eval_batch_size": 8,
        "preflight_batch_size": 4,
        "encoder_hidden_dims": (16, 12),
        "diffsr_f_hidden_dims": (16,),
        "diffsr_g_hidden_dims": (16,),
        "diffsr_mu_hidden_dims": (16,),
        "diffsr_num_noises": 2,
        "device": "cpu",
        "transition_objective": "jepa_ntp",
        "jepa_loss": "sigreg_ebm",
        "jepa_ntp_head": "diff_chunk",
        "jepa_ntp_chunk_span": "boundary_next",
        "jepa_endpoint_coeff": 0.0,
        "encoder_window_mode": "intermediate",
        "source_history_steps": 2,
        "source_anchor": "current",
        "jepa_sigreg_sketches": 4,
    }
    base.update(overrides)
    return HighLevelSkillDiffSRConfig(**base)


def _encoder_snapshot(trainer: HighLevelSkillDiffSRTrainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trainer.skill_encoder.state_dict().items()}


def _encoder_moved(trainer: HighLevelSkillDiffSRTrainer, before: dict[str, torch.Tensor]) -> bool:
    after = trainer.skill_encoder.state_dict()
    return any(not torch.equal(before[k], after[k]) for k in before)


# ---------------------------------------------------------------------------
# Normalized reconstruction head
# ---------------------------------------------------------------------------
def test_reconstruction_head_running_moments_match_concatenated_batches() -> None:
    torch.manual_seed(0)
    head = _NormalizedReconstructionHead(
        z_dim=3, output_dim=4, hidden_dims=(8,), activation="silu", eps=1.0e-4
    )
    a = torch.randn(16, 4) * 3.0 + 1.0
    b = torch.randn(9, 4) * 0.5 - 2.0
    head.update(a)
    head.update(b)
    both = torch.cat([a, b])
    assert torch.allclose(head.mean, both.mean(dim=0), atol=1e-5)
    assert torch.allclose(head.var, both.var(dim=0, unbiased=False), atol=1e-4)
    assert int(head.count.item()) == 25
    # A constant slot normalizes to exactly zero instead of dividing by zero.
    constant = torch.zeros(6, 4)
    fresh = _NormalizedReconstructionHead(
        z_dim=3, output_dim=4, hidden_dims=(8,), activation="silu", eps=1.0e-4
    )
    fresh.update(constant)
    assert torch.equal(fresh.normalize(constant), torch.zeros(6, 4))


# ---------------------------------------------------------------------------
# jepa_ntp + reconstruction, regularizers off
# ---------------------------------------------------------------------------
def test_jepa_reconstruction_term_trains_encoder_and_round_trips(tmp_path: Path) -> None:
    torch.manual_seed(1)
    env = _AnyHorizonEnv(deterministic=False, horizon_steps=3, state_dim=38)
    config = _jepa_config(
        env,
        jepa_reconstruction_coeff=1.0,
        jepa_sigreg_coeff=0.0,
        reg_coeff=0.0,
        # Only the autoencoder term drives the encoder in this step.
        jepa_ntp_coeff=0.0,
    )
    trainer = HighLevelSkillDiffSRTrainer(config, env)
    assert trainer.jepa_reconstruction_head is not None
    # input_window: the state plus the intermediate window (H - 1 slots).
    assert trainer._reconstruction_output_dim() == env.state_dim * env.horizon_steps
    before = _encoder_snapshot(trainer)
    metrics = trainer.train_step()
    assert metrics["train/jepa_reconstruction_loss"] > 0.0
    assert metrics["train/jepa_sigreg"] >= 0.0
    assert _encoder_moved(trainer, before)
    assert int(trainer.jepa_reconstruction_head.count.item()) == config.batch_size
    eval_metrics = trainer.evaluate(prefix="eval")
    for key in (
        "eval/jepa_reconstruction_loss_eval",
        "eval/jepa_reconstruction_loss_shuffled_eval",
        "eval/jepa_reconstruction_z_explained",
        "eval/jepa_ntp_loss_eval",
    ):
        assert key in eval_metrics, key
    resume = tmp_path / "latest.pt"
    trainer.save_checkpoint(resume)
    assert "jepa_reconstruction_head_state_dict" in torch.load(resume, weights_only=False)
    fresh = HighLevelSkillDiffSRTrainer(config, env)
    fresh.load_checkpoint(resume)
    a = trainer.jepa_reconstruction_head.state_dict()
    b = fresh.jepa_reconstruction_head.state_dict()  # type: ignore[union-attr]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_jepa_reconstruction_requires_jepa_objective() -> None:
    with pytest.raises(ValueError, match="jepa_reconstruction_coeff"):
        HighLevelSkillDiffSRConfig(
            transition_objective="endpoint", jepa_reconstruction_coeff=1.0
        ).validate()


def test_default_config_builds_no_new_heads() -> None:
    env = _AnyHorizonEnv(deterministic=False, horizon_steps=3, state_dim=38)
    trainer = HighLevelSkillDiffSRTrainer(_jepa_config(env), env)
    assert trainer.jepa_reconstruction_head is None
    assert trainer.regression_planner is None
    metrics = trainer.train_step()
    assert "train/jepa_reconstruction_loss" not in metrics
    assert "train/regression_planner_nll" not in metrics


# ---------------------------------------------------------------------------
# Jointly trained regression planner
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("detach", [False, True])
def test_regression_planner_gradient_reaches_encoder_only_when_joint(
    tmp_path: Path, detach: bool
) -> None:
    torch.manual_seed(2)
    env = _AnyHorizonEnv(deterministic=False, horizon_steps=3, state_dim=38)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    config = _jepa_config(
        env,
        cotrain_regression_planner=True,
        planner_language_embeddings_path=str(lang_table),
        regression_planner_hidden_dims=(16, 16),
        regression_planner_detach_target=detach,
        # Every other encoder term off: only the planner NLL is live.
        jepa_ntp_coeff=0.0,
        jepa_sigreg_coeff=0.0,
        reg_coeff=0.0,
    )
    trainer = HighLevelSkillDiffSRTrainer(config, env)
    assert trainer.regression_planner is not None
    planner_before = {
        k: v.detach().clone() for k, v in trainer.regression_planner.state_dict().items()
    }
    before = _encoder_snapshot(trainer)
    metrics = trainer.train_step()
    for key in (
        "train/regression_planner_nll",
        "train/regression_planner_mse",
        "train/regression_planner_cosine",
        "train/regression_planner_r2",
    ):
        assert key in metrics, key
    assert _encoder_moved(trainer, before) is (not detach)
    planner_after = trainer.regression_planner.state_dict()
    assert any(not torch.equal(planner_before[k], planner_after[k]) for k in planner_before)


def test_regression_planner_eval_and_checkpoints(tmp_path: Path) -> None:
    torch.manual_seed(3)
    env = _AnyHorizonEnv(deterministic=False, horizon_steps=3, state_dim=38)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    config = _jepa_config(
        env,
        cotrain_regression_planner=True,
        planner_language_embeddings_path=str(lang_table),
        regression_planner_hidden_dims=(16,),
    )
    trainer = HighLevelSkillDiffSRTrainer(config, env)
    assert trainer.planner_state_dim == 3 * env.state_dim
    trainer.train_step()
    eval_metrics = trainer.evaluate(prefix="eval")
    for key in (
        "eval/regression_planner_mse_eval",
        "eval/regression_planner_cosine_eval",
        "eval/regression_planner_r2_eval",
        "eval/regression_planner_nll_eval",
        "eval/regression_planner_cosine_shuffled_lang_eval",
        "eval/regression_planner_pred_loss",
        "eval/regression_planner_pred_loss_encoder_z",
        "eval/regression_planner_pred_loss_shuffled_z",
    ):
        assert key in eval_metrics, key
    standalone = tmp_path / "regression_planner.pt"
    trainer.save_regression_planner_checkpoint(standalone, skill_checkpoint_path="skill.pt")
    saved = torch.load(standalone, weights_only=False)
    restored = LatentRegressionPlanner.from_checkpoint_state_dict(saved)
    assert restored.state_dim == 3 * env.state_dim
    assert saved["detach_target"] is False
    resume = tmp_path / "latest.pt"
    trainer.save_checkpoint(resume)
    fresh = HighLevelSkillDiffSRTrainer(config, env)
    fresh.load_checkpoint(resume)
    a = trainer.regression_planner.state_dict()  # type: ignore[union-attr]
    b = fresh.regression_planner.state_dict()  # type: ignore[union-attr]
    assert all(torch.equal(a[k], b[k]) for k in a)
    again = HighLevelSkillDiffSRConfig.from_dict(config.to_dict())
    assert again.regression_planner_hidden_dims == (16,)


def test_regression_planner_without_language() -> None:
    env = _AnyHorizonEnv(deterministic=False, horizon_steps=3, state_dim=38)
    config = _jepa_config(
        env,
        cotrain_regression_planner=True,
        regression_planner_use_language=False,
        regression_planner_hidden_dims=(16,),
    )
    trainer = HighLevelSkillDiffSRTrainer(config, env)
    assert trainer.regression_planner.lang_embed_dim == 0  # type: ignore[union-attr]
    metrics = trainer.train_step()
    assert "train/regression_planner_nll" in metrics
    eval_metrics = trainer.evaluate(prefix="eval")
    assert "eval/regression_planner_cosine_shuffled_lang_eval" not in eval_metrics


def test_regression_planner_requires_language_table() -> None:
    with pytest.raises(ValueError, match="planner_language_embeddings_path"):
        HighLevelSkillDiffSRConfig(
            transition_objective="jepa_ntp", cotrain_regression_planner=True
        ).validate()


# ---------------------------------------------------------------------------
# Joint-velocity scaling of full_body macro frames
# ---------------------------------------------------------------------------
def test_scale_macro_qvel_touches_only_the_velocity_block() -> None:
    from rlopt.agent.hl_skill_diffsr import _scale_macro_qvel

    frames = torch.randn(4, 3, 67)
    scaled = _scale_macro_qvel(frames, 0.25)
    assert torch.equal(scaled[..., :29], frames[..., :29])
    assert torch.allclose(scaled[..., 29:58], 0.25 * frames[..., 29:58])
    assert torch.equal(scaled[..., 58:], frames[..., 58:])
    history = frames.reshape(4, -1)
    assert torch.allclose(_scale_macro_qvel(history, 0.25), scaled.reshape(4, -1))
    assert _scale_macro_qvel(frames, 1.0) is frames
    with pytest.raises(ValueError, match="67-wide"):
        _scale_macro_qvel(torch.randn(2, 38), 0.5)


def test_macro_qvel_scale_scales_the_sampled_jepa_window() -> None:
    torch.manual_seed(7)
    env = _AnyHorizonEnv(deterministic=False, horizon_steps=3, state_dim=67)
    raw = HighLevelSkillDiffSRTrainer(_jepa_config(env), env)
    scaled = HighLevelSkillDiffSRTrainer(_jepa_config(env, macro_qvel_scale=0.25), env)
    from rlopt.env_interface import require_imitation_interface

    sampler = require_imitation_interface(env, "sample_expert_macro_transition_batch", purpose="test")
    env.calls = 100
    a = raw._sample_jepa_window(sampler, batch_size=4, chunk_steps=6, split="train")
    env.calls = 100
    b = scaled._sample_jepa_window(sampler, batch_size=4, chunk_steps=6, split="train")
    for raw_t, scaled_t in zip(a, b, strict=True):
        frames_raw = raw_t.reshape(*raw_t.shape[:-1], -1, 67)
        frames_scaled = scaled_t.reshape(*scaled_t.shape[:-1], -1, 67)
        assert torch.allclose(frames_scaled[..., 29:58], 0.25 * frames_raw[..., 29:58])
        assert torch.equal(frames_scaled[..., :29], frames_raw[..., :29])
    assert "train/jepa_ntp_loss" in scaled.train_step()
    with pytest.raises(ValueError, match="macro_qvel_scale"):
        HighLevelSkillDiffSRConfig(macro_qvel_scale=0.5).validate()


def test_heads_drop_qvel_keeps_qvel_for_the_encoder_only() -> None:
    torch.manual_seed(8)
    env = _AnyHorizonEnv(deterministic=False, horizon_steps=3, state_dim=67)
    trainer = HighLevelSkillDiffSRTrainer(_jepa_config(env, heads_drop_qvel=True), env)
    assert trainer.state_dim == 67 and trainer.head_state_dim == 38
    # Encoder input: s_t + intermediate window (H - 1 slots), 67 wide.
    assert trainer.skill_encoder.net[0].weight.shape[1] == 67 * env.horizon_steps
    # boundary_next target = H + 1 root_qpos frames.
    assert trainer.planner_state_dim == 3 * 38
    from rlopt.env_interface import require_imitation_interface

    sampler = require_imitation_interface(env, "sample_expert_macro_transition_batch", purpose="test")
    state, window, source = trainer._sample_jepa_window(sampler, batch_size=4, chunk_steps=6, split="train")
    assert state.shape[-1] == 67 and window.shape[-1] == 67 and source.shape[-1] == 3 * 38
    metrics = trainer.train_step()
    assert "train/jepa_ntp_loss" in metrics
    assert "train/jepa_ntp_loss_eval" in trainer.evaluate(prefix="train")
    with pytest.raises(ValueError, match="heads_drop_qvel"):
        HighLevelSkillDiffSRConfig(heads_drop_qvel=True).validate()
    from rlopt.agent.hl_skill_diffsr import _drop_macro_qvel

    frames = torch.randn(2, 67)
    dropped = _drop_macro_qvel(frames)
    assert torch.equal(dropped[:, :29], frames[:, :29])
    assert torch.equal(dropped[:, 29:], frames[:, 58:])
