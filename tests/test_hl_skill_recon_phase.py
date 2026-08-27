"""Reconstruction target variants and the live command-phase clock.

The 2026-08-20 interface design study left two follow-ups:

* the ``reconstruction`` objective tied on success rate but carried +110%
  global drift -- its input-window target is purely local, so nothing binds
  the code to where the motion ends up. ``reconstruction_target`` adds the
  ``endpoint`` and ``full_window`` targets that repair that, as a controlled
  change of the decode target only;
* the ``sin_cos`` phase channel is load-bearing at hold 10 but pinned constant
  at hold 1 (``phase_period == code_period == 1``). ``phase_source="episode"``
  gives a hold-1 interface a live clock: env steps since reset modulo
  ``phase_period``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from rlopt.agent.hl_skill_diffsr import (
    FrozenHighLevelSkillCommandSampler,
    HighLevelSkillDiffSRConfig,
    HighLevelSkillDiffSRTrainer,
)

STATE_DIM = 38
HORIZON = 2
Z_DIM = 4


class _FakeEnv:
    def __init__(self) -> None:
        self.imitation_interface = self
        self.expert_horizons: list[int] = []

    def _macro_batch(self, batch_size: int, horizon_steps: int) -> TensorDict:
        generator = torch.Generator().manual_seed(batch_size + horizon_steps)
        return TensorDict(
            {
                ("hl", "state"): torch.randn(
                    batch_size, STATE_DIM, generator=generator
                ),
                ("hl", "future_window"): torch.randn(
                    batch_size, horizon_steps, STATE_DIM, generator=generator
                ),
                ("hl", "target"): torch.randn(
                    batch_size, STATE_DIM, generator=generator
                ),
            },
            batch_size=[batch_size],
        )

    def sample_expert_macro_transition_batch(
        self, *, batch_size: int, horizon_steps: int, **_: object
    ) -> TensorDict:
        self.expert_horizons.append(int(horizon_steps))
        return self._macro_batch(int(batch_size), int(horizon_steps))

    def current_expert_macro_transition_batch(
        self, horizon_steps: int, env_ids=None, state_history_steps: int = 0
    ) -> TensorDict:
        del state_history_steps
        count = 4 if env_ids is None else int(env_ids.numel())
        return self._macro_batch(count, int(horizon_steps))


def _discover(env: object, name: str):
    return getattr(env, name, None)


def _config(**overrides: object) -> HighLevelSkillDiffSRConfig:
    base = {
        "z_dim": Z_DIM,
        "horizon_steps": HORIZON,
        "encoder_window_mode": "intermediate",
        "transition_objective": "reconstruction",
        "batch_size": 8,
        "encoder_hidden_dims": (32, 32),
        "diffsr_feature_dim": 8,
        "diffsr_embed_dim": 8,
        "diffsr_g_hidden_dims": (16,),
        "diffsr_mu_hidden_dims": (16,),
        "diffsr_f_hidden_dims": (16,),
    }
    base.update(overrides)
    return HighLevelSkillDiffSRConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# reconstruction_target
# ---------------------------------------------------------------------------


def test_reconstruction_target_rejects_unknown_values() -> None:
    # validate() runs at trainer construction and in from_dict.
    with pytest.raises(ValueError, match="reconstruction_target"):
        _config(reconstruction_target="next_chunk").validate()


def test_reconstruction_target_defaults_to_input_window_for_old_checkpoints() -> None:
    # A checkpoint config dict written before the field existed must load as
    # the exact old behaviour.
    config = _config()
    payload = config.to_dict()
    payload.pop("reconstruction_target", None)
    restored = HighLevelSkillDiffSRConfig.from_dict(payload)
    assert restored.reconstruction_target == "input_window"


@pytest.mark.parametrize(
    ("target", "expected_output_dim"),
    [
        # intermediate window mode hides the endpoint: window steps = H - 1.
        ("input_window", STATE_DIM * HORIZON),
        ("endpoint", STATE_DIM),
        ("full_window", STATE_DIM * (HORIZON + 1)),
    ],
)
def test_decoder_output_dim_follows_the_target(
    target: str, expected_output_dim: int
) -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(
        _config(reconstruction_target=target), _FakeEnv()
    )
    assert trainer.reconstruction_decoder is not None
    device = next(trainer.reconstruction_decoder.parameters()).device
    prediction = trainer.reconstruction_decoder(torch.zeros(2, Z_DIM, device=device))
    assert tuple(prediction.shape) == (2, expected_output_dim)


def test_reconstruction_targets_pick_the_right_tensors() -> None:
    torch.manual_seed(0)
    state = torch.randn(3, STATE_DIM)
    window = torch.randn(3, HORIZON, STATE_DIM)
    endpoint = torch.randn(3, STATE_DIM)

    trainer = HighLevelSkillDiffSRTrainer(
        _config(reconstruction_target="endpoint"), _FakeEnv()
    )
    assert torch.equal(
        trainer._reconstruction_target(state, window, endpoint), endpoint
    )

    trainer = HighLevelSkillDiffSRTrainer(
        _config(reconstruction_target="full_window"), _FakeEnv()
    )
    expected = torch.cat((state, window.reshape(3, -1)), dim=-1)
    assert torch.equal(
        trainer._reconstruction_target(state, window, endpoint), expected
    )

    trainer = HighLevelSkillDiffSRTrainer(
        _config(reconstruction_target="input_window"), _FakeEnv()
    )
    # intermediate mode: the endpoint slot is hidden from the encoder input.
    expected = torch.cat((state, window[:, : HORIZON - 1].reshape(3, -1)), dim=-1)
    assert torch.equal(
        trainer._reconstruction_target(state, window, endpoint), expected
    )


@pytest.mark.parametrize("target", ["endpoint", "full_window", "input_window"])
def test_train_step_and_eval_run_for_every_target(target: str) -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(
        _config(reconstruction_target=target), _FakeEnv()
    )
    metrics = trainer.train_step()
    assert math.isfinite(metrics["train/reconstruction_loss"])
    eval_metrics = trainer.evaluate(num_batches=1, batch_size=8)
    assert math.isfinite(eval_metrics["train/reconstruction_loss_eval"])


# ---------------------------------------------------------------------------
# phase_source
# ---------------------------------------------------------------------------


def _pretrained_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(
        _config(transition_objective="endpoint"), _FakeEnv()
    )
    trainer.train_step()
    path = tmp_path / "skill.pt"
    trainer.save_checkpoint(path)
    return path


def _sampler(
    checkpoint: Path, **overrides: object
) -> FrozenHighLevelSkillCommandSampler:
    kwargs = {
        "env": _FakeEnv(),
        "checkpoint_path": checkpoint,
        "latent_dim": Z_DIM + 2,
        "latent_steps_min": 1,
        "latent_steps_max": 1,
        "discover_env_method": _discover,
        "horizon_steps": HORIZON,
        "command_phase_mode": "sin_cos",
        "code_latent_dim": Z_DIM,
        "phase_period": 10,
    }
    kwargs.update(overrides)
    return FrozenHighLevelSkillCommandSampler(**kwargs)  # type: ignore[arg-type]


def _phase_of(latents: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(latents[:, -2], latents[:, -1])
    return torch.remainder(angle / (2.0 * math.pi), 1.0)


def _step_td(done: torch.Tensor | None = None) -> TensorDict:
    td = TensorDict({}, batch_size=[4])
    if done is not None:
        td.set("done", done)
    return td


def test_phase_source_rejects_unknown_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase_source"):
        _sampler(_pretrained_checkpoint(tmp_path), phase_source="motion")


def test_episode_phase_requires_hold_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="latent_steps_max"):
        _sampler(
            _pretrained_checkpoint(tmp_path),
            phase_source="episode",
            latent_steps_min=10,
            latent_steps_max=10,
        )


def test_hold_phase_is_constant_at_hold_one(tmp_path: Path) -> None:
    # The informationally-dead case: with the hold clock and hold 1 the phase
    # pins to (period - 1) / period on every step.
    sampler = _sampler(_pretrained_checkpoint(tmp_path))
    device = torch.device("cpu")
    phases = [
        _phase_of(
            sampler.sample_for_step(_step_td(), device=device, dtype=torch.float32)
        )
        for _ in range(3)
    ]
    for phase in phases:
        assert torch.allclose(phase, torch.full((4,), 0.9), atol=1.0e-6)


def test_episode_phase_advances_and_resets_on_done(tmp_path: Path) -> None:
    sampler = _sampler(_pretrained_checkpoint(tmp_path), phase_source="episode")
    device = torch.device("cpu")

    first = _phase_of(
        sampler.sample_for_step(_step_td(), device=device, dtype=torch.float32)
    )
    second = _phase_of(
        sampler.sample_for_step(_step_td(), device=device, dtype=torch.float32)
    )
    assert torch.allclose(first, torch.zeros(4), atol=1.0e-6)
    assert torch.allclose(second, torch.full((4,), 0.1), atol=1.0e-6)

    done = torch.tensor([True, False, False, False])
    third = _phase_of(
        sampler.sample_for_step(_step_td(done), device=device, dtype=torch.float32)
    )
    assert torch.allclose(third[0], torch.tensor(0.0), atol=1.0e-6)
    assert torch.allclose(third[1:], torch.full((3,), 0.2), atol=1.0e-6)


def test_episode_phase_wraps_at_the_period(tmp_path: Path) -> None:
    sampler = _sampler(
        _pretrained_checkpoint(tmp_path),
        phase_source="episode",
        phase_period=3,
    )
    device = torch.device("cpu")
    phases = [
        _phase_of(
            sampler.sample_for_step(_step_td(), device=device, dtype=torch.float32)
        )[0].item()
        for _ in range(4)
    ]
    assert phases == pytest.approx([0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0], abs=1.0e-6)


# ---------------------------------------------------------------------------
# jepa triplet structure and single-encoder (online) target mode
# ---------------------------------------------------------------------------


def _jepa_config(**overrides: object) -> HighLevelSkillDiffSRConfig:
    base = {
        "transition_objective": "jepa_ntp",
        "jepa_loss": "sigreg",
        "device": "cpu",
    }
    base.update(overrides)
    return _config(**base)


def test_jepa_config_rejects_bad_context_and_mode() -> None:
    with pytest.raises(ValueError, match="jepa_context_chunks"):
        _jepa_config(jepa_context_chunks=2).validate()
    # "stopgrad" became a valid mode on 2026-08-23 (SimSiam asymmetry cell).
    with pytest.raises(ValueError, match="jepa_target_encoder_mode"):
        _jepa_config(jepa_target_encoder_mode="byol").validate()


def test_pair_default_is_unchanged_and_logs_the_copy_gate() -> None:
    torch.manual_seed(0)
    env = _FakeEnv()
    trainer = HighLevelSkillDiffSRTrainer(_jepa_config(), env)
    assert trainer.jepa_target_encoder is not None
    metrics = trainer.train_step()
    # The constructor preflight-samples one plain-horizon batch first.
    assert env.expert_horizons[-1] == 2 * HORIZON
    assert math.isfinite(metrics["train/jepa_copy_mse"])
    assert math.isfinite(metrics["train/jepa_pred_over_copy"])


def test_triplet_widens_the_predictor_and_samples_three_chunks() -> None:
    torch.manual_seed(0)
    env = _FakeEnv()
    trainer = HighLevelSkillDiffSRTrainer(_jepa_config(jepa_context_chunks=1), env)
    first_linear = trainer.jepa_predictor[0]
    assert first_linear.in_features == 2 * Z_DIM
    metrics = trainer.train_step()
    assert env.expert_horizons[-1] == 3 * HORIZON
    assert math.isfinite(metrics["train/loss"])
    assert math.isfinite(metrics["train/jepa_pred_over_copy"])


def test_online_mode_has_no_target_encoder_and_still_trains() -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(
        _jepa_config(jepa_target_encoder_mode="online"), _FakeEnv()
    )
    assert trainer.jepa_target_encoder is None
    metrics = trainer.train_step()
    assert math.isfinite(metrics["train/loss"])


def test_online_checkpoint_refuses_online_finetuning(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(
        _jepa_config(jepa_target_encoder_mode="online"), _FakeEnv()
    )
    trainer.train_step()
    path = tmp_path / "online_jepa.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "target_encoder" not in payload["jepa_state_dict"]
    with pytest.raises(ValueError, match="chunk-pair EMA"):
        FrozenHighLevelSkillCommandSampler(
            env=_FakeEnv(),
            checkpoint_path=path,
            latent_dim=Z_DIM,
            latent_steps_min=1,
            latent_steps_max=1,
            discover_env_method=_discover,
            horizon_steps=HORIZON,
            finetune_enabled=True,
            offline_batch_size=8,
        )


# ---------------------------------------------------------------------------
# jepa_ntp_coeff (the chunk-wise non-JEPA cell: DiffSR + SIGReg, no NTP)
# ---------------------------------------------------------------------------


def test_jepa_ntp_coeff_zero_is_valid_and_default_is_one() -> None:
    cfg = _config(transition_objective="jepa_ntp")
    cfg.validate()
    assert cfg.jepa_ntp_coeff == 1.0
    cfg = _config(transition_objective="jepa_ntp", jepa_ntp_coeff=0.0)
    cfg.validate()
    assert cfg.jepa_ntp_coeff == 0.0


def test_jepa_ntp_coeff_rejects_negative() -> None:
    import pytest

    cfg = _config(transition_objective="jepa_ntp", jepa_ntp_coeff=-0.5)
    with pytest.raises(ValueError, match="jepa_ntp_coeff"):
        cfg.validate()


# ---------------------------------------------------------------------------
# stopgrad target mode + full_body (67-wide) re-anchoring
# ---------------------------------------------------------------------------


def test_target_encoder_mode_accepts_stopgrad_and_rejects_unknown() -> None:
    import pytest

    cfg = _config(transition_objective="jepa_ntp", jepa_target_encoder_mode="stopgrad")
    cfg.validate()
    assert cfg.jepa_target_encoder_mode == "stopgrad"
    bad = _config(transition_objective="jepa_ntp", jepa_target_encoder_mode="byol")
    with pytest.raises(ValueError, match="jepa_target_encoder_mode"):
        bad.validate()


def test_reanchor_accepts_fullbody_and_keeps_qvel_invariant() -> None:
    import torch

    from rlopt.agent.hl_skill_diffsr import _reanchor_heading_frames

    torch.manual_seed(0)
    frames = torch.randn(4, 6, 67)
    # Valid rot6d blocks so the transform is well-defined.
    frames[..., 61:67] = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
    anchor = frames[:, 0].clone()
    out = _reanchor_heading_frames(frames, anchor)
    assert out.shape == frames.shape
    # Joint qpos AND qvel (the 58-wide prefix) are frame-invariant.
    assert torch.allclose(out[..., :58], frames[..., :58])
    # Anchoring on itself puts the anchor frame at the origin (xy) with
    # identity-ish heading.
    assert torch.allclose(out[:, 0, 58:60], torch.zeros(4, 2), atol=1e-5)


def test_reanchor_rejects_unknown_width() -> None:
    import pytest
    import torch

    from rlopt.agent.hl_skill_diffsr import _reanchor_heading_frames

    frames = torch.randn(2, 3, 50)
    with pytest.raises(ValueError, match="38-wide root_qpos or 67-wide"):
        _reanchor_heading_frames(frames, frames[:, 0])


# ---------------------------------------------------------------------------
# generative NTP heads (diff_token / diff_chunk, 2026-08-26)
# ---------------------------------------------------------------------------


def _diff_head_config(head: str, **overrides: object) -> HighLevelSkillDiffSRConfig:
    return _jepa_config(jepa_loss="sigreg_ebm", jepa_ntp_head=head, **overrides)


def test_ntp_head_validation_guards() -> None:
    cfg = _diff_head_config("diff_token")
    cfg.validate()
    assert cfg.jepa_ntp_head == "diff_token"
    with pytest.raises(ValueError, match="jepa_ntp_head"):
        _jepa_config(jepa_ntp_head="transformer").validate()
    # diffusion heads require the grounded loss and the chunk pair
    with pytest.raises(ValueError, match="sigreg_ebm"):
        _jepa_config(jepa_loss="sigreg", jepa_ntp_head="diff_token").validate()
    with pytest.raises(ValueError, match="chunk pair"):
        _diff_head_config("diff_chunk", jepa_context_chunks=1).validate()


def test_diff_token_head_trains_and_targets_the_token_width() -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_diff_head_config("diff_token"), _FakeEnv())
    assert trainer.jepa_ntp_diffsr is not None
    assert trainer.jepa_ntp_diffsr.next_obs_dim == Z_DIM
    metrics = trainer.train_step()
    assert math.isfinite(metrics["train/loss"])


def test_diff_chunk_head_trains_and_targets_the_chunk_width() -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_diff_head_config("diff_chunk"), _FakeEnv())
    assert trainer.jepa_ntp_diffsr is not None
    assert trainer.jepa_ntp_diffsr.next_obs_dim == HORIZON * STATE_DIM
    metrics = trainer.train_step()
    assert math.isfinite(metrics["train/loss"])


def test_diff_head_checkpoint_carries_the_head_and_refuses_finetune(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_diff_head_config("diff_token"), _FakeEnv())
    trainer.train_step()
    path = tmp_path / "diff_token.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "ntp_diffsr" in payload["jepa_state_dict"]
    with pytest.raises(ValueError, match="chunk-pair EMA"):
        FrozenHighLevelSkillCommandSampler(
            env=_FakeEnv(),
            checkpoint_path=path,
            latent_dim=Z_DIM,
            latent_steps_min=1,
            latent_steps_max=1,
            discover_env_method=_discover,
            horizon_steps=HORIZON,
            finetune_enabled=True,
            offline_batch_size=8,
        )


def test_mlp_default_builds_no_extra_head() -> None:
    trainer = HighLevelSkillDiffSRTrainer(
        _jepa_config(jepa_loss="sigreg_ebm"), _FakeEnv()
    )
    assert trainer.jepa_ntp_diffsr is None


def test_diff_pair_head_targets_state_plus_token_width() -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_diff_head_config("diff_pair"), _FakeEnv())
    assert trainer.jepa_ntp_diffsr is not None
    assert trainer.jepa_ntp_diffsr.next_obs_dim == STATE_DIM + Z_DIM
    metrics = trainer.train_step()
    assert math.isfinite(metrics["train/loss"])


def test_diff_chunk_next_anchor_trains_and_validates() -> None:
    with pytest.raises(ValueError, match="jepa_ntp_chunk_anchor"):
        _diff_head_config("diff_chunk", jepa_ntp_chunk_anchor="midpoint").validate()
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(
        _diff_head_config("diff_chunk", jepa_ntp_chunk_anchor="next"), _FakeEnv()
    )
    metrics = trainer.train_step()
    assert math.isfinite(metrics["train/loss"])
