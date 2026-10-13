from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

warnings.filterwarnings(
    "ignore",
    message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
    category=DeprecationWarning,
    append=False,
)

from test_skill_commander import _FakeMacroEnv, _make_language_table  # noqa: E402

from rlopt.agent.hl_skill_diffsr import (  # noqa: E402
    HighLevelSkillDiffSRConfig,
    HighLevelSkillDiffSRTrainer,
)
from rlopt.agent.language_tilt_planner import (  # noqa: E402
    LanguageExpert,
    LanguageTiltPlanner,
    LanguageTiltPlannerConfig,
    load_language_tilt_planner,
)


def _tiny_config(**overrides) -> LanguageTiltPlannerConfig:
    base = {
        "flow_hidden_dims": (32, 32),
        "flow_time_embed_dim": 8,
        "flow_num_inference_steps": 8,
        "expert_hidden_dims": (16,),
        "lang_proj_dim": 4,
        "eval_betas": (0.0, 1.0),
        "eval_num_samples": 3,
    }
    base.update(overrides)
    return LanguageTiltPlannerConfig(**base)


# ---------------------------------------------------------------------------
# Guidance schedule: exact for a Gaussian base
# ---------------------------------------------------------------------------
def test_guidance_matches_analytic_gaussian_tilt() -> None:
    """For endpoint N(0, I) and source N(0, I) the tilt by exp(beta b^T z) is
    N(beta b, I). The analytic base velocity is v(z_u, u) = (2u - 1) z_u /
    ((1-u)^2 + u^2); adding beta * w(u) * b must reproduce that Gaussian."""
    torch.manual_seed(0)
    z_dim, batch, steps = 3, 200_000, 400
    beta = 1.7
    b = torch.tensor([0.8, -0.5, 0.3])

    def base_velocity(z_u: torch.Tensor, u: float) -> torch.Tensor:
        return (2.0 * u - 1.0) * z_u / ((1.0 - u) ** 2 + u**2)

    z = torch.randn(batch, z_dim)
    dt = 1.0 / steps
    for step in range(steps):
        u = step * dt
        u_t = torch.full((batch,), u)
        weight = LanguageTiltPlanner.guidance_weight(u_t, 1.0)[:, None]
        z = z + dt * (base_velocity(z, u) + beta * weight * b)
    mean = z.mean(dim=0)
    cov = torch.cov(z.T)
    assert torch.allclose(mean, beta * b, atol=0.02), mean
    assert torch.allclose(cov, torch.eye(z_dim), atol=0.03), cov


def test_guidance_weight_bounds() -> None:
    u = torch.linspace(0.0, 1.0, 11)
    w = LanguageTiltPlanner.guidance_weight(u, 1.0)
    assert torch.isclose(w[0], torch.tensor(1.0))
    assert torch.isclose(w[-1], torch.tensor(0.0))
    assert bool((w >= 0.0).all())
    # Bounded: the peak of (1-u)/((1-u)^2+u^2) on [0,1] is (1+sqrt(2))/2.
    assert bool((w <= 0.5 * (1.0 + 2.0**0.5) + 1e-6).all())
    w2 = LanguageTiltPlanner.guidance_weight(u, 2.0)
    assert torch.isclose(w2[0], torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Language expert
# ---------------------------------------------------------------------------
def test_language_expert_tilt_is_linear_in_language_and_composes() -> None:
    torch.manual_seed(1)
    expert = LanguageExpert(
        state_dim=6, lang_embed_dim=5, z_dim=4, proj_dim=3, hidden_dims=(16,)
    )
    state = torch.randn(7, 6)
    l1 = torch.randn(7, 5)
    l2 = torch.randn(7, 5)
    tilt_sum = expert.tilt(state, l1 + l2)
    assert torch.allclose(tilt_sum, expert.tilt(state, l1) + expert.tilt(state, l2), atol=1e-6)
    z = torch.randn(7, 4)
    # logits are exactly z^T b(X, l_j) for every instruction j
    logits = expert.logits(state, z, expert.project_language(l1))
    manual = torch.stack(
        [(z * expert.tilt(state, l1[j].expand(7, -1))).sum(-1) for j in range(7)], dim=1
    )
    assert torch.allclose(logits, manual, atol=1e-5)


def test_language_expert_classification_dedupes_and_learns() -> None:
    torch.manual_seed(2)
    expert = LanguageExpert(
        state_dim=4, lang_embed_dim=6, z_dim=8, proj_dim=4, hidden_dims=(32,)
    )
    # Three instructions, each with its own code direction; states are noise.
    instr = F.normalize(torch.randn(3, 6), dim=-1)
    dirs = torch.randn(3, 8) * 3.0
    ids = torch.randint(0, 3, (96,))
    lang = instr[ids]
    z = dirs[ids] + 0.1 * torch.randn(96, 8)
    state = torch.randn(96, 4)
    _loss, info = expert.classification_loss(state, z, lang)
    assert info["lang_unique"] == 3.0
    assert info["lang_chance"] == pytest.approx(1.0 / 3.0)
    optim = torch.optim.Adam(expert.parameters(), lr=1e-2)
    for _ in range(200):
        loss, info = expert.classification_loss(state, z, lang)
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert info["lang_acc"] > 0.95, info


# ---------------------------------------------------------------------------
# Planner: shapes, guidance direction, checkpoint round trip
# ---------------------------------------------------------------------------
def test_planner_sample_shapes_and_beta_zero_is_base() -> None:
    torch.manual_seed(3)
    planner = LanguageTiltPlanner(
        state_dim=5, lang_embed_dim=4, z_dim=6, config=_tiny_config()
    )
    state = torch.randn(9, 5)
    lang = torch.randn(9, 4)
    noise = torch.randn(9, 6)
    z0 = planner.sample(state, lang, beta=0.0, noise=noise)
    assert z0.shape == (9, 6)
    # beta=0 is the untilted base flow, whatever the language says.
    z0_other = planner.sample(state, torch.randn(9, 4), beta=0.0, noise=noise)
    assert torch.allclose(z0, z0_other)
    loss, info = planner.losses(state, torch.randn(9, 6), lang)
    assert loss.ndim == 0
    for key in ("flow_loss", "lang_loss", "lang_acc", "lang_unique"):
        assert key in info


def test_guidance_is_centered_and_bounded() -> None:
    """With a trained expert, guidance raises the target's classifier
    probability, and because the drift is centered it cannot run away: the
    sample norm stays comparable to the base flow's even at large beta."""
    torch.manual_seed(7)
    planner = LanguageTiltPlanner(
        state_dim=4, lang_embed_dim=6, z_dim=8, config=_tiny_config(flow_num_inference_steps=16)
    )
    instr = F.normalize(torch.randn(3, 6), dim=-1)
    dirs = torch.randn(3, 8) * 2.0
    ids = torch.randint(0, 3, (128,))
    lang, z, state = instr[ids], dirs[ids] + 0.3 * torch.randn(128, 8), torch.randn(128, 4)
    optim = torch.optim.Adam(planner.parameters(), lr=3e-3)
    for _ in range(300):
        loss, _info = planner.losses(state, z, lang)
        optim.zero_grad()
        loss.backward()
        optim.step()
    noise = torch.randn(128, 8)
    with torch.no_grad():
        cand_proj = planner.expert.project_language(instr)
        phi = planner.expert.state_matrix(state)
        probs = {}
        norms = {}
        for beta in (0.0, 1.0, 4.0, 16.0):
            samples = planner.sample(state, lang, beta=beta, noise=noise, candidates=instr)
            logits = planner.expert.logits(state, samples, cand_proj)
            probs[beta] = float(torch.softmax(logits, -1)[torch.arange(128), ids].mean())
            norms[beta] = float(samples.norm(dim=-1).mean())
            # The centered drift vanishes where the target already dominates.
            drift = planner.expert.classifier_drift(
                phi, samples, planner.expert.project_language(lang), cand_proj
            )
            assert torch.isfinite(drift).all()
    assert probs[1.0] >= probs[0.0] - 1e-3
    assert probs[4.0] >= probs[1.0] - 1e-3
    assert norms[16.0] < 2.0 * norms[0.0], norms


def test_composition_drift_uses_summed_projection() -> None:
    torch.manual_seed(8)
    expert = LanguageExpert(state_dim=3, lang_embed_dim=4, z_dim=5, proj_dim=2, hidden_dims=(8,))
    state, z = torch.randn(6, 3), torch.randn(6, 5)
    cands = torch.randn(4, 4)
    phi = expert.state_matrix(state)
    cand_proj = expert.project_language(cands)
    p1 = expert.project_language(cands[0].expand(6, -1))
    p2 = expert.project_language(cands[1].expand(6, -1))
    d1 = expert.classifier_drift(phi, z, p1, cand_proj)
    d2 = expert.classifier_drift(phi, z, p2, cand_proj)
    d12 = expert.classifier_drift(phi, z, p1 + p2, cand_proj, normalizer_count=2.0)
    assert torch.allclose(d12, d1 + d2, atol=1e-5)


def test_planner_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(4)
    planner = LanguageTiltPlanner(
        state_dim=5, lang_embed_dim=4, z_dim=6, config=_tiny_config()
    )
    path = tmp_path / "planner.pt"
    torch.save(planner.checkpoint_state_dict(), path)
    restored = load_language_tilt_planner(path)
    state = torch.randn(3, 5)
    lang = torch.randn(3, 4)
    noise = torch.randn(3, 6)
    assert torch.allclose(
        planner.sample(state, lang, beta=1.0, noise=noise),
        restored.sample(state, lang, beta=1.0, noise=noise),
    )
    assert restored.config.to_dict() == planner.config.to_dict()
    bad = planner.checkpoint_state_dict()
    bad["planner_type"] = "other"
    with pytest.raises(ValueError, match="language_tilt_flow"):
        load_language_tilt_planner(bad)


def test_config_round_trip_and_validation() -> None:
    config = _tiny_config(eval_betas=(0.5, 2.0))
    again = LanguageTiltPlannerConfig.from_dict(config.to_dict())
    assert again.to_dict() == config.to_dict()
    with pytest.raises(ValueError, match="eval_betas"):
        LanguageTiltPlannerConfig(eval_betas=()).validate()
    with pytest.raises(ValueError, match="flow_noise_std"):
        LanguageTiltPlannerConfig(flow_noise_std=0.0).validate()


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------
def _trainer_config(env: _FakeMacroEnv, lang_table: Path, **overrides) -> HighLevelSkillDiffSRConfig:
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
        "cotrain_language_planner": True,
        "planner_language_embeddings_path": str(lang_table),
        "planner_flow_hidden_dims": (16,),
        "planner_flow_time_embed_dim": 4,
        "planner_flow_num_inference_steps": 4,
        "planner_expert_hidden_dims": (8,),
        "planner_lang_proj_dim": 3,
        "planner_eval_betas": (0.0, 1.0),
        "planner_eval_num_samples": 2,
    }
    base.update(overrides)
    return HighLevelSkillDiffSRConfig(**base)


def _assert_planner_metrics(metrics: dict[str, float], prefix: str) -> None:
    for key in (
        f"{prefix}/planner_flow_loss_eval",
        f"{prefix}/planner_lang_acc_eval",
        f"{prefix}/planner_lang_acc_shuffled_eval",
        f"{prefix}/planner_pred_loss_encoder_z",
        f"{prefix}/planner_pred_loss_shuffled_z",
        f"{prefix}/planner_beta0/cos_mean",
        f"{prefix}/planner_beta1/cos_best_of_k",
        f"{prefix}/planner_beta1/spread",
        f"{prefix}/planner_beta1/lang_acc",
        f"{prefix}/planner_beta1/pred_loss",
    ):
        assert key in metrics, key


def test_cotrain_language_planner_diffsr_path(tmp_path: Path) -> None:
    torch.manual_seed(5)
    env = _FakeMacroEnv(deterministic=True)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    trainer = HighLevelSkillDiffSRTrainer(_trainer_config(env, lang_table), env)
    assert trainer.planner is not None
    assert trainer.planner_state_dim == env.state_dim
    metrics = trainer.train_step()
    for key in (
        "train/planner_loss",
        "train/planner_flow_loss",
        "train/planner_lang_loss",
        "train/planner_lang_acc",
    ):
        assert key in metrics, key
    eval_metrics = trainer.evaluate(prefix="eval")
    _assert_planner_metrics(eval_metrics, "eval")
    # Standalone planner checkpoint round-trips into a usable planner.
    ckpt = tmp_path / "planner.pt"
    trainer.save_planner_checkpoint(ckpt, skill_checkpoint_path="skill.pt")
    saved = torch.load(ckpt, weights_only=False)
    for key in ("planner_state_dict", "planner_config", "skill_config", "state_dim"):
        assert key in saved
    assert saved["source_history_steps"] == 0
    restored = load_language_tilt_planner(ckpt)
    assert restored.state_dim == env.state_dim
    # The resume checkpoint carries the planner and its optimizer.
    resume = tmp_path / "latest.pt"
    trainer.save_checkpoint(resume)
    state = torch.load(resume, weights_only=False)
    assert "language_planner" in state
    assert "language_planner_optimizer_state_dict" in state
    fresh = HighLevelSkillDiffSRTrainer(_trainer_config(env, lang_table), env)
    fresh.load_checkpoint(resume)
    a = dict(trainer.planner.state_dict())
    b = dict(fresh.planner.state_dict())  # type: ignore[union-attr]
    assert all(torch.equal(a[k], b[k]) for k in a)


class _AnyHorizonEnv(_FakeMacroEnv):
    """``_FakeMacroEnv`` that serves any window length (jepa draws 2H)."""

    def sample_expert_macro_transition_batch(  # type: ignore[override]
        self,
        batch_size: int,
        horizon_steps: int,
        split: str | None = None,
        eval_fraction: float = 0.1,
        split_seed: int = 0,
        state_history_steps: int = 0,
    ) -> TensorDict:
        del split, eval_fraction, split_seed
        self.calls += 1
        gen = torch.Generator().manual_seed(self.calls)
        history = int(state_history_steps)
        state_history = torch.randn(batch_size, history + 1, self.state_dim, generator=gen)
        state = state_history[:, -1, :].clone()
        future_window = torch.randn(batch_size, int(horizon_steps), self.state_dim, generator=gen)
        payload = {
            "state": state,
            "future_window": future_window,
            "target": future_window[:, -1, :].clone(),
            "traj_rank": torch.randint(0, self.num_trajectories, (batch_size,), generator=gen),
        }
        if history > 0:
            payload["state_history"] = state_history
        return TensorDict({"hl": TensorDict(payload, batch_size=[batch_size])}, batch_size=[batch_size])


def test_cotrain_language_planner_jepa_history_path(tmp_path: Path) -> None:
    """The production recipe: sigreg_ebm / diff_chunk with a past chunk."""
    torch.manual_seed(6)
    env = _AnyHorizonEnv(deterministic=False, horizon_steps=3, state_dim=38)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    config = _trainer_config(
        env,
        lang_table,
        transition_objective="jepa_ntp",
        jepa_loss="sigreg_ebm",
        jepa_ntp_head="diff_chunk",
        jepa_ntp_chunk_span="boundary_next",
        jepa_endpoint_coeff=0.0,
        encoder_window_mode="intermediate",
        source_history_steps=2,
        source_anchor="current",
        jepa_sigreg_sketches=4,
    )
    trainer = HighLevelSkillDiffSRTrainer(config, env)
    assert trainer.planner is not None
    assert trainer.planner_state_dim == 3 * env.state_dim
    metrics = trainer.train_step()
    assert "train/planner_flow_loss" in metrics
    eval_metrics = trainer.evaluate(prefix="eval")
    _assert_planner_metrics(eval_metrics, "eval")
    assert eval_metrics["eval/planner_pred_loss_encoder_z"] > 0.0


def test_planner_requires_language_table() -> None:
    with pytest.raises(ValueError, match="planner_language_embeddings_path"):
        HighLevelSkillDiffSRConfig(cotrain_language_planner=True).validate()


def test_planner_config_round_trip_through_skill_config(tmp_path: Path) -> None:
    env = _FakeMacroEnv(deterministic=True)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    config = _trainer_config(env, lang_table, planner_eval_betas=(0.0, 0.5, 3.0))
    again = HighLevelSkillDiffSRConfig.from_dict(config.to_dict())
    assert again.planner_eval_betas == (0.0, 0.5, 3.0)
    assert again.planner_flow_hidden_dims == (16,)
    assert again.language_planner_config().to_dict() == config.language_planner_config().to_dict()
