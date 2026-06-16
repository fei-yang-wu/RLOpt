from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from tensordict import TensorDict

warnings.filterwarnings(
    "ignore",
    message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
    category=DeprecationWarning,
    append=False,
)

from rlopt.agent.hl_skill_diffsr import (  # noqa: E402
    HighLevelSkillDiffSRConfig,
    HighLevelSkillDiffSRTrainer,
)
from rlopt.agent.skill_commander import (  # noqa: E402
    FrozenSkillCommanderSampler,
    SkillCommander,
    SkillCommanderConfig,
    SkillCommanderTrainer,
    build_rank_embedding_lookup,
    load_language_embedding_table,
)


class _FakeMacroEnv:
    """Minimal env exposing the macro-transition surface with trajectory ranks."""

    def __init__(
        self,
        *,
        state_dim: int = 7,
        horizon_steps: int = 3,
        num_trajectories: int = 4,
        deterministic: bool = True,
    ) -> None:
        self.device = torch.device("cpu")
        self.state_dim = int(state_dim)
        self.horizon_steps = int(horizon_steps)
        self.num_trajectories = int(num_trajectories)
        self.deterministic = bool(deterministic)
        self.motion_names = [f"motion_{i}" for i in range(self.num_trajectories)]
        self.calls = 0

    def sample_expert_macro_transition_batch(
        self,
        batch_size: int,
        horizon_steps: int,
        split: str | None = None,
        eval_fraction: float = 0.1,
        split_seed: int = 0,
    ) -> TensorDict:
        assert int(horizon_steps) == self.horizon_steps
        assert split in (None, "all", "train", "eval")
        assert 0.0 < float(eval_fraction) < 1.0
        assert isinstance(split_seed, int)
        self.calls += 1
        # Deterministic mode seeds by batch_size so a fixed batch_size always
        # yields the same batch (enabling overfit tests); otherwise vary by call.
        seed = (1000 + batch_size) if self.deterministic else self.calls
        gen = torch.Generator().manual_seed(int(seed))
        state = torch.randn(batch_size, self.state_dim, generator=gen)
        future_window = torch.randn(
            batch_size, horizon_steps, self.state_dim, generator=gen
        )
        target = future_window[:, -1, :].clone()
        traj_rank = torch.randint(
            0, self.num_trajectories, (batch_size,), generator=gen
        )
        hl = TensorDict(
            {
                "state": state,
                "future_window": future_window,
                "target": target,
                "traj_rank": traj_rank,
            },
            batch_size=[batch_size],
        )
        return TensorDict({"hl": hl}, batch_size=[batch_size])

    def current_expert_macro_transition_batch(
        self,
        *,
        horizon_steps: int,
        env_ids: torch.Tensor | None = None,
    ) -> TensorDict:
        batch_size = 2 if env_ids is None else int(env_ids.numel())
        return self.sample_expert_macro_transition_batch(
            batch_size=batch_size,
            horizon_steps=horizon_steps,
        )

    def current_achieved_macro_transition_batch(
        self,
        *,
        horizon_steps: int,
        env_ids: torch.Tensor | None = None,
    ) -> TensorDict:
        return self.current_expert_macro_transition_batch(
            horizon_steps=horizon_steps, env_ids=env_ids
        )

    def expert_trajectory_motion_names(self) -> list[str]:
        return list(self.motion_names)


def _discover_direct_method(env: object, method_name: str):
    method = getattr(env, method_name, None)
    return method if callable(method) else None


def _make_skill_checkpoint(
    tmp_path: Path,
    env: _FakeMacroEnv,
    *,
    z_dim: int = 5,
) -> Path:
    skill_config = HighLevelSkillDiffSRConfig(
        horizon_steps=env.horizon_steps,
        z_dim=z_dim,
        diffsr_feature_dim=4,
        diffsr_embed_dim=8,
        batch_size=6,
        num_updates=1,
        log_interval=1,
        eval_batches=1,
        eval_batch_size=4,
        preflight_batch_size=4,
        encoder_hidden_dims=(16, 12),
        diffsr_f_hidden_dims=(16,),
        diffsr_g_hidden_dims=(16,),
        diffsr_mu_hidden_dims=(16,),
        diffsr_num_noises=2,
        device="cpu",
    )
    trainer = HighLevelSkillDiffSRTrainer(skill_config, env)
    path = tmp_path / "skill.pt"
    trainer.save_checkpoint(path)
    return path


def _make_language_table(
    tmp_path: Path,
    names: list[str],
    *,
    embed_dim: int = 5,
    seed: int = 0,
) -> Path:
    gen = torch.Generator().manual_seed(int(seed))
    embeddings = torch.randn(len(names), embed_dim, generator=gen)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    table = {
        "names": list(names),
        "phrases": list(names),
        "name_to_index": {name: index for index, name in enumerate(names)},
        "embeddings": embeddings,
        "embed_dim": int(embed_dim),
        "backend": "dummy",
        "model": None,
        "raw_names": True,
        "manifest": "test",
    }
    path = tmp_path / "lang.pt"
    torch.save(table, path)
    return path


def _gen_config(
    skill_ckpt: Path,
    lang_table: Path,
    **overrides: Any,
) -> SkillCommanderConfig:
    values: dict[str, Any] = {
        "skill_checkpoint_path": str(skill_ckpt),
        "language_embeddings_path": str(lang_table),
        "generator_hidden_dims": (32, 32),
        "batch_size": 8,
        "num_updates": 2,
        "log_interval": 1,
        "eval_batches": 1,
        "eval_batch_size": 8,
        "preflight_batch_size": 4,
        "train_split": "all",
        "eval_split": "all",
        "eval_trajectory_fraction": 0.25,
        "device": "cpu",
    }
    values.update(overrides)
    return SkillCommanderConfig(**values)


def test_generator_forward_shapes() -> None:
    generator = SkillCommander(
        state_dim=7, lang_embed_dim=5, z_dim=3, hidden_dims=(16,)
    )
    state = torch.randn(4, 7)
    lang = torch.randn(4, 5)
    z = generator(state, lang)
    assert z.shape == (4, 3)
    with pytest.raises(ValueError, match="state width mismatch"):
        generator(torch.randn(4, 6), lang)
    with pytest.raises(ValueError, match="lang_emb shape mismatch"):
        generator(state, torch.randn(4, 6))


def test_language_table_lookup_and_missing_name(tmp_path) -> None:
    path = _make_language_table(tmp_path, ["walk", "dance"], embed_dim=5)
    table = load_language_embedding_table(path)
    lookup = build_rank_embedding_lookup(table, ["dance", "walk", "dance"], "cpu")
    assert lookup.shape == (3, 5)
    # Repeated names resolve to the same embedding row.
    assert torch.equal(lookup[0], lookup[2])
    with pytest.raises(ValueError, match="no entry"):
        build_rank_embedding_lookup(table, ["walk", "unknown_motion"], "cpu")


def test_trainer_distillation_loss_decreases(tmp_path) -> None:
    torch.manual_seed(0)
    env = _FakeMacroEnv(deterministic=True)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    trainer = SkillCommanderTrainer(_gen_config(skill_ckpt, lang_table), env)

    first = trainer.train_step()["train/z_mse"]
    last = first
    for _ in range(200):
        last = trainer.train_step()["train/z_mse"]
    assert last < first


def test_trainer_evaluate_returns_distill_metrics(tmp_path) -> None:
    torch.manual_seed(1)
    env = _FakeMacroEnv(deterministic=True)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    trainer = SkillCommanderTrainer(_gen_config(skill_ckpt, lang_table), env)
    metrics = trainer.evaluate(prefix="eval")
    assert "eval/z_mse" in metrics
    assert "eval/z_cosine" in metrics
    assert "eval/z_cosine_shuffled_lang" in metrics


def test_trainer_requires_traj_rank_from_env(tmp_path) -> None:
    torch.manual_seed(2)
    env = _FakeMacroEnv(deterministic=True)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    trainer = SkillCommanderTrainer(_gen_config(skill_ckpt, lang_table), env)

    class _NoRankEnv(_FakeMacroEnv):
        def sample_expert_macro_transition_batch(self, *args, **kwargs):  # type: ignore[override]
            batch = super().sample_expert_macro_transition_batch(*args, **kwargs)
            del batch["hl", "traj_rank"]
            return batch

    trainer.env = _NoRankEnv(deterministic=True)
    with pytest.raises(ValueError, match="traj_rank"):
        trainer.train_step()


def test_checkpoint_roundtrips_generator(tmp_path) -> None:
    torch.manual_seed(3)
    env = _FakeMacroEnv(deterministic=True)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    trainer = SkillCommanderTrainer(_gen_config(skill_ckpt, lang_table), env)
    trainer.train_step()
    ckpt_path = tmp_path / "generator.pt"
    trainer.save_checkpoint(ckpt_path)
    saved = torch.load(ckpt_path, weights_only=False)
    for key in (
        "generator_state_dict",
        "skill_config",
        "skill_checkpoint_path",
        "state_dim",
        "lang_embed_dim",
        "z_dim",
    ):
        assert key in saved


@pytest.mark.parametrize(
    ("command_mode", "latent_dim"),
    [("z", 5), ("z_phi", 9)],
)
def test_frozen_language_sampler_returns_latents(
    tmp_path,
    command_mode: str,
    latent_dim: int,
) -> None:
    torch.manual_seed(4)
    env = _FakeMacroEnv(deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    trainer = SkillCommanderTrainer(_gen_config(skill_ckpt, lang_table), env)
    gen_ckpt = tmp_path / "generator.pt"
    trainer.save_checkpoint(gen_ckpt)

    sampler = FrozenSkillCommanderSampler(
        env=env,
        checkpoint_path=gen_ckpt,
        language_embeddings_path=lang_table,
        latent_dim=latent_dim,
        latent_steps_min=2,
        latent_steps_max=2,
        discover_env_method=_discover_direct_method,
        command_mode=command_mode,
        device="cpu",
    )
    td = TensorDict({}, batch_size=[2])
    first = sampler.sample_for_step(
        td, device=torch.device("cpu"), dtype=torch.float32
    ).clone()
    second = sampler.sample_for_step(
        td, device=torch.device("cpu"), dtype=torch.float32
    ).clone()

    assert first.shape == (2, latent_dim)
    # Command is held for latent_steps (=2) before being renewed.
    assert torch.equal(first, second)
    assert not any(param.requires_grad for param in sampler.generator.parameters())
    assert sampler.trainable_parameters() == []


def test_frozen_language_sampler_rejects_wrong_latent_dim(tmp_path) -> None:
    torch.manual_seed(5)
    env = _FakeMacroEnv(deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    trainer = SkillCommanderTrainer(_gen_config(skill_ckpt, lang_table), env)
    gen_ckpt = tmp_path / "generator.pt"
    trainer.save_checkpoint(gen_ckpt)

    with pytest.raises(ValueError, match="command width must match"):
        FrozenSkillCommanderSampler(
            env=env,
            checkpoint_path=gen_ckpt,
            language_embeddings_path=lang_table,
            latent_dim=99,
            latent_steps_min=1,
            latent_steps_max=1,
            discover_env_method=_discover_direct_method,
            command_mode="z",
            device="cpu",
        )


def test_cotrain_commander_in_skill_trainer(tmp_path) -> None:
    torch.manual_seed(6)
    env = _FakeMacroEnv(deterministic=True)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    config = HighLevelSkillDiffSRConfig(
        horizon_steps=env.horizon_steps,
        z_dim=5,
        diffsr_feature_dim=4,
        diffsr_embed_dim=8,
        batch_size=8,
        num_updates=3,
        log_interval=1,
        eval_batches=1,
        eval_batch_size=8,
        preflight_batch_size=4,
        encoder_hidden_dims=(16, 12),
        diffsr_f_hidden_dims=(16,),
        diffsr_g_hidden_dims=(16,),
        diffsr_mu_hidden_dims=(16,),
        diffsr_num_noises=2,
        device="cpu",
        cotrain_commander=True,
        commander_language_embeddings_path=str(lang_table),
        commander_hidden_dims=(16,),
    )
    trainer = HighLevelSkillDiffSRTrainer(config, env)
    assert trainer.commander is not None
    metrics = trainer.train_step()
    assert "train/commander_cosine" in metrics
    assert "train/commander_mse" in metrics
    eval_metrics = trainer.evaluate(prefix="eval")
    assert "eval/commander_cosine" in eval_metrics
    # The co-trained commander saves in the SkillCommander rollout-sampler format.
    ckpt = tmp_path / "commander.pt"
    trainer.save_commander_checkpoint(ckpt, skill_checkpoint_path="skill.pt")
    saved = torch.load(ckpt, weights_only=False)
    for key in (
        "generator_state_dict",
        "skill_config",
        "config",
        "state_dim",
        "lang_embed_dim",
        "z_dim",
    ):
        assert key in saved
    assert "generator_hidden_dims" in saved["config"]


def test_frozen_commander_uses_achieved_state(tmp_path) -> None:
    torch.manual_seed(7)
    env = _FakeMacroEnv(deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    trainer = SkillCommanderTrainer(_gen_config(skill_ckpt, lang_table), env)
    gen_ckpt = tmp_path / "generator.pt"
    trainer.save_checkpoint(gen_ckpt)

    sampler = FrozenSkillCommanderSampler(
        env=env,
        checkpoint_path=gen_ckpt,
        language_embeddings_path=lang_table,
        latent_dim=5,
        latent_steps_min=2,
        latent_steps_max=2,
        discover_env_method=_discover_direct_method,
        command_mode="z",
        use_achieved_state=True,
        device="cpu",
    )
    assert sampler.use_achieved_state
    td = TensorDict({}, batch_size=[2])
    out = sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
    assert out.shape == (2, 5)
