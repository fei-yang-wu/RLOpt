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

from rlopt.agent.causal_interface_planner import (  # noqa: E402
    CausalInterfaceTransformerFlowPlanner,
)
from rlopt.agent.hl_skill_diffsr import (  # noqa: E402
    HighLevelSkillDiffSRConfig,
    HighLevelSkillDiffSRTrainer,
)
from rlopt.agent.skill_commander import (  # noqa: E402
    DiffusionSkillCommander,
    DiTLatentSkillCommander,
    FlowMatchingSkillCommander,
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
        state_history_steps: int = 0,
    ) -> TensorDict:
        assert int(horizon_steps) == self.horizon_steps
        assert split in (None, "all", "train", "eval")
        assert 0.0 < float(eval_fraction) < 1.0
        assert isinstance(split_seed, int)
        assert int(state_history_steps) >= 0
        self.calls += 1
        # Deterministic mode seeds by batch_size so a fixed batch_size always
        # yields the same batch (enabling overfit tests); otherwise vary by call.
        seed = (1000 + batch_size) if self.deterministic else self.calls
        gen = torch.Generator().manual_seed(int(seed))
        if int(state_history_steps) > 0:
            state_history = torch.randn(
                batch_size,
                int(state_history_steps) + 1,
                self.state_dim,
                generator=gen,
            )
            state = state_history[:, -1, :].clone()
        else:
            state_history = None
            state = torch.randn(batch_size, self.state_dim, generator=gen)
        future_window = torch.randn(
            batch_size, horizon_steps, self.state_dim, generator=gen
        )
        target = future_window[:, -1, :].clone()
        traj_rank = torch.randint(
            0, self.num_trajectories, (batch_size,), generator=gen
        )
        hl_payload = {
            "state": state,
            "future_window": future_window,
            "target": target,
            "traj_rank": traj_rank,
        }
        if state_history is not None:
            hl_payload["state_history"] = state_history
        hl = TensorDict(
            hl_payload,
            batch_size=[batch_size],
        )
        return TensorDict({"hl": hl}, batch_size=[batch_size])

    def sample_causal_planner_training_batch(
        self,
        batch_size: int,
        horizon_steps: int,
        split: str | None = None,
        eval_fraction: float = 0.1,
        split_seed: int = 0,
        history_steps: int = 0,
    ) -> TensorDict:
        batch = self.sample_expert_macro_transition_batch(
            batch_size=batch_size,
            horizon_steps=horizon_steps,
            split=split,
            eval_fraction=eval_fraction,
            split_seed=split_seed,
            state_history_steps=history_steps,
        )
        state = batch.get(("hl", "state"))
        history = batch.get(("hl", "state_history"))
        if history is None:
            history = state.unsqueeze(1)
        planner = TensorDict(
            {"state": state, "state_history": history}, batch_size=[batch_size]
        )
        batch.set("planner", planner)
        return batch

    def causal_planner_observation_spec(self, history_steps: int = 0) -> dict[str, Any]:
        return {
            "name": "fake_causal_robot_history",
            "version": 1,
            "feature_names": ["robot_state"],
            "feature_widths": [self.state_dim],
            "frame_dim": self.state_dim,
            "history_steps": int(history_steps),
            "history_frames": int(history_steps) + 1,
            "flat_dim": (int(history_steps) + 1) * self.state_dim,
            "history_order": "oldest_to_newest",
            "reset_padding": "repeat_initial_observation",
            "reference_features": [],
        }

    def current_expert_macro_transition_batch(
        self,
        *,
        horizon_steps: int,
        env_ids: torch.Tensor | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        batch_size = 2 if env_ids is None else int(env_ids.numel())
        return self.sample_expert_macro_transition_batch(
            batch_size=batch_size,
            horizon_steps=horizon_steps,
            state_history_steps=state_history_steps,
        )

    def current_achieved_macro_transition_batch(
        self,
        *,
        horizon_steps: int,
        env_ids: torch.Tensor | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self.current_expert_macro_transition_batch(
            horizon_steps=horizon_steps,
            env_ids=env_ids,
            state_history_steps=state_history_steps,
        )

    def current_causal_planner_observation(
        self,
        *,
        env_ids: torch.Tensor | None = None,
        history_steps: int = 0,
    ) -> TensorDict:
        batch_size = 2 if env_ids is None else int(env_ids.numel())
        gen = torch.Generator().manual_seed(9000 + self.calls)
        history = torch.randn(
            batch_size,
            int(history_steps) + 1,
            self.state_dim,
            generator=gen,
        )
        planner = TensorDict(
            {"state": history[:, -1], "state_history": history},
            batch_size=[batch_size],
        )
        return TensorDict({"planner": planner}, batch_size=[batch_size])

    def expert_trajectory_motion_names(self) -> list[str]:
        return list(self.motion_names)

    def expert_macro_feature_slices(
        self, horizon_steps: int
    ) -> dict[str, tuple[int, int]]:
        assert int(horizon_steps) == self.horizon_steps
        split = max(1, min(3, self.state_dim))
        return {
            "expert_motion": (0, split),
            "anchor": (split, self.state_dim),
        }


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


def test_flow_matching_generator_forward_and_loss_shapes() -> None:
    torch.manual_seed(11)
    generator = FlowMatchingSkillCommander(
        state_dim=7,
        lang_embed_dim=5,
        z_dim=3,
        hidden_dims=(16,),
        time_embed_dim=8,
        num_inference_steps=2,
        inference_noise_std=0.0,
    )
    state = torch.randn(4, 7)
    lang = torch.randn(4, 5)
    z_target = torch.randn(4, 3)
    loss, metrics = generator.flow_matching_loss(state, lang, z_target)
    z = generator(state, lang)

    assert loss.ndim == 0
    assert "flow/velocity_mse" in metrics
    assert z.shape == (4, 3)
    with pytest.raises(ValueError, match="z_target must have shape"):
        generator.flow_matching_loss(state, lang, torch.randn(4, 4))


def test_diffusion_generator_forward_and_loss_shapes() -> None:
    torch.manual_seed(14)
    generator = DiffusionSkillCommander(
        state_dim=7,
        lang_embed_dim=5,
        z_dim=3,
        hidden_dims=(16,),
        time_embed_dim=8,
        num_train_timesteps=10,
        num_inference_steps=2,
        inference_scheduler="ddim",
        ddim_eta=0.0,
        inference_noise_std=0.0,
    )
    state = torch.randn(4, 7)
    lang = torch.randn(4, 5)
    z_target = torch.randn(4, 3)
    loss, metrics = generator.diffusion_loss(state, lang, z_target)
    z = generator(state, lang)

    assert loss.ndim == 0
    assert "diffusion/noise_mse" in metrics
    assert z.shape == (4, 3)
    with pytest.raises(ValueError, match="z_target must have shape"):
        generator.diffusion_loss(state, lang, torch.randn(4, 4))


def test_dit_latent_generator_forward_and_loss_shapes() -> None:
    torch.manual_seed(17)
    generator = DiTLatentSkillCommander(
        state_dim=7,
        lang_embed_dim=5,
        z_dim=5,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        patch_dim=2,
        num_state_tokens=2,
        time_embed_dim=8,
        num_train_timesteps=10,
        num_inference_steps=2,
        inference_scheduler="ddim",
        ddim_eta=0.0,
        inference_noise_std=0.0,
    )
    state = torch.randn(4, 7)
    lang = torch.randn(4, 5)
    z_target = torch.randn(4, 5)
    loss, metrics = generator.diffusion_loss(state, lang, z_target)
    z = generator(state, lang)

    assert loss.ndim == 0
    assert "diffusion/noise_mse" in metrics
    assert z.shape == (4, 5)
    assert generator.padded_z_dim == 6
    with pytest.raises(ValueError, match="z_target must have shape"):
        generator.diffusion_loss(state, lang, torch.randn(4, 4))


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


def test_no_language_history_flow_trainer_checkpoint_and_sampler(tmp_path) -> None:
    torch.manual_seed(16)
    env = _FakeMacroEnv(deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    config = SkillCommanderConfig(
        skill_checkpoint_path=str(skill_ckpt),
        condition_on_language=False,
        state_history_steps=2,
        planner_type="flow_matching",
        generator_hidden_dims=(32,),
        flow_time_embed_dim=8,
        flow_num_inference_steps=2,
        flow_inference_noise_std=0.0,
        batch_size=8,
        num_updates=2,
        log_interval=1,
        eval_batches=1,
        eval_batch_size=8,
        preflight_batch_size=4,
        train_split="all",
        eval_split="all",
        eval_trajectory_fraction=0.25,
        device="cpu",
    )
    trainer = SkillCommanderTrainer(config, env)
    assert trainer.lang_embed_dim == 0
    assert trainer.state_dim == env.state_dim
    assert trainer.planner_state_dim == env.state_dim * 3

    metrics = trainer.train_step()
    assert "train/flow_loss" in metrics
    eval_metrics = trainer.evaluate()
    assert "eval/z_cosine_wrong_lang" not in eval_metrics

    ckpt_path = tmp_path / "no_language_flow_generator.pt"
    trainer.save_checkpoint(ckpt_path)
    saved = torch.load(ckpt_path, weights_only=False)
    assert saved["config"]["condition_on_language"] is False
    assert saved["config"]["state_history_steps"] == 2
    assert saved["state_dim"] == env.state_dim * 3
    assert saved["macro_state_dim"] == env.state_dim
    assert saved["lang_embed_dim"] == 0

    sampler = FrozenSkillCommanderSampler(
        env=env,
        checkpoint_path=ckpt_path,
        language_embeddings_path="",
        latent_dim=5,
        latent_steps_min=1,
        latent_steps_max=1,
        discover_env_method=_discover_direct_method,
        generator_config_overrides={"flow_inference_noise_std": 0.0},
        command_mode="z",
        device="cpu",
    )
    td = TensorDict({}, batch_size=[2])
    z = sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
    assert z.shape == (2, 5)
    assert sampler.condition_on_language is False
    assert sampler.planner_state_dim == env.state_dim * 3


def test_flow_matching_trainer_checkpoint_and_sampler(tmp_path) -> None:
    torch.manual_seed(12)
    env = _FakeMacroEnv(deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    config = _gen_config(
        skill_ckpt,
        lang_table,
        planner_type="flow_matching",
        generator_hidden_dims=(32,),
        flow_time_embed_dim=8,
        flow_num_inference_steps=2,
        flow_inference_noise_std=0.0,
        language_contrastive_coeff=0.25,
        language_contrastive_margin=0.05,
    )
    trainer = SkillCommanderTrainer(config, env)
    metrics = trainer.train_step()
    assert "train/flow_loss" in metrics
    assert "train/flow/velocity_mse" in metrics
    assert "train/language_contrastive/loss" in metrics
    assert "train/language_contrastive_weighted_loss" in metrics

    eval_metrics = trainer.evaluate()
    assert "eval/z_cosine_wrong_lang" in eval_metrics
    assert "eval/z_mse_wrong_lang" in eval_metrics
    assert "eval/z_cosine_language_delta" in eval_metrics

    ckpt_path = tmp_path / "flow_generator.pt"
    trainer.save_checkpoint(ckpt_path)
    saved = torch.load(ckpt_path, weights_only=False)
    assert saved["config"]["planner_type"] == "flow_matching"

    sampler = FrozenSkillCommanderSampler(
        env=env,
        checkpoint_path=ckpt_path,
        language_embeddings_path=lang_table,
        latent_dim=5,
        latent_steps_min=1,
        latent_steps_max=1,
        discover_env_method=_discover_direct_method,
        generator_config_overrides={
            "diffusion_inference_scheduler": "ddpm",
            "diffusion_inference_noise_std": 0.0,
        },
        command_mode="z",
        device="cpu",
    )
    td = TensorDict({}, batch_size=[2])
    z = sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
    assert z.shape == (2, 5)
    assert isinstance(sampler.generator, FlowMatchingSkillCommander)


def test_diffusion_trainer_checkpoint_and_sampler(tmp_path) -> None:
    torch.manual_seed(15)
    env = _FakeMacroEnv(deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    config = _gen_config(
        skill_ckpt,
        lang_table,
        planner_type="diffusion_policy",
        generator_hidden_dims=(32,),
        diffusion_time_embed_dim=8,
        diffusion_num_train_timesteps=10,
        diffusion_num_inference_steps=2,
        diffusion_inference_scheduler="ddim",
        diffusion_ddim_eta=0.0,
        diffusion_inference_noise_std=0.0,
        language_contrastive_coeff=0.25,
        language_contrastive_margin=0.05,
    )
    trainer = SkillCommanderTrainer(config, env)
    metrics = trainer.train_step()
    assert "train/diffusion_loss" in metrics
    assert "train/diffusion/noise_mse" in metrics
    assert "train/language_contrastive/loss" in metrics
    assert "train/language_contrastive_weighted_loss" in metrics

    eval_metrics = trainer.evaluate()
    assert "eval/z_cosine_wrong_lang" in eval_metrics
    assert "eval/z_mse_wrong_lang" in eval_metrics
    assert "eval/z_cosine_language_delta" in eval_metrics

    ckpt_path = tmp_path / "diffusion_generator.pt"
    trainer.save_checkpoint(ckpt_path)
    saved = torch.load(ckpt_path, weights_only=False)
    assert saved["config"]["planner_type"] == "diffusion_policy"
    assert saved["config"]["diffusion_inference_scheduler"] == "ddim"

    sampler = FrozenSkillCommanderSampler(
        env=env,
        checkpoint_path=ckpt_path,
        language_embeddings_path=lang_table,
        latent_dim=5,
        latent_steps_min=1,
        latent_steps_max=1,
        discover_env_method=_discover_direct_method,
        generator_config_overrides={
            "diffusion_inference_scheduler": "ddpm",
            "diffusion_inference_noise_std": 0.0,
        },
        command_mode="z",
        device="cpu",
    )
    td = TensorDict({}, batch_size=[2])
    z = sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
    assert z.shape == (2, 5)
    assert isinstance(sampler.generator, DiffusionSkillCommander)
    assert sampler.generator.inference_scheduler_name == "ddpm"
    assert sampler.generator.inference_noise_std == pytest.approx(0.0)


def test_dit_trainer_checkpoint_and_sampler(tmp_path) -> None:
    torch.manual_seed(18)
    env = _FakeMacroEnv(deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    config = _gen_config(
        skill_ckpt,
        lang_table,
        planner_type="dit",
        diffusion_time_embed_dim=8,
        diffusion_num_train_timesteps=10,
        diffusion_num_inference_steps=2,
        diffusion_inference_scheduler="ddim",
        diffusion_ddim_eta=0.0,
        diffusion_inference_noise_std=0.0,
        dit_model_dim=16,
        dit_num_layers=1,
        dit_num_heads=4,
        dit_feedforward_dim=32,
        dit_patch_dim=2,
        dit_num_state_tokens=2,
        language_contrastive_coeff=0.25,
        language_contrastive_margin=0.05,
    )
    trainer = SkillCommanderTrainer(config, env)
    metrics = trainer.train_step()
    assert "train/diffusion_loss" in metrics
    assert "train/diffusion/noise_mse" in metrics
    assert "train/language_contrastive/loss" in metrics

    ckpt_path = tmp_path / "dit_generator.pt"
    trainer.save_checkpoint(ckpt_path)
    saved = torch.load(ckpt_path, weights_only=False)
    assert saved["config"]["planner_type"] == "dit_diffusion"
    assert saved["config"]["dit_model_dim"] == 16

    sampler = FrozenSkillCommanderSampler(
        env=env,
        checkpoint_path=ckpt_path,
        language_embeddings_path=lang_table,
        latent_dim=5,
        latent_steps_min=1,
        latent_steps_max=1,
        discover_env_method=_discover_direct_method,
        generator_config_overrides={
            "diffusion_inference_scheduler": "ddpm",
            "diffusion_inference_noise_std": 0.0,
        },
        command_mode="z",
        device="cpu",
    )
    td = TensorDict({}, batch_size=[2])
    z = sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
    assert z.shape == (2, 5)
    assert isinstance(sampler.generator, DiTLatentSkillCommander)
    assert sampler.generator.inference_scheduler_name == "ddpm"


def test_state_feature_dropout_targets_named_slices(tmp_path) -> None:
    torch.manual_seed(13)
    env = _FakeMacroEnv(state_dim=7, deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    lang_table = _make_language_table(tmp_path, env.motion_names)
    trainer = SkillCommanderTrainer(
        _gen_config(
            skill_ckpt,
            lang_table,
            state_feature_dropout_prob=1.0,
            state_feature_dropout_terms=("robot_state",),
            state_feature_dropout_mode="zero",
            state_feature_dropout_warmup_updates=2,
        ),
        env,
    )
    state = torch.ones(4, env.state_dim)
    augmented, metrics = trainer._augment_state(state)
    assert torch.equal(augmented, state)
    assert metrics == {}

    trainer.update = 2
    augmented, metrics = trainer._augment_state(state)
    assert torch.count_nonzero(augmented).item() == 0
    assert metrics["state_feature_dropout/active_frac"] == pytest.approx(1.0)
    assert metrics["state_feature_dropout/num_features"] == pytest.approx(
        float(env.state_dim)
    )


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

    def blocked_expert_getter(**_kwargs):
        msg = "deployable inference must not read expert macro state"
        raise AssertionError(msg)

    env.current_expert_macro_transition_batch = blocked_expert_getter  # type: ignore[method-assign]

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
        goal_name="motion_0",
        device="cpu",
    )
    assert sampler.use_achieved_state
    td = TensorDict({}, batch_size=[2])
    out = sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
    assert out.shape == (2, 5)


def test_frozen_commander_loads_shared_interface_planner(tmp_path) -> None:
    env = _FakeMacroEnv(deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    planner = CausalInterfaceTransformerFlowPlanner(
        state_dim=env.state_dim,
        target_dim=5,
        term_widths=(5,),
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        patch_dim=4,
        num_state_tokens=1,
    )
    checkpoint = {
        "planner_config": planner.config_dict(),
        "planner_state_dict": planner.state_dict(),
        "target_spec": {
            "interface": "latent_skill",
            "term_names": ["z"],
            "term_widths": [5],
            "target_dim": 5,
        },
        "metadata": {
            "flow_num_inference_steps": 2,
            "flow_inference_noise_std": 0.0,
            "sample_metadata": {
                "state_history_steps": 0,
                "planner_observation_spec": env.causal_planner_observation_spec(0),
                "provenance": {"skill_checkpoint": str(skill_ckpt)},
            },
        },
    }
    planner_ckpt = tmp_path / "shared_planner.pt"
    torch.save(checkpoint, planner_ckpt)

    def blocked_expert_getter(**_kwargs):
        msg = "shared deployable planner must not read expert state"
        raise AssertionError(msg)

    env.current_expert_macro_transition_batch = blocked_expert_getter  # type: ignore[method-assign]
    sampler = FrozenSkillCommanderSampler(
        env=env,
        checkpoint_path=planner_ckpt,
        language_embeddings_path="",
        latent_dim=5,
        latent_steps_min=env.horizon_steps,
        latent_steps_max=env.horizon_steps,
        horizon_steps=env.horizon_steps,
        discover_env_method=_discover_direct_method,
        command_mode="z",
        use_achieved_state=True,
        device="cpu",
    )
    td = TensorDict({}, batch_size=[2])
    out = sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
    assert out.shape == (2, 5)


def test_frozen_shared_planner_uses_explicit_language_goal(tmp_path) -> None:
    env = _FakeMacroEnv(deterministic=False)
    skill_ckpt = _make_skill_checkpoint(tmp_path, env)
    language_path = _make_language_table(tmp_path, env.motion_names, embed_dim=5)
    planner = CausalInterfaceTransformerFlowPlanner(
        state_dim=env.state_dim,
        target_dim=5,
        term_widths=(5,),
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        patch_dim=4,
        num_state_tokens=1,
        language_dim=5,
        num_language_tokens=1,
    )
    checkpoint = {
        "planner_config": planner.config_dict(),
        "planner_state_dict": planner.state_dict(),
        "target_spec": {
            "interface": "latent_skill",
            "term_names": ["z"],
            "term_widths": [5],
            "target_dim": 5,
        },
        "metadata": {
            "flow_num_inference_steps": 2,
            "flow_inference_noise_std": 0.0,
            "sample_metadata": {
                "state_history_steps": 0,
                "planner_observation_spec": env.causal_planner_observation_spec(0),
                "language_conditioning": {
                    "enabled": True,
                    "embedding_dim": 5,
                    "embedding_path": str(language_path),
                },
                "provenance": {"skill_checkpoint": str(skill_ckpt)},
            },
        },
    }
    planner_ckpt = tmp_path / "shared_language_planner.pt"
    torch.save(checkpoint, planner_ckpt)

    def blocked_expert_getter(**_kwargs):
        msg = "language planner must not read expert state"
        raise AssertionError(msg)

    env.current_expert_macro_transition_batch = blocked_expert_getter  # type: ignore[method-assign]
    sampler = FrozenSkillCommanderSampler(
        env=env,
        checkpoint_path=planner_ckpt,
        language_embeddings_path=language_path,
        latent_dim=5,
        latent_steps_min=env.horizon_steps,
        latent_steps_max=env.horizon_steps,
        horizon_steps=env.horizon_steps,
        discover_env_method=_discover_direct_method,
        command_mode="z",
        use_achieved_state=True,
        goal_name=env.motion_names[0],
        device="cpu",
    )
    td = TensorDict({}, batch_size=[2])
    out = sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)

    assert out.shape == (2, 5)
    assert sampler.condition_on_language
    assert sampler.forced_language_embedding is not None
