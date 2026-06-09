from __future__ import annotations

import warnings
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
warnings.filterwarnings(
    "ignore",
    message="invalid escape sequence.*",
    category=DeprecationWarning,
    append=False,
)
warnings.filterwarnings(
    "ignore",
    message="invalid escape sequence.*",
    category=SyntaxWarning,
    append=False,
)

from rlopt.agent.hl_skill_diffsr import (  # noqa: E402
    FrozenHighLevelSkillCommandSampler,
    HighLevelSkillDiffSRConfig,
    HighLevelSkillDiffSRTrainer,
)


class _FakeMacroEnv:
    def __init__(
        self,
        *,
        state_dim: int = 7,
        horizon_steps: int = 3,
        shape_error: str | None = None,
    ) -> None:
        self.device = torch.device("cpu")
        self.state_dim = int(state_dim)
        self.horizon_steps = int(horizon_steps)
        self.shape_error = shape_error
        self.calls = 0
        self.requested_splits: list[str | None] = []

    def sample_expert_macro_transition_batch(
        self,
        batch_size: int,
        horizon_steps: int,
        split: str | None = None,
        eval_fraction: float = 0.1,
        split_seed: int = 0,
    ) -> TensorDict:
        assert int(horizon_steps) == self.horizon_steps
        assert 0.0 < float(eval_fraction) < 1.0
        assert isinstance(split_seed, int)
        self.calls += 1
        self.requested_splits.append(split)
        base = torch.arange(batch_size * self.state_dim, dtype=torch.float32).reshape(
            batch_size,
            self.state_dim,
        )
        base = base / 100.0 + float(self.calls) * 0.01
        offsets = torch.arange(1, horizon_steps + 1, dtype=torch.float32).view(
            1,
            horizon_steps,
            1,
        )
        future_window = base.unsqueeze(1) + offsets / 10.0
        target = future_window[:, -1, :].clone()
        state = base.clone()
        if self.shape_error == "state":
            state = state.unsqueeze(-1)
        elif self.shape_error == "window":
            future_window = future_window[:, :, :-1]
        elif self.shape_error == "target":
            target = target[:, :-1]
        hl = TensorDict(
            {
                "state": state,
                "future_window": future_window,
                "target": target,
            },
            batch_size=[batch_size],
        )
        return TensorDict({"hl": hl}, batch_size=[batch_size])

    def expert_macro_feature_slices(
        self,
        horizon_steps: int,
    ) -> dict[str, tuple[int, int]]:
        assert int(horizon_steps) == self.horizon_steps
        return {
            "expert_motion": (0, 2),
            "expert_anchor_pos_b": (2, 5),
            "expert_anchor_ori_b": (5, self.state_dim),
        }


def _config(**overrides: Any) -> HighLevelSkillDiffSRConfig:
    values: dict[str, Any] = {
        "horizon_steps": 3,
        "z_dim": 5,
        "diffsr_feature_dim": 4,
        "diffsr_embed_dim": 8,
        "batch_size": 6,
        "num_updates": 2,
        "log_interval": 1,
        "eval_batches": 1,
        "eval_batch_size": 4,
        "preflight_batch_size": 4,
        "encoder_hidden_dims": (16, 12),
        "diffsr_f_hidden_dims": (16,),
        "diffsr_g_hidden_dims": (16,),
        "diffsr_mu_hidden_dims": (16,),
        "diffsr_num_noises": 2,
        "device": "cpu",
    }
    values.update(overrides)
    return HighLevelSkillDiffSRConfig(**values)


def _module_parameters(trainer: HighLevelSkillDiffSRTrainer) -> list[torch.Tensor]:
    return [*trainer.skill_encoder.parameters(), *trainer.diffsr.parameters()]


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_skill_encoder_outputs_z_and_diffsr_gradients_reach_encoder() -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_config(), _FakeMacroEnv())
    state, future_window, target = trainer._sample_and_validate_macro_batch(
        4,
        split=trainer.config.train_split,
    )

    assert trainer.feature_slices == {
        "expert_motion": (0, 2),
        "expert_anchor_pos_b": (2, 5),
        "expert_anchor_ori_b": (5, trainer.state_dim),
    }
    z = trainer.skill_encoder(state, future_window)
    assert z.shape == (4, trainer.config.z_dim)

    trainer.diffsr.update_obs_norm(target)
    loss = trainer._diffsr_loss_for_z(state, z, target)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    encoder_grad = sum(
        float(param.grad.abs().sum().item())
        for param in trainer.skill_encoder.parameters()
        if param.grad is not None
    )
    assert encoder_grad > 0.0


def test_intermediate_encoder_mode_hides_macro_target_from_encoder() -> None:
    torch.manual_seed(3)
    config = _config(encoder_window_mode="intermediate")
    trainer = HighLevelSkillDiffSRTrainer(config, _FakeMacroEnv())
    state, future_window, target = trainer._sample_and_validate_macro_batch(
        4,
        split=trainer.config.train_split,
    )

    assert trainer.encoder_window_steps == config.horizon_steps - 1
    assert trainer.skill_encoder.window_steps == config.horizon_steps - 1
    with pytest.raises(ValueError, match="state/future_window shape mismatch"):
        trainer.skill_encoder(state, future_window)

    z = trainer._encode_skill(state, future_window)
    manual_z = trainer.skill_encoder(state, future_window[:, :-1, :])
    assert z.shape == (4, trainer.config.z_dim)
    assert torch.equal(z, manual_z)

    trainer.diffsr.update_obs_norm(target)
    loss = trainer._diffsr_loss_for_z(state, z, target)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    encoder_grad = sum(
        float(param.grad.abs().sum().item())
        for param in trainer.skill_encoder.parameters()
        if param.grad is not None
    )
    assert encoder_grad > 0.0


def test_intermediate_encoder_mode_requires_horizon_greater_than_one() -> None:
    config = _config(horizon_steps=1, encoder_window_mode="intermediate")
    with pytest.raises(ValueError, match="requires horizon_steps > 1"):
        config.validate()


@pytest.mark.parametrize(
    ("shape_error", "match"),
    [
        ("state", "hl/state"),
        ("window", "hl/future_window"),
        ("target", "hl/target"),
    ],
)
def test_trainer_preflight_rejects_wrong_macro_shapes(
    shape_error: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        HighLevelSkillDiffSRTrainer(
            _config(),
            _FakeMacroEnv(shape_error=shape_error),
        )


def test_zero_and_shuffled_eval_do_not_step_optimizer() -> None:
    torch.manual_seed(1)
    env = _FakeMacroEnv()
    trainer = HighLevelSkillDiffSRTrainer(_config(), env)
    trainer.train_step()
    before = [param.detach().clone() for param in _module_parameters(trainer)]

    metrics = trainer.evaluate(
        num_batches=1,
        batch_size=4,
        prefix="train",
        include_reconstruction=True,
    )

    after = [param.detach().clone() for param in _module_parameters(trainer)]
    for before_param, after_param in zip(before, after, strict=True):
        assert torch.equal(before_param, after_param)
    assert "train/loss_real_z_eval" in metrics
    assert "train/loss_zero_z_eval" in metrics
    assert "train/loss_shuffled_z_eval" in metrics
    assert "train/sample_recon_l1" in metrics
    assert "train/sample_recon_mse" in metrics
    assert "train/sample_recon_dim_mse" in metrics
    assert "train/sample_recon_expert_motion_dim_mse" in metrics
    assert "train/sample_recon_expert_anchor_pos_b_dim_mse" in metrics
    assert "train/sample_recon_expert_anchor_ori_b_dim_mse" in metrics
    assert "train/norm_sample_recon_mse" in metrics
    assert "train/norm_sample_recon_dim_mse" in metrics
    assert "train/norm_sample_recon_expert_motion_dim_mse" in metrics
    assert "train/norm_sample_recon_expert_anchor_pos_b_dim_mse" in metrics
    assert "train/norm_sample_recon_expert_anchor_ori_b_dim_mse" in metrics
    assert "train" in env.requested_splits
    assert "eval" in env.requested_splits


def test_window_probe_eval_does_not_step_optimizer() -> None:
    torch.manual_seed(2)
    env = _FakeMacroEnv()
    trainer = HighLevelSkillDiffSRTrainer(_config(), env)
    before = [param.detach().clone() for param in _module_parameters(trainer)]

    metrics = trainer.evaluate_window_probe(
        train_batches=2,
        eval_batches=1,
        batch_size=4,
        prefix="eval",
    )

    after = [param.detach().clone() for param in _module_parameters(trainer)]
    for before_param, after_param in zip(before, after, strict=True):
        assert torch.equal(before_param, after_param)
    assert "eval/window_probe_z_dim_mse" in metrics
    assert "eval/window_probe_z_shuffled_dim_mse" in metrics
    assert "eval/window_probe_state_dim_mse" in metrics
    assert "eval/window_probe_mean_dim_mse" in metrics
    assert "eval/window_probe_z_norm_dim_mse" in metrics
    assert "eval/window_probe_z_step_final_dim_mse" in metrics
    assert metrics["eval/window_probe_train_samples"] == 8.0
    assert metrics["eval/window_probe_eval_samples"] == 4.0
    assert "train" in env.requested_splits
    assert "eval" in env.requested_splits


@pytest.mark.parametrize("encoder_window_mode", ["full", "intermediate"])
def test_checkpoint_save_load_restores_trainer_state(
    tmp_path,
    encoder_window_mode: str,
) -> None:
    torch.manual_seed(2)
    config = _config(encoder_window_mode=encoder_window_mode)
    trainer = HighLevelSkillDiffSRTrainer(config, _FakeMacroEnv())
    trainer.train_step()
    checkpoint_path = tmp_path / "latest.pt"

    trainer.save_checkpoint(checkpoint_path)
    loaded = HighLevelSkillDiffSRTrainer(config, _FakeMacroEnv())
    loaded.load_checkpoint(checkpoint_path)

    assert loaded.update == trainer.update
    assert loaded.config.to_dict() == trainer.config.to_dict()
    _assert_nested_equal(
        trainer.skill_encoder.state_dict(),
        loaded.skill_encoder.state_dict(),
    )
    _assert_nested_equal(trainer.diffsr.state_dict(), loaded.diffsr.state_dict())
    _assert_nested_equal(
        trainer.optimizer.state_dict(),
        loaded.optimizer.state_dict(),
    )
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert "feature_normalization_state_dict" in checkpoint


class _FakeCurrentMacroEnv(_FakeMacroEnv):
    def __init__(self, *, state_dim: int = 7, horizon_steps: int = 3) -> None:
        super().__init__(state_dim=state_dim, horizon_steps=horizon_steps)
        self.published_latents: torch.Tensor | None = None

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
            split=None,
        )

    def set_agent_latent_command(self, latent_command: torch.Tensor) -> None:
        self.published_latents = latent_command.detach().clone()


def _discover_direct_method(env: object, method_name: str):
    method = getattr(env, method_name, None)
    return method if callable(method) else None


def test_frozen_hl_skill_command_sampler_loads_checkpoint_and_holds_latents(
    tmp_path,
) -> None:
    torch.manual_seed(4)
    config = _config(encoder_window_mode="intermediate")
    trainer = HighLevelSkillDiffSRTrainer(config, _FakeMacroEnv())
    checkpoint_path = tmp_path / "latest.pt"
    trainer.save_checkpoint(checkpoint_path)
    env = _FakeCurrentMacroEnv()

    sampler = FrozenHighLevelSkillCommandSampler(
        env=env,
        checkpoint_path=checkpoint_path,
        latent_dim=config.z_dim,
        latent_steps_min=2,
        latent_steps_max=2,
        discover_env_method=_discover_direct_method,
        device="cpu",
    )
    td = TensorDict({}, batch_size=[2])

    first = sampler.sample_for_step(
        td,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).clone()
    second = sampler.sample_for_step(
        td,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).clone()
    third = sampler.sample_for_step(
        td,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).clone()

    assert first.shape == (2, config.z_dim)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert not any(param.requires_grad for param in sampler.skill_encoder.parameters())


def test_frozen_hl_skill_command_sampler_appends_sin_cos_phase(tmp_path) -> None:
    torch.manual_seed(5)
    config = _config(encoder_window_mode="intermediate")
    trainer = HighLevelSkillDiffSRTrainer(config, _FakeMacroEnv())
    checkpoint_path = tmp_path / "latest.pt"
    trainer.save_checkpoint(checkpoint_path)
    env = _FakeCurrentMacroEnv()

    sampler = FrozenHighLevelSkillCommandSampler(
        env=env,
        checkpoint_path=checkpoint_path,
        latent_dim=config.z_dim + 2,
        latent_steps_min=3,
        latent_steps_max=3,
        discover_env_method=_discover_direct_method,
        command_phase_mode="sin_cos",
        code_latent_dim=config.z_dim,
        phase_period=3,
        device="cpu",
    )
    td = TensorDict({}, batch_size=[2])

    first = sampler.sample_for_step(
        td,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).clone()
    second = sampler.sample_for_step(
        td,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).clone()
    third = sampler.sample_for_step(
        td,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).clone()
    fourth = sampler.sample_for_step(
        td,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).clone()

    assert first.shape == (2, config.z_dim + 2)
    assert torch.equal(first[:, : config.z_dim], second[:, : config.z_dim])
    assert torch.equal(first[:, : config.z_dim], third[:, : config.z_dim])
    assert not torch.equal(first[:, : config.z_dim], fourth[:, : config.z_dim])
    assert torch.allclose(
        first[:, -2:],
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        second[:, -2:],
        torch.tensor([[0.8660254, -0.5], [0.8660254, -0.5]]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        third[:, -2:],
        torch.tensor([[-0.8660254, -0.5], [-0.8660254, -0.5]]),
        atol=1.0e-6,
    )


def test_frozen_hl_skill_command_sampler_rejects_latent_dim_mismatch(tmp_path) -> None:
    config = _config()
    trainer = HighLevelSkillDiffSRTrainer(config, _FakeMacroEnv())
    checkpoint_path = tmp_path / "latest.pt"
    trainer.save_checkpoint(checkpoint_path)

    with pytest.raises(ValueError, match="command width must match"):
        FrozenHighLevelSkillCommandSampler(
            env=_FakeCurrentMacroEnv(),
            checkpoint_path=checkpoint_path,
            latent_dim=config.z_dim + 1,
            latent_steps_min=1,
            latent_steps_max=1,
            discover_env_method=_discover_direct_method,
            device="cpu",
        )
