"""Variable-window skill encoders: horizon sets, stride sets, code layouts.

The hub encoder reads one fixed window. These tests pin the extension that
lets one encoder serve several window lengths:

* the config accepts a horizon SET (max = ``horizon_steps``) or a stride SET
  (must contain 1), never both, and only on the merged chunk recipe;
* the ``padded`` and ``sequence`` trunks ignore slots past a row's length;
* every variant (padded / sequence / fixed target / stride / nested / block)
  trains, evaluates per variant, checkpoints its heads, and deploys through
  the frozen sampler under each live policy;
* a fixed-window config still builds the flat trunk, byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from rlopt.agent.hl_skill_diffsr import (
    FrozenHighLevelSkillCommandSampler,
    HighLevelSkillDiffSRConfig,
    HighLevelSkillDiffSRTrainer,
    _code_mask,
    _encoder_build_kwargs,
    _gather_slots,
    _strided_window,
)
from rlopt.agent.hl_skill_encoder import SkillLatentSpec, build_skill_encoder

STATE_DIM = 38
HMAX = 4
CHOICES = (2, 3, 4)
Z_DIM = 6


class _FakeEnv:
    """Serves random macro batches of any horizon; records the requests."""

    def __init__(self, num_envs: int = 5) -> None:
        self.imitation_interface = self
        self.num_envs = int(num_envs)
        self.expert_horizons: list[int] = []
        self.current_horizons: list[int] = []

    def _macro_batch(
        self, batch_size: int, horizon_steps: int, state_history_steps: int = 0
    ) -> TensorDict:
        payload = {
            ("hl", "state"): torch.randn(batch_size, STATE_DIM),
            ("hl", "future_window"): torch.randn(batch_size, horizon_steps, STATE_DIM),
            ("hl", "target"): torch.randn(batch_size, STATE_DIM),
        }
        if state_history_steps > 0:
            history = torch.randn(batch_size, state_history_steps + 1, STATE_DIM)
            history[:, -1].copy_(payload[("hl", "state")])
            payload[("hl", "state_history")] = history
        return TensorDict(payload, batch_size=[batch_size])

    def sample_expert_macro_transition_batch(
        self, *, batch_size: int, horizon_steps: int, **kwargs: object
    ) -> TensorDict:
        self.expert_horizons.append(int(horizon_steps))
        return self._macro_batch(
            int(batch_size),
            int(horizon_steps),
            int(kwargs.get("state_history_steps", 0)),
        )

    def current_expert_macro_transition_batch(
        self, horizon_steps: int, env_ids=None, state_history_steps: int = 0
    ) -> TensorDict:
        self.current_horizons.append(int(horizon_steps))
        count = self.num_envs if env_ids is None else int(env_ids.numel())
        return self._macro_batch(count, int(horizon_steps), int(state_history_steps))


def _discover(env: object, name: str):
    return getattr(env, name, None)


def _config(**overrides: object) -> HighLevelSkillDiffSRConfig:
    base = {
        "z_dim": Z_DIM,
        "horizon_steps": HMAX,
        "encoder_window_mode": "intermediate",
        "transition_objective": "jepa_ntp",
        "jepa_loss": "sigreg_ebm",
        "jepa_ntp_head": "diff_chunk",
        "jepa_ntp_chunk_span": "boundary_next",
        "jepa_endpoint_coeff": 0.0,
        "batch_size": 12,
        "encoder_hidden_dims": (32, 32),
        "diffsr_feature_dim": 8,
        "diffsr_embed_dim": 8,
        "diffsr_g_hidden_dims": (16,),
        "diffsr_mu_hidden_dims": (16,),
        "diffsr_f_hidden_dims": (16,),
        "sequence_width": 16,
        "sequence_depth": 1,
        "sequence_heads": 2,
        "eval_batches": 1,
        "eval_batch_size": 6,
    }
    base.update(overrides)
    config = HighLevelSkillDiffSRConfig(**base)  # type: ignore[arg-type]
    config.validate()
    return config


VARIANTS: dict[str, dict[str, object]] = {
    "padded": {"horizon_choices": CHOICES},
    "sequence": {"horizon_choices": CHOICES, "horizon_encoder": "sequence"},
    "fixed_target": {
        "horizon_choices": CHOICES,
        "horizon_target_mode": "fixed",
        "horizon_fixed_target_steps": 3,
    },
    "stride": {"stride_choices": (1, 2)},
    "nested": {
        "horizon_choices": CHOICES,
        "horizon_input_mode": "full",
        "horizon_code_layout": "nested",
    },
    "block": {
        "horizon_choices": CHOICES,
        "horizon_input_mode": "full",
        "horizon_code_layout": "block",
    },
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_fixed_window_config_builds_the_flat_trunk() -> None:
    assert _encoder_build_kwargs(_config()) == {"trunk": "flat", "num_variants": 0}


def test_horizon_choices_max_must_equal_horizon_steps() -> None:
    with pytest.raises(ValueError, match="max\\(horizon_choices\\)"):
        _config(horizon_choices=(2, 3))


def test_horizon_and_stride_sets_are_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(horizon_choices=CHOICES, stride_choices=(1, 2))


def test_stride_set_must_contain_the_base_stride() -> None:
    with pytest.raises(ValueError, match="must contain 1"):
        _config(stride_choices=(2, 3))


def test_code_layouts_need_the_full_input() -> None:
    with pytest.raises(ValueError, match="horizon_input_mode='full'"):
        _config(horizon_choices=CHOICES, horizon_code_layout="nested")


def test_variant_sets_need_the_merged_chunk_recipe() -> None:
    with pytest.raises(ValueError, match="merged chunk recipe"):
        _config(horizon_choices=CHOICES, jepa_ntp_head="mlp")


def test_variant_config_round_trips_through_dict() -> None:
    config = _config(horizon_choices=CHOICES, horizon_encoder="sequence")
    restored = HighLevelSkillDiffSRConfig.from_dict(config.to_dict())
    assert restored.horizon_choices == CHOICES
    assert restored.horizon_encoder == "sequence"
    assert (
        HighLevelSkillDiffSRConfig.from_dict(_config().to_dict()).horizon_choices == ()
    )


def test_code_masks_are_prefixes_or_disjoint_blocks() -> None:
    nested = _config(
        horizon_choices=CHOICES, horizon_input_mode="full", horizon_code_layout="nested"
    )
    masks = [_code_mask(nested, i, device=torch.device("cpu")) for i in range(3)]
    assert [int(m.sum()) for m in masks] == [2, 4, 6]
    assert torch.all(masks[0] <= masks[1])
    assert torch.all(masks[1] <= masks[2])
    block = _config(
        horizon_choices=CHOICES, horizon_input_mode="full", horizon_code_layout="block"
    )
    blocks = torch.stack(
        [_code_mask(block, i, device=torch.device("cpu")) for i in range(3)]
    )
    assert torch.equal(blocks.sum(dim=0), torch.ones(Z_DIM))


# --------------------------------------------------------------------------- #
# Window helpers and trunks
# --------------------------------------------------------------------------- #
def test_gather_and_stride_helpers_index_as_documented() -> None:
    window = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1)  # s[t+1..t+8]
    start = torch.tensor([2])
    assert _gather_slots(window, start, 3).flatten().tolist() == [3.0, 4.0, 5.0]
    strided = _strided_window(window, torch.tensor([2]), 4)
    assert strided.flatten().tolist() == [2.0, 4.0, 6.0, 8.0]
    with pytest.raises(ValueError, match="needs"):
        _strided_window(window, torch.tensor([3]), 4)


@pytest.mark.parametrize("trunk", ["padded", "sequence"])
def test_variant_trunks_ignore_slots_past_the_length(trunk: str) -> None:
    torch.manual_seed(0)
    encoder = build_skill_encoder(
        state_dim=STATE_DIM,
        window_steps=HMAX - 1,
        z_dim=Z_DIM,
        hidden_dims=(32,),
        spec=SkillLatentSpec(latent_mode="deterministic"),
        trunk=trunk,
        num_variants=3,
        sequence_width=16,
        sequence_depth=1,
        sequence_heads=2,
    ).eval()
    state = torch.randn(4, STATE_DIM)
    window = torch.randn(4, HMAX - 1, STATE_DIM)
    lengths = torch.tensor([1, 2, 3, 1])
    variant = torch.tensor([0, 1, 2, 0])
    z = encoder(state, window, lengths, variant)
    noisy = window.clone()
    noisy[0, 1:] += 10.0
    noisy[1, 2:] += 10.0
    noisy[3, 1:] += 10.0
    assert torch.allclose(z, encoder(state, noisy, lengths, variant), atol=1e-5)
    # A change inside the visible slots does move the code.
    visible = window.clone()
    visible[0, 0] += 1.0
    assert not torch.allclose(z[0], encoder(state, visible, lengths, variant)[0])


def test_flat_trunk_refuses_lengths() -> None:
    encoder = build_skill_encoder(
        state_dim=STATE_DIM,
        window_steps=2,
        z_dim=Z_DIM,
        hidden_dims=(8,),
        spec=SkillLatentSpec(latent_mode="deterministic"),
    )
    with pytest.raises(ValueError, match="flat trunk"):
        encoder(
            torch.randn(2, STATE_DIM),
            torch.randn(2, 2, STATE_DIM),
            torch.tensor([1, 2]),
        )


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_variant_trains_evaluates_and_checkpoints(
    name: str, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    env = _FakeEnv()
    trainer = HighLevelSkillDiffSRTrainer(_config(**VARIANTS[name]), env)
    assert trainer.jepa_ntp_heads is not None
    expected_heads = 1 if name == "fixed_target" else len(trainer.variant_values)
    assert len(trainer.jepa_ntp_heads) == expected_heads
    for _ in range(2):
        metrics = trainer.train_step()
        assert torch.isfinite(torch.tensor(metrics["train/loss"]))
    labels = trainer.variant_labels
    assert any(
        key.endswith(labels[0])
        for key in metrics
        if key.startswith("train/jepa_ntp_loss/")
    )
    # The macro batch is drawn at the widest span the set needs.
    widest = 2 * HMAX * (2 if name == "stride" else 1)
    assert max(env.expert_horizons) == widest
    evaluation = trainer.evaluate()
    for label in labels:
        assert f"train/jepa_ntp_loss_eval/{label}" in evaluation
        assert f"train/z_effective_rank/{label}" in evaluation
    assert "train/jepa_ntp_z_explained" in evaluation
    path = tmp_path / f"{name}.pt"
    trainer.save_checkpoint(path)
    blob = torch.load(path, weights_only=False)
    assert set(blob["jepa_state_dict"]["ntp_diffsr_heads"]) == set(
        trainer.jepa_ntp_heads
    )
    assert tuple(blob["config"]["horizon_choices"]) == tuple(
        trainer.config.horizon_choices
    )


def test_nested_heads_see_only_their_prefix() -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_config(**VARIANTS["nested"]), _FakeEnv())
    batch = trainer._variant_batch(
        batch_size=6, split=None, variant=torch.zeros(6, dtype=torch.long)
    )
    z = torch.randn(6, Z_DIM, device=trainer.device)
    loss_a, _ = trainer._variant_ntp_terms(batch, z, update_norm=False)
    z_tail = z.clone()
    z_tail[:, 2:] = 7.0
    torch.manual_seed(1)
    loss_a_again, _ = trainer._variant_ntp_terms(batch, z, update_norm=False)
    torch.manual_seed(1)
    loss_b, _ = trainer._variant_ntp_terms(batch, z_tail, update_norm=False)
    assert torch.allclose(loss_a_again, loss_b)
    del loss_a


# --------------------------------------------------------------------------- #
# Frozen sampler
# --------------------------------------------------------------------------- #
def _checkpoint(tmp_path: Path, name: str) -> Path:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_config(**VARIANTS[name]), _FakeEnv())
    trainer.train_step()
    path = tmp_path / f"{name}.pt"
    trainer.save_checkpoint(path)
    return path


def _sampler(checkpoint: Path, env: _FakeEnv, **overrides: object):
    kwargs = {
        "env": env,
        "checkpoint_path": checkpoint,
        "latent_dim": Z_DIM + 2,
        "latent_steps_min": 1,
        "latent_steps_max": 1,
        "discover_env_method": _discover,
        "horizon_steps": HMAX,
        "command_phase_mode": "sin_cos",
        "finetune_enabled": False,
    }
    kwargs.update(overrides)
    return FrozenHighLevelSkillCommandSampler(**kwargs)  # type: ignore[arg-type]


def _step(sampler, env: _FakeEnv, done: bool = False) -> torch.Tensor:
    td = TensorDict(
        {"done": torch.full((env.num_envs,), done, dtype=torch.bool)},
        batch_size=[env.num_envs],
    )
    return sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)


@pytest.mark.parametrize("name", sorted(VARIANTS))
@pytest.mark.parametrize("policy", ["base", "episode", "step"])
def test_sampler_deploys_every_variant_under_every_policy(
    name: str, policy: str, tmp_path: Path
) -> None:
    env = _FakeEnv()
    sampler = _sampler(_checkpoint(tmp_path, name), env, live_horizon=policy)
    codes = _step(sampler, env)
    assert codes.shape == (env.num_envs, Z_DIM + 2)
    assert torch.isfinite(codes).all()
    assert sampler._live_variant is not None
    if policy == "base":
        expected = (
            -1 if name == "block" else (0 if name == "stride" else len(CHOICES) - 1)
        )
        assert torch.all(sampler._live_variant == expected)
    request = max(env.current_horizons)
    assert request == (HMAX * 2 if name == "stride" else HMAX)


def test_sampler_fixed_member_and_bad_members(tmp_path: Path) -> None:
    env = _FakeEnv()
    checkpoint = _checkpoint(tmp_path, "padded")
    sampler = _sampler(checkpoint, env, live_horizon="3")
    _step(sampler, env)
    assert torch.all(sampler._live_variant == CHOICES.index(3))
    with pytest.raises(ValueError, match="not in the checkpoint"):
        _sampler(checkpoint, env, live_horizon="7")
    with pytest.raises(ValueError, match="must be 'base'"):
        _sampler(checkpoint, env, live_horizon="soon")


def test_episode_policy_redraws_only_on_reset(tmp_path: Path) -> None:
    torch.manual_seed(3)
    env = _FakeEnv(num_envs=64)
    sampler = _sampler(_checkpoint(tmp_path, "padded"), env, live_horizon="episode")
    _step(sampler, env)
    first = sampler._live_variant.clone()
    _step(sampler, env, done=False)
    assert torch.equal(sampler._live_variant, first)
    _step(sampler, env, done=True)
    assert not torch.equal(sampler._live_variant, first)


def test_step_policy_redraws_every_renewal(tmp_path: Path) -> None:
    torch.manual_seed(4)
    env = _FakeEnv(num_envs=64)
    sampler = _sampler(_checkpoint(tmp_path, "padded"), env, live_horizon="step")
    _step(sampler, env)
    first = sampler._live_variant.clone()
    _step(sampler, env, done=False)
    assert not torch.equal(sampler._live_variant, first)


def test_nested_truncation_zeroes_the_tail_dims(tmp_path: Path) -> None:
    env = _FakeEnv()
    sampler = _sampler(_checkpoint(tmp_path, "nested"), env, live_horizon="2")
    codes = _step(sampler, env)
    assert torch.all(codes[:, 2:Z_DIM] == 0.0)
    assert torch.any(codes[:, :2] != 0.0)


def test_block_base_keeps_every_block(tmp_path: Path) -> None:
    env = _FakeEnv()
    sampler = _sampler(_checkpoint(tmp_path, "block"), env, live_horizon="base")
    codes = _step(sampler, env)
    assert torch.all(codes[:, :Z_DIM].abs().sum(dim=0) > 0.0)
    single = _sampler(_checkpoint(tmp_path, "block"), env, live_horizon="4")
    codes = _step(single, env)
    assert torch.all(codes[:, :4] == 0.0)


def test_fixed_window_checkpoint_refuses_a_live_policy(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_config(), _FakeEnv())
    trainer.train_step()
    path = tmp_path / "fixed.pt"
    trainer.save_checkpoint(path)
    env = _FakeEnv()
    _step(_sampler(path, env), env)
    with pytest.raises(ValueError, match="fixed window"):
        _sampler(path, env, live_horizon="episode")


def test_variant_checkpoint_refuses_finetune_and_phi(tmp_path: Path) -> None:
    env = _FakeEnv()
    checkpoint = _checkpoint(tmp_path, "padded")
    with pytest.raises(ValueError, match="Online finetuning"):
        _sampler(checkpoint, env, finetune_enabled=True)
    with pytest.raises(ValueError, match="command_mode='z'"):
        _sampler(
            checkpoint, env, command_mode="phi", latent_dim=8 + 2, code_latent_dim=8
        )
