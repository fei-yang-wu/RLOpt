"""Tests for the high-level skill encoders, focused on the SONIC-motivated FSQ mode.

``SONICFSQSkillEncoder`` exists so the planner -> tracker command boundary sits at
the quantizer output, which is what makes the interface genuinely quantized rather
than merely trained through a quantizer. These tests pin that contract and guard
the pre-existing ``fsq``/``FSQQuantizer`` paths against regression, because the
qualified ``latent_skill`` oracle checkpoint depends on them.
"""

from __future__ import annotations

import pytest
import torch

from rlopt.agent.hl_skill_diffsr import HighLevelSkillDiffSRConfig
from rlopt.agent.hl_skill_encoder import (
    FSQQuantizer,
    FSQSkillEncoder,
    SkillLatentSpec,
    SONICFSQSkillEncoder,
    build_skill_encoder,
)

# gear_sonic all_mlp_v1.yaml: num_fsq_levels=32, max_num_tokens=2 -> 64 values.
SONIC_LEVELS: tuple[int, ...] = (32,) * 64
STATE_DIM = 93
WINDOW_STEPS = 10


def _build(levels: tuple[int, ...], z_dim: int | None = None):
    return build_skill_encoder(
        state_dim=STATE_DIM,
        window_steps=WINDOW_STEPS,
        z_dim=len(levels) if z_dim is None else z_dim,
        hidden_dims=(256, 128),
        spec=SkillLatentSpec(latent_mode="sonic_fsq", sonic_fsq_levels=levels),
    )


def _batch(batch_size: int = 8):
    torch.manual_seed(0)
    return (
        torch.randn(batch_size, STATE_DIM),
        torch.randn(batch_size, WINDOW_STEPS, STATE_DIM),
    )


def test_factory_builds_sonic_fsq_encoder():
    assert isinstance(_build(SONIC_LEVELS), SONICFSQSkillEncoder)


def test_command_width_equals_token_width():
    """The published command is the token itself, not a projection of it."""
    encoder = _build(SONIC_LEVELS)
    z, _, _ = encoder.encode(*_batch(), deterministic=True)
    assert z.shape == (8, len(SONIC_LEVELS))


def test_command_lies_on_normalized_lattice():
    """Matches gear_sonic's LATENT_INITIAL_MOTION_TOKEN: exact multiples of 1/16."""
    encoder = _build(SONIC_LEVELS)
    z, _, _ = encoder.encode(*_batch(), deterministic=True)
    half = SONIC_LEVELS[0] // 2
    scaled = z * half
    assert torch.allclose(scaled, scaled.round(), atol=1e-5)
    assert bool((z.abs() <= 1.0 + 1e-6).all())


def test_lattice_holds_for_odd_and_mixed_levels():
    for levels in [(5,) * 8, (8, 8, 8, 5, 5)]:
        encoder = _build(levels)
        z, _, _ = encoder.encode(*_batch(), deterministic=True)
        half = torch.tensor([max(level // 2, 1) for level in levels], dtype=z.dtype)
        scaled = z * half
        assert torch.allclose(scaled, scaled.round(), atol=1e-5), levels
        assert bool((z.abs() <= 1.0 + 1e-6).all()), levels


def test_no_learned_projection_at_the_command_boundary():
    encoder = _build(SONIC_LEVELS)
    assert isinstance(encoder.code_to_latent, torch.nn.Identity)


def test_straight_through_gradient_reaches_the_trunk():
    """Rounding must not block gradient: this is what co-training depends on."""
    encoder = _build(SONIC_LEVELS)
    z, _, _ = encoder.encode(*_batch(), deterministic=True)
    z.pow(2).mean().backward()
    grads = [
        param.grad
        for param in encoder.net.parameters()
        if param.grad is not None and bool(param.grad.abs().sum() > 0)
    ]
    assert grads, "no gradient reached the encoder trunk through the quantizer"


def test_regularizer_is_zero():
    """FSQ needs no commitment/KL term; reg_coeff must have nothing to scale."""
    _, reg, _ = _build(SONIC_LEVELS).encode(*_batch(), deterministic=True)
    assert float(reg.detach()) == 0.0


def test_z_dim_must_equal_token_width():
    with pytest.raises(ValueError, match="z_dim must equal"):
        _build(SONIC_LEVELS, z_dim=256)


def test_sonic_token_space_capacity():
    """64 dims x 32 levels = 2**320; a flat int64 code index cannot exist."""
    encoder = _build(SONIC_LEVELS)
    assert encoder.fsq.codebook_size == 32**64
    assert not encoder.fsq.flat_code_supported
    # Falls back to per-dim level indices, pooled by the usage metrics.
    assert encoder.num_codes == 32


# --------------------------------------------------------------------------- #
# Regression guards: the pre-existing paths must be untouched.
# --------------------------------------------------------------------------- #
def test_fsq_quantizer_still_returns_the_unnormalized_integer_grid():
    quantizer = FSQQuantizer((32,) * 4)
    torch.manual_seed(0)
    z_q, _ = quantizer(torch.randn(16, 4) * 5.0)
    assert torch.allclose(z_q, z_q.round(), atol=1e-5)
    assert float(z_q.abs().max()) > 1.0


def test_fsq_encoder_still_projects_to_z_dim():
    encoder = build_skill_encoder(
        state_dim=STATE_DIM,
        window_steps=WINDOW_STEPS,
        z_dim=256,
        hidden_dims=(256, 128),
        spec=SkillLatentSpec(latent_mode="fsq", fsq_levels=(8, 8, 8, 5, 5)),
    )
    assert isinstance(encoder, FSQSkillEncoder)
    assert isinstance(encoder.code_to_latent, torch.nn.Linear)
    z, _, _ = encoder.encode(*_batch(), deterministic=True)
    assert z.shape == (8, 256)


# --------------------------------------------------------------------------- #
# Config plumbing.
# --------------------------------------------------------------------------- #
def _config(**overrides) -> HighLevelSkillDiffSRConfig:
    """Validation lives in an explicit ``validate()``, so call it here."""
    base = {
        "horizon_steps": WINDOW_STEPS,
        "latent_mode": "sonic_fsq",
        "z_dim": len(SONIC_LEVELS),
        "sonic_fsq_levels": SONIC_LEVELS,
    }
    base.update(overrides)
    config = HighLevelSkillDiffSRConfig(**base)
    config.validate()
    return config


def test_config_rejects_z_dim_mismatch():
    with pytest.raises(ValueError, match="z_dim must equal len"):
        _config(z_dim=256)


def test_config_projects_levels_into_the_encoder_spec():
    spec = _config().latent_spec()
    assert spec.latent_mode == "sonic_fsq"
    assert spec.sonic_fsq_levels == SONIC_LEVELS


def test_config_round_trips_through_dict():
    restored = HighLevelSkillDiffSRConfig.from_dict(_config().to_dict())
    assert restored.sonic_fsq_levels == SONIC_LEVELS
    assert restored.latent_mode == "sonic_fsq"


def test_config_leaves_other_latent_modes_alone():
    """sonic_fsq_levels must not constrain z_dim for the pre-existing modes."""
    config = HighLevelSkillDiffSRConfig(
        horizon_steps=WINDOW_STEPS, latent_mode="fsq", z_dim=256
    )
    config.validate()
    assert config.latent_spec().latent_mode == "fsq"
