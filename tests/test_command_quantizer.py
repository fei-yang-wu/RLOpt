"""The command-boundary quantizer: SONIC's lattice at the tracker input.

`latent_mode="sonic_fsq"` makes the ENCODER an FSQ bottleneck at pretrain
time. This is the other placement: a continuous encoder whose published
command is snapped to the same lattice, so the interface is quantized without
retraining the representation.
"""

from __future__ import annotations

import torch

from rlopt.agent.hl_skill_encoder import FSQQuantizer
from rlopt.agent.ipmd.ipmd import IPMDLatentLearningConfig


def test_default_is_off_and_matches_sonic_width() -> None:
    cfg = IPMDLatentLearningConfig()
    assert cfg.command_quantizer == "none"
    # SONIC publishes 64 dimensions at 32 levels.
    assert list(cfg.command_fsq_levels) == [32] * 64


def test_sonic_normalization_lands_on_multiples_of_one_sixteenth() -> None:
    """Every published value must be an exact multiple of 1/(32 // 2).

    That is the property the released SONIC token space has: each entry of
    `gear_sonic`'s LATENT_INITIAL_MOTION_TOKEN is a multiple of 1/16.
    """
    quantizer = FSQQuantizer((32,) * 64)
    codes = torch.randn(64, 64) * 3.0
    quantized, _ = quantizer(codes)
    published = quantized / 16.0
    scaled = published * 16.0
    torch.testing.assert_close(scaled, torch.round(scaled), atol=1e-5, rtol=0)
    assert float(published.min()) >= -1.0
    assert float(published.max()) <= 1.0


def test_static_input_gives_a_stable_command() -> None:
    """The property the arm depends on: a still input publishes a still command.

    NOT idempotence -- FSQ's `_bound` applies a tanh, so re-quantizing an
    already-quantized value compresses it again. The pipeline never does that:
    the quantizer always sees the raw continuous code and publishes once. What
    must hold is that repeated near-identical inputs give bit-identical output.
    """
    quantizer = FSQQuantizer((32,) * 8)
    code = torch.randn(1, 8)
    first, _ = quantizer(code)
    second, _ = quantizer(code.clone())
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_nearby_codes_collapse_to_one_lattice_point() -> None:
    """The point of the arm: small dither must vanish, not survive.

    Two codes a hair apart map to the same published value, which is what
    should remove per-step command flicker during a static pose.
    """
    quantizer = FSQQuantizer((32,) * 4)
    base = torch.zeros(1, 4)
    nudged = base + 1.0e-3
    a, _ = quantizer(base)
    b, _ = quantizer(nudged)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
