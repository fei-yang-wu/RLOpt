"""``wm.skill_bf16``: the frozen skill encoder under bf16 autocast, a latent-task keyword."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options


def config(**wm) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


def test_off_by_default() -> None:
    assert WMBPTTConfig().skill_bf16 is False
    validate_options(config(), SimpleNamespace())


def test_needs_a_task_that_took_it() -> None:
    with pytest.raises(ValueError, match="latent-goal task"):
        validate_options(config(skill_bf16=True), SimpleNamespace())
    with pytest.raises(ValueError, match="latent-goal task"):
        validate_options(config(skill_bf16=True), SimpleNamespace(skill_bf16=False))
    validate_options(config(skill_bf16=True, goal="z", goal_encoder="enc.pt"), SimpleNamespace(skill_bf16=True))


def test_latent_v3_task_takes_it() -> None:
    import inspect

    from imitation_experiments.world_model.g1_v3_task import G1V3LatentTrackingTask

    assert "skill_bf16" in inspect.signature(G1V3LatentTrackingTask.__init__).parameters
