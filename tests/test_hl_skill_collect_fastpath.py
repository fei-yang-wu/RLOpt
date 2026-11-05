"""The collection fast path skips the frozen anchor encoder only when unread.

`FrozenHighLevelSkillCommandSampler.sample_for_step` used to run the frozen
anchor encoder (`initial_skill_encoder`) on every renewed command even with
finetuning off, where its output is discarded. These tests pin that the fast
path returns the same commands without that forward, and that it still runs
whenever something can read the anchor latent: legacy finetuning, an
instance-level replacement of `_encode_current_macro_batch` (planner
adapters), or the fast path switched off.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from test_hl_skill_online_jepa import _discover, _FakeEnv, _pretrained_checkpoint

import rlopt.agent.hl_skill_diffsr as hl
from rlopt.agent.hl_skill_diffsr import FrozenHighLevelSkillCommandSampler

Z_DIM = 4
HORIZON = 2


def _sampler(
    checkpoint: Path, **overrides: object
) -> FrozenHighLevelSkillCommandSampler:
    kwargs = {
        "env": _FakeEnv(),
        "checkpoint_path": checkpoint,
        "latent_dim": Z_DIM,
        "latent_steps_min": 1,
        "latent_steps_max": 1,
        "discover_env_method": _discover,
        "horizon_steps": HORIZON,
        "finetune_enabled": False,
    }
    kwargs.update(overrides)
    return FrozenHighLevelSkillCommandSampler(**kwargs)  # type: ignore[arg-type]


def _count_anchor_forwards(sampler: FrozenHighLevelSkillCommandSampler) -> list[int]:
    calls = [0]

    def hook(*_: object) -> None:
        calls[0] += 1

    sampler.initial_skill_encoder.register_forward_hook(hook)
    return calls


def _step(sampler: FrozenHighLevelSkillCommandSampler) -> torch.Tensor:
    td = TensorDict({}, batch_size=[4])
    return sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    return _pretrained_checkpoint(tmp_path)


def test_fast_path_returns_the_same_commands_without_the_anchor_forward(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hl, "COLLECT_FASTPATH", False)
    eager = _sampler(checkpoint)
    eager_calls = _count_anchor_forwards(eager)
    eager_latents = [_step(eager) for _ in range(3)]

    monkeypatch.setattr(hl, "COLLECT_FASTPATH", True)
    fast = _sampler(checkpoint)
    fast_calls = _count_anchor_forwards(fast)
    fast_latents = [_step(fast) for _ in range(3)]

    assert eager_calls[0] == 3
    assert fast_calls[0] == 0
    for a, b in zip(eager_latents, fast_latents, strict=True):
        assert torch.equal(a, b)


def test_anchor_forward_still_runs_for_an_instance_level_encode_replacement(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hl, "COLLECT_FASTPATH", True)
    sampler = _sampler(checkpoint)
    calls = _count_anchor_forwards(sampler)
    original = sampler._encode_current_macro_batch
    seen: list[torch.Tensor | None] = []

    def adapter(env_ids: torch.Tensor):
        out = original(env_ids)
        seen.append(out[4])
        return out

    sampler._encode_current_macro_batch = adapter  # type: ignore[method-assign]
    _step(sampler)
    assert calls[0] == 1
    assert seen
    assert isinstance(seen[0], torch.Tensor)


def test_encode_keeps_the_five_tensor_contract_by_default(checkpoint: Path) -> None:
    sampler = _sampler(checkpoint)
    env_ids = torch.arange(4)
    z, state, window, _target, initial_z = sampler._encode_current_macro_batch(env_ids)
    assert isinstance(initial_z, torch.Tensor)
    assert torch.equal(
        initial_z,
        sampler.initial_skill_encoder(
            state, hl._encoder_input_window(sampler.config, window)
        ),
    )
    z2, *_rest, skipped = sampler._encode_current_macro_batch(
        env_ids, with_initial_z=False
    )
    assert skipped is None
    assert torch.equal(z, z2)
