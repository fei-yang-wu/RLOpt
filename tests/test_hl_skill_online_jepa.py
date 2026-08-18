"""Online finetuning of a ``jepa_ntp`` skill encoder.

Before 2026-08-17 the online path computed one loss only -- the DiffSR
endpoint head -- so pairing online finetuning with a JEPA-pretrained encoder
was refused outright. Refusing was right: finetuning an encoder under an
objective it was never trained on changes what its token means, silently.
These tests pin the wiring that lets the SAME objective continue online:

* the pretrain heads (EMA target encoder, predictor, energy pair) are restored
  from the checkpoint rather than re-initialized;
* the loss reads adjacent chunk PAIRS (2H frames), for expert and achieved
  windows alike;
* the EMA target moves only after an optimizer step, toward the online encoder.
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
)

# jepa_ntp re-anchors the next chunk in heading frame, so the macro state is
# the 38-wide root_qpos frame; anything else is refused at construction.
STATE_DIM = 38
HORIZON = 2
Z_DIM = 4


class _FakeEnv:
    """Serves macro batches; records the horizons it was asked for."""

    def __init__(self) -> None:
        # `sample_achieved_chunk_windows` has no legacy-name mapping, so the
        # env must present itself as the imitation interface, exactly as the
        # real ImitationRLEnv does.
        self.imitation_interface = self
        self.expert_horizons: list[int] = []
        self.achieved_horizons: list[int] = []
        self.achieved_available = True

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
        del env_ids, state_history_steps
        return self._macro_batch(4, int(horizon_steps))

    def sample_achieved_chunk_windows(
        self, batch_size: int, horizon_steps: int
    ) -> TensorDict | None:
        self.achieved_horizons.append(int(horizon_steps))
        if not self.achieved_available:
            return None
        return self._macro_batch(int(batch_size), int(horizon_steps))


def _discover(env: object, name: str):
    return getattr(env, name, None)


def _config(**overrides: object) -> HighLevelSkillDiffSRConfig:
    base = {
        "z_dim": Z_DIM,
        "horizon_steps": HORIZON,
        "encoder_window_mode": "intermediate",
        "transition_objective": "jepa_ntp",
        "jepa_loss": "sigreg_ebm",
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


def _pretrained_checkpoint(tmp_path: Path, **config_overrides: object) -> Path:
    torch.manual_seed(0)
    trainer = HighLevelSkillDiffSRTrainer(_config(**config_overrides), _FakeEnv())
    trainer.train_step()
    path = tmp_path / "skill.pt"
    trainer.save_checkpoint(path)
    return path


def _sampler(
    checkpoint: Path, env: _FakeEnv, **overrides: object
) -> FrozenHighLevelSkillCommandSampler:
    kwargs = {
        "env": env,
        "checkpoint_path": checkpoint,
        "latent_dim": Z_DIM,
        "latent_steps_min": 1,
        "latent_steps_max": 1,
        "discover_env_method": _discover,
        "horizon_steps": HORIZON,
        "finetune_enabled": True,
        "achieved_coeff": 1.0,
        "pg_coeff": 0.0,
        "offline_batch_size": 8,
        "train_diffsr": True,
    }
    kwargs.update(overrides)
    return FrozenHighLevelSkillCommandSampler(**kwargs)  # type: ignore[arg-type]


def test_jepa_heads_are_restored_not_reinitialized(tmp_path: Path) -> None:
    """Random heads would drag a 50k-update encoder in a random direction."""
    checkpoint_path = _pretrained_checkpoint(tmp_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sampler = _sampler(checkpoint_path, _FakeEnv())

    assert sampler.jepa_predictor is not None
    assert sampler.jepa_target_encoder is not None
    for name, module in (
        ("predictor", sampler.jepa_predictor),
        ("g", sampler.jepa_g),
        ("f", sampler.jepa_f),
        ("target_encoder", sampler.jepa_target_encoder),
    ):
        assert module is not None
        for key, value in module.state_dict().items():
            expected = saved["jepa_state_dict"][name][key].to(value.device)
            assert torch.equal(value, expected), f"{name}.{key} was not restored"
    # The EMA target is a reference, never optimized.
    trainable = {id(p) for p in sampler.trainable_parameters()}
    assert not any(id(p) in trainable for p in sampler.jepa_target_encoder.parameters())
    assert all(id(p) in trainable for p in sampler.jepa_predictor.parameters())


def test_online_loss_reads_chunk_pairs_and_reaches_the_heads(tmp_path: Path) -> None:
    env = _FakeEnv()
    sampler = _sampler(_pretrained_checkpoint(tmp_path), env)

    loss, metrics = sampler.compute_online_finetune_loss(
        TensorDict({}, batch_size=[]),
        latent_key="latent",
        actor_loss_fn=lambda _batch: torch.zeros(()),
    )

    assert torch.isfinite(loss)
    # Two adjacent chunks per row, expert and achieved alike.
    assert env.expert_horizons[-1] == 2 * HORIZON
    assert env.achieved_horizons[-1] == 2 * HORIZON
    assert metrics["hl_skill_achieved_loss"] != 0.0
    assert "hl_skill_jepa_ntp" in metrics

    loss.backward()
    assert sampler.jepa_predictor is not None
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in sampler.jepa_predictor.parameters()
    )
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in sampler.skill_encoder.parameters()
    )


def test_unfilled_achieved_ring_contributes_zero(tmp_path: Path) -> None:
    """A ring that has not filled must not stall or poison the update."""
    env = _FakeEnv()
    env.achieved_available = False
    sampler = _sampler(_pretrained_checkpoint(tmp_path), env)

    loss, metrics = sampler.compute_online_finetune_loss(
        TensorDict({}, batch_size=[]),
        latent_key="latent",
        actor_loss_fn=lambda _batch: torch.zeros(()),
    )

    assert torch.isfinite(loss)
    assert float(metrics["hl_skill_achieved_loss"]) == 0.0


def test_ema_target_follows_the_online_encoder_after_a_step(tmp_path: Path) -> None:
    """EMA is per finetune update, and must see post-step weights."""
    sampler = _sampler(_pretrained_checkpoint(tmp_path), _FakeEnv())
    assert sampler.jepa_target_encoder is not None
    target_parameter = next(iter(sampler.jepa_target_encoder.parameters()))
    online_parameter = next(iter(sampler.skill_encoder.parameters()))
    before = target_parameter.detach().clone()

    with torch.no_grad():
        online_parameter.add_(1.0)
    sampler.on_after_finetune_step()

    momentum = float(sampler.config.jepa_ema_momentum)
    if momentum < 1.0:
        assert not torch.equal(target_parameter, before)
        moved = (target_parameter - before).abs().mean()
        gap = (online_parameter - before).abs().mean()
        assert moved <= gap, "EMA moved further than the online weights"
    else:
        assert torch.equal(target_parameter, before)


def test_endpoint_encoder_still_uses_the_endpoint_path(tmp_path: Path) -> None:
    """The endpoint objective must keep its single-chunk sampling."""
    env = _FakeEnv()
    checkpoint_path = _pretrained_checkpoint(tmp_path, transition_objective="endpoint")
    sampler = _sampler(checkpoint_path, env)

    loss, _metrics = sampler.compute_online_finetune_loss(
        TensorDict({}, batch_size=[]),
        latent_key="latent",
        actor_loss_fn=lambda _batch: torch.zeros(()),
    )

    assert torch.isfinite(loss)
    assert env.expert_horizons[-1] == HORIZON
    assert env.achieved_horizons[-1] == HORIZON


def test_jepa_checkpoint_without_heads_is_refused(tmp_path: Path) -> None:
    checkpoint_path = _pretrained_checkpoint(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload.pop("jepa_state_dict")
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="jepa_state_dict"):
        _sampler(checkpoint_path, _FakeEnv())
