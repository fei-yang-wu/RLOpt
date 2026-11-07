"""``validate_options``: every option dependency fails at construction (CPU)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness

from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options
from rlopt.agent.wm_bptt.speed_options import ActorLR

TASK = SimpleNamespace(max_step_reward=0.1)


def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


def test_the_default_config_passes() -> None:
    validate_options(config(), TASK)


REFUSED = [
    ({"actor_source": "dream"}, "actor_source"),
    ({"critic_source": "dream"}, "critic_source"),
    ({"actor_source": "real", "actor_horizon": 4}, "actor_horizon"),
    ({"actor_estimator": "magic"}, "actor_estimator"),
    ({"actor_estimator": "rebar"}, "rebar"),
    ({"actor_estimator": "score_function", "actor_source": "real"}, "score_function"),
    ({"actor_ema": -0.1}, "actor_ema"),
    ({"actor_ema": 1.0}, "actor_ema"),
    ({"actor_opt": "sgd"}, "actor_opt"),
    ({"wm_opt": "sgd"}, "wm_opt"),
    ({"wm_opt": "muon", "dynamics_model": "diffsr"}, "wm_opt=muon"),
    ({"wm_opt": "muon", "wm_weight_decay": 0.01}, "wm_weight_decay"),
    ({"wm_opt": "muon", "wm_hidden": [64]}, "wm_hidden"),
    ({"actor_opt": "muon", "actor_hidden": [64]}, "actor_hidden"),
    ({"actor_lr_mult": 0.0}, "actor_lr_mult"),
    ({"actor_lr_decay_return": 0.75, "actor_lr_decay_len": 300}, "which is 1"),
    ({"actor_lr_decay_by": 1400}, "need wm.actor_lr_decay_return"),
    ({"actor_lr_mult": 4.0, "actor_lr_decay_by": 1400, "actor_lr_decay_len": 300}, "need wm.actor_lr_decay_return"),
    ({"actor_lr_decay_len": 300}, "need wm.actor_lr_decay_return"),
    ({"actor_lr_mult": 4.0, "actor_lr_decay_return": 0.0, "actor_lr_decay_len": 300}, "actor_lr_decay_return must be"),
    ({"actor_lr_mult": 4.0, "actor_lr_decay_return": 0.75}, "actor_lr_decay_len"),
    ({"actor_lr_mult": 4.0, "actor_lr_decay_return": 0.75, "actor_lr_decay_len": 300, "actor_lr_decay_by": 0}, "actor_lr_decay_by"),
    ({"last_every_min": 5.0}, "needs wm.ckpt_light"),
    ({"ckpt_light": True, "last_every_min": -1.0}, "last_every_min"),
]


@pytest.mark.parametrize(("wm", "match"), REFUSED)
def test_refused_options(wm: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(**wm), TASK)


def test_ckpt_light_needs_periodic_checkpoints() -> None:
    cfg = config(ckpt_light=True)
    cfg.save_interval = 0
    with pytest.raises(ValueError, match="save_interval"):
        validate_options(cfg, TASK)
    cfg.save_interval = 100
    validate_options(cfg, TASK)


def test_construction_validates() -> None:
    with pytest.raises(ValueError, match="actor_source"):
        harness.make_agent(actor_source="dream")


def test_a_return_trigger_needs_a_task_that_scales_the_return() -> None:
    wm = {"actor_lr_mult": 4.0, "actor_lr_decay_return": 0.75, "actor_lr_decay_len": 300}
    validate_options(config(**wm), TASK)
    with pytest.raises(ValueError, match="max_step_reward"):
        validate_options(config(**wm), SimpleNamespace())


def test_the_g1_v3_task_scales_returns_by_the_env_step_reward_maximum() -> None:
    agent = harness.make_agent()
    assert agent.task.max_step_reward == pytest.approx(0.1)


# ------------------------------------------------------------ actor_lr_mult
def recipe_schedule() -> ActorLR:
    return ActorLR(4.0, 0.75, 300, 1400)


def test_the_multiplier_holds_then_ramps_to_one_once() -> None:
    lr = recipe_schedule()
    assert lr.on
    assert lr.at(0) == 4.0
    assert lr.observe(50, 0.5) is None
    assert lr.at(100) == 4.0
    assert "training return 0.800 >= 0.75 at iteration 100" in lr.observe(100, 0.8)
    assert lr.observe(150, 0.9) is None
    assert (lr.at(100), lr.at(250), lr.at(400), lr.at(5000)) == (4.0, 2.5, 1.0, 1.0)


def test_the_deadline_fires_without_the_return() -> None:
    lr = recipe_schedule()
    assert lr.observe(1350, 0.1) is None
    assert "deadline 1400 at iteration 1400" in lr.observe(1400, 0.1)
    late = recipe_schedule()
    assert "deadline" in late.observe(1450, None)


def test_a_missing_return_does_not_fire_the_trigger() -> None:
    lr = recipe_schedule()
    assert lr.observe(100, None) is None
    assert lr.observe(200, float("nan")) is None
    assert lr.decay_at is None


def test_the_trigger_state_round_trips_and_is_required() -> None:
    lr = recipe_schedule()
    lr.observe(100, 0.8)
    resumed = recipe_schedule()
    resumed.load_state_dict(lr.state_dict())
    assert resumed.at(250) == 2.5
    with pytest.raises(ValueError, match="actor_lr_decay_at"):
        recipe_schedule().load_state_dict({})


def test_off_is_one_at_every_iteration_and_saves_nothing() -> None:
    lr = ActorLR()
    assert not lr.on
    assert not lr.watches_return
    assert lr.at(10**6) == 1.0
    assert lr.observe(1400, 0.9) is None
    assert lr.state_dict() == {}
    lr.load_state_dict({})


def test_a_constant_multiplier_has_no_trigger_state() -> None:
    lr = ActorLR(2.0)
    assert lr.on
    assert not lr.watches_return
    assert lr.at(5000) == 2.0
    assert lr.state_dict() == {}


def actor_lrs(agent: Any) -> list[float]:
    return [g["lr"] for g in agent.actor_optim.param_groups]


@pytest.mark.parametrize("actor_opt", ["adam", "muon"])
def test_the_multiplier_reaches_only_the_actor(actor_opt: str) -> None:
    wm = {"actor_lr": 1e-3, "wm_lr": 3e-3, "critic_lr": 5e-3, "lr_cosine_iters": 10, "actor_opt": actor_opt}
    agent = harness.make_agent(actor_lr_mult=4.0, **wm)
    agent._iteration = 5
    agent._set_lrs()
    f = agent._lr_factor()
    assert all(lr == pytest.approx(4e-3 * f) for lr in actor_lrs(agent))
    assert agent.wm_optim.param_groups[0]["lr"] == 3e-3 * f
    assert agent.critic_optim.param_groups[0]["lr"] == 5e-3 * f


def track(agent: Any, rewards: list[list[float]], done: list[list[bool]]) -> None:
    for reward, ended in zip(rewards, done, strict=True):
        agent._track_episodes(torch.tensor(reward), torch.tensor(ended))


def with_return_sums(num_envs: int) -> Any:
    agent = harness.make_agent(num_envs=num_envs, actor_lr_mult=4.0, actor_lr_decay_return=0.75, actor_lr_decay_len=3)
    agent.ensure_replay()
    return agent


def test_the_return_is_the_mean_of_per_episode_ratios_averaged_over_steps() -> None:
    agent = with_return_sums(3)
    scale = agent.task.max_step_reward
    track(
        agent,
        [[0.05, 0.1, 0.02], [0.05, 0.1, 0.02], [0.08, 0.1, 0.02], [0.08, 0.1, 0.02]],
        [[False, False, False], [True, True, False], [False, False, False], [True, False, True]],
    )
    # Step 2 ends env 0 (0.10 over 2 steps) and env 1 (0.20 over 2 steps).
    first = (0.10 / (2 * scale) + 0.20 / (2 * scale)) / 2
    # Step 4 ends env 0 (0.16 over 2 steps) and env 2 (0.08 over 4 steps).
    second = (0.16 / (2 * scale) + 0.08 / (4 * scale)) / 2
    total, steps = agent._return_sums.tolist()
    assert steps == 2
    assert total == pytest.approx(first + second, rel=1e-5)
    row: dict[str, float] = {}
    agent._observe_actor_lr(row)
    assert row["train/normalized_return"] == pytest.approx((first + second) / 2, rel=1e-5)
    assert agent._return_sums.tolist() == [0.0, 0.0]


def test_a_log_row_without_an_ended_episode_reports_no_return() -> None:
    agent = with_return_sums(2)
    track(agent, [[0.1, 0.1]], [[False, False]])
    row: dict[str, float] = {}
    agent._observe_actor_lr(row)
    assert "train/normalized_return" not in row
    assert row["train/actor_lr_mult"] == 4.0


def test_the_return_trigger_fires_in_a_run_and_the_ramp_lowers_the_lr() -> None:
    agent = harness.make_agent(
        frames=harness.frames_for(9), actor_lr_mult=4.0, actor_lr_decay_return=0.05, actor_lr_decay_len=3,
        wm_warmup_iters=0, log_interval_iters=2,
    )  # fmt: skip
    agent.train()
    assert agent._actor_lr.decay_at == 2
    assert agent._actor_lr.at(agent._iteration) == 1.0
    assert actor_lrs(agent)[0] == pytest.approx(agent.config.wm.actor_lr * agent._lr_factor(), rel=1e-3)


def test_resume_restores_the_trigger(tmp_path: Path) -> None:
    wm = {"actor_lr_mult": 4.0, "actor_lr_decay_return": 0.05, "actor_lr_decay_len": 3, "log_interval_iters": 2}
    trained = harness.make_agent(frames=harness.frames_for(4), **wm)
    trained.train()
    path = tmp_path / "a.pt"
    trained.save_model(path)
    resumed = harness.make_agent(**wm)
    resumed.load_model(str(path))
    assert resumed._actor_lr.decay_at == trained._actor_lr.decay_at == 2


def test_resume_refuses_a_checkpoint_without_trigger_state(tmp_path: Path) -> None:
    trained = harness.make_agent(frames=harness.frames_for(3))
    trained.train()
    path = tmp_path / "a.pt"
    trained.save_model(path)
    agent = harness.make_agent(actor_lr_mult=4.0, actor_lr_decay_return=0.75, actor_lr_decay_len=3)
    with pytest.raises(ValueError, match="actor_lr_decay_at"):
        agent.load_model(str(path))


def test_the_off_path_allocates_nothing_and_logs_nothing() -> None:
    agent = harness.make_agent(frames=harness.frames_for(2))
    agent.train()
    assert not hasattr(agent, "_return_sums")
    assert agent._actor_lr.state_dict() == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_per_step_option_paths_do_not_synchronize() -> None:
    """Contract 6: the return sums and the average stay on the device."""
    agent = harness.make_agent(
        device="cuda:0", num_envs=4, actor_ema=0.9, actor_lr_mult=4.0, actor_lr_decay_return=0.75, actor_lr_decay_len=3
    )
    agent.ensure_replay()
    reward = torch.full((4,), 0.05, device="cuda:0")
    done = torch.tensor([True, False, True, False], device="cuda:0")
    agent.actor_avg.update()  # the copying call
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        agent._track_episodes(reward, done)
        agent.actor_avg.update()
        agent._set_lrs()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert agent._return_sums.tolist()[1] == 1.0
