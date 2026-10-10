# ruff: noqa: E402 -- filter upstream import deprecations before importing TorchRL.
"""FlashSAC agent shell: config checks, update schedule, CPU/GPU end-to-end smoke."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script_method` is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
    category=DeprecationWarning,
)
import math
from pathlib import Path

import pytest
import torch
from tensordict.nn import InteractionType, set_interaction_type
from torchrl.data import Unbounded
from torchrl.envs import (
    Compose,
    GymEnv,
    RewardSum,
    SerialEnv,
    StepCounter,
    TransformedEnv,
)

from rlopt.agent import FlashSAC, FlashSACRLOptConfig
from rlopt.agent.flashsac.agent import ARCH_TAG, update_rate, updates_due
from rlopt.agent.flashsac.config import FlashSACConfig

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The env passed to Collector is missing transforms.*:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Synchronization debug mode is a prototype.*:UserWarning"
    ),
]

NUM_ENVS = 4
CUDA = torch.cuda.is_available()


def _make_env(device: str = "cpu", num_envs: int = NUM_ENVS) -> TransformedEnv:
    return TransformedEnv(
        SerialEnv(
            num_envs,
            lambda: GymEnv("Pendulum-v1", device=device),
            device=device,
        ),
        Compose(RewardSum(), StepCounter(500)),
    )


def _make_cfg(
    tmp_path: Path,
    *,
    device: str = "cpu",
    total_frames: int = NUM_ENVS * 500,
    num_envs: int = NUM_ENVS,
) -> FlashSACRLOptConfig:
    cfg = FlashSACRLOptConfig()
    cfg.device = cfg.env.device = device
    cfg.env.num_envs = num_envs
    cfg.env.env_name = "Pendulum-v1"
    cfg.logger.backend = ""
    cfg.logger.log_dir = str(tmp_path)
    cfg.policy.input_keys = ["observation"]
    cfg.collector.frames_per_batch = num_envs
    cfg.collector.total_frames = total_frames
    cfg.save_interval = 800
    fs = cfg.flashsac
    fs.batch_size = 64
    fs.rho = 2.0
    fs.warmup_transitions = 100
    fs.replay.capacity_steps = 256
    fs.log_interval_steps = 100
    fs.learner.actor_dim = 32
    fs.learner.critic_dim = 32
    fs.learner.amp_dtype = "fp32"
    return cfg


def _checkpoints(agent: FlashSAC) -> list[Path]:
    return sorted(
        agent.log_dir.glob("**/model_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("flashsac_train")
    cfg = _make_cfg(tmp_path)
    agent = FlashSAC(_make_env(), cfg)
    agent.train()
    return agent, cfg


# ----------------------------------------------------------------- config checks
@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("rho", 0.0, "rho must be positive"),
        ("rho", float("inf"), "rho must be positive"),
        ("batch_size", 0, "batch_size must be >= 1"),
        ("warmup_mode", "gauss", "warmup_mode"),
        ("warmup_transitions", -1, "warmup_transitions"),
        ("action_bound", -1.0, "action_bound entries"),
        ("action_bound", [1.0, 0.0], "action_bound entries"),
        ("log_interval_steps", 0, "log_interval_steps"),
        ("noise_max_repeat", 0, "noise_max_repeat"),
    ],
)
def test_config_rejects_bad_scalar(field, value, match):
    cfg = FlashSACConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=match):
        cfg.validate(NUM_ENVS)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_step", 0, "n_step must be >= 1"),
        ("capacity_steps", 4, "below 2\\*\\(n_step\\+1\\)"),
        ("obs_dtype", "int8", "obs_dtype"),
        ("pool_fraction", 0.0, "pool_fraction"),
        ("terminal_obs", "both", "terminal_obs"),
    ],
)
def test_config_rejects_bad_replay(field, value, match):
    cfg = FlashSACConfig()
    setattr(cfg.replay, field, value)
    with pytest.raises(ValueError, match=match):
        cfg.validate(NUM_ENVS)


def test_config_rejects_ring_smaller_than_batch():
    cfg = FlashSACConfig()
    cfg.batch_size = 8192
    cfg.replay.capacity_steps = 16
    with pytest.raises(ValueError, match="holds 64 slots"):
        cfg.validate(NUM_ENVS)


def test_config_default_is_valid_and_collector_defaults():
    FlashSACConfig().validate(1024)
    cfg = FlashSACRLOptConfig()
    assert cfg.collector.init_random_frames == 0
    assert cfg.collector.frames_per_batch == 1
    assert cfg.compile.compile is False


def test_agent_rejects_bad_config_at_construction(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.flashsac.rho = -1.0
    with pytest.raises(ValueError, match="rho must be positive"):
        FlashSAC(_make_env(), cfg)


# --------------------------------------------------------------- update schedule
@pytest.mark.parametrize(
    ("rho", "n_envs", "batch"),
    [
        (4.0, 1024, 2048),  # U = 2
        (4.0, 4096, 2048),  # U = 8
        (4.0, 20_000, 8192),  # U = 9.765625
        (16.0, 20_000, 8192),  # U = 39.0625
        (1.0, 16, 256),  # U = 0.0625: one update per 16 steps
        (0.7, 100, 64),  # U = 1.09375: non-dyadic rho
    ],
)
def test_update_schedule_floor_rule_never_drifts(rho, n_envs, batch):
    rate = update_rate(rho, n_envs, batch)
    steps = 10_000
    total = 0
    lo = math.floor(float(rate))
    hi = math.ceil(float(rate))
    for t in range(steps):
        due = updates_due(t, rate)
        assert lo <= due <= hi
        total += due
        # The running total equals floor((t+1) * U) exactly at every step.
        assert total == ((t + 1) * rate.numerator) // rate.denominator
    assert total == math.floor(steps * rate)
    # On average U per step (no drift): within one update over the whole run.
    assert abs(total / steps - float(rate)) <= 1.0 / steps


# --------------------------------------------------------------- end-to-end CPU
def test_end_to_end_cpu_smoke(trained):
    agent, cfg = trained
    steps = cfg.collector.total_frames // NUM_ENVS
    assert agent.data_buffer.filled_rows == min(steps, 256)
    # warm-up 100 transitions = 25 steps; updates run on the remaining steps.
    train_steps = steps - 100 // NUM_ENVS
    rate = update_rate(cfg.flashsac.rho, NUM_ENVS, cfg.flashsac.batch_size)
    expected = (train_steps * rate.numerator) // rate.denominator
    assert agent.learner.update_count == expected
    assert expected > 50
    for param in agent.learner.parameters():
        assert torch.isfinite(param).all()
    ckpts = _checkpoints(agent)
    assert [int(p.stem.rsplit("_", 1)[1]) for p in ckpts] == [800, 1600, 2000]


def test_collector_uses_the_learner_actor(trained):
    agent, _ = trained
    assert agent.collector.policy is agent.policy
    assert agent.collector_policy is agent.policy
    assert agent.policy.module.actor is agent.learner.actor
    assert agent.policy.module.action_bound.data_ptr() == (
        agent.learner.action_bound.data_ptr()
    )


def test_checkpoint_contents(trained):
    agent, _ = trained
    path = _checkpoints(agent)[-1]
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert "vec_norm_msg" not in data
    assert data["cumulative_env_frames"] == 2000
    meta = data["policy_meta"]
    assert meta["arch"] == ARCH_TAG
    assert meta["in_keys"] == [["observation"]]
    assert meta["squash"] == "tanh"
    assert meta["action_bound"] == [2.0]  # derived from the Pendulum action spec
    state = data["policy_state_dict"]
    assert "module.action_bound" in state
    # BatchNorm running statistics are in the policy state dict.
    assert any(k.endswith("running_mean") for k in state)
    assert any(k.endswith("running_var") for k in state)
    assert data["learner_state_dict"]["update_count"] == agent.learner.update_count


def test_reload_reproduces_actions(trained, tmp_path):
    agent, _ = trained
    path = _checkpoints(agent)[-1]
    cfg2 = _make_cfg(tmp_path)
    fresh = FlashSAC(_make_env(), cfg2)
    obs = torch.randn(7, 3)
    assert not torch.allclose(fresh.predict(obs), agent.predict(obs))
    fresh.load_model(str(path))
    expected = agent.predict(obs)
    torch.testing.assert_close(fresh.predict(obs), expected, rtol=0, atol=0)
    # deterministic action = bound * tanh(mean)
    mean, _ = agent.learner.actor.mean_and_std(obs, training=False)
    torch.testing.assert_close(expected, 2.0 * torch.tanh(mean))
    # The collector policy, under DETERMINISTIC exploration, gives the same action.
    from tensordict import TensorDict

    with set_interaction_type(InteractionType.DETERMINISTIC):
        out = fresh.collector_policy(TensorDict({"observation": obs}, batch_size=[7]))
    torch.testing.assert_close(out["action"], expected, rtol=0, atol=0)
    # BatchNorm buffers and learner counters came back.
    for key, value in agent.policy.state_dict().items():
        torch.testing.assert_close(fresh.policy.state_dict()[key], value)
    assert fresh.learner.update_count == agent.learner.update_count
    assert fresh._resume_frame_offset == 2000


def test_policy_only_strict_load_matches_dev_evaluator_contract(trained, tmp_path):
    """The dev evaluators strict-load ``policy_state_dict`` into ``agent.policy``."""
    agent, _ = trained
    data = torch.load(_checkpoints(agent)[-1], map_location="cpu", weights_only=False)
    fresh = FlashSAC(_make_env(), _make_cfg(tmp_path))
    incompatible = fresh.policy.load_state_dict(data["policy_state_dict"], strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    obs = torch.randn(5, 3)
    torch.testing.assert_close(fresh.predict(obs), agent.predict(obs), rtol=0, atol=0)


def test_load_model_rejects_wrong_checkpoint(trained, tmp_path):
    agent, _ = trained
    data = torch.load(_checkpoints(agent)[-1], map_location="cpu", weights_only=False)
    fresh = FlashSAC(_make_env(), _make_cfg(tmp_path))
    bad = tmp_path / "bad_arch.pt"
    data["policy_meta"] = {**data["policy_meta"], "in_keys": [["other"]]}
    torch.save(data, bad)
    with pytest.raises(ValueError, match="in_keys"):
        fresh.load_model(str(bad))
    not_flash = tmp_path / "ppo_like.pt"
    torch.save({"policy_state_dict": {}}, not_flash)
    with pytest.raises(ValueError, match="policy_meta"):
        fresh.load_model(str(not_flash))


def test_resume_keeps_cumulative_env_frames(trained, tmp_path):
    agent, cfg = trained
    first = _checkpoints(agent)[-1]  # model_step_2000
    cfg2 = _make_cfg(tmp_path, total_frames=2400)
    resumed = FlashSAC(_make_env(), cfg2)
    resumed.load_model(str(first))
    resumed.train()
    frames = [int(p.stem.rsplit("_", 1)[1]) for p in _checkpoints(resumed)]
    assert frames[-1] == 2400
    last = torch.load(_checkpoints(resumed)[-1], map_location="cpu", weights_only=False)
    assert last["cumulative_env_frames"] == 2400
    # 100 more env steps ran, not 500: the budget continued from 2000.
    assert resumed.learner.update_count > agent.learner.update_count
    steps_after = (2400 - 2000) // NUM_ENVS
    rate = update_rate(cfg.flashsac.rho, NUM_ENVS, cfg.flashsac.batch_size)
    # The ring was empty at resume: warm-up (100 transitions, with the loaded policy)
    # precedes updates again.
    train_steps = steps_after - 100 // NUM_ENVS
    gained = resumed.learner.update_count - agent.learner.update_count
    assert gained == (train_steps * rate.numerator) // rate.denominator


def test_resume_does_not_use_random_warmup(trained, tmp_path):
    agent, _ = trained
    resumed = FlashSAC(_make_env(), _make_cfg(tmp_path))
    assert resumed._random_phase() is True
    resumed.load_model(str(_checkpoints(agent)[-1]))
    assert resumed._random_phase() is False


def test_interrupt_writes_final_checkpoint(tmp_path):
    cfg = _make_cfg(tmp_path, total_frames=NUM_ENVS * 200)
    cfg.save_interval = 10_000_000  # no periodic checkpoint in this run
    agent = FlashSAC(_make_env(), cfg)
    original = agent.collect
    calls = {"n": 0}

    def interrupted(run, idx):
        calls["n"] += 1
        if calls["n"] == 31:
            raise KeyboardInterrupt
        return original(run, idx)

    agent.collect = interrupted  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        agent.train()
    frames = [int(p.stem.rsplit("_", 1)[1]) for p in _checkpoints(agent)]
    assert frames == [30 * NUM_ENVS]
    data = torch.load(_checkpoints(agent)[0], map_location="cpu", weights_only=False)
    assert data["cumulative_env_frames"] == 30 * NUM_ENVS


def test_warmup_actions_are_random_then_policy(tmp_path):
    cfg = _make_cfg(tmp_path, total_frames=NUM_ENVS * 40)
    cfg.flashsac.warmup_transitions = 80  # 20 steps
    agent = FlashSAC(_make_env(), cfg)
    run = agent.init_metadata()
    for i in range(40):
        it = agent.collect(run, i)
        agent.prepare(it, run)
        agent.iterate(it, run)
    assert agent._random_phase() is False
    assert agent._gate_open is True
    assert agent.data_buffer.filled_rows == 40


# --------------------------------------------------------------- validate_training
def test_validate_rejects_unbounded_action_spec_without_bound(tmp_path):
    class Unb(FlashSAC):
        def _action_spec(self):
            return Unbounded(shape=(1,))

    cfg = _make_cfg(tmp_path)
    cfg.flashsac.action_bound = None
    with pytest.raises(ValueError, match="unbounded"):
        Unb(_make_env(), cfg)
    cfg.flashsac.action_bound = 3.0
    agent = Unb(_make_env(), cfg)  # an explicit bound is accepted
    agent.validate_training()
    assert agent.learner.action_bound.tolist() == [3.0]


def test_action_bound_vector_and_shape_check(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.flashsac.action_bound = [1.5]
    agent = FlashSAC(_make_env(), cfg)
    assert agent.learner.action_bound.tolist() == [1.5]
    cfg = _make_cfg(tmp_path)
    cfg.flashsac.action_bound = [1.0, 2.0]
    with pytest.raises(ValueError, match="shape"):
        FlashSAC(_make_env(), cfg)


def test_validate_training_errors(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    agent = FlashSAC(_make_env(), cfg)
    agent.validate_training()

    agent.config.collector.frames_per_batch = 1
    with pytest.raises(ValueError, match="frames_per_batch"):
        agent.validate_training()
    agent.config.collector.frames_per_batch = NUM_ENVS

    agent.config.collector.total_frames = 100
    with pytest.raises(ValueError, match="no update"):
        agent.validate_training()
    agent.config.collector.total_frames = 2000

    agent.config.compile.compile = True
    with pytest.raises(ValueError, match="compile_mode"):
        agent.validate_training()
    agent.config.compile.compile = False

    agent.config.flashsac.learner.compile_mode = "default"
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    with pytest.raises(ValueError, match="TORCHDYNAMO_DISABLE"):
        agent.validate_training()
    agent.config.flashsac.learner.compile_mode = "none"

    agent.config.flashsac.learner.use_cuda_graph = True
    with pytest.raises(ValueError, match="CUDA device"):
        agent.validate_training()
    agent.config.flashsac.learner.use_cuda_graph = False

    agent.config.policy.input_keys = ["missing_key"]
    with pytest.raises(ValueError, match="missing_key"):
        agent.validate_training()


def test_construction_rejects_missing_input_key(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.policy.input_keys = ["nope"]
    with pytest.raises(ValueError, match="nope"):
        FlashSAC(_make_env(), cfg)


# ----------------------------------------------------------------- host syncs
@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_no_host_sync_in_prepare_and_iterate(tmp_path):
    """prepare() and iterate() must not synchronise with the host."""
    cfg = _make_cfg(tmp_path, device="cuda:0")
    cfg.flashsac.warmup_transitions = 16
    cfg.flashsac.learner.amp_dtype = "bf16"  # GradScaler (fp16) reads found_inf
    agent = FlashSAC(_make_env("cuda:0"), cfg)
    run = agent.init_metadata()
    # Warm up outside the checked region: lazy allocations, optimizer state.
    for i in range(12):
        it = agent.collect(run, i)
        agent.prepare(it, run)
        agent.iterate(it, run)
    assert agent._gate_open
    torch.cuda.synchronize()
    for i in range(12, 24):
        it = agent.collect(run, i)  # the env step may sync (CPU gym): not checked
        torch.cuda.set_sync_debug_mode("error")
        try:
            agent.prepare(it, run)
            agent.iterate(it, run)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    # The metric flush is the one allowed copy.
    metrics = agent._flush_metrics(run)
    assert math.isfinite(metrics["train/critic/loss"])
    assert 0.0 <= metrics["health/c51_edge_mass"] <= 1.0
    assert 0.0 <= metrics["health/action_saturation"] <= 1.0
    assert metrics["health/updates_per_env_step_target"] == pytest.approx(
        cfg.flashsac.rho * NUM_ENVS / cfg.flashsac.batch_size
    )


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_cuda_graph_update_path(tmp_path):
    cfg = _make_cfg(tmp_path, device="cuda:0", total_frames=NUM_ENVS * 120)
    cfg.flashsac.warmup_transitions = 16
    cfg.flashsac.learner.amp_dtype = "bf16"
    cfg.flashsac.learner.use_cuda_graph = True
    agent = FlashSAC(_make_env("cuda:0"), cfg)
    agent.train()
    assert agent._graph_ready
    assert agent.learner.update_count > 0
    for param in agent.learner.parameters():
        assert torch.isfinite(param).all()


def test_metrics_flush_reports_health_keys(trained):
    agent, _ = trained
    # trained fixture already flushed; run one more flush on a fresh window.
    run = agent.init_metadata()
    agent._env_acc.zero_()
    metrics = agent._flush_metrics(run)
    for key in (
        "health/c51_edge_mass",
        "health/action_saturation",
        "health/replay_age_rows",
        "health/bn_gap",
        "health/actor_bn_running_mean_norm",
        "health/critic_bn_running_var_norm",
        "health/replay_pool_miss",
        "health/final_obs_missing",
        "replay/rows",
    ):
        assert key in metrics, key
    assert metrics["health/replay_pool_miss"] == 0.0


def test_zero_terminal_obs_is_counted_in_next_mode(tmp_path):
    """``health/final_obs_missing`` guards the ``next`` terminal mode too.

    Regression guard for the dropped-VecGymEnvTransform wrapper bug, where the
    terminal observation of done rows arrived as zeros in ``next``.
    """
    cfg = _make_cfg(tmp_path)
    agent = FlashSAC(_make_env(), cfg)
    run = agent.init_metadata()
    it = agent.collect(run, 0)
    r = it.rollout
    done = torch.zeros_like(r.get(("next", "done")))
    done[0] = True
    r.set(("next", "done"), done)
    r.set(("next", "truncated"), done.clone())
    r.set(("next", "terminated"), torch.zeros_like(done))
    obs = r.get(("next", "observation")).clone()
    obs[0] = 0.0
    r.set(("next", "observation"), obs)
    agent.prepare(it, run)
    assert agent._terminal_mode == "next"
    metrics = agent._flush_metrics(run)
    assert metrics["health/final_obs_missing"] == 1.0
    agent.collector.shutdown()


# --------------------------------------------------------------------- warm-up
def _warmup_actions(mode: str, bound: list[float], n: int = 200_000):
    from rlopt.agent.flashsac.agent import FlashSACPolicyNet

    # The actor is never called in the random phase.
    net = FlashSACPolicyNet(
        torch.nn.Identity(), torch.tensor(bound), None, warmup_mode=mode
    )
    net.collecting = True
    net.random_actions = True
    torch.manual_seed(0)
    with set_interaction_type(InteractionType.RANDOM):
        return net(torch.zeros(n, 3))


def test_normal_warmup_is_clipped_unit_normal_in_env_units():
    """DV3: N(0, 1) per joint in env units, clipped to [-B_j, B_j]."""
    bound = [0.5, 3.0, 24.5]
    action, tanh_a = _warmup_actions("normal", bound)
    b = torch.tensor(bound)
    assert bool((action.abs() <= b).all())
    torch.testing.assert_close(tanh_a, action / b)  # stored tanh-space action
    assert bool((tanh_a.abs() <= 1.0).all())
    # Narrow joint: P(|z| > 0.5) = 0.617 of the samples sit on the bound.
    clipped = (action[:, 0].abs() == 0.5).float().mean().item()
    assert clipped == pytest.approx(0.617, abs=0.01)
    # Wide joints: the unit normal is untouched (std 1, not B / sqrt(3)).
    assert action[:, 2].std().item() == pytest.approx(1.0, abs=0.01)
    assert action[:, 2].abs().max().item() < 7.0
    # Clipping at 3 sigma moves 0.27 percent of the mass onto the bound.
    assert (action[:, 1].abs() == 3.0).float().mean().item() == pytest.approx(
        0.0027, abs=0.001
    )


def test_uniform_warmup_spans_the_bound():
    bound = [0.5, 24.5]
    action, tanh_a = _warmup_actions("uniform", bound)
    assert action[:, 1].std().item() == pytest.approx(24.5 / math.sqrt(3), rel=0.02)
    assert bool((tanh_a.abs() <= 1.0).all())


def test_warmup_mode_reaches_the_collector(tmp_path):
    cfg = _make_cfg(tmp_path, total_frames=NUM_ENVS * 10)
    cfg.flashsac.warmup_mode = "normal"
    cfg.flashsac.action_bound = 24.5
    agent = FlashSAC(_make_env(), cfg)
    assert agent.policy_net.warmup_mode == "normal"
    run = agent.init_metadata()
    it = agent.collect(run, 0)
    action = it.rollout.get("action")
    torch.testing.assert_close(it.rollout.get("action_tanh"), action / 24.5)
    assert action.abs().max().item() < 6.0  # uniform would reach ~24
    agent.collector.shutdown()


# ------------------------------------------------------------ clocks and timers
def test_checkpoint_timer_starts_after_the_cold_start(tmp_path):
    """``checkpoint_interval_s`` must not fire just because the first step was slow."""
    import time

    cfg = _make_cfg(tmp_path, total_frames=NUM_ENVS * 20)
    cfg.save_interval = 10_000_000
    cfg.flashsac.warmup_transitions = NUM_ENVS * 100  # no updates: a fast run
    cfg.collector.total_frames = NUM_ENVS * 20
    cfg.flashsac.checkpoint_interval_s = 1.0
    agent = FlashSAC(_make_env(), cfg)
    original = agent.collect

    def slow_first(run, idx):
        if idx == 0:
            time.sleep(1.2)  # a cold start longer than the interval
        return original(run, idx)

    agent.collect = slow_first  # type: ignore[method-assign]
    # validate_training requires total_frames > warm-up; skip it for this probe.
    agent.validate_training = lambda: None  # type: ignore[method-assign]
    start = time.monotonic()
    agent.train()
    assert time.monotonic() - start < 1.2 + 0.9, "run too slow for this probe"
    # Only the end-of-run checkpoint (save_interval > 0), no timer checkpoint.
    assert [int(p.stem.rsplit("_", 1)[1]) for p in _checkpoints(agent)] == [
        NUM_ENVS * 20
    ]
    assert agent._train_clock_start is not None
    assert agent._train_clock_start >= start + 1.2


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_flush_reports_learner_time_and_memory(tmp_path):
    cfg = _make_cfg(tmp_path, device="cuda:0", total_frames=NUM_ENVS * 60)
    cfg.flashsac.warmup_transitions = 16
    cfg.flashsac.learner.amp_dtype = "bf16"
    agent = FlashSAC(_make_env("cuda:0"), cfg)
    run = agent.init_metadata()
    for i in range(30):
        it = agent.collect(run, i)
        agent.prepare(it, run)
        agent.iterate(it, run)
    assert agent._learn_events_used > 0
    metrics = agent._flush_metrics(run)
    assert metrics["time/learner_ms_per_update"] > 0.0
    assert metrics["perf/cuda_max_memory_allocated_gib"] > 0.0
    assert "time/train_elapsed_s" in metrics
    assert metrics["train/cumulative_env_frames"] == NUM_ENVS * 30
    # The window is drained: the next flush without updates has no timing.
    assert agent._learn_events_used == 0
    assert "time/learner_ms_per_update" not in agent._flush_metrics(run)
    agent.collector.shutdown()


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.skipif(
    __import__("os").environ.get("TORCHDYNAMO_DISABLE") == "1",
    reason="compile is disabled by TORCHDYNAMO_DISABLE",
)
def test_compiled_graph_fp16_update_path_has_no_host_sync(tmp_path):
    """The G1 recipe path: compile 'default' + CUDA graphs + fp16 GradScaler."""
    cfg = _make_cfg(tmp_path, device="cuda:0", total_frames=NUM_ENVS * 120)
    cfg.flashsac.warmup_transitions = 16
    cfg.flashsac.learner.amp_dtype = "fp16"
    cfg.flashsac.learner.compile_mode = "default"
    cfg.flashsac.learner.use_cuda_graph = True
    agent = FlashSAC(_make_env("cuda:0"), cfg)
    agent.validate_training()
    run = agent.init_metadata()
    # Warm up outside the checked region: compile, graph capture, lazy state.
    for i in range(12):
        it = agent.collect(run, i)
        agent.prepare(it, run)
        agent.iterate(it, run)
    assert agent._graph_ready
    torch.cuda.synchronize()
    before = agent.learner.update_count
    for i in range(12, 30):
        it = agent.collect(run, i)
        torch.cuda.set_sync_debug_mode("error")
        try:
            agent.prepare(it, run)
            agent.iterate(it, run)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert agent.learner.update_count > before
    metrics = agent._flush_metrics(run)
    assert math.isfinite(metrics["train/critic/loss"])
    assert metrics["time/learner_ms_per_update"] > 0.0
    for param in agent.learner.parameters():
        assert torch.isfinite(param).all()
    agent.collector.shutdown()
