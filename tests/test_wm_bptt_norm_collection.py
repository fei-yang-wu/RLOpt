"""``wm.norm_from_collection``: the world-model normalizers fit what collection saw.

The reference is a direct fit, in float64, of every transition of every env, with
the true next state the env held before its reset. NDM Gym does this at
``ndm_gym/train.py`` (``update_wm_norms`` after ``buffer.add``).
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from torch import Tensor

from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
ON = {"norm_from_collection": True}
NO_SYNC = {"no_sync": True, "fused_adam": True}
STEPS = 70
ENVS = 32


def collected(steps: int = STEPS, **wm: Any) -> tuple[Any, dict[str, Any], list[Tensor]]:
    """The agent after ``steps`` collection steps, and the env's true next states."""
    agent = harness.make_agent(env_cls=harness.FinalObsFakeEnv, num_envs=ENVS, **wm)
    agent.ensure_replay()
    env, truth = agent._isaac, []
    step = env.step

    def recording(action: Tensor) -> Any:
        out = step(action)
        truth.append(env.true_next_state.clone())
        return out

    env.step = recording
    obs, _ = env.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(steps):
        obs = agent._collect_step(obs, stats)
    return agent, obs, truth


def direct_fit(agent: Any, truth: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Input and increment features of every collected transition."""
    rp, t, T = agent.replay, agent.task, len(truth)
    state = rp.state[:T].flatten(0, 1)
    x = t.wm_input(state, rp.action[:T].flatten(0, 1), rp.context[:T].flatten(0, 1))
    return x.double(), t.increment(state, torch.stack(truth).flatten(0, 1)).double()


def assert_fits(norm: Any, x: Tensor) -> None:
    torch.testing.assert_close(norm.mean.double(), x.mean(0), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(norm.var.double(), x.var(0, unbiased=False), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("wm", [{}, NO_SYNC], ids=["default", "no_sync"])
def test_the_statistics_are_a_direct_fit_of_every_transition(wm: dict[str, Any]) -> None:
    agent, _, truth = collected(**ON, **wm)
    x, inc = direct_fit(agent, truth)
    assert agent.dynamics.in_norm.count == agent.dynamics.out_norm.count == STEPS * ENVS
    assert_fits(agent.dynamics.in_norm, x)
    assert_fits(agent.dynamics.out_norm, inc)


def test_the_reset_rows_would_fit_something_else() -> None:
    """Control: reading the next row, the reset state, does not give these statistics."""
    agent, _, truth = collected(**ON)
    rp, t, T = agent.replay, agent.task, len(truth)
    inc = t.increment(rp.state[: T - 1].flatten(0, 1), rp.state[1:T].flatten(0, 1)).double()
    assert not torch.allclose(agent.dynamics.out_norm.var.double(), inc.var(0, unbiased=False), rtol=0.1)


def test_a_true_terminal_transition_moves_the_statistics() -> None:
    """A failed episode's fall reaches the height increment's variance."""
    agent, _, truth = collected(**ON)
    _, inc = direct_fit(agent, truth)
    without = inc[inc[:, 2].abs() < 0.2]
    assert without.shape[0] < inc.shape[0]  # the run holds failures
    assert agent.dynamics.out_norm.var[2] > 2 * without[:, 2].var(unbiased=False)


def test_it_needs_no_terminal_ring() -> None:
    both, _, _ = collected(**ON, fit_terminal_transitions=True)
    alone, _, _ = collected(**ON)
    assert alone.replay.terminal_state is None
    assert harness.state_equal(both.dynamics.state_dict(), alone.dynamics.state_dict())


@pytest.mark.parametrize("wm", [{}, NO_SYNC], ids=["default", "no_sync"])
def test_an_update_leaves_the_normalizers_to_collection(wm: dict[str, Any]) -> None:
    agent, _, _ = collected(**ON, **wm)
    d = agent.dynamics
    before = {name: {k: v.clone() for k, v in n.state_dict().items()} for name, n in (("in", d.in_norm), ("out", d.out_norm))}
    agent._update_world_model(defaultdict(list))
    assert harness.state_equal(before["in"], d.in_norm.state_dict())
    assert harness.state_equal(before["out"], d.out_norm.state_dict())


@pytest.mark.parametrize("wm", [{}, NO_SYNC], ids=["default", "no_sync"])
def test_off_the_collection_does_not_touch_them(wm: dict[str, Any]) -> None:
    agent, _, _ = collected(**wm)
    assert agent.dynamics.in_norm.count == 0
    assert agent.dynamics.out_norm.count == 0


@pytest.mark.parametrize("wm", [{}, NO_SYNC], ids=["default", "no_sync"])
def test_the_normalizers_freeze_at_the_sample_count(wm: dict[str, Any]) -> None:
    limit = 10 * ENVS
    agent, obs, _ = collected(steps=12, norm_freeze_samples=float(limit), **ON, **wm)
    d = agent.dynamics
    stopped = float(d.out_norm.count)
    assert limit <= stopped <= limit + ENVS  # the update that crosses the limit lands
    agent._update_world_model(defaultdict(list))
    assert bool(d.out_norm.frozen)
    assert bool(d.in_norm.frozen)
    stats: dict[str, list] = defaultdict(list)
    for _ in range(4):
        obs = agent._collect_step(obs, stats)
    assert float(d.out_norm.count) == stopped


@pytest.mark.parametrize("wm", [{}, NO_SYNC], ids=["default", "no_sync"])
def test_a_non_finite_row_counts_nowhere(wm: dict[str, Any]) -> None:
    class Diverging(harness.FinalObsFakeEnv):
        def step(self, action: Tensor) -> Any:
            out = super().step(action)
            final = out[-1].get("final_obs")
            if final is not None:
                final["obs"]["wm"]["root_state"][:1, -13] = float("nan")  # the first ended episode's newest x
                self.diverged = getattr(self, "diverged", 0) + 1
            return out

    agent = harness.make_agent(env_cls=Diverging, num_envs=ENVS, **ON, **wm)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(STEPS):
        obs = agent._collect_step(obs, stats)
    d = agent.dynamics
    assert agent._isaac.diverged > 0
    assert float(d.out_norm.count) == STEPS * ENVS - agent._isaac.diverged
    assert torch.isfinite(d.out_norm.mean).all()
    assert torch.isfinite(d.out_norm.var).all()
    assert torch.isfinite(d.in_norm.var).all()


def test_a_pushed_row_counts_nowhere() -> None:
    class Pushing(harness.FinalObsFakeEnv):
        def _obs(self) -> dict[str, Any]:
            obs = super()._obs()
            obs["wm"]["push_flag"] = torch.ones_like(obs["wm"]["push_flag"])
            return obs

    agent = harness.make_agent(env_cls=Pushing, num_envs=ENVS, **ON)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(4):
        obs = agent._collect_step(obs, stats)
    assert agent.dynamics.out_norm.count == 0


def test_a_task_must_say_its_final_obs_is_post_step() -> None:
    config = WMBPTTRLOptConfig(wm=WMBPTTConfig(**ON))
    with pytest.raises(ValueError, match="norm_from_collection"):
        validate_options(config, SimpleNamespace())
    validate_options(config, SimpleNamespace(final_obs_is_post_step=True))


def test_an_env_without_final_obs_stops_the_run_at_a_log_row() -> None:
    agent = harness.make_agent(frames=harness.frames_for(6), **ON)
    with pytest.raises(RuntimeError, match="final_obs"):
        agent.train()


@pytest.mark.parametrize(
    "wm",
    [{}, NO_SYNC, {"dynamics_model": "diffsr", "diffsr_feature_dim": 16, "diffsr_embed_dim": 32}],
    ids=["default", "no_sync", "diffsr"],
)
def test_a_run_trains_with_both_options(wm: dict[str, Any]) -> None:
    agent = harness.make_agent(
        env_cls=harness.FinalObsFakeEnv, num_envs=ENVS, frames=harness.frames_for(6, num_envs=ENVS),
        fit_terminal_transitions=True, **ON, **wm,
    )
    agent.train()
    assert float(agent.dynamics.out_norm.count) == agent._frames_processed
    assert all(torch.isfinite(p).all() for p in agent.dynamics.parameters())


# ------------------------------------------------------------------ recipes run on the GPU
def census(**wm: Any) -> tuple[Any, dict[str, Any], dict[str, list]]:
    agent = harness.make_agent(device="cuda:0", num_envs=64, env_cls=harness.SyncFreeFinalObsEnv, **ON, **NO_SYNC, **wm)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(60):
        obs = agent._collect_step(obs, stats)
    return agent, obs, stats


def update(agent: Any, obs: dict[str, Any], stats: dict[str, list]) -> dict[str, Any]:
    metrics: dict[str, list] = defaultdict(list)
    for _ in range(3):
        obs = agent._collect_step(obs, stats)
    agent._update_world_model(metrics)
    agent._update_actor_critic(metrics)
    return obs


@cuda
@pytest.mark.parametrize("wm", [{}, {"fit_terminal_transitions": True}], ids=["alone", "with_terminals"])
def test_the_collection_path_does_not_synchronize(wm: dict[str, Any]) -> None:
    agent, obs, stats = census(**wm)
    obs = update(agent, obs, stats)  # first calls: cuBLAS handles, optimizer state
    torch.cuda.synchronize()
    before = float(agent.dynamics.out_norm.count)
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(2):
            obs = update(agent, obs, stats)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert float(agent.dynamics.out_norm.count) == before + 6 * 64


@cuda
def test_the_collection_path_compiles_without_recompiling() -> None:
    """Steps with and without an ended episode must share one compiled variant."""
    agent, obs, stats = census(compile=True, compile_eager=True, fit_terminal_transitions=True)
    for _ in range(2):
        obs = update(agent, obs, stats)
    with torch.compiler.set_stance("fail_on_recompile"):
        for _ in range(2):
            obs = update(agent, obs, stats)
    assert all(torch.isfinite(p).all() for p in agent.dynamics.parameters())
