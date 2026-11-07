"""``wm.fit_terminal_transitions``: the world model fits the step into a failure.

Replay level first, on rows written by hand, then the agent on the real
``G1V3TrackingTask`` with an env that publishes ``final_obs``. The reference for
the mask is NDM Gym's formula (``ndm_gym/buffer.py``, ``sample_windows``).
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
from rlopt.agent.wm_bptt.replay import SequenceReplay, rollout_mask

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
ON = {"fit_terminal_transitions": True}
NO_SYNC = {"no_sync": True, "fused_adam": True}
EP = [0, 1, 2, 0, 1, 2, 3, 4]  # one env: the episode ends on the transition out of row 2


def filled(episode: list[int] = EP, *, capacity: int = 16, terminal_states: bool = True) -> SequenceReplay:
    """State t at row t, and the true last state 100 + t where the next row restarts."""
    rp = SequenceReplay(1, capacity, 2, 1, 1, torch.device("cpu"), terminal_states=terminal_states)
    for t, step in enumerate(episode):
        rp.add(
            torch.tensor([[float(t), 0.0]]), torch.full((1, 1), float(t)), torch.zeros(1, 1), torch.zeros(1, 1),
            torch.zeros(1, dtype=torch.long), torch.tensor([t]), torch.tensor([step]), torch.zeros(1, dtype=torch.bool),
        )
        if terminal_states and t + 1 < len(episode) and episode[t + 1] != step + 1:
            rp.set_terminal(torch.tensor([0]), torch.tensor([[100.0 + t, 0.0]]), torch.tensor([True]))
    return rp


def whole(rp: SequenceReplay, length: int = len(EP)) -> dict[str, Tensor]:
    return rp.window(torch.tensor([rp.oldest]), torch.tensor([0]), length)


# ------------------------------------------------------------------ replay
def test_off_the_window_keeps_its_rows_and_its_mask() -> None:
    rp = filled(terminal_states=False)
    assert rp.terminal_state is None
    assert rp.has_terminal is None
    win = whole(rp)
    assert "ended" not in win
    assert torch.equal(win["next_state"], win["state"][:, 1:])
    assert win["next_state"].data_ptr() == win["state"][:, 1:].data_ptr()  # a view: no copy on the default path
    assert win["valid"][0].tolist() == [True, True, False, True, True, True, True]
    assert rollout_mask(win, 7)[0].tolist() == [1, 1, 0, 0, 0, 0, 0]


def test_the_last_transition_gets_its_true_state_and_the_window_stops_after_it() -> None:
    win = whole(filled())
    assert win["next_state"][0, :, 0].tolist() == [1, 2, 102, 4, 5, 6, 7]
    assert win["ended"][0].tolist() == [False, False, True, False, False, False, False]
    assert win["valid"][0].tolist() == [True] * 7
    assert rollout_mask(win, 7)[0].tolist() == [1, 1, 1, 0, 0, 0, 0]


def test_only_the_terminal_rows_differ_from_the_default_window() -> None:
    on, off = whole(filled()), whole(filled(terminal_states=False))
    for key in off:
        if key not in ("next_state", "valid"):
            assert torch.equal(on[key], off[key]), key
    same = ~on["ended"]
    assert torch.equal(on["next_state"][same], off["next_state"][same])


def test_a_new_episode_starts_a_new_window() -> None:
    win = filled().window(torch.tensor([3]), torch.tensor([0]), 5)  # rows 3..7, one episode
    assert rollout_mask(win, 4)[0].tolist() == [1, 1, 1, 1]
    assert win["next_state"][0, :, 0].tolist() == [4, 5, 6, 7]


def test_the_mask_is_ndm_gyms_on_random_episodes() -> None:
    """Transition j counts unless an episode ended on one of 0 .. j - 1."""
    gen = torch.Generator().manual_seed(0)
    done = torch.rand(64, 9, generator=gen) < 0.3
    ours = rollout_mask({"valid": torch.ones_like(done), "ended": done}, 9)
    d = done.float()
    ndm = torch.cumprod(torch.cat((torch.ones(64, 1), 1 - d[:, :-1]), -1), -1)
    assert torch.equal(ours, ndm)


def test_an_invalid_transition_still_stops_the_window() -> None:
    win = {
        "valid": torch.tensor([[True, False, True, True]]),
        "ended": torch.tensor([[False, False, False, False]]),
    }
    assert rollout_mask(win, 4)[0].tolist() == [1, 0, 0, 0]


def test_a_stale_terminal_state_from_the_last_lap_of_the_ring_is_not_read() -> None:
    rp = filled([0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], capacity=8)
    assert rp.terminal_state[1, 0, 0] == 101.0  # slot 1, row 1, overwritten at row 9
    win = whole(rp, 8)  # rows 4..11
    assert not win["ended"].any()
    assert win["next_state"][0, :, 0].tolist() == [5, 6, 7, 8, 9, 10, 11]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_row_left_out_behaves_as_without_the_option(value: float) -> None:
    """``keep`` false: not a fit target, and its value reaches no window."""
    rp = filled()
    rp.steps = 3  # write through the row of the ended episode, as the step out of it does
    rp.set_terminal(torch.tensor([0]), torch.tensor([[value, 0.0]]), torch.tensor([False]))
    rp.steps = len(EP)
    win = whole(rp)
    assert torch.isfinite(win["next_state"]).all()
    assert win["valid"][0].tolist() == [True, True, False, True, True, True, True]
    assert rollout_mask(win, 7)[0].tolist() == [1, 1, 0, 0, 0, 0, 0]


def test_the_ring_holds_one_more_state_per_row_only_with_the_option() -> None:
    on, off = filled(), filled(terminal_states=False)
    assert on.terminal_state.shape == (*on.state.shape,)
    assert off.terminal_state is None


# ------------------------------------------------------------------ construction
def test_a_task_must_say_its_final_obs_is_post_step() -> None:
    config = WMBPTTRLOptConfig(wm=WMBPTTConfig(**ON))
    with pytest.raises(ValueError, match="final_obs"):
        validate_options(config, SimpleNamespace())
    validate_options(config, SimpleNamespace(final_obs_is_post_step=True))


def test_the_default_agent_allocates_nothing_for_it() -> None:
    agent = harness.make_agent(frames=harness.frames_for(2))
    agent.train()
    assert agent.replay.terminal_state is None
    assert agent._unpublished_ends is None


def test_an_env_without_final_obs_stops_the_run_at_a_log_row() -> None:
    agent = harness.make_agent(frames=harness.frames_for(6), **ON)
    with pytest.raises(RuntimeError, match="final_obs"):
        agent.train()


# ------------------------------------------------------------------ the agent
def collected(steps: int = 70, **wm: Any) -> tuple[Any, list[tuple[int, Tensor, Tensor]]]:
    """The agent after ``steps`` collection steps, and what the env published per row."""
    agent = harness.make_agent(env_cls=harness.FinalObsFakeEnv, num_envs=32, **ON, **wm)
    agent.ensure_replay()
    env, published = agent._isaac, []
    step = env.step

    def recording(action: Tensor) -> Any:
        out = step(action)
        final = out[-1].get("final_obs")
        if final is not None:
            state = agent.task.read_obs(final["obs"]["wm"])["state"]
            published.append((agent.replay.steps - 1, final["_env_ids"], state))
        return out

    env.step = recording
    obs, _ = env.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(steps):
        obs = agent._collect_step(obs, stats)
    return agent, published


def test_every_published_state_lands_on_its_row() -> None:
    agent, published = collected()
    rp = agent.replay
    assert len(published) > 5
    count = 0
    for row, ids, state in published:
        assert rp.has_terminal[row, ids].all()
        assert torch.equal(rp.terminal_state[row, ids], state)
        count += ids.numel()
    assert int(rp.has_terminal.sum()) == count


def test_a_window_fits_the_published_state_and_nothing_past_it() -> None:
    agent, published = collected()
    rp = agent.replay
    row, ids, state = published[len(published) // 2]
    k = 3
    tau = torch.full_like(ids, row - 1)  # the terminal transition is the window's second
    win = rp.window(tau, ids, k + 1)
    assert win["ended"][:, 1].all()
    assert torch.equal(win["next_state"][:, 1], state)
    # A failed row's true state is below the reference; the reset row is not.
    errs, mask = agent._wm_rollout(win, k, grad=False)
    assert mask[:, 1].eq(1).all()
    assert mask[:, 2].eq(0).all()
    pred = win["state"][:, 0]
    ctx = win["context"][:, 0]
    with torch.no_grad():
        for j in range(2):
            pred, err = agent._wm_step(pred, win["action"][:, j], ctx, win["next_state"][:, j])
    assert torch.equal(errs[:, 1], err)
    assert torch.equal(err, agent.task.state_error(pred, state))


def test_the_world_model_gets_a_gradient_from_the_transition_into_a_failure() -> None:
    agent, published = collected()
    row, ids, _ = published[len(published) // 2]
    win = agent.replay.window(torch.full_like(ids, row), ids, 2)  # the terminal transition alone
    errs, mask = agent._wm_rollout(win, 1, grad=True)
    assert mask.eq(1).all()
    (errs.square().mean(-1) * mask).sum().backward()
    assert sum(p.grad.abs().sum() for p in agent.dynamics.parameters() if p.grad is not None) > 0


def test_the_first_transition_of_a_terminal_window_feeds_the_normalizers_its_true_state() -> None:
    agent, published = collected()
    row, ids, state = published[len(published) // 2]
    win = agent.replay.window(torch.full_like(ids, row), ids, 2)
    assert torch.equal(agent._first_transition(win)[1], state)
    assert win["valid"][:, 0].all()


def test_real_windows_keep_their_own_episode_masks() -> None:
    """They read the episode counter, not the new validity."""
    agent, published = collected(replay_steps=120, actor_source="real", critic_source="real")
    row, ids, _ = published[len(published) // 2]
    starts = (torch.full_like(ids, row - 1), ids)
    with_terminals = agent._real_rollout(ids.numel(), grad=False, starts=starts)
    agent.replay.has_terminal.zero_()
    without = agent._real_rollout(ids.numel(), grad=False, starts=starts)
    for key in ("inep", "term", "trunc", "states", "rewards"):
        assert torch.equal(with_terminals[key], without[key]), key


@pytest.mark.parametrize(
    "wm",
    [{}, NO_SYNC, {"dynamics_model": "diffsr", "diffsr_feature_dim": 16, "diffsr_embed_dim": 32}],
    ids=["default", "no_sync", "diffsr"],
)
def test_a_run_fits_terminal_transitions_and_logs_their_share(wm: dict[str, Any]) -> None:
    agent = harness.make_agent(
        env_cls=harness.FinalObsFakeEnv, num_envs=32, frames=harness.frames_for(6, num_envs=32), **ON, **wm
    )
    rows: list[dict[str, float]] = []
    log = agent.log_training_metrics
    agent.log_training_metrics = lambda flushed, **kw: (rows.append(flushed), log(flushed, **kw))[1]
    agent.train()
    assert rows
    assert all(row["wm/terminal_frac"] > 0 for row in rows)
    assert all(row["wm/terminal_nonfinite"] == 0 for row in rows)


def test_a_non_finite_final_state_is_counted_and_not_fit() -> None:
    class Diverging(harness.FinalObsFakeEnv):
        def step(self, action: Tensor) -> Any:
            out = super().step(action)
            final = out[-1].get("final_obs")
            if final is not None:
                final["obs"]["wm"]["root_state"][:, -13] = float("nan")  # the newest frame's x
            return out

    agent = harness.make_agent(env_cls=Diverging, num_envs=32, **ON)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(70):
        obs = agent._collect_step(obs, stats)
    assert sum(stats["wm/terminal_nonfinite"]) > 0
    assert not agent.replay.has_terminal.any()
    tau, env = agent.replay.sample_starts(64, 3)
    errs, _ = agent._wm_rollout(agent.replay.window(tau, env, 4), 3, grad=False)
    assert torch.isfinite(errs).all()


def test_a_pushed_final_state_is_not_fit() -> None:
    class Pushing(harness.FinalObsFakeEnv):
        def step(self, action: Tensor) -> Any:
            out = super().step(action)
            final = out[-1].get("final_obs")
            if final is not None:
                final["obs"]["wm"]["push_flag"] = torch.ones_like(final["obs"]["wm"]["push_flag"])
            return out

    agent = harness.make_agent(env_cls=Pushing, num_envs=32, **ON)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(70):
        obs = agent._collect_step(obs, stats)
    assert not agent.replay.has_terminal.any()


def test_the_option_stays_out_of_resume_matching(tmp_path: Any) -> None:
    """The replay is not in a checkpoint, so a run may turn the option on when it resumes."""
    trained = harness.make_agent(frames=harness.frames_for(2))
    trained.train()
    path = tmp_path / "a.pt"
    trained.save_model(path)
    harness.make_agent(env_cls=harness.FinalObsFakeEnv, **ON).load_model(str(path))


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
def test_the_terminal_path_does_not_synchronize() -> None:
    agent, obs, stats = census()
    obs = update(agent, obs, stats)  # first calls: cuBLAS handles, optimizer state
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(2):
            obs = update(agent, obs, stats)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert agent.replay.has_terminal.any()


@cuda
def test_the_terminal_path_compiles_without_recompiling() -> None:
    agent, obs, stats = census(compile=True, compile_eager=True)
    for _ in range(2):
        obs = update(agent, obs, stats)
    with torch.compiler.set_stance("fail_on_recompile"):
        for _ in range(2):
            obs = update(agent, obs, stats)
    assert all(torch.isfinite(p).all() for p in agent.dynamics.parameters())
