"""``wm.imag_start_perturb``: kicks on the root's horizontal velocity at imagined starts.

NDM Gym's ``perturb_start`` (``ndm_gym/algos/common.py``) adds N(0, std^2) m/s to the
start state's horizontal root linear velocity after the observation history is read,
so the history stays the recorded one; the policy update's rollout, which the critic
also fits, starts from the kicked state. The references here are the replay rows
themselves. The default path is the golden test's.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from torch import Tensor

from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
EXACT = {"rtol": 0, "atol": 0}
VELOCITY_XY = slice(7, 9)  # the 71-D state: position 3, WXYZ 4, linear velocity 3, ...


def collected(agent: Any, steps: int = 45) -> dict[str, Any]:
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(steps):
        obs = agent._collect_step(obs, stats)
    return obs


def starts(agent: Any, n: int) -> tuple[Tensor, Tensor]:
    """``n`` replay rows whose horizon stays inside the clip."""
    rp, t = agent.replay, agent.task
    taus = torch.arange(rp.oldest + t.history - 1, rp.steps - 1, device=agent.device)
    tau, env = torch.meshgrid(taus, torch.arange(agent._num_envs, device=agent.device), indexing="ij")
    slot = rp._slot(tau)
    valid = t.valid_imagination_start(rp.rank[slot, env], rp.frame[slot, env], agent.config.wm.horizon)
    pick = valid.nonzero()[:n]
    assert pick.shape[0] == n
    return tau[pick[:, 0], pick[:, 1]], env[pick[:, 0], pick[:, 1]]


def recorded_start(agent: Any, tau: Tensor, env: Tensor) -> Tensor:
    return agent.replay.history(tau, env, agent.task.history)["hist_states"][:, -1]


# ------------------------------------------------------------------ the kick
def test_the_kick_moves_only_the_horizontal_root_velocity() -> None:
    agent = harness.make_agent(imag_start_perturb=0.3)
    collected(agent)
    tau, env = starts(agent, 8)
    out = agent._imagine(8, grad=True, starts=(tau, env), start_perturb=0.3)
    kicked, recorded = out["states"][:, 0].detach(), recorded_start(agent, tau, env)
    moved = (kicked != recorded).any(0)
    assert moved[VELOCITY_XY].all()
    assert not moved[: VELOCITY_XY.start].any()
    assert not moved[VELOCITY_XY.stop :].any()


def test_the_kick_is_gaussian_with_the_given_std_and_no_vertical_part() -> None:
    agent = harness.make_agent(env_cls=harness.FakeIsaacEnv)
    state = agent.task.library.ref_state(torch.zeros(200_000, dtype=torch.long), torch.zeros(200_000, dtype=torch.long))
    g = torch.Generator().manual_seed(3)
    kick = agent.task.perturb_start(state, 0.15, g) - state
    torch.testing.assert_close(kick[:, 9], torch.zeros(200_000), **EXACT)
    xy = kick[:, VELOCITY_XY]
    torch.testing.assert_close(xy.std(0), torch.full((2,), 0.15), rtol=0.01, atol=0)
    assert float(xy.mean(0).abs().max()) < 0.002
    assert float(torch.corrcoef(xy.T)[0, 1].abs()) < 0.01


def test_the_history_stays_the_recorded_one() -> None:
    """The actor's first input does not see the kick; the model's first step does.
    Without sensor noise the kick is the only random draw of these rollouts."""
    agent = harness.make_agent(imag_start_perturb=0.5, sensor_noise=False)
    collected(agent)
    tau, env = starts(agent, 8)
    plain = agent._imagine(8, grad=False, starts=(tau, env))
    kicked = agent._imagine(8, grad=False, starts=(tau, env), start_perturb=0.5)
    torch.testing.assert_close(kicked["mus"][:, 0], plain["mus"][:, 0], **EXACT)
    assert (kicked["states"][:, 1] != plain["states"][:, 1]).any()
    torch.testing.assert_close(plain["states"][:, 0], recorded_start(agent, tau, env), **EXACT)


def test_the_kick_draws_from_the_given_generator() -> None:
    agent = harness.make_agent(sensor_noise=False)
    collected(agent)
    tau, env = starts(agent, 8)

    def run(seed: int) -> Tensor:
        g = torch.Generator().manual_seed(seed)
        return agent._imagine(8, grad=False, starts=(tau, env), start_perturb=0.2, generator=g)["states"]

    before = torch.get_rng_state()
    a, b, c = run(5), run(5), run(6)
    assert torch.equal(torch.get_rng_state(), before)
    torch.testing.assert_close(a, b, **EXACT)
    assert not torch.equal(a, c)


# ------------------------------------------------------------------ where it applies
def spy(agent: Any, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    perturb = agent.task.perturb_start

    def record(state: Tensor, std: float, generator: Any = None) -> Tensor:
        calls.append(std)
        return perturb(state, std, generator)

    monkeypatch.setattr(agent.task, "perturb_start", record)
    return calls


def test_every_actor_update_kicks_and_the_critic_fits_that_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = harness.make_agent(frames=harness.frames_for(5), imag_start_perturb=0.15)
    calls = spy(agent, monkeypatch)
    actor_rollouts, critic_rollouts = [], []
    imagine, update_critic = agent._imagine, agent._update_critic

    def imagine_spy(*args: Any, **kwargs: Any) -> dict[str, Tensor]:
        out = imagine(*args, **kwargs)
        actor_rollouts.append((kwargs.get("start_perturb"), out))
        return out

    def critic_spy(out: dict[str, Tensor], metrics: Any) -> None:
        critic_rollouts.append(out)
        update_critic(out, metrics)

    monkeypatch.setattr(agent, "_imagine", imagine_spy)
    monkeypatch.setattr(agent, "_update_critic", critic_spy)
    agent.train()
    assert actor_rollouts
    assert calls == [0.15] * len(actor_rollouts)
    assert [kick for kick, _ in actor_rollouts] == [0.15] * len(actor_rollouts)
    assert [id(out) for _, out in actor_rollouts] == [id(out) for out in critic_rollouts]


def test_the_default_agent_never_kicks(monkeypatch: pytest.MonkeyPatch) -> None:
    from imitation_experiments.world_model.g1_v3_task import G1V3TrackingTask

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        msg = "kicked a start with the option off"
        raise AssertionError(msg)

    monkeypatch.setattr(G1V3TrackingTask, "perturb_start", refuse)
    agent = harness.make_agent(frames=harness.frames_for(4))
    agent.train()
    assert agent._iteration == 4


# ------------------------------------------------------------------ refused
def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


class Capable:
    def perturb_start(self, state: Tensor, _std: float, _generator: Any = None) -> Tensor:
        return state


@pytest.mark.parametrize(
    ("wm", "task", "match"),
    [
        ({"imag_start_perturb": -0.1}, Capable(), "standard deviation"),
        ({"imag_start_perturb": 0.15}, object(), "perturb_start"),
        ({"imag_start_perturb": 0.15, "actor_source": "real", "critic_source": "real"}, Capable(), "recorded windows"),
        ({"imag_start_perturb": 0.15, "actor_source": "real"}, Capable(), "recorded windows"),
    ],
)
def test_refused(wm: dict[str, Any], task: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(**wm), task)


def test_allowed() -> None:
    validate_options(config(imag_start_perturb=0.15), Capable())
    validate_options(config(imag_start_perturb=0.15, critic_source="real"), Capable())
    validate_options(config(imag_start_perturb=0.15, actor_estimator="score_function"), Capable())
    validate_options(config(), object())


# ------------------------------------------------------------------ the recipe's systems options
class SyncFreeEnv(harness.FakeIsaacEnv):
    def step(self, action: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor, dict[str, Any]]:
        zeros = torch.zeros(self.n, device=self.device)
        self.frame = self.frame + 1
        self.episode = self.episode + 1
        self.frame = torch.where(self.frame >= self.v3_library.length[self.rank] - 2, 0, self.frame)
        self.hist_action = torch.cat((self.hist_action[:, 1:], action[:, None]), 1)
        return self._obs(), zeros, zeros.bool(), zeros.bool(), {}


RECIPE = {"no_sync": True, "fused_adam": True, "compile": True, "compile_eager": True, "imag_start_perturb": 0.15}


def rounds(agent: Any, obs: dict[str, Any], n: int) -> dict[str, Any]:
    stats: dict[str, list] = defaultdict(list)
    for _ in range(n):
        for _ in range(3):
            obs = agent._collect_step(obs, stats)
        agent._set_lrs()
        agent._run_update("world_model", agent._update_world_model, defaultdict(list))
        agent._run_update("actor_critic", agent._update_actor_critic, defaultdict(list))
    return obs


@cuda
@pytest.mark.parametrize("graphs", [False, True], ids=["no_sync_compile", "update_graphs"])
def test_the_kick_adds_no_host_read_and_no_recompile(graphs: bool) -> None:
    wm = {**RECIPE, **({"update_graphs": True, "graph_early": True} if graphs else {})}
    agent = harness.make_agent(device="cuda:0", num_envs=64, env_cls=SyncFreeEnv, **wm)
    obs = collected(agent, 44)
    obs = rounds(agent, obs, 5)  # compile, warm-up and capture
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.compiler.set_stance("fail_on_recompile"):
            obs = rounds(agent, obs, 3)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    if graphs:
        assert agent._graphs["world_model"].captures == 1
        assert agent._graphs["actor_critic"].captures == 1
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())


@cuda
def test_a_captured_update_draws_new_kicks_on_every_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The graph replays the draw with the default CUDA generator, whose offset moves."""
    wm = {**RECIPE, "update_graphs": True, "graph_early": True}
    agent = harness.make_agent(device="cuda:0", num_envs=64, env_cls=SyncFreeEnv, **wm)
    kicks: list[Tensor] = []
    perturb = agent.task.perturb_start

    def keep(state: Tensor, std: float, generator: Any = None) -> Tensor:
        out = perturb(state, std, generator)
        kicks.append(out[:, VELOCITY_XY] - state[:, VELOCITY_XY])  # a graph output buffer once captured
        return out

    monkeypatch.setattr(agent.task, "perturb_start", keep)
    obs = collected(agent, 44)
    obs = rounds(agent, obs, 5)
    assert agent._graphs["actor_critic"].captures == 1
    captured = kicks[-1]  # the capture's own buffer, which every replay rewrites
    before = captured.clone()
    rounds(agent, obs, 1)
    torch.cuda.synchronize()
    assert not torch.equal(captured, before)
    assert 0.05 < float(captured.std()) < 0.3  # 0.15 over 64 draws
