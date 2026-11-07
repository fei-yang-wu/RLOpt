"""``wm.update_graphs``: the world-model and actor-critic updates as CUDA graphs.

What compares how:

- Bit for bit: a replayed graph against the same agent's eager update, over warm-up,
  capture and replays; the device-head draws against ``torch.randint`` with host
  bounds; the capturable optimizers against the ``no_sync`` loop when every learning
  rate is a float32 value (Adam).
- ``rtol``: the same comparison at the configured rates, where the plain fused
  kernel reads a float64 rate and the capturable one a float32 tensor, and with
  Muon, whose capturable step is a separate implementation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from torch import Tensor

from rlopt.agent.wm_bptt.agent import WMBPTT
from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options
from rlopt.agent.wm_bptt.muon import MuonAdam
from rlopt.agent.wm_bptt.replay import SequenceReplay
from rlopt.agent.wm_bptt.speed_options import set_lr

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEVICES = ["cpu", pytest.param("cuda:0", marks=cuda)]
NO_SYNC = {"no_sync": True, "fused_adam": True}
GRAPHS = {**NO_SYNC, "update_graphs": True}
FLOAT32_RATES = {"wm_lr": 2.0**-12, "actor_lr": 2.0**-11, "critic_lr": 2.0**-10}
"""Learning rates float32 holds exactly, so a device rate equals the float one."""


# ------------------------------------------------------------------ validation
def config(device: str = "cuda:0", **wm: Any) -> WMBPTTRLOptConfig:
    cfg = WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))
    cfg.device = device
    return cfg


@pytest.mark.parametrize(
    ("wm", "match"),
    [
        ({"update_graphs": True}, "no_sync"),
        ({"update_graphs": True, "no_sync": True}, "fused_adam"),
        ({**GRAPHS, "critic_update_every": 2}, "critic_update_every"),
        ({**GRAPHS, "actor_estimator": "score_function"}, "pathwise"),
        ({**GRAPHS, "dynamics_model": "diffsr"}, "MLP world model"),
        ({**NO_SYNC, "graph_early": True}, "set wm.update_graphs"),
    ],
)
def test_refused_combinations(wm: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(**wm), object())


def test_graphs_need_a_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA graphs"):
        validate_options(config("cpu", **GRAPHS), object())
    validate_options(config("cuda:0", **GRAPHS), object())


# ------------------------------------------------------------------ device head
def ring(device: str, capacity: int = 50, num_envs: int = 3) -> SequenceReplay:
    return SequenceReplay(num_envs, capacity, 2, 1, 1, torch.device(device))


def add_rows(rp: SequenceReplay, n: int) -> None:
    N = rp.num_envs
    for _ in range(n):
        step = torch.full((N,), rp.steps, device=rp.device)
        rp.add(
            torch.zeros(N, 2, device=rp.device), torch.zeros(N, 1, device=rp.device), torch.zeros(N, 1, device=rp.device),
            torch.zeros(N, 1, device=rp.device), step, step, step, torch.zeros(N, dtype=torch.bool, device=rp.device),
        )


@pytest.mark.parametrize("device", DEVICES)
def test_device_head_draws_equal_host_randint(device: str) -> None:
    """Before the ring is full, when it fills and after it wrapped: the same rows."""
    host, head = ring(device), ring(device)
    head.use_device_head()
    for n in (12, 30, 8, 77):
        add_rows(host, n)
        add_rows(head, n)
        assert int(head.steps_t) == head.steps == host.steps
        for low, high in ((0, 8), (9, 0), (4, 3)):
            gen = [torch.Generator(device=device).manual_seed(5) for _ in range(2)]
            lo, hi = host.oldest + low, host.steps - 1 - high
            want = torch.randint(lo, hi + 1, (4096,), device=device, generator=gen[0])
            got = head.uniform_rows(4096, low, high, gen[1])
            assert torch.equal(want, got)
            assert int(got.min()) == lo
            assert int(got.max()) == hi
        torch.manual_seed(1)
        a = host.sample_starts(256, 8)
        torch.manual_seed(1)
        b = head.sample_starts(256, 8)
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("device", DEVICES)
def test_device_fill_draws_are_uniform_over_the_rows(device: str) -> None:
    """``graph_early``'s ``rand * n`` form: inside the bounds, and uniform over them
    (chi-square) before the ring is full, when it fills and after it wrapped."""
    rp = ring(device)
    rp.use_device_head(fill=True)
    gen = torch.Generator(device=device).manual_seed(2)
    for n in (12, 38, 30):
        add_rows(rp, n)
        for low, high in ((0, 8), (9, 0)):
            lo, hi = rp.oldest + low, rp.steps - 1 - high
            rows = rp.uniform_rows(200_000, low, high, gen)
            assert int(rows.min()) == lo
            assert int(rows.max()) == hi
            counts = torch.bincount(rows - lo, minlength=hi - lo + 1).double()
            cells = counts.numel()
            expected = rows.numel() / cells
            chi2 = float(((counts - expected) ** 2 / expected).sum())
            assert chi2 < cells + 5 * (2 * cells) ** 0.5, (chi2, cells)


@cuda
def test_a_captured_device_fill_draw_follows_the_ring() -> None:
    """Captured once while the ring fills, the draw keeps covering exactly the rows
    the ring holds, as it fills and after it wraps."""
    rp = ring("cuda:0")
    rp.use_device_head(fill=True)
    add_rows(rp, 20)
    for _ in range(2):
        rp.uniform_rows(65_536, 4, 3)  # warm-up allocations
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rows = rp.uniform_rows(65_536, 4, 3)
    for n in (0, 10, 25, 40):
        add_rows(rp, n)
        graph.replay()
        assert int(rows.min()) == rp.oldest + 4
        assert int(rows.max()) == rp.steps - 1 - 3


def test_a_ring_too_short_for_the_bounds_raises() -> None:
    rp = ring("cpu")
    rp.use_device_head()
    add_rows(rp, 5)
    with pytest.raises(RuntimeError, match="need more"):
        rp.uniform_rows(4, 3, 2)


def test_the_default_replay_has_no_device_head() -> None:
    agent = harness.make_agent()
    assert agent.ensure_replay().steps_t is None
    assert agent._graphs is None


# ------------------------------------------------------------------ helpers
def filled(steps: int = 100, env_cls: type = harness.FakeIsaacEnv, **wm: Any) -> WMBPTT:
    """An agent whose ring (80 rows in ``harness.SMALL``) is full: graphs can capture."""
    agent = harness.make_agent(device="cuda:0", env_cls=env_cls, **wm)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(steps):
        obs = agent._collect_step(obs, stats)
    agent._obs = obs  # type: ignore[attr-defined]
    return agent


def graphed(agent: WMBPTT, metrics: dict[str, list]) -> None:
    agent._run_update("world_model", agent._update_world_model, metrics)
    agent._run_update("actor_critic", agent._update_actor_critic, metrics)


def eager(agent: WMBPTT, metrics: dict[str, list]) -> None:
    agent._update_world_model(metrics)
    agent._update_actor_critic(metrics)


def optimizer_state(opt: Any) -> Any:
    """The moments and step counts, without the group settings that the option changes."""
    if isinstance(opt, MuonAdam):
        return {"muon": opt.muon.state_dict()["state"], "adam": opt.adam.state_dict()["state"]}
    return opt.state_dict()["state"]


def training_state(agent: WMBPTT) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dynamics": agent.dynamics.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "optim_wm": optimizer_state(agent.wm_optim),
        "optim_actor": optimizer_state(agent.actor_optim),
        "optim_critic": optimizer_state(agent.critic_optim),
        "counts": agent._device_counts,
    }
    if agent.actor_avg is not None:
        out["average"] = agent.actor_avg.module.state_dict()
        out["average_updates"] = agent.actor_avg.updates
    return out


def stacked(metrics: dict[str, list]) -> dict[str, Tensor]:
    return {key: torch.stack([torch.as_tensor(v) for v in values]) for key, values in metrics.items()}


def run(
    agent: WMBPTT, iteration: Any, calls: int, seed: int = 3, *, schedule: bool = False, collect: int = 0
) -> dict[str, Tensor]:
    """``calls`` iterations' updates; ``schedule``: the learning rates ``train`` sets
    before each, advancing the iteration; ``collect``: env steps before each, so the
    ring's head moves under a captured graph."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    metrics: dict[str, list] = defaultdict(list)
    stats: dict[str, list] = defaultdict(list)
    for _ in range(calls):
        for _ in range(collect):
            agent._obs = agent._collect_step(agent._obs, stats)  # type: ignore[attr-defined]
        if schedule:
            agent._set_lrs()
        iteration(agent, metrics)
        if schedule:
            agent._iteration += 1
    torch.cuda.synchronize()
    return stacked(metrics)


# ------------------------------------------------------------------ bitwise
VARIANTS = [
    pytest.param({}, harness.FakeIsaacEnv, id="adam"),
    pytest.param({"actor_opt": "muon", "wm_opt": "muon", "actor_ema": 0.99}, harness.FakeIsaacEnv, id="muon_ema"),
    pytest.param({"compile": True}, harness.FakeIsaacEnv, id="compile"),
    pytest.param(
        {"fit_terminal_transitions": True, "norm_from_collection": True, "actor_lr_mult": 2.0,
         "actor_lr_decay_by": 4, "actor_lr_decay_len": 3},
        harness.FinalObsFakeEnv, id="terminal_lr_mult",
    ),
]


@cuda
@pytest.mark.parametrize(("wm", "env_cls"), VARIANTS)
def test_replayed_graphs_equal_the_eager_updates(wm: dict[str, Any], env_cls: type) -> None:
    """Three warm-up calls, the capture and three replays against seven eager calls
    of the same agent, with collection moving the ring and the schedule writing the
    learning rates before each: every weight, normalizer, optimizer state, policy
    average, device count, replay row and metric, bit for bit."""
    calls = 7
    graph_agent, eager_agent = filled(env_cls=env_cls, **GRAPHS, **wm), filled(env_cls=env_cls, **GRAPHS, **wm)
    assert harness.state_equal(training_state(graph_agent), training_state(eager_agent))
    for agent in (graph_agent, eager_agent):
        if agent._actor_lr.has_trigger:
            agent._actor_lr.decay_at = 0  # the ramp moves the actor's rate at every iteration
        if agent.actor_avg is not None:  # the average's first copy is eager: both agents take it here
            run(agent, eager, 1, schedule=True)
    got = run(graph_agent, graphed, calls, schedule=True, collect=3)
    want = run(eager_agent, eager, calls, schedule=True, collect=3)
    assert graph_agent._graphs is not None
    assert [g.captures for g in graph_agent._graphs.values()] == [1, 1]
    assert harness.state_equal(training_state(graph_agent), training_state(eager_agent))
    assert harness.state_equal(harness.replay_tensors(graph_agent), harness.replay_tensors(eager_agent))
    assert harness.state_equal(got, want)
    assert set(got) >= {"wm/loss", "actor/objective", "critic/loss"}
    per_iteration = {k: graph_agent.config.wm.wm_batches_per_iter if k.startswith("wm/") else 1 for k in got}
    assert all(v.numel() == calls * per_iteration[k] for k, v in got.items())


@cuda
@pytest.mark.parametrize("wm", [{}, {"actor_opt": "muon", "wm_opt": "muon", "actor_ema": 0.99, "compile": True}], ids=["adam", "muon_ema_compile"])
def test_early_graphs_equal_the_eager_updates_while_the_ring_fills(wm: dict[str, Any]) -> None:
    """``graph_early``: captured with 45 of 80 rows, the graphs follow the fill level
    on the device while collection fills the ring and wraps it: bit for bit the
    same agent's eager updates."""
    early = {**GRAPHS, "graph_early": True, **wm}
    graph_agent, eager_agent = filled(45, **early), filled(45, **early)
    for agent in (graph_agent, eager_agent):
        if agent.actor_avg is not None:
            run(agent, eager, 1, schedule=True)
    got = run(graph_agent, graphed, 8, schedule=True, collect=6)
    want = run(eager_agent, eager, 8, schedule=True, collect=6)
    rp = graph_agent.replay
    assert rp is not None
    assert rp.steps > rp.capacity  # the ring filled under the captured graphs
    assert [g.captures for g in graph_agent._graphs.values()] == [1, 1]  # type: ignore[union-attr]
    assert harness.state_equal(training_state(graph_agent), training_state(eager_agent))
    assert harness.state_equal(got, want)


@cuda
@pytest.mark.parametrize("early", [False, True])
def test_graph_early_captures_before_the_ring_is_full(early: bool) -> None:
    """Eight iterations hold 57 of the 80 rows: only ``graph_early`` captures."""
    agent = harness.make_agent(device="cuda:0", frames=harness.frames_for(8), **GRAPHS, graph_early=early)
    agent.train()
    rp = agent.replay
    assert rp is not None
    assert rp.steps < rp.capacity
    assert [g.captures for g in agent._graphs.values()] == ([1, 1] if early else [0, 0])  # type: ignore[union-attr]
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())


@cuda
@pytest.mark.parametrize("device_rates", [True, False], ids=["float32_rates", "configured_rates"])
def test_the_graphed_updates_equal_the_no_sync_loop(device_rates: bool) -> None:
    """The option against the loop it captures: bit for bit with Adam when every rate
    is a float32 value, ``rtol`` at the configured ones (float64 in the plain kernel)."""
    rates = FLOAT32_RATES if device_rates else {}
    graph_agent, loop_agent = filled(**GRAPHS, **rates), filled(**NO_SYNC, **rates)
    got = run(graph_agent, graphed, 6)
    want = run(loop_agent, eager, 6)
    if device_rates:
        assert harness.state_equal(training_state(graph_agent), training_state(loop_agent))
        assert harness.state_equal(got, want)
        return
    a, b = training_state(graph_agent), training_state(loop_agent)
    for name in ("dynamics", "actor", "critic"):
        for key, value in b[name].items():
            if value.is_floating_point():
                torch.testing.assert_close(a[name][key], value, rtol=1e-4, atol=1e-6)


@cuda
def test_muon_graphs_match_the_no_sync_loop_within_rtol() -> None:
    wm = {"actor_opt": "muon", "wm_opt": "muon", **FLOAT32_RATES}
    graph_agent, loop_agent = filled(**GRAPHS, **wm), filled(**NO_SYNC, **wm)
    run(graph_agent, graphed, 6)
    run(loop_agent, eager, 6)
    a, b = training_state(graph_agent), training_state(loop_agent)
    for name in ("dynamics", "actor", "critic"):
        for key, value in b[name].items():
            if value.is_floating_point():
                torch.testing.assert_close(a[name][key], value, rtol=1e-4, atol=1e-6)


# ------------------------------------------------------------------ learning rates
@cuda
def test_every_optimizer_has_its_own_device_rate() -> None:
    for wm in ({}, {"actor_opt": "muon", "wm_opt": "muon"}):
        agent = harness.make_agent(device="cuda:0", **GRAPHS, **wm)
        groups = [g for opt in (agent.wm_optim, agent.actor_optim, agent.critic_optim) for g in opt.param_groups]
        assert all(torch.is_tensor(g["lr"]) and g["lr"].is_cuda for g in groups)
        assert len({id(g["lr"]) for g in groups}) == len(groups)
        assert all(g.get("capturable", True) for g in groups)


@cuda
def test_replays_read_the_rates_the_schedule_writes() -> None:
    """After capture, ``_set_lrs`` reaches the graphs: the multiplier's ramp, and a
    zero rate that leaves the actor's and the world model's weights where they are."""
    agent = filled(**GRAPHS, actor_lr_mult=4.0, actor_lr_decay_by=1, actor_lr_decay_len=10)
    run(agent, graphed, 4)  # warm-up and capture
    agent._iteration = 6
    agent._actor_lr.decay_at = 1
    agent._set_lrs()
    wm = agent.config.wm
    want = wm.actor_lr * agent._actor_lr.at(6) * agent._lr_factor()
    assert all(float(g["lr"]) == pytest.approx(want, rel=1e-6) for g in agent.actor_optim.param_groups)
    for opt in (agent.wm_optim, agent.actor_optim):
        set_lr(opt, 0.0)
    nets = [p.detach().clone() for p in (*agent.dynamics.net.parameters(), *agent.actor.net.parameters())]
    critic = [p.detach().clone() for p in agent.critic.net.parameters()]
    run(agent, graphed, 2)
    after = [*agent.dynamics.net.parameters(), *agent.actor.net.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(nets, after, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(critic, agent.critic.net.parameters(), strict=True))


# ------------------------------------------------------------------ checkpoints
@cuda
@pytest.mark.parametrize("wm", [{}, {"actor_opt": "muon", "wm_opt": "muon", "actor_ema": 0.99}], ids=["adam", "muon_ema"])
def test_a_checkpoint_load_recaptures_and_continues_bit_for_bit(tmp_path: Path, wm: dict[str, Any]) -> None:
    """Load into an agent whose graphs are captured: the old graphs would write the
    replaced optimizer state, so new ones warm up and capture. They then continue
    the checkpoint exactly as an eager agent that loads it."""
    graph_agent, eager_agent = filled(**GRAPHS, **wm), filled(**GRAPHS, **wm)
    for agent in (graph_agent, eager_agent):
        run(agent, eager, 1)
    run(graph_agent, graphed, 5)
    run(eager_agent, eager, 5)
    path = tmp_path / "ckpt.pt"
    graph_agent.save_model(path)
    old = dict(graph_agent._graphs or {})
    run(graph_agent, graphed, 2)  # the state moves on past the checkpoint
    for agent in (graph_agent, eager_agent):
        agent.load_model(str(path))
    assert graph_agent._graphs is not None
    assert all(graph_agent._graphs[name] is not old[name] for name in old)
    assert harness.state_equal(training_state(graph_agent), training_state(eager_agent))
    got = run(graph_agent, graphed, 5, seed=9)
    want = run(eager_agent, eager, 5, seed=9)
    assert [g.captures for g in graph_agent._graphs.values()] == [1, 1]
    assert harness.state_equal(training_state(graph_agent), training_state(eager_agent))
    assert harness.state_equal(got, want)


@cuda
def test_a_checkpoint_without_graphs_resumes_with_them_and_back(tmp_path: Path) -> None:
    """Group settings follow the resuming agent: device rates and ``capturable`` with
    the option, float rates without it."""
    plain = filled(**NO_SYNC)
    run(plain, eager, 2)
    path = tmp_path / "plain.pt"
    plain.save_model(path)
    graph_agent = filled(**GRAPHS)
    graph_agent.load_model(str(path))
    for opt in (graph_agent.wm_optim, graph_agent.actor_optim, graph_agent.critic_optim):
        assert all(torch.is_tensor(g["lr"]) and g["capturable"] for g in opt.param_groups)
    run(graph_agent, graphed, 5)
    assert [g.captures for g in graph_agent._graphs.values()] == [1, 1]  # type: ignore[union-attr]
    path = tmp_path / "graphs.pt"
    graph_agent.save_model(path)
    back = filled(**NO_SYNC)
    back.load_model(str(path))
    for opt in (back.wm_optim, back.actor_optim, back.critic_optim):
        assert all(isinstance(g["lr"], float) and not g["capturable"] for g in opt.param_groups)
    run(back, eager, 1)


@cuda
def test_a_graph_run_trains_and_resumes(tmp_path: Path) -> None:
    """Through ``train``: the ring fills, both graphs capture, log rows read them;
    a resumed run refills its ring and captures again."""
    iterations = 26  # 40 rows, then 2.5 per iteration: the 80-row ring is full at 16
    wm = {"actor_ema": 0.99, "actor_opt": "muon"}
    agent = harness.make_agent(device="cuda:0", frames=harness.frames_for(iterations), **GRAPHS, **wm)
    rows: list[dict[str, float]] = []
    agent.log_training_metrics = lambda row, **_kwargs: rows.append(row)  # type: ignore[method-assign]
    agent.train()
    assert agent._graphs is not None
    assert [g.captures for g in agent._graphs.values()] == [1, 1]
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())
    assert all(row["wm/nonfinite"] == 0 and row["actor/nonfinite"] == 0 for row in rows)
    assert all(torch.isfinite(torch.tensor(row["wm/loss"])) for row in rows if "wm/loss" in row)
    assert agent.actor_avg is not None
    assert agent.actor_avg.updates == iterations - agent.config.wm.wm_warmup_iters
    path = tmp_path / "run.pt"
    agent.save_model(path)
    resumed = harness.make_agent(device="cuda:0", frames=2 * harness.frames_for(iterations), **GRAPHS, **wm)
    resumed.load_model(str(path))
    resumed.train()
    assert [g.captures for g in resumed._graphs.values()] == [1, 1]  # type: ignore[union-attr]
    assert resumed.actor_avg is not None
    assert resumed.actor_avg.updates > agent.actor_avg.updates
    scorer = harness.make_agent()  # what the evaluator builds
    scorer.load_policy(str(path))


# ------------------------------------------------------------------ host reads
class SyncFreeEnv(harness.FakeIsaacEnv):
    def step(self, _action: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor, dict[str, Any]]:
        zeros = torch.zeros(self.n, device=self.device)
        self.frame = self.frame + 1
        self.episode = self.episode + 1
        self.frame = torch.where(self.frame >= self.v3_library.length[self.rank] - 2, 0, self.frame)
        return self._obs(), zeros, zeros.bool(), zeros.bool(), {}


@cuda
@pytest.mark.parametrize("wm", [{}, {"actor_opt": "muon", "wm_opt": "muon", "actor_ema": 0.99, "compile": True}], ids=["adam", "muon_ema_compile"])
def test_replays_do_not_synchronize(wm: dict[str, Any]) -> None:
    agent = filled(**GRAPHS, **wm)
    agent._isaac = SyncFreeEnv(agent._num_envs, "cuda:0")
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    run(agent, eager, 1)
    run(agent, graphed, 4)  # warm-up and capture, which synchronizes once
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            for _ in range(3):
                obs = agent._collect_step(obs, stats)
            agent._set_lrs()
            graphed(agent, defaultdict(list))
    finally:
        torch.cuda.set_sync_debug_mode("default")
