"""``wm.no_sync``: the same training with no host read in the update loop.

What compares how:

- Bit for bit: collection and the replay; the unmasked normalizer update; the gated
  step on finite gradients; a whole world-model update and a whole actor-critic update
  against the host path with the same starts and fused Adam on both sides.
- ``rtol``: the masked normalizer update against boolean indexing (reduction order),
  and the same updates against the default foreach Adam (a different kernel).
- Statistically: the start distribution, because the random stream differs.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from torch import Tensor, nn

from rlopt.agent.wm_bptt.agent import DEVICE_COUNTS, WMBPTT
from rlopt.agent.wm_bptt.config import WMBPTTConfig
from rlopt.agent.wm_bptt.muon import MuonAdam
from rlopt.agent.wm_bptt.networks import RunningNorm
from rlopt.agent.wm_bptt.replay import SequenceReplay
from rlopt.agent.wm_bptt.speed_options import gated_step, update_norm_on_device, zero_nonfinite_grads

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEVICES = ["cpu", pytest.param("cuda:0", marks=cuda)]
ON = {"no_sync": True, "fused_adam": True}


def norm_state(norm: RunningNorm) -> dict[str, Tensor]:
    return {k: v.clone() for k, v in norm.state_dict().items()}


# ------------------------------------------------------------------ normalizers
@pytest.mark.parametrize("device", DEVICES)
def test_the_unmasked_update_equals_update_bit_for_bit(device: str) -> None:
    a, b = RunningNorm(5).to(device), RunningNorm(5).to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    for rows in (7, 64, 1, 300):
        x = 3 + 2 * torch.randn(rows, 5, device=device, generator=gen)
        a.update(x)
        update_norm_on_device(b, x)
        assert harness.state_equal(a.state_dict(), b.state_dict())


@pytest.mark.parametrize("device", DEVICES)
def test_a_frozen_norm_stays_put(device: str) -> None:
    norm = RunningNorm(3).to(device)
    update_norm_on_device(norm, torch.randn(16, 3, device=device))
    norm.frozen.fill_(True)
    before = norm_state(norm)
    update_norm_on_device(norm, torch.randn(16, 3, device=device))
    update_norm_on_device(norm, torch.randn(16, 3, device=device), torch.ones(16, dtype=torch.bool, device=device))
    assert harness.state_equal(before, norm.state_dict())


@pytest.mark.parametrize("device", DEVICES)
def test_the_masked_update_matches_the_indexed_rows(device: str) -> None:
    """Reduction order differs, so this is an ``rtol`` comparison. An excluded row
    may hold anything, including NaN, and must not reach the statistics."""
    a, b = RunningNorm(6).to(device), RunningNorm(6).to(device)
    gen = torch.Generator(device=device).manual_seed(1)
    for _ in range(5):
        x = 3 + 2 * torch.randn(512, 6, device=device, generator=gen)
        valid = torch.rand(512, device=device, generator=gen) < 0.7
        a.update(x[valid])
        x[~valid] = float("nan")
        update_norm_on_device(b, x, valid)
    assert a.count.item() == b.count.item()
    torch.testing.assert_close(b.mean, a.mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b.var, a.var, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_a_batch_without_a_valid_row_changes_nothing(device: str) -> None:
    norm = RunningNorm(3).to(device)
    before = norm_state(norm)  # count 0: the merge would divide by 0
    update_norm_on_device(norm, torch.randn(8, 3, device=device), torch.zeros(8, dtype=torch.bool, device=device))
    assert harness.state_equal(before, norm.state_dict())


@pytest.mark.parametrize("device", DEVICES)
def test_a_false_gate_holds_the_statistics(device: str) -> None:
    norm = RunningNorm(3).to(device)
    before = norm_state(norm)
    valid = torch.ones(8, dtype=torch.bool, device=device)
    update_norm_on_device(norm, torch.randn(8, 3, device=device), valid, torch.tensor(False, device=device))
    assert harness.state_equal(before, norm.state_dict())
    update_norm_on_device(norm, torch.randn(8, 3, device=device), valid, torch.tensor(True, device=device))
    assert norm.count.item() == 8


# ------------------------------------------------------------------ gated step
def net(device: str) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 4)).to(device)


def backward(model: nn.Module, device: str) -> None:
    model.zero_grad(set_to_none=True)
    model(torch.ones(3, 4, device=device)).square().sum().backward()


def optimizer(kind: str, model: nn.Module) -> Any:
    if kind == "muon":
        return MuonAdam(model, [model], lr=1e-2, fused=True)
    return torch.optim.Adam(model.parameters(), lr=1e-2, fused=True)


def test_a_gated_step_needs_the_fused_adam() -> None:
    model = net("cpu")
    backward(model, "cpu")
    with pytest.raises(ValueError, match="fused_adam"):
        gated_step(torch.optim.Adam(model.parameters(), lr=1e-2), torch.tensor(True))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["adam", "muon"])
def test_a_true_gate_is_the_plain_step(device: str, kind: str) -> None:
    a, b = net(device), net(device)
    opt_a, opt_b = optimizer(kind, a), optimizer(kind, b)
    ok = torch.tensor(True, device=device)
    for _ in range(3):
        backward(a, device)
        backward(b, device)
        opt_a.step()
        gated_step(opt_b, ok)
    if kind == "adam":  # Muon's gate is a ``where``: the arithmetic of ``step`` is the same
        assert harness.state_equal(opt_a.state_dict(), opt_b.state_dict())
    assert harness.state_equal(a.state_dict(), b.state_dict())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["adam", "muon"])
def test_a_non_finite_gradient_changes_no_state(device: str, kind: str) -> None:
    model = net(device)
    opt = optimizer(kind, model)
    backward(model, device)
    opt.step()
    weights, state = harness.digest(model.state_dict()), harness.digest(opt.state_dict())
    backward(model, device)
    next(model.parameters()).grad[0, 0] = float("nan")
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    finite = torch.isfinite(norm)
    zero_nonfinite_grads(model.parameters(), finite)
    assert not any(p.grad.isnan().any() for p in model.parameters())
    gated_step(opt, finite)
    assert harness.digest(model.state_dict()) == weights
    assert harness.digest(opt.state_dict()) == state


# ------------------------------------------------------------------ the updates
def collected(device: str = "cpu", steps: int = 60, **wm: Any) -> WMBPTT:
    """An agent with a filled replay, as ``train`` has it before its first update."""
    agent = harness.make_agent(device=device, **wm)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(steps):
        obs = agent._collect_step(obs, stats)
    return agent


def world_model_state(agent: WMBPTT) -> dict[str, Any]:
    return {"dynamics": agent.dynamics.state_dict(), "optim": agent.wm_optim.state_dict()}


def policy_state(agent: WMBPTT) -> dict[str, Any]:
    return {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "optim_actor": agent.actor_optim.state_dict(),
        "optim_critic": agent.critic_optim.state_dict(),
    }


def fix_starts(agent: WMBPTT, starts: tuple[Tensor, Tensor]) -> None:
    imagine = agent._imagine
    agent._imagine = lambda batch, **kwargs: imagine(batch, starts=starts, **kwargs)  # type: ignore[method-assign]


def assert_close_nest(got: dict[str, Any], want: dict[str, Any], **tolerance: float) -> None:
    for name in want:
        for key, value in want[name].items():
            if isinstance(value, Tensor) and value.is_floating_point():
                torch.testing.assert_close(got[name][key], value, **tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_collection_is_the_same_bit_for_bit(device: str) -> None:
    host, on_device = collected(device, fused_adam=True), collected(device, **ON)
    assert harness.state_equal(harness.replay_tensors(host), harness.replay_tensors(on_device))
    assert harness.state_equal(host.actor.state_dict(), on_device.actor.state_dict())


@pytest.mark.parametrize("device", DEVICES)
def test_the_world_model_update_is_the_same_bit_for_bit(device: str) -> None:
    """Fused Adam on both sides, normalizers frozen: only the gate and the freeze
    test differ. A run with live normalizers differs in their masked sums, below."""
    wm = {"norm_freeze_samples": 0.0}
    host, on_device = collected(device, fused_adam=True, **wm), collected(device, **ON, **wm)
    for agent in (host, on_device):
        torch.manual_seed(3)
        torch.cuda.manual_seed_all(3)
        for _ in range(3):
            agent._update_world_model(defaultdict(list))
    assert harness.state_equal(world_model_state(host), world_model_state(on_device))


@pytest.mark.parametrize("device", DEVICES)
def test_live_normalizers_match_within_rtol(device: str) -> None:
    host, on_device = collected(device, fused_adam=True), collected(device, **ON)
    for agent in (host, on_device):
        torch.manual_seed(3)
        torch.cuda.manual_seed_all(3)
        agent._update_world_model(defaultdict(list))
    for name in ("in_norm", "out_norm"):
        a, b = getattr(host.dynamics, name), getattr(on_device.dynamics, name)
        assert a.count.item() == b.count.item()
        torch.testing.assert_close(b.mean, a.mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(b.var, a.var, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_the_actor_critic_update_is_the_same_bit_for_bit(device: str) -> None:
    """Fused Adam on both sides, with the same starts: their sampler is the one
    part whose random stream differs."""
    host, on_device = collected(device, fused_adam=True), collected(device, **ON)
    starts = on_device._device_starts(32, on_device.config.wm.horizon, None)
    for agent in (host, on_device):
        fix_starts(agent, starts)
        torch.manual_seed(3)
        torch.cuda.manual_seed_all(3)
        agent._update_actor_critic(defaultdict(list))
    assert harness.state_equal(policy_state(host), policy_state(on_device))


@pytest.mark.parametrize("device", DEVICES)
def test_against_the_default_adam_within_rtol(device: str) -> None:
    """Fused against the default foreach (CUDA) or single-tensor (CPU) Adam: the same
    update up to rounding, so no comparison here is bitwise."""
    wm = {"norm_freeze_samples": 0.0}
    control, on_device = collected(device, **wm), collected(device, **ON, **wm)
    starts = on_device._device_starts(32, on_device.config.wm.horizon, None)
    tolerance = {"rtol": 1e-4, "atol": 1e-5}
    for agent in (control, on_device):
        fix_starts(agent, starts)
        torch.manual_seed(3)
        torch.cuda.manual_seed_all(3)
        agent._update_world_model(defaultdict(list))
        agent._update_actor_critic(defaultdict(list))
    for get in (world_model_state, policy_state):
        want, got = get(control), get(on_device)
        assert_close_nest({k: v for k, v in got.items() if "optim" not in k}, {k: v for k, v in want.items() if "optim" not in k}, **tolerance)


def test_normalizers_freeze_at_the_same_sample_count() -> None:
    wm = {"norm_freeze_samples": 64.0}
    host, on_device = collected(fused_adam=True, **wm), collected(**ON, **wm)
    for agent in (host, on_device):
        torch.manual_seed(3)
        for _ in range(3):
            agent._update_world_model(defaultdict(list))
        assert bool(agent.dynamics.out_norm.frozen) and bool(agent.dynamics.in_norm.frozen)
    assert host.dynamics.out_norm.count.item() == on_device.dynamics.out_norm.count.item()


# ------------------------------------------------------------------ skipped steps
def poison_rollout(agent: WMBPTT) -> None:
    rollout = agent._wm_rollout

    def poisoned(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        errs, mask = rollout(*args, **kwargs)
        return errs * float("nan"), mask

    agent._wm_rollout = poisoned  # type: ignore[method-assign]


def poison_imagination(agent: WMBPTT) -> None:
    imagine = agent._imagine

    def poisoned(batch: int, **kwargs: Any) -> dict[str, Tensor]:
        out = imagine(batch, **kwargs)
        out["objective"] = out["objective"] * float("nan")
        return out

    agent._imagine = poisoned  # type: ignore[method-assign]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("opt", ["adam", "muon"])
def test_a_non_finite_world_model_step_is_skipped_and_counted(device: str, opt: str) -> None:
    agent = collected(device, wm_opt=opt, norm_freeze_samples=0.0, **ON)
    agent._update_world_model(defaultdict(list))
    before = harness.digest(world_model_state(agent))
    poison_rollout(agent)
    agent._update_world_model(defaultdict(list))
    assert harness.digest(world_model_state(agent)) == before
    assert agent._device_counts.tolist() == [agent.config.wm.wm_batches_per_iter, 0.0, 0.0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("opt", ["adam", "muon"])
def test_a_non_finite_actor_step_is_skipped_and_counted(device: str, opt: str) -> None:
    agent = collected(device, actor_opt=opt, critic_fit=False, **ON)
    agent._update_actor_critic(defaultdict(list))
    before = harness.digest({"actor": agent.actor.net.state_dict(), "optim": agent.actor_optim.state_dict()})
    poison_imagination(agent)
    agent._update_actor_critic(defaultdict(list))
    assert harness.digest({"actor": agent.actor.net.state_dict(), "optim": agent.actor_optim.state_dict()}) == before
    assert agent._device_counts.tolist() == [0.0, 1.0, 0.0]


def test_a_log_row_reports_the_counts_and_clears_them() -> None:
    agent = collected(**ON)
    agent._device_counts += torch.tensor([2.0, 1.0, 0.0])
    row = agent._flush(defaultdict(list), defaultdict(list), 0.1)
    assert (row["wm/nonfinite"], row["actor/nonfinite"], row["imagine/invalid_starts"]) == (2.0, 1.0, 0.0)
    assert agent._device_counts.tolist() == [0.0] * len(DEVICE_COUNTS)


# ------------------------------------------------------------------ starts
class StartTask:
    """Clips of known lengths; replay rows hold their cursors."""

    history = 3
    action_dim = 1
    sensor_noise = False
    lengths = torch.tensor([40, 25, 60])

    def valid_imagination_start(self, rank: Tensor, frame: Tensor, horizon: int) -> Tensor:
        return self.lengths[rank] - 1 - frame >= horizon


def start_agent(task: Any = None, *, steps: int = 200, capacity: int = 150, num_envs: int = 12) -> WMBPTT:
    agent = WMBPTT.__new__(WMBPTT)
    agent.config = SimpleNamespace(device="cpu", wm=WMBPTTConfig(horizon=8, no_sync=True, fused_adam=True))
    agent.task = task or StartTask()
    agent.replay = SequenceReplay(num_envs, capacity, 1, 1, 1, torch.device("cpu"))
    agent._num_envs = num_envs
    agent._device_counts = torch.zeros(len(DEVICE_COUNTS))
    rank = torch.arange(num_envs) % StartTask.lengths.numel()
    for step in range(steps):
        frame = (step + 7 * torch.arange(num_envs)) % StartTask.lengths[rank]
        agent.replay.add(
            frame.float().unsqueeze(-1), torch.zeros(num_envs, 1), torch.zeros(num_envs, 1), torch.zeros(num_envs, 1),
            rank, frame, torch.full((num_envs,), step), torch.zeros(num_envs, dtype=torch.bool),
        )
    return agent


def cell_counts(agent: WMBPTT, tau: Tensor, env: Tensor) -> Counter:
    rp = agent.replay
    assert rp is not None
    slot = rp._slot(tau)
    return Counter(zip(rp.rank[slot, env].tolist(), (rp.frame[slot, env] // 4).tolist(), strict=True))


def test_device_starts_are_valid_and_inside_the_ring() -> None:
    agent = start_agent()
    rp = agent.replay
    assert rp is not None
    gen = torch.Generator().manual_seed(0)
    tau, env = agent._device_starts(256, 8, gen)
    assert tau.shape == env.shape == (256,)
    assert int(tau.min()) >= rp.oldest + agent.task.history - 1
    assert int(tau.max()) <= rp.steps - 1
    slot = rp._slot(tau)
    assert agent.task.valid_imagination_start(rp.rank[slot, env], rp.frame[slot, env], 8).all()
    assert agent._device_counts.tolist() == [0.0, 0.0, 0.0]


def test_device_starts_follow_the_rejection_sampler() -> None:
    """Chi-square of (clip, frame bin) counts: both samplers against the exact
    distribution, uniform over the valid rows, and against each other."""
    agent = start_agent()
    rp = agent.replay
    assert rp is not None
    lo, hi = rp.oldest + agent.task.history - 1, rp.steps - 1
    rows = torch.arange(lo, hi + 1)
    grid_tau = rows.repeat_interleave(agent._num_envs)
    grid_env = torch.arange(agent._num_envs).repeat(rows.numel())
    slot = rp._slot(grid_tau)
    ok = agent.task.valid_imagination_start(rp.rank[slot, grid_env], rp.frame[slot, grid_env], 8)
    exact = cell_counts(agent, grid_tau[ok], grid_env[ok])
    total = sum(exact.values())

    gen = torch.Generator().manual_seed(11)
    batch, draws = 2048, 100
    fast: Counter = Counter()
    slow: Counter = Counter()
    for _ in range(draws):
        fast.update(cell_counts(agent, *agent._device_starts(batch, 8, gen)))
        tau = torch.randint(lo, hi + 1, (batch,), generator=gen)
        env = torch.randint(0, agent._num_envs, (batch,), generator=gen)
        slow.update(cell_counts(agent, *agent._filter_imagination_starts(tau, env, 8, resample=True, generator=gen)))

    n = batch * draws
    cells = len(exact)
    bound = cells + 5 * math.sqrt(2 * cells)
    assert set(fast) <= set(exact) and set(slow) <= set(exact)
    for counts in (fast, slow):
        chi2 = sum((counts[c] - n * e / total) ** 2 / (n * e / total) for c, e in exact.items())
        assert chi2 < bound, (chi2, bound)
    two_sample = sum((fast[c] - slow[c]) ** 2 / (fast[c] + slow[c]) for c in exact if fast[c] + slow[c])
    assert two_sample < bound, (two_sample, bound)


class MostlyInvalidTask(StartTask):
    def valid_imagination_start(self, rank: Tensor, frame: Tensor, horizon: int) -> Tensor:
        return frame == 0


def test_too_few_valid_starts_raise_at_the_log_row() -> None:
    agent = start_agent(MostlyInvalidTask())
    agent._device_starts(256, 8, None)
    assert agent._device_counts[2] > 0
    with pytest.raises(RuntimeError, match="imagination starts"):
        agent._read_device_counts({})


def test_explicit_starts_keep_the_raising_check() -> None:
    agent = collected(**ON)
    bad = torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)
    rp = agent.replay
    rp.frame[:] = 10_000  # type: ignore[union-attr, index]
    with pytest.raises(ValueError, match="Explicit imagination starts"):
        agent._imagine(4, grad=False, starts=bad)


# ------------------------------------------------------------------ whole runs
def test_a_no_sync_run_trains_and_resumes(tmp_path: Path) -> None:
    frames = harness.frames_for(4)
    agent = harness.make_agent(frames=frames, **ON)
    rows: list[dict[str, float]] = []
    agent.log_training_metrics = lambda row, **_kwargs: rows.append(row)  # type: ignore[method-assign]
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    agent.train()
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())
    assert not harness.state_equal(before, agent.actor.state_dict())
    assert rows and all(row["wm/nonfinite"] == 0 and row["actor/nonfinite"] == 0 for row in rows)
    path = tmp_path / "run.pt"
    agent.save_model(path)
    resumed = harness.make_agent(frames=harness.frames_for(6), **ON)
    resumed.load_model(str(path))
    assert harness.state_equal(agent.actor.state_dict(), resumed.actor.state_dict())
    resumed.train()


def test_the_default_run_has_no_device_counts() -> None:
    agent = harness.make_agent(frames=harness.frames_for(2))
    agent.train()
    assert not hasattr(agent, "_device_counts")


# ------------------------------------------------------------------ host reads
def loop(agent: WMBPTT, obs: dict[str, Any], stats: dict[str, list]) -> dict[str, Any]:
    metrics: dict[str, list] = defaultdict(list)
    for _ in range(3):
        obs = agent._collect_step(obs, stats)
    agent._update_world_model(metrics)
    agent._update_actor_critic(metrics)
    return obs


class SyncFreeEnv(harness.FakeIsaacEnv):
    """The stand-in's ``step`` reads the host itself; the agent's reads are the census."""

    def step(self, action: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor, dict[str, Any]]:
        zeros = torch.zeros(self.n, device=self.device)
        self.frame = self.frame + 1
        self.episode = self.episode + 1
        self.frame = torch.where(self.frame >= self.v3_library.length[self.rank] - 2, 0, self.frame)
        return self._obs(), zeros, zeros.bool(), zeros.bool(), {"log": {"host_value": 1.0}}


def census_agent(**wm: Any) -> tuple[WMBPTT, dict[str, Any], dict[str, list]]:
    agent = harness.make_agent(device="cuda:0", num_envs=64, **wm)
    agent._isaac = SyncFreeEnv(64, "cuda:0")
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(60):
        obs = agent._collect_step(obs, stats)
    obs = loop(agent, obs, stats)  # first calls: cuBLAS handles, optimizer state
    torch.cuda.synchronize()
    return agent, obs, stats


@cuda
def test_the_census_sees_the_default_loop() -> None:
    """Control: under the same check the default loop reads the host."""
    agent, obs, stats = census_agent()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError, match="synchroniz"):
            loop(agent, obs, stats)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@cuda
@pytest.mark.parametrize(
    "wm",
    [
        {},
        {"actor_opt": "muon", "wm_opt": "muon", "actor_ema": 0.99},
        {"actor_estimator": "score_function"},
        {"dynamics_model": "diffsr", "diffsr_feature_dim": 16, "diffsr_embed_dim": 32},
    ],
    ids=["adam", "muon_ema", "score_function", "diffsr"],
)
def test_the_update_loop_does_not_synchronize(wm: dict[str, Any]) -> None:
    agent, obs, stats = census_agent(**ON, **wm)
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(2):
            obs = loop(agent, obs, stats)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert stats["host_value"][-1] == 1.0  # a host value waits for the log row
    row = agent._flush(defaultdict(list), stats, 0.1)
    assert row["env/host_value"] == 1.0
