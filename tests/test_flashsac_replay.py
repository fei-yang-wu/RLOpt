# ruff: noqa: PT018, PT006, PT011, B905
"""Tests for the FlashSAC time-major replay ring (replay.py)."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest
import torch

_REPLAY_PATH = (
    Path(__file__).resolve().parents[1] / "rlopt" / "agent" / "flashsac" / "replay.py"
)
_spec = importlib.util.spec_from_file_location(
    "flashsac_replay_under_test", _REPLAY_PATH
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TimeMajorReplay = _mod.TimeMajorReplay

DO, DC, A = 4, 6, 3
CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")


def _mk_obs(step: float, env: int, dim: int, terminal: bool = False) -> torch.Tensor:
    """Obs encoding (step, env) so a sample can be decoded exactly (fp32)."""
    v = torch.zeros(dim)
    v[0] = step + (0.5 if terminal else 0.0)
    v[1] = env
    return v


class Sim:
    """Random episodic data with explicit next obs, time-outs and terminations."""

    def __init__(self, n_envs, p_term=0.08, p_trunc=0.08, p_both=0.02, seed=0):
        self.n = n_envs
        self.g = torch.Generator().manual_seed(seed)
        self.p = (p_term, p_trunc, p_both)
        self.steps: list[dict] = []

    def step(self) -> dict:
        t = len(self.steps)
        N = self.n
        r = torch.rand(N, generator=self.g)
        p_term, p_trunc, p_both = self.p
        term = r < p_term
        trunc = (r >= p_term) & (r < p_term + p_trunc)
        both = (r >= p_term + p_trunc) & (r < p_term + p_trunc + p_both)
        term = term | both
        trunc = trunc | both
        done = term | trunc
        d = {
            "obs": torch.stack([_mk_obs(t, e, DO) for e in range(N)]),
            "critic_obs": torch.stack([_mk_obs(t, e, DC) for e in range(N)]),
            "action": torch.rand(N, A, generator=self.g) * 2 - 1,
            "reward": torch.randn(N, generator=self.g),
            "terminated": term,
            "truncated": trunc,
        }
        nxt = torch.stack(
            [_mk_obs(t + 1, e, DO, terminal=bool(done[e])) for e in range(N)]
        )
        # Done rows: terminal obs (marker +0.5); others: obs of the next step.
        nxt_c = torch.stack(
            [_mk_obs(t + 1, e, DC, terminal=bool(done[e])) for e in range(N)]
        )
        d["next_obs"], d["next_critic_obs"] = nxt, nxt_c
        self.steps.append(d)
        return d


def reference_nstep(steps, t, e, n, gamma):
    """Slow pure-Python n-step return with explicit next obs."""
    ret, term_any, k_eff = 0.0, False, 0
    for k in range(n):
        s = steps[t + k]
        ret += gamma**k * float(s["reward"][e])
        k_eff += 1
        if bool(s["terminated"][e]) or bool(s["truncated"][e]):
            term_any = bool(s["terminated"][e])
            last = s
            break
        last = s
    # A terminated window never bootstraps: the buffer returns the terminating
    # row's own observation as filler instead of pooling the terminal obs.
    src_obs = "obs" if term_any else "next_obs"
    src_critic = "critic_obs" if term_any else "next_critic_obs"
    return {
        "reward_n": ret,
        "terminated_n": 1.0 if term_any else 0.0,
        "discount_n": gamma**k_eff,
        "n_eff": k_eff,
        "next_obs": last[src_obs][e],
        "next_critic_obs": last[src_critic][e],
    }


def make_buf(N, T, n=3, gamma=0.9, device="cpu", dtype=torch.float32, **kw):
    kw.setdefault("pool_fraction", 0.5)  # the sim ends episodes far more often than G1
    return TimeMajorReplay(
        N, T, DO, DC, A, n_step=n, gamma=gamma, device=device, obs_dtype=dtype, **kw
    )


def _fill(buf, sim, count, device="cpu"):
    for _ in range(count):
        d = sim.step()
        buf.add({k: v.to(device) for k, v in d.items()})


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("steps,T", [(30, 64), (90, 24)])  # no wrap, wrap
def test_matches_reference_exhaustively(n, steps, T):
    N, gamma = 6, 0.9
    sim = Sim(N, seed=n + steps)
    buf = make_buf(N, T, n=n, gamma=gamma)
    _fill(buf, sim, steps)
    valid = buf.valid_rows
    assert valid == min(steps, T) - (n + 1)
    assert len(buf) == valid * N
    oldest = buf.oldest_logical
    assert oldest == max(0, steps - T)
    u = torch.arange(valid).repeat_interleave(N)
    env = torch.arange(N).repeat(valid)
    out = buf._window(u, env, oldest % T, oldest)
    for i in range(u.numel()):
        t, e = oldest + int(u[i]), int(env[i])
        ref = reference_nstep(sim.steps, t, e, n, gamma)
        assert out["obs"][i, 0].item() == t and out["obs"][i, 1].item() == e
        assert out["reward_n"][i].item() == pytest.approx(
            ref["reward_n"], rel=1e-5, abs=1e-6
        )
        assert out["terminated_n"][i].item() == ref["terminated_n"]
        assert out["discount_n"][i].item() == pytest.approx(ref["discount_n"], rel=1e-6)
        assert int(out["n_eff"][i]) == ref["n_eff"]
        assert torch.equal(out["next_obs"][i], ref["next_obs"])
        assert torch.equal(out["next_critic_obs"][i], ref["next_critic_obs"])
        assert out["reward_1"][i].item() == pytest.approx(
            float(sim.steps[t]["reward"][e])
        )
    assert buf.pool_miss_count() == 0


def test_cases_covered_by_sim():
    """The random data hits time-outs, terminations, both flags and short episodes."""
    sim = Sim(6, seed=3)
    for _ in range(60):
        sim.step()
    term = torch.stack([s["terminated"] for s in sim.steps])
    trunc = torch.stack([s["truncated"] for s in sim.steps])
    assert (term & ~trunc).any() and (trunc & ~term).any() and (term & trunc).any()
    # Two done flags within n=3 steps of one env (episode shorter than n).
    done = term | trunc
    assert (done[:-1] & done[1:]).any()


def test_sample_decodes_and_respects_guard():
    N, T, n = 5, 20, 3
    sim = Sim(N, seed=11)
    buf = make_buf(N, T, n=n)
    _fill(buf, sim, 77)  # wraps several times
    total = 77
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        b = buf.sample(256, generator=g)
        t = b["obs"][:, 0].long()
        # not overwritten, and the window incl. next-obs row is already written
        assert int(t.min()) >= total - T
        assert int(t.max()) <= total - 1 - (n + 1)
        assert int((t + b["n_eff"]).max()) <= total - 1
        for i in range(0, 256, 17):
            ref = reference_nstep(sim.steps, int(t[i]), int(b["env"][i]), n, 0.9)
            assert b["reward_n"][i].item() == pytest.approx(
                ref["reward_n"], rel=1e-5, abs=1e-6
            )


def test_guard_rows_exclude_head_before_full():
    buf = make_buf(2, 16, n=3)
    sim = Sim(2)
    with pytest.raises(RuntimeError):
        buf.sample(4)
    _fill(buf, sim, 4)  # n+1 = 4 rows: still none valid
    assert buf.valid_rows == 0 and len(buf) == 0
    _fill(buf, sim, 1)
    assert buf.valid_rows == 1
    b = buf.sample(32)
    assert (b["obs"][:, 0] == 0).all()


def test_terminal_pool_timeout_uses_terminal_obs_not_reset_obs():
    N, T, n = 1, 16, 2
    buf = make_buf(N, T, n=n, gamma=0.5)
    # Env time-outs at step 2; step 3 is the first step of a new episode.
    for t in range(10):
        trunc = t == 2
        buf.add(
            {
                "obs": _mk_obs(t, 0, DO)[None],
                "critic_obs": _mk_obs(t, 0, DC)[None],
                "action": torch.zeros(1, A),
                "reward": torch.tensor([1.0]),
                "terminated": torch.tensor([False]),
                "truncated": torch.tensor([trunc]),
                "next_obs": _mk_obs(t + 1, 0, DO, terminal=trunc)[None],
                "next_critic_obs": _mk_obs(t + 1, 0, DC, terminal=trunc)[None],
            }
        )
    # Window starting at 1: steps 1 and 2 (truncating). n_eff = 2.
    out = buf._window(torch.tensor([1]), torch.tensor([0]), 0, 0)
    assert out["terminated_n"].item() == 0.0  # time-out keeps the bootstrap
    assert out["n_eff"].item() == 2
    assert out["discount_n"].item() == pytest.approx(0.25)
    assert out["reward_n"].item() == pytest.approx(1.5)
    assert out["next_obs"][0, 0].item() == 3.5  # terminal obs, not reset obs (3.0)
    assert out["next_critic_obs"][0, 0].item() == 3.5
    # Window starting at 2 ends immediately at the truncating step.
    out = buf._window(torch.tensor([2]), torch.tensor([0]), 0, 0)
    assert out["n_eff"].item() == 1 and out["next_obs"][0, 0].item() == 3.5


def test_next_obs_equals_explicit_next_layout_n1():
    N, T = 4, 32
    sim = Sim(N, seed=5)
    buf = make_buf(N, T, n=1)
    _fill(buf, sim, 28)
    valid = buf.valid_rows
    u = torch.arange(valid).repeat_interleave(N)
    env = torch.arange(N).repeat(valid)
    out = buf._window(u, env, 0, 0)

    def _expected(a, b, key):
        st = sim.steps[int(a)]
        own = "obs" if key == "next_obs" else "critic_obs"
        return st[own if bool(st["terminated"][int(b)]) else key][int(b)]

    exp = torch.stack([_expected(a, b, "next_obs") for a, b in zip(u, env)])
    expc = torch.stack([_expected(a, b, "next_critic_obs") for a, b in zip(u, env)])
    assert torch.equal(out["next_obs"], exp)
    assert torch.equal(out["next_critic_obs"], expc)


def test_pool_overflow_is_detected():
    N, T = 4, 20
    # Every row times out; the pool (2N slots) cannot cover the ring.
    buf = make_buf(N, T, n=1, pool_fraction=0.01)
    sim = Sim(N, p_term=0.0, p_trunc=1.0, p_both=0.0)
    _fill(buf, sim, 18)
    buf.sample(512)
    assert buf.pool_miss_count() > 0


def test_extra_keys_and_state_roundtrip(tmp_path):
    N, T = 3, 12
    sim = Sim(N, seed=2)
    a = make_buf(N, T, n=2, extra_keys={"cmd": 2})
    for _ in range(15):
        d = sim.step()
        d["cmd"] = torch.stack([d["obs"][:, 0], d["obs"][:, 1]], 1)
        a.add(d)
    b = a.sample(16, generator=torch.Generator().manual_seed(1))
    assert torch.equal(b["cmd"][:, 0], b["obs"][:, 0])
    p = tmp_path / "ring.pt"
    a.save_storage(str(p))
    c = make_buf(N, T, n=2, extra_keys={"cmd": 2})
    c.load_storage(str(p))
    assert len(c) == len(a) and c.valid_rows == a.valid_rows
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    s1, s2 = a.sample(64, generator=g1), c.sample(64, generator=g2)
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    # counters-only state: empty ring after load
    d_ = make_buf(N, T, n=2, extra_keys={"cmd": 2})
    d_.load_state_dict(a.state_dict())
    assert len(d_) == 0
    bad = make_buf(N, T + 1, n=2, extra_keys={"cmd": 2})
    with pytest.raises(ValueError):
        bad.load_state_dict(a.state_dict())


def test_determinism_with_seeded_generator():
    N, T = 5, 24
    sim = Sim(N, seed=9)
    buf = make_buf(N, T, n=3)
    _fill(buf, sim, 40)
    s1 = buf.sample(128, generator=torch.Generator().manual_seed(3))
    s2 = buf.sample(128, generator=torch.Generator().manual_seed(3))
    s3 = buf.sample(128, generator=torch.Generator().manual_seed(4))
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    assert not torch.equal(s1["row"], s3["row"]) or not torch.equal(
        s1["env"], s3["env"]
    )


def test_sample_does_not_mutate_storage():
    N, T = 4, 24
    sim = Sim(N, seed=1)
    buf = make_buf(N, T, n=3)
    _fill(buf, sim, 30)
    snap = {k: v.clone() for k, v in buf._storage_tensors().items()}
    for _ in range(5):
        buf.sample(64)
        buf.sample_graph_safe(64)
    for k, v in buf._storage_tensors().items():
        assert torch.equal(v, snap[k]), k


def test_graph_safe_matches_window_distribution_and_range():
    N, T, n = 5, 24, 3
    sim = Sim(N, seed=6)
    buf = make_buf(N, T, n=n)
    _fill(buf, sim, 50)
    vt = buf.valid_rows_tensor
    assert vt.item() == buf.valid_rows
    b = buf.sample_graph_safe(4096, vt, buf.oldest_row_tensor)
    t = b["obs"][:, 0].long()
    assert int(t.min()) >= 50 - T and int(t.max()) <= 50 - 1 - (n + 1)
    # uniform over rows (loose)
    counts = torch.bincount(t - (50 - T), minlength=buf.valid_rows).float()
    assert counts.min() > 0.5 * counts.mean()
    # the range follows later adds (not baked)
    _fill(buf, sim, 3)
    assert vt.item() == buf.valid_rows
    b = buf.sample_graph_safe(1024)
    assert int(b["obs"][:, 0].max()) <= 53 - 1 - (n + 1)


def test_bf16_roundtrip_tolerance():
    N, T = 3, 12
    buf = make_buf(N, T, n=1, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, DO, generator=g) * 3
    xc = torch.randn(N, DC, generator=g) * 3
    for _ in range(8):
        buf.add(
            {
                "obs": x,
                "critic_obs": xc,
                "action": torch.zeros(N, A),
                "reward": torch.zeros(N),
                "terminated": torch.zeros(N, dtype=torch.bool),
                "truncated": torch.zeros(N, dtype=torch.bool),
                "next_obs": x,
                "next_critic_obs": xc,
            }
        )
    b = buf.sample(64)
    # bf16 has an 8-bit significand: relative error <= 2^-8
    assert b["obs"].dtype == torch.float32
    idx = b["env"]
    assert torch.allclose(b["obs"], x[idx], rtol=2**-8, atol=1e-6)
    assert torch.allclose(b["next_critic_obs"], xc[idx], rtol=2**-8, atol=1e-6)


def test_bytes_formula_and_memory():
    per = TimeMajorReplay.bytes_per_transition(351, 544, 29)
    assert per == 1916
    buf = make_buf(4, 16, n=3, dtype=torch.bfloat16, extra_keys={"c": 2})
    exp = 16 * 4 * ((DO + DC) * 4 // 2 + A * 4 + 4 + 2 + 4 + 8) + (
        buf.pool_slots + 1
    ) * ((DO + DC) * 2 + 8)
    assert buf.memory_bytes() == exp


@needs_cuda
def test_cuda_matches_cpu_reference_and_no_host_sync():
    dev = "cuda"
    N, T, n = 64, 24, 3
    sim = Sim(N, seed=21)
    buf = make_buf(N, T, n=n, gamma=0.95, device=dev)
    for _ in range(10):  # warm up allocator / kernels
        d = sim.step()
        buf.add({k: v.to(dev) for k, v in d.items()})
    buf.sample(128)
    steps = []
    for _ in range(40):
        d = sim.step()
        steps.append({k: v.to(dev) for k, v in d.items()})
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for s in steps:
            buf.add(s)
        out = buf.sample(512)
        out2 = buf.sample_graph_safe(512)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    for o in (out, out2):
        t = o["obs"][:, 0].long().cpu()
        for i in range(0, 512, 29):
            ref = reference_nstep(sim.steps, int(t[i]), int(o["env"][i]), n, 0.95)
            assert o["reward_n"][i].item() == pytest.approx(
                ref["reward_n"], rel=1e-4, abs=1e-5
            )
            assert o["terminated_n"][i].item() == ref["terminated_n"]
            assert torch.equal(o["next_obs"][i].cpu(), ref["next_obs"])
    assert buf.pool_miss_count() == 0


@needs_cuda
def test_cuda_graph_capture_follows_range():
    dev = "cuda"
    N, T, n = 16, 32, 3
    sim = Sim(N, seed=8)
    buf = make_buf(N, T, n=n, device=dev)
    for _ in range(10):
        buf.add({k: v.to(dev) for k, v in sim.step().items()})
    buf.sample_graph_safe(64)  # warm up
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buf.sample_graph_safe(64)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = buf.sample_graph_safe(512)
    for _ in range(15):  # grow the range after capture
        buf.add({k: v.to(dev) for k, v in sim.step().items()})
    graph.replay()
    torch.cuda.synchronize()
    t = static["obs"][:, 0].long()
    assert int(t.max()) > 10 - 1 - (n + 1)  # new rows are reachable
    assert int(t.max()) <= 25 - 1 - (n + 1)


@needs_cuda
def test_throughput_microbenchmark(capsys):
    dev = "cuda"
    N, B, rows = 20000, 8192, 16
    free = torch.cuda.mem_get_info()[0]
    per = TimeMajorReplay.bytes_per_transition(351, 544, 29) + 100
    if per * N * rows * 1.5 > free:
        pytest.skip("not enough GPU memory for N=20000")
    buf = TimeMajorReplay(N, rows, 351, 544, 29, n_step=3, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    step = {
        "obs": torch.randn(N, 351, device=dev),
        "critic_obs": torch.randn(N, 544, device=dev),
        "action": torch.rand(N, 29, device=dev) * 2 - 1,
        "reward": torch.randn(N, device=dev),
        "terminated": torch.rand(N, device=dev) < 0.001,
        "truncated": torch.rand(N, device=dev) < 0.001,
        "next_obs": torch.randn(N, 351, device=dev),
        "next_critic_obs": torch.randn(N, 544, device=dev),
    }
    for _ in range(rows + 4):
        buf.add(step)
    for _ in range(3):
        buf.sample(B, generator=g)
    torch.cuda.synchronize()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        out = buf.sample(B, generator=g)
    torch.cuda.synchronize()
    ts = (time.perf_counter() - t0) / iters
    nbytes = sum(v.numel() * v.element_size() for v in out.values())
    t0 = time.perf_counter()
    for _ in range(iters):
        buf.add(step)
    torch.cuda.synchronize()
    ta = (time.perf_counter() - t0) / iters
    with capsys.disabled():
        print(
            f"\n[replay bench] N={N} B={B} ring={buf.memory_bytes() / 2**30:.2f} GiB | "
            f"sample {ts * 1e3:.2f} ms, output {nbytes / 2**20:.1f} MiB "
            f"({nbytes / ts / 1e9:.1f} GB/s out) | add {ta * 1e3:.2f} ms"
        )


def test_terminations_do_not_consume_the_pool():
    """A termination flood must not overwrite the time-out entries of the pool.

    Regression: a G1 smoke with random warm-up ended 28 percent of rows by
    termination; the pool (5 percent of slots) was overwritten and every
    time-out window older than a few rows read another row's observation.
    """
    N, T, n = 8, 40, 1
    buf = make_buf(N, T, n=n, pool_fraction=0.01)  # pool = 2N = 16 entries
    g = torch.Generator().manual_seed(0)
    timeouts = []
    for t in range(36):
        term = torch.rand(N, generator=g) < 0.9  # termination flood
        trunc = torch.zeros(N, dtype=torch.bool)
        if t == 3:  # one time-out early in the ring, on env 5
            term[5] = False
            trunc[5] = True
            timeouts.append((t, 5))
        buf.add(
            {
                "obs": torch.stack([_mk_obs(t, e, DO) for e in range(N)]),
                "critic_obs": torch.stack([_mk_obs(t, e, DC) for e in range(N)]),
                "action": torch.zeros(N, A),
                "reward": torch.ones(N),
                "terminated": term,
                "truncated": trunc,
                "next_obs": torch.stack(
                    [_mk_obs(t + 1, e, DO, terminal=True) for e in range(N)]
                ),
                "next_critic_obs": torch.stack(
                    [_mk_obs(t + 1, e, DC, terminal=True) for e in range(N)]
                ),
            }
        )
    ((t, e),) = timeouts
    out = buf._window(torch.tensor([t]), torch.tensor([e]), 0, 0)
    assert out["terminated_n"].item() == 0.0
    assert out["next_obs"][0, 0].item() == t + 1.5  # its own terminal obs
    assert out["next_obs"][0, 1].item() == e
    # Terminated windows read the terminating row's own observation.
    rows = torch.arange(buf.valid_rows).repeat_interleave(N)
    envs = torch.arange(N).repeat(buf.valid_rows)
    win = buf._window(rows, envs, 0, 0)
    term_rows = win["terminated_n"] > 0
    assert term_rows.any()
    assert torch.equal(win["next_obs"][term_rows], win["obs"][term_rows])
    assert torch.isfinite(win["next_obs"]).all()
    assert buf.pool_miss_count() == 0
