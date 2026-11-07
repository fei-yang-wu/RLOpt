"""Golden test: a default WMBPTT run stays bit-identical to the baseline.

Contract 1 of ``wiki/code-opt-ndm-port.md``: every option is off by default, and
a default run keeps the bits of RLOpt ``71d9f8f`` with the parent at ``a78d8ab``.
Each case runs the real ``WMBPTT.__init__`` and ``train`` on the real
``G1V3TrackingTask`` (``wm_bptt_harness``) and compares a SHA-256 per group of
weights, optimizer state, RNG state, replay contents and counters against
``data/wm_bptt_golden.json``.

Exact bits depend on the kernels, so references are stored per platform (torch
build plus CPU capability or GPU name). On a platform without one, the case
skips: it then protects nothing. To add a platform, run the baseline commits
there with ``WM_BPTT_GOLDEN_WRITE=1``. Never regenerate from changed code: a
digest that moves means the default run moved.

GPU cases use the configured sizes, with bf16 autocast, and run in a subprocess
so test order cannot change cuBLAS or compile state. They need about 10 GB.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness

from rlopt.agent.wm_bptt.config import WMBPTTConfig

REFERENCE = Path(__file__).parent / "data" / "wm_bptt_golden.json"
BASELINE = {"rlopt": "71d9f8f", "parent": "a78d8ab"}
WRITE = os.environ.get("WM_BPTT_GOLDEN_WRITE") == "1"
CPU_ITERATIONS = 6
CUDA_ITERATIONS = 5
CUDA_ENVS = 256

CPU_CASES: dict[str, dict[str, Any]] = {
    "default": {},
    "real_rebar": {"actor_source": "real", "critic_source": "real", "actor_estimator": "rebar"},
    "score_function": {"actor_estimator": "score_function"},
    "diffsr": {"dynamics_model": "diffsr", "diffsr_feature_dim": 16, "diffsr_embed_dim": 32},
}
CUDA_CASES: dict[str, dict[str, Any]] = {
    "default": {},
    "compile": {"compile": True},  # what the campaigns run
}


def platform(device: str) -> str:
    if device == "cpu":
        return f"cpu|torch={torch.__version__}|{torch.backends.cpu.get_cpu_capability()}"
    return f"cuda|torch={torch.__version__}|{torch.cuda.get_device_name(0)}"


def references() -> dict[str, Any]:
    if REFERENCE.exists():
        return json.loads(REFERENCE.read_text())
    return {"baseline": BASELINE, "platforms": {}}


def write_reference(key: str, case: str, fingerprint: dict[str, str]) -> None:
    data = references()
    data["platforms"].setdefault(key, {})[case] = fingerprint
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def checkpoint_keys(agent: Any) -> str:
    """Top-level names of the checkpoint a default run writes."""
    newest = max(agent._checkpoint_dir().glob("model_step_*.pt"), key=lambda p: p.stat().st_mtime)
    return harness.digest(sorted(torch.load(newest, map_location="cpu", weights_only=False)))


def run_cpu(case: str) -> dict[str, str]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        frames = harness.frames_for(CPU_ITERATIONS)
        agent = harness.make_agent(frames=frames, save_interval=frames // 2, **CPU_CASES[case])
        agent.train()
        return {**harness.fingerprint(agent), "checkpoint_keys": checkpoint_keys(agent)}
    finally:
        torch.set_num_threads(threads)


def run_cuda(case: str) -> dict[str, str]:
    torch.set_num_threads(1)
    frames = harness.frames_for(CUDA_ITERATIONS, num_envs=CUDA_ENVS, wm=WMBPTTConfig(min_replay_steps=40))
    agent = harness.make_agent(
        device="cuda:0",
        num_envs=CUDA_ENVS,
        small=False,
        frames=frames,
        wm_warmup_iters=1,  # the default 500 would never reach the actor
        replay_steps=64,
        log_interval_iters=2,
        **CUDA_CASES[case],
    )
    agent.train()
    return harness.fingerprint(agent)


def compare(key: str, case: str, got: dict[str, str]) -> None:
    if WRITE:
        write_reference(key, case, got)
        return
    want = references()["platforms"].get(key, {}).get(case)
    if want is None:
        pytest.skip(f"no golden reference for {key} / {case}: this platform has no drift protection")
    moved = sorted(name for name in want.keys() | got.keys() if want.get(name) != got.get(name))
    assert not moved, f"default WMBPTT run drifted from {BASELINE} in {moved} ({key}, case {case})"


@pytest.mark.parametrize("case", CPU_CASES)
def test_cpu_run_is_bit_identical_to_the_baseline(case: str) -> None:
    compare(platform("cpu"), case, run_cpu(case))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("case", CUDA_CASES)
def test_cuda_run_is_bit_identical_to_the_baseline(case: str) -> None:
    done = subprocess.run(
        [sys.executable, __file__, case], capture_output=True, text=True, check=False, cwd=Path(__file__).parent
    )
    assert done.returncode == 0, done.stderr[-4000:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    compare(platform("cuda"), case, got)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"actor_lr": 2.0000001e-4}, {"actor", "optim_actor"}),
        ({"wm_lr": 2.0000001e-4}, {"dynamics", "optim_wm"}),
        ({"critic_lr": 2.0000001e-4}, {"critic", "optim_critic"}),
    ],
)
def test_the_fingerprint_sees_a_small_change(change: dict[str, float], expected: set[str]) -> None:
    """A learning rate moved by parts in 1e7: the digests must name it."""
    frames = harness.frames_for(CPU_ITERATIONS)
    fingerprints = []
    for wm in ({}, change):
        agent = harness.make_agent(frames=frames, **wm)
        agent.train()
        fingerprints.append(harness.fingerprint(agent))
    moved = {name for name, value in fingerprints[0].items() if fingerprints[1][name] != value}
    assert expected <= moved


def test_the_fingerprint_sees_one_extra_random_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(hook: Callable[[], Any] | None) -> dict[str, str]:
        agent = harness.make_agent(frames=harness.frames_for(CPU_ITERATIONS))
        if hook is not None:
            update = agent._update_world_model
            monkeypatch.setattr(agent, "_update_world_model", lambda metrics: (hook(), update(metrics))[1])
        agent.train()
        return harness.fingerprint(agent)

    clean, drawn = run(None), run(lambda: torch.rand(1))
    assert clean["rng_cpu"] != drawn["rng_cpu"]


if __name__ == "__main__":
    print(json.dumps(run_cuda(sys.argv[1])))
