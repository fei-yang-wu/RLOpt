"""A log row reads the device a fixed number of times, not once per value.

The real-env census of 2026-11-07 (G1-v3, ``no_sync`` with ``update_graphs``) had
the agent's only host reads at log rows: 62 per row, one per logged value. The row
now reads its values together. Its numbers are unchanged, bit for bit.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness

from rlopt.agent.wm_bptt.agent import _read_scalars

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEVICES = ["cpu", pytest.param("cuda:0", marks=cuda)]


@pytest.mark.parametrize("device", DEVICES)
def test_read_scalars_equals_float_of_each_value(device: str) -> None:
    gen = torch.Generator(device=device).manual_seed(0)
    values: dict[str, Any] = {
        "f32": torch.randn((), device=device, generator=gen) * 1e-3,
        "f64": torch.randn((), device=device, dtype=torch.float64, generator=gen),
        "one": torch.randn(1, device=device, generator=gen),
        "long": torch.tensor(2**40 + 3, device=device),
        "flag": torch.tensor(True, device=device),
        "bf16": torch.randn((), device=device, generator=gen).bfloat16(),
        "host_tensor": torch.tensor(0.1),
        "python": 0.25,
        "int": 7,
    }
    got = _read_scalars(values)
    assert list(got) == list(values)
    for key, value in values.items():
        assert got[key] == float(value), key
        assert type(got[key]) is float


def collected(device: str, **wm: Any):
    agent = harness.make_agent(device=device, **wm)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(60):
        obs = agent._collect_step(obs, stats)
    metrics: dict[str, list] = defaultdict(list)
    agent._update_world_model(metrics)
    agent._update_actor_critic(metrics)
    return agent, metrics, stats


@pytest.mark.parametrize("device", DEVICES)
def test_a_row_holds_the_values_of_one_read_per_value(device: str) -> None:
    agent, metrics, stats = collected(device)
    want = {}
    for src in (metrics, stats):
        for key, values in src.items():
            name = key if "/" in key else f"env/{key}"
            want[name] = float(torch.stack([torch.as_tensor(v, device=agent.device, dtype=torch.float32) for v in values]).mean())
    torch.manual_seed(5)
    row = agent._flush(metrics, stats, 0.1)
    for name, value in want.items():
        assert row[name] == value, name
    # The k-step errors, as one host read per value gave them, on the same windows.
    torch.manual_seed(5)
    rp, wm = agent.replay, agent.config.wm
    tau, env = rp.sample_starts(wm.eval_batch, wm.eval_window)  # type: ignore[union-attr]
    errs, mask = agent._wm_rollout(rp.window(tau, env, wm.eval_window + 1), wm.eval_window, grad=False)  # type: ignore[union-attr]
    m = mask[:, 7] > 0
    e = errs[m, 7]
    assert row["wm_eval/k8_root_pos_mm"] == float(e[:, 0:3].norm(dim=-1).mean() * 1000)
    assert row["wm_eval/k8_joint_pos_deg"] == float(e[:, 12:41].abs().mean() * 180 / torch.pi)
    assert row["wm_eval/k8_windows"] == float(m.float().sum())


def host_reads(fn: Any) -> int:
    """Synchronizing calls of ``fn`` (``set_sync_debug_mode`` warns once of itself,
    before the record starts)."""
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return sum("synchroniz" in str(w.message) for w in caught)


@cuda
def test_a_row_reads_the_device_a_fixed_number_of_times() -> None:
    """With the options of the census on: one read for the means, one per k-step
    horizon (its selection) and one for its errors, and one each for the device
    counts, the return trigger's sums and the episode sums. Twenty more logged
    names add none."""
    wm = {"no_sync": True, "fused_adam": True, "actor_lr_mult": 2.0, "actor_lr_decay_return": 0.5, "actor_lr_decay_len": 3}
    agent, metrics, stats = collected("cuda:0", **wm)
    base = host_reads(lambda: agent._flush(metrics, stats, 0.1))
    agent, metrics, stats = collected("cuda:0", **wm)
    for i in range(20):
        metrics[f"extra/{i}"].append(torch.rand((), device="cuda:0"))
    more = host_reads(lambda: agent._flush(metrics, stats, 0.1))
    horizons = sum(k <= agent.config.wm.eval_window for k in (1, 8, 32))
    assert base == more == 1 + horizons + 1 + 3
