"""``wm.elementwise_3x3``, ``wm.compile_eager`` and ``wm.combo_kernels``.

The first and last change process-wide compiler state, so every check runs in a
subprocess of this file and prints one JSON line. Compiled results differ from eager
ones by rounding, so these comparisons use ``rtol``; none is bitwise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from imitation_experiments.teleop.g1_kinematics import G1_TRACKED_BODY_NAMES, G1Kinematics
from imitation_experiments.world_model.g1_v3_math import rotate
from torch import Tensor

from rlopt.agent.wm_bptt.speed_options import combo_kernels, elementwise_3x3_products

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEVICE = "cuda:0"
ALL_ON = {
    "compile": True,
    "compile_eager": True,
    "elementwise_3x3": True,
    "fused_adam": True,
    "goal_table": True,
    "reference_body_table": True,
}


def run(*args: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, __file__, *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent,
        env={**os.environ, **(env or {})},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ the checks
def table_entry() -> Any:
    from torch._inductor import decomposition  # noqa: PLC0415

    return decomposition.decompositions[torch.ops.aten.bmm.default]


def check_default() -> dict[str, Any]:
    """A default agent leaves the process's compiler state alone."""
    entry = table_entry()
    harness.make_agent()
    return {
        "same_entry": table_entry() is entry,
        "tag": torch.compiler.config.cache_key_tag,
        "combo": torch._inductor.config.combo_kernels,
    }


def check_patch() -> dict[str, Any]:
    default = table_entry()
    torch.compiler.config.cache_key_tag = "mine"
    elementwise_3x3_products()
    patched = table_entry()
    tag = torch.compiler.config.cache_key_tag
    elementwise_3x3_products()
    return {
        "patched": patched is not default,
        "tag": tag,
        "second_call_changes_nothing": table_entry() is patched and torch.compiler.config.cache_key_tag == tag,
    }


def check_agent_applies() -> dict[str, Any]:
    harness.make_agent(compile=True, elementwise_3x3=True, combo_kernels=True)
    return {
        "patched": getattr(table_entry(), "elementwise_3x3", False),
        "combo": torch._inductor.config.combo_kernels,
        "limit": torch._dynamo.config.recompile_limit,
    }


def check_late_patch() -> dict[str, Any]:
    torch.compile(lambda x: x + 1, backend="eager")(torch.ones(2))
    try:
        elementwise_3x3_products()
    except RuntimeError as error:
        return {"refused": str(error)}
    return {"refused": ""}


def fk_inputs() -> tuple[Tensor, ...]:
    torch.manual_seed(0)
    n = 512
    p, v, w = (torch.randn(n, 3, device=DEVICE) for _ in range(3))
    rotation = torch.linalg.qr(torch.randn(n, 3, 3, device=DEVICE))[0]
    q, qd = 0.3 * torch.randn(n, 29, device=DEVICE), torch.randn(n, 29, device=DEVICE)
    return p, rotation, q, v, w, qd


def check_products(on: str) -> dict[str, Any]:
    """Rotations and FK, compiled at matmul precision ``high`` as training runs them,
    against fp32 eager."""
    from torch._inductor.utils import run_and_get_code  # noqa: PLC0415

    if on == "on":
        elementwise_3x3_products()
    torch.set_float32_matmul_precision("high")
    kin = G1Kinematics().to(DEVICE)
    args = fk_inputs()
    rotation, v = args[1], args[3]

    def fk(*fk_args: Tensor) -> tuple[Tensor, ...]:
        return kin.forward_with_velocity(*fk_args, G1_TRACKED_BODY_NAMES)

    with torch.no_grad():
        got, code = run_and_get_code(torch.compile(fk, dynamic=False), *args)
        turned, turned_code = run_and_get_code(torch.compile(rotate, dynamic=False), rotation, v)
        torch.set_float32_matmul_precision("highest")
        want, turned_want = fk(*args), rotate(rotation, v)
    return {
        "bmm_calls": sum("extern_kernels.bmm" in c for c in (*code, *turned_code)),
        "fk_error": max(relative(a, b) for a, b in zip(got, want, strict=True)),
        "rotate_error": relative(turned, turned_want),
    }


def check_cache(on: str) -> dict[str, Any]:
    """One compile, reporting the FX graph cache's verdict."""
    from torch._dynamo.utils import counters  # noqa: PLC0415

    if on == "on":
        elementwise_3x3_products()
    a = torch.ones(64, 3, 3, device=DEVICE)
    with torch.no_grad():
        torch.compile(lambda x, y: torch.bmm(x, y).sin(), dynamic=False)(a, a)
    return {"hit": counters["inductor"]["fxgraph_cache_hit"], "miss": counters["inductor"]["fxgraph_cache_miss"]}


def loop(agent: Any, obs: dict[str, Any], stats: dict[str, list]) -> dict[str, Any]:
    metrics: dict[str, list] = defaultdict(list)
    for _ in range(3):
        obs = agent._collect_step(obs, stats)
    agent._update_world_model(metrics)
    agent._update_actor_critic(metrics)
    return obs


def relative(a: Tensor, b: Tensor) -> float:
    return ((a - b).abs() / (1e-3 + b.abs())).max().item()


def check_compile_eager(no_sync: str) -> dict[str, Any]:
    """After a warm-up every compiled site holds its shapes: no recompile. Each site
    then agrees with its eager function."""
    agent = harness.make_agent(device=DEVICE, num_envs=64, **{**ALL_ON, "no_sync": no_sync == "no_sync"})
    agent.ensure_replay()
    t, mdp = agent.task, agent.task.mdp
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(50):
        obs = agent._collect_step(obs, stats)

    rp = agent.replay
    tau, env = rp.sample_starts(32, 2)
    win = rp.window(tau, env, 2)
    state, action, ctx = win["state"][:, 0], win["action"][:, 0], win["context"][:, 0]
    rank, frame = win["rank"][:, 0], win["frame"][:, 0]
    body = mdp.body_states(state)

    def measured() -> Tensor:
        with torch.no_grad():  # as ``_collect_step`` runs the env
            return mdp.reward(state, action, action, rank, frame, body_state=body)[0]

    for _ in range(2):
        obs = loop(agent, obs, stats)
        measured()
    with torch.compiler.set_stance("fail_on_recompile"):
        for _ in range(2):
            obs = loop(agent, obs, stats)
        reward = measured()
    with torch.no_grad():
        return {
            "limit": torch._dynamo.config.recompile_limit,
            "reward_error": relative(reward, mdp._reward(state, action, action, rank, frame, body, None)[0]),
            "critic_error": relative(agent._critic_obs_fn()(state, ctx, rank, frame), t.critic_obs(state, ctx, rank, frame)),
            "input_error": relative(agent._norm_features_fn()(*agent._first_transition(win))[0], t.wm_input(state, action, ctx)),
            "increment_error": relative(agent._norm_features_fn()(*agent._first_transition(win))[1], t.increment(state, win["state"][:, 1])),
            "finite": all(torch.isfinite(p).all().item() for p in agent.actor.parameters()),
        }


def check_combo() -> dict[str, Any]:
    combo_kernels()
    agent = harness.make_agent(device=DEVICE, num_envs=16, compile=True, combo_kernels=True)
    agent.ensure_replay()
    t = agent.task
    obs, _ = agent._isaac.reset()
    rows = t.read_obs(obs["wm"])
    args = (rows["state"], rows["context"], rows["rank"], rows["frame"])
    with torch.no_grad():
        error = relative(torch.compile(t.critic_obs, dynamic=False)(*args), t.critic_obs(*args))
    return {"combo": torch._inductor.config.combo_kernels, "error": error}


CHECKS = {
    "default": check_default,
    "patch": check_patch,
    "agent_applies": check_agent_applies,
    "late_patch": check_late_patch,
    "products": check_products,
    "cache": check_cache,
    "compile_eager": check_compile_eager,
    "combo": check_combo,
}


# ------------------------------------------------------------------ the tests
def test_a_default_agent_changes_no_compiler_state() -> None:
    assert run("default") == {"same_entry": True, "tag": "", "combo": False}


def test_the_patch_applies_once_and_keeps_a_tag() -> None:
    got = run("patch")
    assert got["patched"] and got["second_call_changes_nothing"]
    assert got["tag"] == "mine:rlopt:elementwise_3x3"


def test_construction_applies_the_process_options() -> None:
    assert run("agent_applies") == {"patched": True, "combo": True, "limit": 64}


def test_the_patch_refuses_a_process_that_has_compiled() -> None:
    assert "before the first compile" in run("late_patch")["refused"]


@cuda
def test_the_patch_replaces_cublas_within_tolerance() -> None:
    off, on = run("products", "off"), run("products", "on")
    assert off["bmm_calls"] > 0, "the control must show the products this option removes"
    assert on["bmm_calls"] == 0
    assert on["fk_error"] < 1e-4
    assert on["rotate_error"] < 1e-4


@cuda
def test_the_tag_keeps_the_two_graphs_apart(tmp_path: Path) -> None:
    env = {"TORCHINDUCTOR_CACHE_DIR": str(tmp_path)}
    assert run("cache", "off", env=env)["miss"] == 1
    tagged = run("cache", "on", env=env)  # a hit here would bypass the patch
    assert (tagged["hit"], tagged["miss"]) == (0, 1)
    again = run("cache", "on", env=env)
    assert (again["hit"], again["miss"]) == (1, 0)


@cuda
@pytest.mark.parametrize("path", ["no_sync", "host"])
def test_compile_eager_sites_hold_their_shapes_and_values(path: str) -> None:
    got = run("compile_eager", path)
    assert got["limit"] == 64
    assert got["finite"]
    for site in ("reward_error", "critic_error", "input_error", "increment_error"):
        assert got[site] < 1e-3, (site, got)


@cuda
def test_combo_kernels_keep_the_values() -> None:
    got = run("combo")
    assert got["combo"] is True
    assert got["error"] < 1e-3


if __name__ == "__main__":
    print(json.dumps(CHECKS[sys.argv[1]](*sys.argv[2:])))
