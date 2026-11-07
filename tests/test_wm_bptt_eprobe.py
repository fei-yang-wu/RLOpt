"""``wm.eprobe_*``: NDM Gym's evaluation-guided resets, the agent's side.

The agent validates the options (NDM Gym's ``check_eprobe_args``), builds the env's
probe at the start of ``train``, ticks it before every iteration's collection (NDM
Gym's ``probe.tick(it, ...)``) with the deployed policy, logs its numbers, and keeps
its state in full checkpoints (``eprobe_fail``, ``eprobe_focus``,
``eprobe_focus_share``). ``deployed_policy`` builds what ``load_policy`` deploys, which
the probe's worker runs. The probe itself (sweeps, counts, focus, timing) is tested
in the parent repository (``test_g1_v3_eval_probe.py``); off, the golden test holds.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from tensordict import TensorDict

from rlopt.agent.wm_bptt.agent import deployed_policy
from rlopt.agent.wm_bptt.config import (
    EPROBE_FIELDS,
    WMBPTTConfig,
    WMBPTTRLOptConfig,
    validate_options,
)

EXACT = {"rtol": 0, "atol": 0}
ON = {"eprobe_share": 0.15, "eprobe_start": 2, "eprobe_every": 2, "actor_ema": 0.9}


class StubProbe:
    """Records what the agent hands the env's probe."""

    def __init__(self, env: Any, wm: Any, run_dir: Any) -> None:
        self.env, self.wm, self.run_dir = env, wm, run_dir
        self.ticks: list[tuple[int, int]] = []
        self.payloads: list[dict[str, Any]] = []
        self.loaded: dict[str, Any] | None = None
        self.closed = False

    def tick(self, iteration: int, payload: Any) -> None:
        self.ticks.append((iteration, self.env.steps))
        if iteration >= self.wm.eprobe_start and iteration % self.wm.eprobe_every == 0:
            sent = payload()
            agent = self.env.agent
            want = agent.collector_policy.state_dict()
            assert set(sent["policy_state_dict"]) == set(want)
            for key, value in want.items():
                assert sent["policy_state_dict"][key].device.type == "cpu"
                torch.testing.assert_close(sent["policy_state_dict"][key], value.cpu(), **EXACT)
            assert sent["policy_meta"] == agent.policy_meta()
            self.payloads.append({**sent, "average": agent._deploys_average})

    def log_metrics(self) -> dict[str, float]:
        return {"eprobe/sweeps": float(len(self.payloads))}

    def state_dict(self) -> dict[str, Any]:
        return {"eprobe_fail": torch.arange(3.0), "eprobe_focus": None, "eprobe_focus_share": 0.0, "ticks": len(self.ticks)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded = state

    def close(self) -> None:
        self.closed = True


class ProbeEnv(harness.FakeIsaacEnv):
    """The stand-in env with an evaluation probe, counting its steps."""

    def __init__(self, num_envs: int, device: Any) -> None:
        super().__init__(num_envs, device)
        self.steps = 0
        self.probes: list[StubProbe] = []
        self.agent: Any = None

    def step(self, action: torch.Tensor):
        self.steps += 1
        return super().step(action)

    def make_eval_probe(self, wm: Any, run_dir: Any) -> StubProbe:
        probe = StubProbe(self, wm, run_dir)
        self.probes.append(probe)
        return probe


def agent_on(tmp_path: Path, iterations: int, *, save_interval: int = 0, **wm: Any) -> Any:
    agent = harness.make_agent(
        frames=harness.frames_for(iterations), log_dir=tmp_path, env_cls=ProbeEnv, save_interval=save_interval, **{**ON, **wm}
    )
    agent._isaac.agent = agent
    return agent


def last_full(agent: Any) -> Path:
    return max(agent._checkpoint_dir().glob("model_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))


# ------------------------------------------------------------------ off
def test_off_by_default_builds_no_probe(tmp_path: Path) -> None:
    agent = harness.make_agent(frames=harness.frames_for(4), log_dir=tmp_path, env_cls=ProbeEnv)
    assert WMBPTTConfig().eprobe_share == 0.0
    assert agent._eprobe is None
    agent.train()
    assert agent._isaac.probes == []
    path = last_full(agent)
    assert "eprobe" not in torch.load(path, weights_only=False)


# ------------------------------------------------------------------ on
def test_the_probe_ticks_before_each_iterations_collection(tmp_path: Path) -> None:
    agent = agent_on(tmp_path, 7)
    agent.train()
    (probe,) = agent._isaac.probes
    assert [it for it, _ in probe.ticks] == list(range(7))
    # Iteration i's tick sees the steps of iterations 0 .. i - 1 only (frames_for counts env steps x envs).
    assert [steps for _, steps in probe.ticks] == [harness.frames_for(i) // 8 for i in range(7)]
    assert probe.run_dir == agent.log_dir
    assert probe.wm is agent.config.wm
    assert probe.closed
    assert agent._eprobe is None


def test_a_sweep_gets_the_deployed_policy(tmp_path: Path) -> None:
    """The live actor before the average's first update, the average after it."""
    agent = agent_on(tmp_path, 9, eprobe_start=0)
    agent.train()
    (probe,) = agent._isaac.probes
    assert [p["average"] for p in probe.payloads] == [False, True, True, True, True]


def test_the_log_row_has_the_probes_numbers(tmp_path: Path) -> None:
    agent = agent_on(tmp_path, 6)
    rows: list[dict[str, Any]] = []
    agent.log_training_metrics = lambda row, **_: rows.append(row)  # type: ignore[method-assign]
    agent.train()
    assert rows
    assert all("eprobe/sweeps" in row for row in rows)


def test_full_checkpoints_hold_the_probe_state_and_a_resume_restores_it(tmp_path: Path) -> None:
    agent = agent_on(tmp_path, 6)
    agent.train()
    path = last_full(agent)
    saved = torch.load(path, weights_only=False)["eprobe"]
    assert torch.equal(saved["eprobe_fail"], torch.arange(3.0))
    assert saved["eprobe_focus"] is None
    resumed = agent_on(tmp_path / "resumed", 8)
    resumed.load_model(str(path))
    resumed.train()
    (probe,) = resumed._isaac.probes
    assert probe.loaded is not None
    assert probe.loaded["ticks"] == saved["ticks"]
    # The probe continues on the iteration count, as NDM Gym's does.
    assert probe.ticks[0][0] == torch.load(path, weights_only=False)["iteration"]


def test_a_resume_without_probe_state_is_refused(tmp_path: Path) -> None:
    agent = harness.make_agent(frames=harness.frames_for(4), log_dir=tmp_path, actor_ema=0.9)
    agent.train()
    path = last_full(agent)
    with pytest.raises(ValueError, match="probe state"):
        agent_on(tmp_path / "resumed", 6).load_model(str(path))


def test_light_checkpoints_leave_the_probe_state_out(tmp_path: Path) -> None:
    agent = agent_on(tmp_path, 8, save_interval=8, ckpt_light=True)
    agent.train()
    light = sorted(agent._checkpoint_dir().glob("policy_step_*.pt"))
    assert light
    assert all("eprobe" not in torch.load(p, weights_only=False) for p in light)
    full = last_full(agent)
    assert "eprobe" in torch.load(full, weights_only=False)


# ------------------------------------------------------------------ the deployed policy
def policy_td(obs: dict[str, Any], n: int) -> TensorDict:
    return TensorDict({"wm": TensorDict(obs["wm"], batch_size=[n])}, batch_size=[n])


@pytest.mark.parametrize(
    "wm",
    [{}, {"actor_ema": 0.9}, {"actor_ema": 0.9, "actor_residual": True}, {"actor_arch": "enc_dec", "actor_hidden": [64, 48, 32], "actor_residual": True}],
)
def test_deployed_policy_is_what_load_policy_deploys(tmp_path: Path, wm: dict[str, Any]) -> None:
    source = harness.make_agent(frames=harness.frames_for(4), log_dir=tmp_path, **wm)
    source.train()
    paths = [last_full(source)]
    evaluator = harness.make_agent()  # the default config, as an evaluator builds it
    evaluator.load_policy(str(paths[-1]))
    policy = deployed_policy(evaluator._isaac, torch.load(paths[-1], weights_only=False), "cpu")
    assert not policy.training
    assert policy.module.residual == bool(wm.get("actor_residual"))
    obs, _ = evaluator._isaac.reset()
    td = policy_td(obs, evaluator._num_envs)
    with torch.no_grad():
        want = evaluator.collector_policy(td.clone())["action"]
        torch.testing.assert_close(policy(td.clone())["action"], want, **EXACT)
        torch.testing.assert_close(source.collector_policy(td.clone())["action"], want, **EXACT)


def test_a_later_sweep_reloads_the_weights_only(tmp_path: Path) -> None:
    source = harness.make_agent(frames=harness.frames_for(4), log_dir=tmp_path, actor_ema=0.9)
    first = source._deployed_payload()
    policy = deployed_policy(source._isaac, first, "cpu")
    source.train()
    later = source._deployed_payload()
    policy.load_state_dict(later["policy_state_dict"], strict=True)
    obs, _ = source._isaac.reset()
    td = policy_td(obs, source._num_envs)
    with torch.no_grad():
        torch.testing.assert_close(policy(td.clone())["action"], source.collector_policy(td.clone())["action"], **EXACT)


def test_deployed_policy_refuses_another_interface() -> None:
    source = harness.make_agent()
    data = source._deployed_payload()
    data["policy_meta"]["policy_obs_dim"] = 7
    with pytest.raises(ValueError, match="policy_obs_dim"):
        deployed_policy(source._isaac, data, "cpu")
    data["policy_meta"]["arch"] = "other"
    with pytest.raises(ValueError, match="not a"):
        deployed_policy(source._isaac, data, "cpu")


# ------------------------------------------------------------------ validation
def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


TASK = SimpleNamespace(isaac_env=SimpleNamespace(make_eval_probe=lambda *_: None))
PRESET = {"eprobe_share": 0.15, "eprobe_start": 1000, "eprobe_every": 125, "eprobe_decay": 0.7, "eprobe_graph": True, "eprobe_async": 20, "actor_ema": 0.995}


@pytest.mark.parametrize(
    ("wm", "task", "match"),
    [
        ({**PRESET, "actor_ema": 0.0}, TASK, "actor_ema"),
        ({**PRESET, "eprobe_graph": False}, TASK, "eprobe_graph"),
        ({**PRESET, "eprobe_async": 5}, TASK, "too small"),
        ({**PRESET, "eprobe_share": 1.0}, TASK, "eprobe_share"),
        ({**PRESET, "eprobe_share": -0.1}, TASK, "eprobe_share"),
        ({**PRESET, "eprobe_tile": 0}, TASK, "eprobe_tile"),
        ({**PRESET, "eprobe_lead_min": 500}, TASK, "eprobe_lead_min"),
        ({**PRESET, "eprobe_decay": 1.5}, TASK, "eprobe_decay"),
        ({**PRESET, "eprobe_every": 0}, TASK, "eprobe_every"),
        ({**PRESET, "eprobe_start": -1}, TASK, "eprobe_start"),
        ({**PRESET, "goal": "z"}, TASK, "explicit"),
        (PRESET, SimpleNamespace(isaac_env=SimpleNamespace()), "make_eval_probe"),
        ({"eprobe_graph": True}, TASK, "eprobe_graph"),
        ({"eprobe_async": 20}, TASK, "eprobe_async"),
        ({"eprobe_start": 1000}, TASK, "eprobe_start"),
        ({"eprobe_decay": 0.7}, TASK, "eprobe_decay"),
    ],
)
def test_refused(wm: dict[str, Any], task: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(**wm), task)


def test_allowed() -> None:
    validate_options(config(**PRESET), TASK)  # BlinkMimic's PRESET[main]
    validate_options(config(**{**PRESET, "eprobe_graph": False, "eprobe_async": 0}), TASK)  # synchronous eager sweeps
    validate_options(config(), SimpleNamespace())  # off, on any task


def test_every_probe_field_is_refused_while_off() -> None:
    default = WMBPTTConfig()
    for name in EPROBE_FIELDS:
        value = getattr(default, name)
        changed = (not value) if isinstance(value, bool) else value + 1
        with pytest.raises(ValueError, match=name):
            validate_options(config(**{name: changed}), TASK)
