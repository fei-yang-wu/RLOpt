"""``wm.wm_mirror``: a share of every world-model batch replaced by its left-right mirror.

The map itself is tested in the parent (``test_g1_v3_mirror.py``: involution,
reflection, mirrored forward kinematics and increments). Here, on the real
``G1V3TrackingTask``: the first int(B p) windows of each batch, and only they, are
mirrored on every physical key, a terminal window included; the normalizers never
see mirrored data (NDM Gym's ``Mirror.windows`` after ``update_wm_norms``); the
default path never mirrors.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness

from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
ON = {"wm_mirror": 0.5}
EXACT = {"rtol": 0, "atol": 0}
PHYSICAL = ("state", "next_state", "action", "policy_action", "prev_action", "context")


def collected(agent: Any, steps: int = 45) -> dict[str, Any]:
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(steps):
        obs = agent._collect_step(obs, stats)
    return obs


def captured(agent: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[list, list]:
    """The windows the replay hands out and the windows the rollout fits, per batch."""
    raw, fit = [], []
    window, rollout = agent.replay.window, agent._wm_rollout
    monkeypatch.setattr(agent.replay, "window", lambda *a: (raw.append(window(*a)), raw[-1])[1])
    monkeypatch.setattr(agent, "_wm_rollout", lambda win, k, grad: (fit.append(win), rollout(win, k, grad=grad))[1])
    return raw, fit


def test_the_first_rows_of_each_batch_are_mirrored_on_every_physical_key(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = harness.make_agent(**ON)
    collected(agent)
    raw, fit = captured(agent, monkeypatch)
    agent._update_world_model(defaultdict(list))
    n = agent._mirror_rows
    assert n == harness.SMALL["wm_batch_size"] // 2
    assert len(fit) == harness.SMALL["wm_batches_per_iter"]
    mirror = agent.task._mirror
    for before, after in zip(raw, fit, strict=True):
        for key in before:
            if key in PHYSICAL:
                fn = {"action": mirror.joints, "policy_action": mirror.joints, "prev_action": mirror.joints,
                      "context": mirror.context}.get(key, mirror.state)  # fmt: skip
                torch.testing.assert_close(after[key][:n], fn(before[key][:n]), **EXACT)
                torch.testing.assert_close(after[key][n:], before[key][n:], **EXACT)
                if key != "context":  # the stand-in env's context has no offsets to mirror
                    assert not torch.equal(after[key][:n], before[key][:n]), key
            else:
                assert after[key] is before[key], key  # masks, cursors and outcomes stay


def test_a_terminal_window_mirrors_its_true_last_state() -> None:
    agent = harness.make_agent(env_cls=harness.FinalObsFakeEnv, fit_terminal_transitions=True, **ON)
    collected(agent, 70)
    rp = agent.replay
    tau, env = rp.sample_starts(256, 8)
    win = rp.window(tau, env, 9)
    assert win["ended"].any()
    out = agent.task.mirror_window(win, 256)
    mirror = agent.task._mirror
    ended = win["ended"]
    torch.testing.assert_close(out["next_state"][ended], mirror.state(win["next_state"][ended]), **EXACT)
    torch.testing.assert_close(out["next_state"], mirror.state(win["next_state"]), **EXACT)
    for key in ("valid", "ended", "rank", "frame", "episode_step", "push", "terminated", "env_reward"):
        assert out[key] is win[key]


def test_the_normalizers_never_see_mirrored_data() -> None:
    agents = []
    for wm in ({}, ON):
        torch.manual_seed(5)
        agents.append(harness.make_agent(**wm))
    for agent in agents:
        torch.manual_seed(11)
        collected(agent)
        torch.manual_seed(12)
        agent._update_world_model(defaultdict(list))
    plain, mirrored = (agent.dynamics for agent in agents)
    for name in ("in_norm", "out_norm"):
        assert harness.state_equal(getattr(plain, name).state_dict(), getattr(mirrored, name).state_dict()), name
    assert not harness.state_equal(plain.net.state_dict(), mirrored.net.state_dict())  # the fit did see them


def test_the_default_agent_never_mirrors(monkeypatch: pytest.MonkeyPatch) -> None:
    from imitation_experiments.world_model.g1_v3_task import G1V3TrackingTask

    def refuse(*_args: Any) -> None:
        msg = "mirrored with the option off"
        raise AssertionError(msg)

    monkeypatch.setattr(G1V3TrackingTask, "mirror_window", refuse)
    monkeypatch.setattr(G1V3TrackingTask, "bind_mirror", refuse)
    agent = harness.make_agent(frames=harness.frames_for(4))
    agent.train()
    assert agent._mirror_rows == 0


@pytest.mark.parametrize(
    ("wm", "env_cls"),
    [(ON, harness.FakeIsaacEnv), ({"wm_mirror": 1.0, "fit_terminal_transitions": True, "wm_structured": True}, harness.FinalObsFakeEnv)],
    ids=["half", "all_terminal_structured"],
)
def test_a_mirrored_run_trains(wm: dict[str, Any], env_cls: type) -> None:
    agent = harness.make_agent(frames=harness.frames_for(6), env_cls=env_cls, **wm)
    agent.train()
    assert all(torch.isfinite(p).all() for p in agent.dynamics.parameters())


def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


CAPABLE = SimpleNamespace(mirror_window=lambda win, _n: win, bind_mirror=lambda: None)


@pytest.mark.parametrize(
    ("wm", "task", "match"),
    [
        ({"wm_mirror": -0.1}, CAPABLE, "wm_mirror must be in"),
        ({"wm_mirror": 1.5}, CAPABLE, "wm_mirror must be in"),
        ({"wm_mirror": 0.0001}, CAPABLE, "no window"),
        ({"wm_mirror": 0.5}, SimpleNamespace(), "mirror_window"),
    ],
)
def test_refused(wm: dict[str, Any], task: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(**wm), task)


def test_allowed() -> None:
    validate_options(config(wm_mirror=0.5), CAPABLE)
    validate_options(config(wm_mirror=1.0), CAPABLE)
    validate_options(config(), SimpleNamespace())


def test_an_asymmetric_action_scale_is_refused() -> None:
    class Lopsided(harness.FakeIsaacEnv):
        def __init__(self, num_envs: int, device: Any) -> None:
            super().__init__(num_envs, device)
            scale = harness.action_scale(self.device).clone()
            scale[harness.G1_ISAACLAB_JOINT_NAMES.index("left_knee_joint")] *= 1.5
            self.action_manager.get_term("joint_pos")._scale = scale

    with pytest.raises(ValueError, match="action scale"):
        harness.make_agent(env_cls=Lopsided, **ON)


# ------------------------------------------------------------------ the recipe's systems options
class SyncFreeEnv(harness.FakeIsaacEnv):
    def step(self, _action: Any) -> Any:
        zeros = torch.zeros(self.n, device=self.device)
        self.frame = self.frame + 1
        self.episode = self.episode + 1
        self.frame = torch.where(self.frame >= self.v3_library.length[self.rank] - 2, 0, self.frame)
        return self._obs(), zeros, zeros.bool(), zeros.bool(), {}


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
def test_the_mirror_adds_no_host_read_and_no_recompile(graphs: bool) -> None:
    wm = {"no_sync": True, "fused_adam": True, "compile": True, "compile_eager": True, **ON}
    if graphs:
        wm.update(update_graphs=True, graph_early=True)
    agent = harness.make_agent(device="cuda:0", num_envs=64, env_cls=SyncFreeEnv, **wm)
    obs = collected(agent, 44)
    obs = rounds(agent, obs, 5)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.compiler.set_stance("fail_on_recompile"):
            obs = rounds(agent, obs, 3)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    if graphs:
        assert agent._graphs["world_model"].captures == 1
    assert all(torch.isfinite(p).all() for p in agent.dynamics.parameters())
