"""``wm.wm_structured``: the semi-implicit world model on the real ``G1V3TrackingTask``.

The task's coordinates are tested in the parent (``test_g1_v3_structured.py``).
Here: the agent fits the outputs on the semi-implicit targets and divides the state
error by the plain increments' scale, as NDM Gym's ``SuperMimic`` does
(``tests/test_blinkmimic.py``, ``test_semi_implicit_in_supermimic``); every model
step (fit, k-step evaluation, imagination) integrates the same way; the default
model keeps its keys; checkpoints carry the option.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness

from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options
from rlopt.agent.wm_bptt.networks import DynamicsModel, RunningNorm

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
ON = {"wm_structured": True}
EXACT = {"rtol": 0, "atol": 0}


def collected(agent: Any, steps: int = 45) -> dict[str, Any]:
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(steps):
        obs = agent._collect_step(obs, stats)
    return obs


def window(agent: Any, batch: int = 64, k: int = 8) -> dict[str, Any]:
    tau, env = agent.replay.sample_starts(batch, k)
    return agent.replay.window(tau, env, k + 1)


def test_only_the_structured_model_has_an_increment_normalizer() -> None:
    plain, structured = DynamicsModel(10, 70, [16, 16]), DynamicsModel(10, 70, [16, 16], structured=True)
    assert not any(key.startswith("inc_norm.") for key in plain.state_dict())
    extra = set(structured.state_dict()) - set(plain.state_dict())
    assert extra == {"inc_norm.mean", "inc_norm.var", "inc_norm.count", "inc_norm.frozen"}
    agent = harness.make_agent()
    assert not hasattr(agent.dynamics, "inc_norm")
    assert not agent.dynamics.structured


def test_the_agent_takes_the_drift_weight_from_the_env() -> None:
    agent = harness.make_agent(**ON)
    assert agent.task._drift_weight == 0.625
    assert agent.task.describe()["wm_structured_drift_weight"] == 0.625
    assert "wm_structured_drift_weight" not in harness.make_agent().task.describe()


@pytest.mark.parametrize("no_sync", [False, True])
def test_the_three_normalizers_fit_inputs_targets_and_increments(no_sync: bool) -> None:
    wm = {"no_sync": True, "fused_adam": True} if no_sync else {}
    agent = harness.make_agent(**ON, **wm)
    collected(agent)
    win = window(agent, 128)
    s0, s1, a, ctx = agent._first_transition(win)
    valid = win["valid"][:, 0]
    agent._update_wm_norms((s0, s1, a, ctx), valid)
    t, d = agent.task, agent.dynamics
    for norm, x in (
        (d.in_norm, t.wm_input(s0, a, ctx)),
        (d.out_norm, t.structured_targets(s0, s1)),
        (d.inc_norm, t.increment(s0, s1)),
    ):
        ref = RunningNorm(x.shape[-1], eps=norm.eps)
        ref.update(x[valid])
        torch.testing.assert_close(norm.mean, ref.mean, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(norm.var, ref.var, atol=1e-6, rtol=1e-4)
        assert float(norm.count) == float(valid.sum())
    assert not torch.allclose(d.out_norm.mean, d.inc_norm.mean)


def test_the_fit_divides_by_the_plain_increments_scale() -> None:
    agent = harness.make_agent(wm_batches_per_iter=1, **ON)
    collected(agent)
    agent._update_world_model(defaultdict(list))
    d = agent.dynamics
    for norm in (d.in_norm, d.out_norm, d.inc_norm):
        norm.frozen.fill_(True)  # the next update leaves the scale as it is
    scale = torch.maximum(d.inc_norm.std(), agent.task.increment_floor)
    torch.testing.assert_close(agent._error_scale(), scale, **EXACT)
    assert not torch.equal(scale, torch.maximum(d.out_norm.std(), agent.task.increment_floor))
    torch.manual_seed(3)
    tau, env = agent.replay.sample_starts(agent.config.wm.wm_batch_size, 8)
    errs, mask = agent._wm_rollout(agent.replay.window(tau, env, 9), 8, grad=False)
    want = ((errs / scale).square().mean(-1) * mask).sum() / mask[:, 0].sum().clamp_min(1.0)
    torch.manual_seed(3)
    metrics: dict[str, list] = defaultdict(list)
    agent._update_world_model(metrics)
    torch.testing.assert_close(metrics["wm/loss"][0], want, rtol=1e-6, atol=0)


def test_every_model_step_integrates_semi_implicitly() -> None:
    agent = harness.make_agent(**ON)
    collected(agent)
    agent._update_world_model(defaultdict(list))
    t = agent.task
    win = window(agent, 32)
    errs, _ = agent._wm_rollout(win, 4, grad=False)
    state, ctx = win["state"][:, 0], win["context"][:, 0]
    for j in range(4):
        out = agent.dynamics(t.wm_input(state, win["action"][:, j], ctx))
        state = t.structured_integrate(state, out)
        torch.testing.assert_close(errs[:, j], t.state_error(state, win["next_state"][:, j]), **EXACT)
    # an untrained (zero) network: positions advance by the drift of the mean output
    fresh = harness.make_agent(**ON)
    collected(fresh)
    s = win["state"][:, 0]
    zero = torch.zeros(s.shape[0], 70)
    step = fresh._next_state(s, zero)
    torch.testing.assert_close(step[:, 13:42], s[:, 13:42] + 0.02 * s[:, 42:71], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(step[:, :3], s[:, :3] + 0.02 * s[:, 7:10], atol=1e-6, rtol=1e-6)
    # imagination
    start = fresh.replay.history(torch.full((4,), fresh.replay.steps - 2), torch.arange(4), t.history)
    from rlopt.agent.wm_bptt.task import history_proprio

    args = (start["hist_states"][:, -1], history_proprio(t, start, False), start["hist_actions"][:, -1],
            start["context"], start["rank"], start["frame"], torch.zeros(4, 29), False)  # fmt: skip
    action, _, nxt, *_ = fresh._imagine_step(*args)
    out = fresh.dynamics(t.wm_input(args[0], action, args[3]))
    torch.testing.assert_close(nxt, t.structured_integrate(args[0], out), **EXACT)


def test_the_k_step_evaluation_reports_with_the_structured_model() -> None:
    agent = harness.make_agent(**ON, eval_window=8)
    collected(agent, 60)
    agent._update_world_model(defaultdict(list))
    row = agent._wm_eval()
    assert {"wm_eval/k1_root_pos_mm", "wm_eval/k8_joint_pos_deg", "wm_eval/k8_normalized"} <= set(row)
    assert all(torch.isfinite(torch.tensor(v)) for v in row.values())


def test_the_default_agent_never_reaches_the_structured_task_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    from imitation_experiments.world_model.g1_v3_task import G1V3TrackingTask

    def refuse(*_args: Any) -> None:
        msg = "structured function reached with the option off"
        raise AssertionError(msg)

    for name in ("bind_structured", "structured_targets", "structured_integrate"):
        monkeypatch.setattr(G1V3TrackingTask, name, refuse)
    harness.make_agent(frames=harness.frames_for(4)).train()


@pytest.mark.parametrize(
    ("wm", "env_cls"),
    [({}, harness.FakeIsaacEnv), ({"fit_terminal_transitions": True, "norm_from_collection": True}, harness.FinalObsFakeEnv)],
    ids=["plain", "terminal_collection"],
)
def test_a_structured_run_trains(wm: dict[str, Any], env_cls: type) -> None:
    agent = harness.make_agent(frames=harness.frames_for(6), env_cls=env_cls, **ON, **wm)
    agent.train()
    assert all(torch.isfinite(p).all() for p in agent.dynamics.parameters())
    assert float(agent.dynamics.inc_norm.count) == float(agent.dynamics.out_norm.count) > 0


def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


CAPABLE = SimpleNamespace(structured_targets=lambda _s, n: n, structured_integrate=lambda s, _o: s, bind_structured=lambda: 0.625)


@pytest.mark.parametrize(
    ("wm", "task", "match"),
    [
        ({}, SimpleNamespace(), "structured_targets"),
        ({"dynamics_model": "diffsr"}, CAPABLE, "MLP world model"),
        ({"actor_source": "real", "critic_source": "real"}, CAPABLE, "plain-increment residuals"),
    ],
)
def test_refused(wm: dict[str, Any], task: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(wm_structured=True, **wm), task)
    validate_options(config(wm_structured=True), CAPABLE)


def test_an_env_without_physics_settings_is_refused() -> None:
    class Bare(harness.FakeIsaacEnv):
        def __init__(self, num_envs: int, device: Any) -> None:
            super().__init__(num_envs, device)
            del self.cfg

    with pytest.raises(ValueError, match="num_substeps"):
        harness.make_agent(env_cls=Bare, **ON)


def trained(tmp_path: Any, **wm: Any) -> Any:
    agent = harness.make_agent(frames=harness.frames_for(4), log_dir=tmp_path, **wm)
    agent.train()
    return agent, max(agent._checkpoint_dir().glob("model_step_*.pt"))


def test_checkpoints_carry_the_option(tmp_path: Any) -> None:
    agent, path = trained(tmp_path, **ON)
    data = torch.load(path, weights_only=False)
    assert "inc_norm.mean" in data["dynamics_state_dict"]
    resumed = harness.make_agent(**ON)
    resumed.load_model(str(path))
    assert harness.state_equal(resumed.dynamics.state_dict(), agent.dynamics.state_dict())
    with pytest.raises(ValueError, match="wm_structured"):
        harness.make_agent().load_model(str(path))
    evaluator = harness.make_agent()  # an inference load reads the policy only
    evaluator.load_policy(str(path))
    _, plain = trained(tmp_path / "plain")
    with pytest.raises(ValueError, match="wm_structured"):
        harness.make_agent(**ON).load_model(str(plain))


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
def test_the_structured_model_adds_no_host_read_and_no_recompile(graphs: bool) -> None:
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
    assert float(agent.dynamics.inc_norm.count) > 0
    assert all(torch.isfinite(p).all() for p in agent.dynamics.parameters())
