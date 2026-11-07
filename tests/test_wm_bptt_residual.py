"""``wm.actor_residual``: the actor outputs an offset about the next reference pose.

The reference is independent of the code under test: ``ref_state(rank, frame + 1)``'s
joint positions, the default pose and the env's action scale, as NDM Gym's
``reference_setpoints`` forms it (``tests/test_action_parameterization.py`` there,
``test_supermimic_residual_setpoints``). Every path that turns the actor's output
into an action is checked: collection, imagination, the deployed policy and
``predict``. The default path is the golden test's.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from tensordict import TensorDict
from torch import Tensor

from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
ON = {"actor_residual": True}
EXACT = {"rtol": 0, "atol": 0}


def expected_setpoint(agent: Any, rank: Tensor, frame: Tensor) -> Tensor:
    """(q_ref[f + 1] - default_q) / scale, from the bank rows."""
    q = agent.task.library.ref_state(rank, frame + 1)[..., 13:42]
    return (q - agent.task.default_q) / harness.action_scale(agent.device)


def zero_actor(agent: Any) -> None:
    last = agent.actor.net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()


def collected(agent: Any, steps: int = 45) -> tuple[dict[str, Any], dict[str, list]]:
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(steps):
        obs = agent._collect_step(obs, stats)
    return obs, stats


def policy_td(obs: dict[str, Any], n: int) -> TensorDict:
    return TensorDict({"wm": TensorDict(obs["wm"], batch_size=[n])}, batch_size=[n])


# ------------------------------------------------------------------ the setpoint
def test_the_setpoint_is_the_reference_pose_of_the_next_frame_clip_end_included() -> None:
    agent = harness.make_agent(**ON)
    lib = agent.task.library
    rank = torch.tensor([0, 0, 0, 1, 1, 1])
    frame = torch.cat((torch.tensor([0, 37]), lib.length[:1] - 1, torch.tensor([5]), lib.length[1:] - 2, lib.length[1:] - 1))
    prop = torch.randn(6, agent.task.history, 99)
    obs = agent.task.policy_obs_from_parts(prop, lib.ref_state(rank, frame), rank, frame)
    torch.testing.assert_close(agent.task.reference_setpoint(obs), expected_setpoint(agent, rank, frame), **EXACT)
    # At a clip's last frame, f + 1 is the last frame again.
    last = rank == 1
    torch.testing.assert_close(
        agent.task.reference_setpoint(obs)[last][-1], expected_setpoint(agent, rank[last][-1:], frame[last][-1:] - 1)[0], **EXACT
    )


def test_a_zero_actor_executes_the_reference_pose_on_every_path() -> None:
    agent = harness.make_agent(action_noise_std=0.0, **ON)
    zero_actor(agent)
    obs, stats = collected(agent, 44)
    # collection: the replay row, the env's action and the next observation's last action
    rows = agent.task.read_obs(obs["wm"])
    want = expected_setpoint(agent, rows["rank"], rows["frame"])
    obs = agent._collect_step(obs, stats)
    slot = (agent.replay.steps - 1) % agent.replay.capacity
    torch.testing.assert_close(agent.replay.policy_action[slot], want, **EXACT)
    torch.testing.assert_close(agent.replay.action[slot], want, **EXACT)
    kept = agent._isaac.episode > 0  # the env zeroes the history of a reset env
    torch.testing.assert_close(agent._isaac.hist_action[kept, -1], want[kept], **EXACT)
    # imagination: every step's mean, with the cursor advancing one frame per step
    out = agent._imagine(16, grad=False)
    H = out["mus"].shape[1]
    frames = out["frame0"].unsqueeze(1) + torch.arange(H)
    torch.testing.assert_close(out["mus"], expected_setpoint(agent, out["rank"].unsqueeze(1).expand_as(frames), frames), **EXACT)
    torch.testing.assert_close(out["actions"], out["mus"], **EXACT)
    # the deployed policy and predict
    rows = agent.task.read_obs(obs["wm"])
    want = expected_setpoint(agent, rows["rank"], rows["frame"])
    deployed = agent.collector_policy(policy_td(obs, agent._num_envs))["action"]
    torch.testing.assert_close(deployed, want, **EXACT)
    feat = agent.task.policy_obs_from_env(*(obs["wm"][key] for _, key in agent.task.policy_in_keys))
    torch.testing.assert_close(agent.predict(feat), want, **EXACT)


def test_the_executed_action_is_the_actor_output_plus_the_setpoint() -> None:
    """With a trained-looking actor: the sum reaches the world model, the reward and
    the next observation; noise and the clamp come after it."""
    agent = harness.make_agent(action_noise_std=0.0, action_clip=0.3, **ON)
    with torch.no_grad():
        agent.actor.net[-1].weight.normal_(0, 0.05)
    obs, stats = collected(agent, 44)
    rows = agent.task.read_obs(obs["wm"])
    feat = agent._policy_features(rows)  # the step's own noise draw, keyed to its row
    agent._collect_step(obs, stats)
    mean = agent.actor(feat) + expected_setpoint(agent, rows["rank"], rows["frame"])  # after the step's normalizer update
    slot = (agent.replay.steps - 1) % agent.replay.capacity
    torch.testing.assert_close(agent.replay.policy_action[slot], mean, **EXACT)
    torch.testing.assert_close(agent.replay.action[slot], mean.clamp(-0.3, 0.3), **EXACT)
    assert (mean.abs() > 0.3).any()  # the clamp acted on the sum
    # one imagined step
    start = agent.replay.history(torch.full((4,), agent.replay.steps - 2), torch.arange(4), agent.task.history)
    state, ctx, rank, frame = start["hist_states"][:, -1], start["context"], start["rank"], start["frame"]
    from rlopt.agent.wm_bptt.task import history_proprio

    prop = history_proprio(agent.task, start, False)
    prev = start["hist_actions"][:, -1]
    eps = torch.zeros(4, 29)
    action, mu, nxt, r, _, new_prop = agent._imagine_step(state, prop, prev, ctx, rank, frame, eps, False)
    pobs = agent.task.policy_obs_from_parts(prop, state, rank, frame)
    torch.testing.assert_close(mu, agent.actor(pobs) + expected_setpoint(agent, rank, frame), **EXACT)
    torch.testing.assert_close(action, mu, **EXACT)
    inc = agent.dynamics(agent.task.wm_input(state, action, ctx))
    torch.testing.assert_close(nxt, agent.task.integrate(state, inc), **EXACT)
    torch.testing.assert_close(r, agent.task.reward(nxt, action, prev, ctx, rank, frame)[0], **EXACT)
    # the last-action observation (columns without sensor noise)
    torch.testing.assert_close(new_prop[:, -1, 64:93], action, **EXACT)


def test_score_function_actions_keep_the_setpoint_inside_the_mean() -> None:
    agent = harness.make_agent(actor_estimator="score_function", **ON)
    zero_actor(agent)
    collected(agent)
    out = agent._imagine(16, grad=True)
    H = out["mus"].shape[1]
    frames = out["frame0"].unsqueeze(1) + torch.arange(H)
    torch.testing.assert_close(out["mus"].detach(), expected_setpoint(agent, out["rank"].unsqueeze(1).expand_as(frames), frames), **EXACT)
    noise = out["actions"] - out["mus"].detach()
    assert 0.05 < float(noise.std()) < 0.2  # sf_action_std 0.1 about the setpoint


# ------------------------------------------------------------------ off and refused
def test_the_default_agent_never_reads_the_setpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from imitation_experiments.world_model.g1_v3_task import G1V3TrackingTask

    def refuse(*_args: Any) -> None:
        msg = "read the reference setpoint with the option off"
        raise AssertionError(msg)

    monkeypatch.setattr(G1V3TrackingTask, "reference_setpoint", refuse)
    monkeypatch.setattr(G1V3TrackingTask, "bind_reference_setpoint", refuse)
    agent = harness.make_agent(frames=harness.frames_for(4))
    agent.train()
    obs, _ = agent._isaac.reset()
    agent.collector_policy(policy_td(obs, agent._num_envs))
    assert "actor_residual" not in agent.policy_meta()  # the default metadata is the baseline's
    assert agent.policy_net.residual is False


def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


CAPABLE = SimpleNamespace(reference_setpoint=lambda obs: obs, bind_reference_setpoint=lambda: None)


@pytest.mark.parametrize(
    ("wm", "task", "match"),
    [
        ({"goal": "z"}, CAPABLE, "latent goal"),
        ({"goal_encoder": "encoder.pt"}, CAPABLE, "latent goal"),
        ({"actor_source": "real", "critic_source": "real"}, CAPABLE, "real windows"),
        ({"actor_source": "real", "actor_estimator": "rebar"}, CAPABLE, "real windows and rebar"),
        ({}, SimpleNamespace(), "reference_setpoint"),
        ({}, SimpleNamespace(reference_setpoint=lambda obs: obs), "bind_reference_setpoint"),
    ],
)
def test_refused(wm: dict[str, Any], task: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(actor_residual=True, **wm), task)


def test_allowed_with_a_capable_task() -> None:
    validate_options(config(actor_residual=True), CAPABLE)
    validate_options(config(actor_residual=True, critic_source="real"), CAPABLE)
    validate_options(config(actor_residual=True, actor_estimator="score_function"), CAPABLE)


def test_the_latent_v3_task_has_no_setpoint() -> None:
    from imitation_experiments.world_model.g1_v3_task import G1V3LatentTrackingTask

    with pytest.raises(ValueError, match="reference_setpoint"):
        validate_options(config(actor_residual=True), G1V3LatentTrackingTask)


# ------------------------------------------------------------------ the binding
def test_a_shifted_goal_layout_fails_the_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    import imitation_experiments.world_model.g1_v3_task as v3

    monkeypatch.setattr(v3, "GOAL_FRAME_JOINT_POS", v3.GOAL_FRAME_JOINT_POS + 1)
    with pytest.raises(ValueError, match="not the reference joint positions of frame f \\+ 1"):
        harness.make_agent(**ON)


def test_a_goal_one_frame_late_fails_the_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = harness.make_agent()
    goal = agent.task.library.goal
    monkeypatch.setattr(agent.task.library, "goal", lambda rank, frame, count=10: goal(rank, frame + 1, count))
    with pytest.raises(ValueError, match="frame f \\+ 1"):
        agent.task.bind_reference_setpoint()


def env_with(names: list[str] | None = None, scale: Any = None) -> type:
    """The stand-in env with another joint_pos action term."""

    class Env(harness.FakeIsaacEnv):
        def __init__(self, num_envs: int, device: Any) -> None:
            super().__init__(num_envs, device)
            term = self.action_manager.get_term("joint_pos")
            if names is not None:
                term._joint_names = names
            if scale is not None:
                term._scale = scale(self.n)

    return Env


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"names": list(reversed(harness.G1_ISAACLAB_JOINT_NAMES))}, "G1_ISAACLAB_JOINT_NAMES order"),
        ({"scale": lambda n: torch.rand(n, 29) + 0.1}, "one action scale for every env"),
        ({"scale": lambda n: torch.zeros(n, 29)}, "positive action scale"),
    ],
)
def test_the_binding_refuses_an_action_term_it_cannot_invert(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        harness.make_agent(env_cls=env_with(**kw), **ON)


def test_a_scalar_action_scale_binds() -> None:
    agent = harness.make_agent(env_cls=env_with(scale=lambda _n: 0.25), **ON)
    torch.testing.assert_close(agent.task._setpoint_scale, torch.full((29,), 0.25), **EXACT)


def test_an_env_without_an_action_manager_is_refused() -> None:
    class Bare(harness.FakeIsaacEnv):
        def __init__(self, num_envs: int, device: Any) -> None:
            super().__init__(num_envs, device)
            del self.action_manager

    with pytest.raises(ValueError, match="action manager"):
        harness.make_agent(env_cls=Bare, **ON)


# ------------------------------------------------------------------ checkpoints
def trained(tmp_path: Any, **wm: Any) -> Any:
    agent = harness.make_agent(frames=harness.frames_for(4), log_dir=tmp_path, **wm)
    agent.train()
    return agent, max(agent._checkpoint_dir().glob("model_step_*.pt"))


def test_an_inference_load_into_a_default_agent_keeps_the_residual(tmp_path: Any) -> None:
    source, path = trained(tmp_path, actor_ema=0.9, **ON)
    assert torch.load(path, weights_only=False)["policy_meta"]["actor_residual"] is True
    evaluator = harness.make_agent()
    evaluator.load_policy(str(path))
    assert evaluator._residual
    assert evaluator.policy_net.residual
    obs, _ = evaluator._isaac.reset()
    td = policy_td(obs, evaluator._num_envs)
    want = source.collector_policy(td.clone())["action"]
    torch.testing.assert_close(evaluator.collector_policy(td.clone())["action"], want, **EXACT)
    feat = evaluator.task.policy_obs_from_env(*(obs["wm"][key] for _, key in evaluator.task.policy_in_keys))
    torch.testing.assert_close(evaluator.predict(feat), source.predict(feat), **EXACT)
    # Without the setpoint the same weights would command near-default offsets.
    plain = evaluator.policy_net.actor(feat, bf16=evaluator.policy_net.bf16)
    assert (plain - want).abs().max() > 0.1


def test_an_inference_load_of_a_plain_checkpoint_turns_it_off(tmp_path: Any) -> None:
    source, path = trained(tmp_path)
    assert "actor_residual" not in torch.load(path, weights_only=False)["policy_meta"]
    evaluator = harness.make_agent(**ON)
    evaluator.load_policy(str(path))
    assert not evaluator._residual
    assert not evaluator.policy_net.residual
    obs, _ = evaluator._isaac.reset()
    td = policy_td(obs, evaluator._num_envs)
    torch.testing.assert_close(evaluator.collector_policy(td.clone())["action"], source.collector_policy(td.clone())["action"], **EXACT)


@pytest.mark.parametrize(("saved", "resumed"), [(ON, {}), ({}, ON)])
def test_a_resume_must_match_the_action_semantics(tmp_path: Any, saved: dict[str, Any], resumed: dict[str, Any]) -> None:
    _, path = trained(tmp_path, **saved)
    with pytest.raises(ValueError, match="actor_residual"):
        harness.make_agent(**resumed).load_model(str(path))
    harness.make_agent(**saved).load_model(str(path))


def test_a_residual_run_trains_and_resumes(tmp_path: Any) -> None:
    agent, path = trained(tmp_path, **ON)
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())
    resumed = harness.make_agent(frames=harness.frames_for(6), log_dir=tmp_path, **ON)
    resumed.load_model(str(path))
    resumed.train()
    assert resumed._iteration > 4


# ------------------------------------------------------------------ the recipe's systems options
class SyncFreeEnv(harness.FakeIsaacEnv):
    def step(self, action: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor, dict[str, Any]]:
        zeros = torch.zeros(self.n, device=self.device)
        self.frame = self.frame + 1
        self.episode = self.episode + 1
        self.frame = torch.where(self.frame >= self.v3_library.length[self.rank] - 2, 0, self.frame)
        self.hist_action = torch.cat((self.hist_action[:, 1:], action[:, None]), 1)
        return self._obs(), zeros, zeros.bool(), zeros.bool(), {}


RECIPE = {"no_sync": True, "fused_adam": True, "compile": True, "compile_eager": True, **ON}


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
def test_the_residual_adds_no_host_read_and_no_recompile(graphs: bool) -> None:
    wm = {**RECIPE, **({"update_graphs": True, "graph_early": True} if graphs else {})}
    agent = harness.make_agent(device="cuda:0", num_envs=64, env_cls=SyncFreeEnv, **wm)
    obs, _ = collected(agent, 44)
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
def test_a_zero_actor_tracks_the_reference_under_the_recipe_options() -> None:
    agent = harness.make_agent(device="cuda:0", num_envs=64, action_noise_std=0.0, env_cls=SyncFreeEnv, **RECIPE)
    zero_actor(agent)
    obs, stats = collected(agent, 44)
    rows = agent.task.read_obs(obs["wm"])
    want = expected_setpoint(agent, rows["rank"], rows["frame"])
    agent._collect_step(obs, stats)
    slot = (agent.replay.steps - 1) % agent.replay.capacity
    torch.testing.assert_close(agent.replay.action[slot], want, **EXACT)
    out = agent._step_fn()  # the compiled imagination step
    assert out is not agent._imagine_step
    imagined = agent._imagine(32, grad=False)
    H = imagined["mus"].shape[1]
    frames = imagined["frame0"].unsqueeze(1) + torch.arange(H, device=agent.device)
    want = expected_setpoint(agent, imagined["rank"].unsqueeze(1).expand_as(frames), frames)
    # compiled: the same sum up to the compiled division's rounding
    torch.testing.assert_close(imagined["mus"], want, rtol=1e-6, atol=1e-6)
