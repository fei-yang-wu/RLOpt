"""``wm.contact_reward`` and ``wm.ref_contact_fix``: a foot-contact term in imagined rewards.

NDM Gym adds w * dt * contact_term to SuperMimic's imagined reward only
(``SuperMimic._reward_term``); the simulator's reward, which evaluation uses, is
unchanged. Here the plain fields reach the agent's task as keywords, never the env's
own task instance, and the expected rewards are the task's ``reward`` (the simulator
parity path) plus the term from FK of the imagined states and an independently
built reference contact. The default path is the golden test's.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from torch import Tensor

from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
ON = {"contact_reward": 0.15, "ref_contact_fix": True}


class EnvWithItsOwnTask(harness.FakeIsaacEnv):
    """As ``G1V3RLEnv``: the env builds its task first, with no keywords, and the
    agent's task reuses its library and MDP."""

    def __init__(self, num_envs: int, device: Any) -> None:
        super().__init__(num_envs, device)
        from imitation_experiments.world_model.g1_v3_task import G1V3TrackingTask

        self.v3_task = G1V3TrackingTask(self, self.device)
        self.v3_mdp = self.v3_task.mdp


def collected(agent: Any, steps: int = 45) -> None:
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(steps):
        obs = agent._collect_step(obs, stats)


def starts(agent: Any, n: int) -> tuple[Tensor, Tensor]:
    rp, t = agent.replay, agent.task
    taus = torch.arange(rp.oldest + t.history - 1, rp.steps - 1, device=agent.device)
    tau, env = torch.meshgrid(taus, torch.arange(agent._num_envs, device=agent.device), indexing="ij")
    slot = rp._slot(tau)
    valid = t.valid_imagination_start(rp.rank[slot, env], rp.frame[slot, env], agent.config.wm.horizon)
    pick = valid.nonzero()[:n]
    return tau[pick[:, 0], pick[:, 1]], env[pick[:, 0], pick[:, 1]]


# ------------------------------------------------------------------ where the keywords go
def test_the_fields_reach_the_agents_task_only() -> None:
    agent = harness.make_agent(env_cls=EnvWithItsOwnTask, **ON)
    env_task = agent._isaac.v3_task
    assert (agent.task.contact_reward, agent.task.ref_contact_fix) == (0.15, True)
    assert (env_task.contact_reward, env_task.ref_contact_fix) == (0.0, False)
    assert agent.task.mdp is env_task.mdp  # one MDP, one simulator reward
    assert agent.task.describe()["contact_reward"] == 0.15


def test_the_default_agent_passes_no_contact_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    import rlopt.agent.wm_bptt.agent as module

    seen: list[dict[str, Any]] = []
    load = module.load_task

    def spy(spec: str, env: Any, device: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return load(spec, env, device, **kwargs)

    monkeypatch.setattr(module, "load_task", spy)
    agent = harness.make_agent()
    assert seen
    assert not {"contact_reward", "ref_contact_fix"} & set(seen[0])
    assert agent.task.contact_reward == 0.0
    assert "contact_reward" not in agent.task.describe()


# ------------------------------------------------------------------ the imagined reward
@pytest.mark.parametrize("calibrated", [False, True])
def test_imagined_rewards_are_the_task_reward_plus_w_dt_times_the_term(calibrated: bool) -> None:
    from imitation_experiments.world_model.g1_v3_contact import (
        FEET,
        contact_term,
        reference_contact,
    )
    from imitation_experiments.world_model.g1_v3_math import CONTROL_DT

    agent = harness.make_agent(contact_reward=0.25, ref_contact_fix=calibrated)
    collected(agent)
    t, lib = agent.task, agent.task.library
    tau, env = starts(agent, 12)
    out = agent._imagine(12, grad=False, starts=(tau, env))
    states, actions, rank, frame0 = out["states"], out["actions"], out["rank"], out["frame0"]
    prev = agent.replay.history(tau, env, t.history)["hist_actions"][:, -1]
    c_ref = reference_contact(lib, calibrated)
    feet = lib.body_index(FEET)
    ctx = torch.zeros(12, t.context_dim)
    for step in range(out["rewards"].shape[1]):
        frame = frame0 + step
        nxt, action = states[:, step + 1], actions[:, step]
        plain, _ = t.reward(nxt, action, prev, ctx, rank, frame)
        bp, _, bv, _ = t.mdp.body_states(nxt)
        row = lib.start[rank] + (frame + 1).minimum(lib.length[rank] - 1)
        term = contact_term(bp[:, feet], bv[:, feet], c_ref[row].float())
        torch.testing.assert_close(out["rewards"][:, step], plain + 0.25 * term * CONTROL_DT)
        prev = action
    assert (out["rewards"] > 0).all()


def test_the_actor_and_critic_train_on_it(tmp_path: Any) -> None:
    agent = harness.make_agent(frames=harness.frames_for(5), log_dir=tmp_path, **ON)
    agent.train()
    assert agent._iteration == 5
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())
    assert all(torch.isfinite(p).all() for p in agent.critic.parameters())


# ------------------------------------------------------------------ refused
def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


TAKES_IT = SimpleNamespace(contact_reward=0.15, ref_contact_fix=True)


@pytest.mark.parametrize(
    ("wm", "task", "match"),
    [
        ({"contact_reward": -0.1}, TAKES_IT, "a weight"),
        ({"ref_contact_fix": True}, TAKES_IT, "set wm.contact_reward"),
        (ON, SimpleNamespace(), "a task that takes it"),
        (ON, SimpleNamespace(contact_reward=0.15, ref_contact_fix=False), "a task that takes it"),
        ({**ON, "actor_source": "real", "critic_source": "real"}, TAKES_IT, "real windows"),
        ({**ON, "critic_source": "real"}, TAKES_IT, "real windows"),
    ],
)
def test_refused(wm: dict[str, Any], task: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(**wm), task)


def test_allowed() -> None:
    validate_options(config(**ON), TAKES_IT)
    validate_options(config(contact_reward=0.15), SimpleNamespace(contact_reward=0.15, ref_contact_fix=False))


def test_the_latent_v3_task_takes_it_too() -> None:
    from imitation_experiments.world_model.g1_v3_task import G1V3LatentTrackingTask

    assert G1V3LatentTrackingTask.contact_reward == 0.0  # inherited keyword, off by default


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


def rounds(agent: Any, n: int) -> None:
    stats: dict[str, list] = defaultdict(list)
    for _ in range(n):
        for _ in range(3):
            agent._obs = agent._collect_step(agent._obs, stats)
        agent._set_lrs()
        agent._run_update("world_model", agent._update_world_model, defaultdict(list))
        agent._run_update("actor_critic", agent._update_actor_critic, defaultdict(list))


@cuda
@pytest.mark.parametrize("graphs", [False, True], ids=["no_sync_compile", "update_graphs"])
def test_the_term_adds_no_host_read_and_no_recompile(graphs: bool) -> None:
    wm = {**RECIPE, **({"update_graphs": True, "graph_early": True} if graphs else {})}
    agent = harness.make_agent(device="cuda:0", num_envs=64, env_cls=SyncFreeEnv, **wm)
    agent.ensure_replay()
    agent._obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(44):
        agent._obs = agent._collect_step(agent._obs, stats)
    rounds(agent, 5)  # compile, warm-up and capture
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.compiler.set_stance("fail_on_recompile"):
            rounds(agent, 3)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    if graphs:
        assert agent._graphs["world_model"].captures == 1
        assert agent._graphs["actor_critic"].captures == 1
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())


@cuda
def test_the_compiled_step_computes_the_term() -> None:
    """The compiled imagination step's rewards against the eager step's, same inputs
    (float32 matmuls, so the two differ only by the compiled kernels' rounding)."""
    agent = harness.make_agent(device="cuda:0", num_envs=64, env_cls=SyncFreeEnv, compile=True, bf16=False, **ON)
    plain = harness.make_agent(device="cuda:0", num_envs=64, env_cls=SyncFreeEnv, bf16=False)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(44):
        obs = agent._collect_step(obs, stats)
    t = agent.task
    start = agent.replay.history(torch.full((32,), 40, device="cuda:0"), torch.arange(32, device="cuda:0"), t.history)
    from rlopt.agent.wm_bptt.task import history_proprio

    args = (
        start["hist_states"][:, -1], history_proprio(t, start, False), start["hist_actions"][:, -1], start["context"],
        start["rank"], start["frame"], torch.zeros(32, t.action_dim, device="cuda:0"), False,
    )
    compiled = agent._step_fn()(*args)
    eager = agent._imagine_step(*args)
    torch.testing.assert_close(compiled[3], eager[3], rtol=1e-5, atol=1e-6)
    plain.actor.load_state_dict(agent.actor.state_dict())
    plain.dynamics.load_state_dict(agent.dynamics.state_dict())
    without = plain._imagine_step(*args)
    assert (eager[3] - without[3] > 0).all()  # the term is positive
