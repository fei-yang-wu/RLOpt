"""``wm.actor_arch=enc_dec``: NDM Gym's TrackingPolicy encoder-decoder as the actor.

References are NDM Gym's (``ndm_gym/policies/tracking_policy.py``, its
``test_muon_on_the_hidden_matrices`` in ``tests/test_blinkmimic.py``): at its sizes
(goal 730 -> 2048, 1024, 512, 256 -> 32, tanh; latent and 990 proprio columns ->
2048, 1024, 512, 256 -> 29, ELU, last layer x 0.01, bias 0) the policy has 9,116,477
parameters, and Muon updates six matrices, each MLP's hidden ones. The flat default is
the golden test's.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from tensordict import TensorDict
from torch import Tensor, nn

from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options
from rlopt.agent.wm_bptt.muon import MuonAdam
from rlopt.agent.wm_bptt.networks import Actor, EncDecActor, build_actor

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
EXACT = {"rtol": 0, "atol": 0}
NDM = {"in_dim": 1720, "out_dim": 29, "goal_dim": 730, "hidden": [2048, 1024, 512, 256], "latent_dim": 32, "out_gain": 0.01}
ON = {"actor_arch": "enc_dec", "actor_hidden": [64, 48, 32]}


def linears(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


# ------------------------------------------------------------------ the network
def test_ndm_sizes_give_ndms_parameter_count_and_shapes() -> None:
    actor = EncDecActor(**NDM)
    assert sum(p.numel() for p in actor.net.parameters()) == 9_116_477
    enc, dec = linears(actor.net.enc), linears(actor.net.dec)
    assert [(m.in_features, m.out_features) for m in enc] == [(730, 2048), (2048, 1024), (1024, 512), (512, 256), (256, 32)]
    assert [(m.in_features, m.out_features) for m in dec] == [(1022, 2048), (2048, 1024), (1024, 512), (512, 256), (256, 29)]
    assert actor(torch.randn(5, 1720)).shape == (5, 29)


def test_the_decoders_last_layer_is_scaled_and_has_no_bias() -> None:
    torch.manual_seed(0)
    actor = EncDecActor(**NDM)
    last = linears(actor.net.dec)[-1]
    assert torch.equal(last.bias, torch.zeros(29))
    assert float(last.weight.abs().max()) <= 0.01 / 256**0.5  # nn.Linear's bound 1 / sqrt(fan_in), x 0.01
    assert float(linears(actor.net.enc)[-1].bias.abs().max()) > 0  # the encoder keeps the default init


def test_forward_is_dec_of_tanh_enc_of_the_goal_and_the_proprio() -> None:
    torch.manual_seed(1)
    actor = EncDecActor(40, 6, 15, [16, 12], 4, 0.5)
    with torch.no_grad():
        actor.obs_norm.mean.normal_()
        actor.obs_norm.var.uniform_(0.5, 2.0)
    obs = torch.randn(9, 40)
    z = (obs - actor.obs_norm.mean) / actor.obs_norm.std()
    latent = torch.tanh(actor.net.enc(z[:, 25:]))
    torch.testing.assert_close(actor(obs), actor.net.dec(torch.cat((latent, z[:, :25]), -1)), **EXACT)
    assert float(latent.abs().max()) < 1


def test_only_the_goal_columns_reach_the_encoder() -> None:
    torch.manual_seed(2)
    actor = EncDecActor(40, 6, 15, [16, 12], 4, 1.0)
    seen: list[Tensor] = []
    actor.net.enc.register_forward_hook(lambda _m, inputs, _out: seen.append(inputs[0]))
    obs = torch.randn(3, 40)
    actor(obs)
    torch.testing.assert_close(seen[0], actor.obs_norm(obs)[:, 25:], **EXACT)


def test_muon_takes_each_mlps_hidden_matrices() -> None:
    actor = EncDecActor(**NDM)
    opt = MuonAdam(actor.net, actor.muon_mlps(), lr=1e-3, betas=(0.9, 0.95))
    muon = {id(p) for group in opt.muon.param_groups for p in group["params"]}
    adam = {id(p) for group in opt.adam.param_groups for p in group["params"]}
    want = {id(m.weight) for mlp in (actor.net.enc, actor.net.dec) for m in linears(mlp)[1:-1]}
    assert muon == want
    assert len(muon) == 6
    assert not muon & adam
    assert muon | adam == {id(p) for p in actor.net.parameters()}
    # One MLP over both would hand Muon the encoder's output and the decoder's input layer.
    wrong = MuonAdam(actor.net, [actor.net], lr=1e-3)
    assert {id(p) for group in wrong.muon.param_groups for p in group["params"]} != want


def test_the_flat_actor_keeps_its_muon_split_and_keys() -> None:
    actor = Actor(1720, 29, [1024, 1024, 512, 256], 0.01)
    assert actor.muon_mlps() == [actor.net]
    assert sorted(actor.state_dict()) == sorted(
        ["obs_norm.mean", "obs_norm.var", "obs_norm.count", "obs_norm.frozen"]
        + [f"net.{i}.{k}" for i in (0, 2, 4, 6, 8) for k in ("weight", "bias")]
    )
    torch.manual_seed(3)
    a = build_actor("flat", 1720, 29, [64, 32], 0.01)
    torch.manual_seed(3)
    b = Actor(1720, 29, [64, 32], 0.01)
    assert harness.state_equal(a.state_dict(), b.state_dict())


# ------------------------------------------------------------------ the agent
def test_the_agent_builds_and_trains_it_with_muon(tmp_path: Any) -> None:
    agent = harness.make_agent(frames=harness.frames_for(4), log_dir=tmp_path, actor_opt="muon", actor_ema=0.9, **ON)
    assert isinstance(agent.actor, EncDecActor)
    assert agent.actor.split == agent.task.history * 99
    muon = {id(p) for group in agent.actor_optim.muon.param_groups for p in group["params"]}
    assert muon == {id(m.weight) for mlp in agent.actor.muon_mlps() for m in linears(mlp)[1:-1]}
    assert len(muon) == 4
    agent.train()
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())
    meta = agent.policy_meta()
    assert (meta["actor_arch"], meta["actor_latent_dim"], meta["actor_goal_dim"]) == ("enc_dec", 32, 730)


def test_the_default_agent_is_flat_and_its_meta_unchanged() -> None:
    agent = harness.make_agent()
    assert type(agent.actor) is Actor
    assert not {"actor_arch", "actor_latent_dim", "actor_goal_dim"} & set(agent.policy_meta())


def trained(tmp_path: Any, **wm: Any) -> tuple[Any, Any]:
    agent = harness.make_agent(frames=harness.frames_for(4), log_dir=tmp_path, **wm)
    agent.train()
    return agent, max(agent._checkpoint_dir().glob("model_step_*.pt"))


def policy_td(obs: dict[str, Any], n: int) -> TensorDict:
    return TensorDict({"wm": TensorDict(obs["wm"], batch_size=[n])}, batch_size=[n])


@pytest.mark.parametrize("ema", [0.0, 0.9])
def test_a_default_evaluator_scores_an_enc_dec_checkpoint(tmp_path: Any, ema: float) -> None:
    source, path = trained(tmp_path, actor_ema=ema, **ON)
    evaluator = harness.make_agent(actor_ema=0.5)  # a flat actor with its own average
    evaluator.load_policy(str(path))
    assert isinstance(evaluator.actor, EncDecActor)
    assert evaluator.policy_net.actor is evaluator.actor
    assert evaluator.policy_meta()["actor_arch"] == "enc_dec"
    obs, _ = evaluator._isaac.reset()
    td = policy_td(obs, evaluator._num_envs)
    torch.testing.assert_close(evaluator.collector_policy(td.clone())["action"], source.collector_policy(td.clone())["action"], **EXACT)
    feat = evaluator.task.policy_obs_from_env(*(obs["wm"][key] for _, key in evaluator.task.policy_in_keys))
    torch.testing.assert_close(evaluator.predict(feat), source.predict(feat), **EXACT)


def test_an_enc_dec_evaluator_scores_a_flat_checkpoint(tmp_path: Any) -> None:
    source, path = trained(tmp_path)
    evaluator = harness.make_agent(**ON)
    evaluator.load_policy(str(path))
    assert type(evaluator.actor) is Actor
    assert "actor_arch" not in evaluator.policy_meta()
    obs, _ = evaluator._isaac.reset()
    td = policy_td(obs, evaluator._num_envs)
    torch.testing.assert_close(evaluator.collector_policy(td.clone())["action"], source.collector_policy(td.clone())["action"], **EXACT)


@pytest.mark.parametrize(("saved", "resumed"), [(ON, {}), ({}, ON), (ON, {**ON, "actor_latent_dim": 16})])
def test_a_resume_must_match_the_actor(tmp_path: Any, saved: dict[str, Any], resumed: dict[str, Any]) -> None:
    _, path = trained(tmp_path, **saved)
    with pytest.raises(ValueError, match="actor_"):
        harness.make_agent(**resumed).load_model(str(path))


def test_an_enc_dec_run_resumes(tmp_path: Any) -> None:
    agent, path = trained(tmp_path, actor_opt="muon", **ON)
    resumed = harness.make_agent(frames=harness.frames_for(6), log_dir=tmp_path, actor_opt="muon", **ON)
    resumed.load_model(str(path))
    assert harness.state_equal(resumed.actor.state_dict(), agent.actor.state_dict())
    assert harness.state_equal(resumed.actor_optim.state_dict(), agent.actor_optim.state_dict())
    resumed.train()
    assert resumed._iteration > 4


# ------------------------------------------------------------------ refused
def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


GOAL = SimpleNamespace(policy_goal_dim=730, policy_obs_dim=1720)


@pytest.mark.parametrize(
    ("wm", "task", "match"),
    [
        ({"actor_arch": "transformer"}, GOAL, "actor_arch"),
        ({"actor_arch": "enc_dec"}, SimpleNamespace(policy_obs_dim=1720), "policy_goal_dim"),
        ({"actor_arch": "enc_dec"}, SimpleNamespace(policy_goal_dim=1720, policy_obs_dim=1720), "no other input"),
        ({"actor_arch": "enc_dec", "actor_latent_dim": 0}, GOAL, "actor_latent_dim"),
        ({"actor_arch": "enc_dec", "actor_opt": "muon", "actor_hidden": [64]}, GOAL, "actor_hidden"),
    ],
)
def test_refused(wm: dict[str, Any], task: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(**wm), task)


def test_allowed() -> None:
    validate_options(config(actor_arch="enc_dec", actor_opt="muon"), GOAL)
    validate_options(config(actor_arch="flat"), SimpleNamespace())


def test_the_v3_task_names_its_goal_columns() -> None:
    agent = harness.make_agent()
    t = agent.task
    assert t.policy_goal_dim == 730
    rank, frame = torch.tensor([0, 1]), torch.tensor([3, 5])
    obs = t.policy_obs_from_parts(torch.randn(2, t.history, 99), t.library.ref_state(rank, frame), rank, frame)
    torch.testing.assert_close(obs[:, -t.policy_goal_dim :], t.goal(None, rank, frame), **EXACT)


# ------------------------------------------------------------------ the recipe's systems options
class SyncFreeEnv(harness.FakeIsaacEnv):
    def step(self, action: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor, dict[str, Any]]:
        zeros = torch.zeros(self.n, device=self.device)
        self.frame = self.frame + 1
        self.episode = self.episode + 1
        self.frame = torch.where(self.frame >= self.v3_library.length[self.rank] - 2, 0, self.frame)
        self.hist_action = torch.cat((self.hist_action[:, 1:], action[:, None]), 1)
        return self._obs(), zeros, zeros.bool(), zeros.bool(), {}


RECIPE = {"no_sync": True, "fused_adam": True, "compile": True, "compile_eager": True, "actor_opt": "muon", "actor_ema": 0.9, **ON}


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
def test_the_enc_dec_actor_adds_no_host_read_and_no_recompile(graphs: bool) -> None:
    wm = {**RECIPE, **({"update_graphs": True, "graph_early": True} if graphs else {})}
    agent = harness.make_agent(device="cuda:0", num_envs=64, env_cls=SyncFreeEnv, **wm)
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats: dict[str, list] = defaultdict(list)
    for _ in range(44):  # the replay's first fill
        obs = agent._collect_step(obs, stats)
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
