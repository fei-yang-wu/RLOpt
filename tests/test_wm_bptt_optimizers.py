"""``wm.fused_adam``, ``wm.actor_opt``, ``wm.wm_opt`` (CPU unless marked)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness
from torch import nn

from rlopt.agent.wm_bptt.muon import MuonAdam
from rlopt.agent.wm_bptt.speed_options import set_lr

RUN = {"frames": harness.frames_for(4)}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def checkpoint(tmp_path: Path, **wm: Any) -> tuple[Any, Path]:
    agent = harness.make_agent(**RUN, **wm)
    agent.train()
    path = tmp_path / "run.pt"
    agent.save_model(path)
    return agent, path


def adam_settings(agent: Any, setting: str) -> list[Any]:
    groups = [g for opt in (agent.wm_optim, agent.actor_optim, agent.critic_optim) for g in opt.param_groups]
    return [g[setting] for g in groups]


def test_default_optimizers_are_unfused() -> None:
    agent = harness.make_agent()
    assert type(agent.wm_optim) is torch.optim.AdamW
    assert type(agent.actor_optim) is torch.optim.Adam and type(agent.critic_optim) is torch.optim.Adam
    assert not any(adam_settings(agent, "fused"))


def test_fused_adam_reaches_every_optimizer() -> None:
    assert all(adam_settings(harness.make_agent(fused_adam=True), "fused"))


def test_a_fused_run_trains() -> None:
    agent = harness.make_agent(fused_adam=True, **RUN)
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    agent.train()
    assert all(torch.isfinite(p).all() for p in agent.actor.parameters())
    assert not harness.state_equal(before, agent.actor.state_dict())


@pytest.mark.parametrize("saved", [False, True])
def test_resume_refuses_another_fused_setting(tmp_path: Path, saved: bool) -> None:
    _, path = checkpoint(tmp_path, fused_adam=saved)
    with pytest.raises(ValueError, match="fused_adam"):
        harness.make_agent(fused_adam=not saved).load_model(str(path))


def test_resume_keeps_the_fused_kernels(tmp_path: Path) -> None:
    """``load_state_dict`` replaces the groups, including their ``fused`` entry."""
    trained, path = checkpoint(tmp_path, fused_adam=True)
    resumed = harness.make_agent(fused_adam=True)
    resumed.load_model(str(path))
    assert all(adam_settings(resumed, "fused"))
    assert harness.state_equal(trained.actor_optim.state_dict(), resumed.actor_optim.state_dict())


def test_a_fused_checkpoint_scores_on_a_default_agent(tmp_path: Path) -> None:
    trained, path = checkpoint(tmp_path, fused_adam=True)
    evaluator = harness.make_agent()
    evaluator.load_policy(str(path))
    assert harness.state_equal(evaluator.actor.state_dict(), trained.actor.state_dict())


@cuda
def test_one_fused_step_matches_foreach_on_gpu() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 32, device="cuda")
    nets = [torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ELU(), torch.nn.Linear(32, 8)).cuda() for _ in range(2)]
    nets[1].load_state_dict(nets[0].state_dict())
    opts = [torch.optim.Adam(nets[0].parameters(), lr=1e-3), torch.optim.Adam(nets[1].parameters(), lr=1e-3, fused=True)]
    for net, opt in zip(nets, opts, strict=True):
        net(x).square().mean().backward()
        opt.step()
    for a, b in zip(nets[0].parameters(), nets[1].parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------- Muon
def small_net() -> nn.Sequential:
    return nn.Sequential(nn.Linear(6, 16), nn.ELU(), nn.Linear(16, 16), nn.ELU(), nn.Linear(16, 16), nn.ELU(), nn.Linear(16, 3))


def linears(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def muon_ids(opt: MuonAdam) -> set[int]:
    return {id(p) for g in opt.muon.param_groups for p in g["params"]}


def adam_ids(opt: MuonAdam) -> set[int]:
    return {id(p) for g in opt.adam.param_groups for p in g["params"]}


def step_on(net: nn.Module, opt: MuonAdam, x: torch.Tensor) -> None:
    opt.zero_grad()
    net(x).square().mean().backward()
    opt.step()


def test_muon_takes_exactly_the_hidden_matrices() -> None:
    net = small_net()
    opt = MuonAdam(net, [net], lr=1e-3, betas=(0.9, 0.95))
    assert muon_ids(opt) == {id(m.weight) for m in linears(net)[1:-1]}
    assert not muon_ids(opt) & adam_ids(opt)
    assert muon_ids(opt) | adam_ids(opt) == {id(p) for p in net.parameters()}
    assert opt.muon.defaults["adjust_lr_fn"] == "match_rms_adamw" and opt.muon.defaults["weight_decay"] == 0
    assert opt.adam.defaults["betas"] == (0.9, 0.95)


def test_muon_refuses_an_mlp_without_hidden_matrices() -> None:
    net = nn.Sequential(nn.Linear(4, 4), nn.ELU(), nn.Linear(4, 2))
    with pytest.raises(ValueError, match="no matrices"):
        MuonAdam(net, [net], lr=1e-3)


def test_muon_refuses_matrices_outside_the_model() -> None:
    net, other = small_net(), small_net()
    with pytest.raises(ValueError, match="parameters of the model"):
        MuonAdam(net, [other], lr=1e-3)


def test_set_lr_reaches_float_and_tensor_learning_rates() -> None:
    net = small_net()
    opt = MuonAdam(net, [net], lr=1e-3)
    set_lr(opt, 5e-4)
    assert [g["lr"] for g in opt.param_groups] == [5e-4, 5e-4]
    lr = torch.tensor(1.0)
    opt.adam.param_groups[0]["lr"] = lr
    set_lr(opt, 2e-4)
    assert opt.adam.param_groups[0]["lr"] is lr and lr.item() == pytest.approx(2e-4)


def test_a_restored_muon_steps_as_the_original() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 6)
    net = small_net()
    opt = MuonAdam(net, [net], lr=1e-3)
    before = [p.detach().clone() for p in net.parameters()]
    step_on(net, opt, x)
    assert all(not torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))
    other = copy.deepcopy(net)
    restored = MuonAdam(other, [other], lr=1e-3)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    step_on(net, opt, x)
    step_on(other, restored, x)
    assert harness.state_equal(net.state_dict(), other.state_dict())


def test_the_gated_step_skips_entirely_or_equals_the_plain_step() -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 6)
    nets = [small_net() for _ in range(2)]
    nets[1].load_state_dict(nets[0].state_dict())
    plain, gated = (MuonAdam(n, [n], lr=1e-2, fused=True) for n in nets)
    for ok in (True, False, True):
        for net, opt in zip(nets, (plain, gated), strict=True):
            opt.zero_grad()
            net(x).square().mean().backward()
        before = copy.deepcopy(gated.state_dict())
        weights = {k: v.clone() for k, v in nets[1].state_dict().items()}
        if ok:
            plain.step()
        gated.step(ok=torch.tensor(ok))
        if ok:
            assert harness.state_equal(nets[0].state_dict(), nets[1].state_dict())
        else:
            assert harness.state_equal(weights, nets[1].state_dict())
            assert harness.state_equal(before, gated.state_dict())


def test_the_gated_step_needs_the_fused_adam() -> None:
    net = small_net()
    opt = MuonAdam(net, [net], lr=1e-2)
    net(torch.randn(4, 6)).sum().backward()
    with pytest.raises(ValueError, match="fused"):
        opt.step(ok=torch.tensor(True))


def test_the_tensor_lr_step_equals_torch_muon_with_a_float_lr() -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 6)
    nets = [small_net() for _ in range(2)]
    nets[1].load_state_dict(nets[0].state_dict())
    float_lr, tensor_lr = (MuonAdam(n, [n], lr=1e-2) for n in nets)
    for g in tensor_lr.muon.param_groups:
        g["lr"] = torch.tensor(1e-2)
    for _ in range(3):
        for net in nets:
            net.zero_grad()
            net(x).square().mean().backward()
        float_lr.muon.step()
        tensor_lr._muon_step_tensor_lr()
    for a, b in zip(nets[0].parameters(), nets[1].parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


@cuda
def test_capturable_muon_replays_in_a_cuda_graph_and_reads_the_filled_lr() -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 6, device="cuda")
    net = small_net().cuda()
    opt = MuonAdam(net, [net], lr=1e-2, capturable=True)
    assert opt.adam.defaults["capturable"] and opt.adam.defaults["fused"]
    for _ in range(2):  # warm up the state tensors before capture
        step_on(net, opt, x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        opt.zero_grad(set_to_none=False)
        net(x).square().mean().backward()
        opt.step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=False)
    with torch.cuda.graph(graph):
        net(x).square().mean().backward()
        opt.step()
    params = list(net.parameters())

    def change_after_replay() -> float:
        before = [p.detach().clone() for p in params]
        graph.replay()
        return max(float((a - b).abs().max()) for a, b in zip(before, params, strict=True))

    set_lr(opt, 1e-2)
    full = change_after_replay()
    set_lr(opt, 0.0)
    assert full > 0 and change_after_replay() == 0.0


@cuda
def test_capturable_muon_keeps_its_device_learning_rates_across_a_load() -> None:
    net = small_net().cuda()
    opt = MuonAdam(net, [net], lr=1e-2, capturable=True)
    step_on(net, opt, torch.randn(8, 6, device="cuda"))
    tensors = [g["lr"] for g in opt.param_groups]
    set_lr(opt, 3e-3)
    opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert all(g["lr"] is t for g, t in zip(opt.param_groups, tensors, strict=True))
    assert all(float(t) == pytest.approx(3e-3) for t in tensors)


def test_the_agent_builds_muon_on_the_actor_and_the_world_model_only() -> None:
    agent = harness.make_agent(actor_opt="muon", wm_opt="muon")
    assert muon_ids(agent.actor_optim) == {id(m.weight) for m in linears(agent.actor.net)[1:-1]}
    assert muon_ids(agent.wm_optim) == {id(m.weight) for m in linears(agent.dynamics.net)[1:-1]}
    assert agent.actor_optim.adam.defaults["betas"] == tuple(agent.config.wm.actor_betas)
    assert agent.wm_optim.adam.defaults["betas"] == (0.9, 0.999)
    assert type(agent.critic_optim) is torch.optim.Adam


def test_learning_rates_reach_the_muon_groups() -> None:
    agent = harness.make_agent(actor_opt="muon", wm_opt="muon", actor_lr=1e-3, wm_lr=3e-3, lr_cosine_iters=10)
    agent._iteration = 5
    agent._set_lrs()
    f = agent._lr_factor()
    assert [g["lr"] for g in agent.actor_optim.param_groups] == [1e-3 * f, 1e-3 * f]
    assert [g["lr"] for g in agent.wm_optim.param_groups] == [3e-3 * f, 3e-3 * f]


@pytest.mark.parametrize("wm", [{"actor_opt": "muon"}, {"wm_opt": "muon"}, {"actor_opt": "muon", "wm_opt": "muon", "fused_adam": True}])
def test_a_muon_run_trains_and_resumes(tmp_path: Path, wm: dict[str, Any]) -> None:
    trained, path = checkpoint(tmp_path, **wm)
    assert all(torch.isfinite(p).all() for module in (trained.actor, trained.dynamics) for p in module.parameters())
    resumed = harness.make_agent(**wm)
    resumed.load_model(str(path))
    for name in ("wm_optim", "actor_optim"):
        assert harness.state_equal(getattr(trained, name).state_dict(), getattr(resumed, name).state_dict())


@pytest.mark.parametrize(("name", "saved"), [("actor_opt", "muon"), ("actor_opt", "adam"), ("wm_opt", "muon"), ("wm_opt", "adam")])
def test_resume_refuses_another_optimizer(tmp_path: Path, name: str, saved: str) -> None:
    _, path = checkpoint(tmp_path, **{name: saved})
    with pytest.raises(ValueError, match=name):
        harness.make_agent(**{name: "adam" if saved == "muon" else "muon"}).load_model(str(path))


def test_a_muon_checkpoint_of_other_sizes_with_an_average_scores_on_a_default_agent(tmp_path: Path) -> None:
    """The arm SCORE meets: other world-model and critic sizes, Muon, a policy average."""
    trained, path = checkpoint(tmp_path, actor_opt="muon", wm_opt="muon", wm_hidden=[48, 48, 48], critic_hidden=[24], actor_ema=0.9)
    evaluator = harness.make_agent()
    evaluator.load_policy(str(path))
    assert harness.state_equal(evaluator.actor.state_dict(), trained.actor_avg.module.state_dict())
    with pytest.raises(ValueError, match="(actor_ema|policy_live|actor_opt|wm_opt)"):
        harness.make_agent().load_model(str(path))


@cuda
def test_every_option_of_this_stage_trains_on_the_gpu_under_bf16(tmp_path: Path) -> None:
    """Muon on both networks, fused Adam, the average and the multiplier together, then
    the end file resumes and scores."""
    wm = {
        "actor_opt": "muon", "wm_opt": "muon", "fused_adam": True, "actor_ema": 0.99, "actor_lr_mult": 2.0,
        "actor_lr_decay_return": 0.05, "actor_lr_decay_len": 3, "ckpt_light": True, "last_every_min": 1e-9,
    }  # fmt: skip
    agent = harness.make_agent(device="cuda:0", frames=harness.frames_for(8), save_interval=40, log_dir=tmp_path, **wm)
    assert agent.config.wm.bf16
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    agent.train()
    assert all(torch.isfinite(p).all() for m in (agent.actor, agent.dynamics, agent.critic) for p in m.parameters())
    assert not harness.state_equal(before, agent.actor.state_dict())
    final = agent._checkpoint_dir() / f"model_step_{agent._frames_processed}.pt"
    resumed = harness.make_agent(device="cuda:0", save_interval=40, **wm)
    resumed.load_model(str(final))
    assert harness.state_equal(resumed.actor_avg.module.state_dict(), agent.actor_avg.module.state_dict())
    evaluator = harness.make_agent(device="cuda:0")
    evaluator.load_policy(str(final))
    assert harness.state_equal(evaluator.actor.state_dict(), agent.actor_avg.module.state_dict())
