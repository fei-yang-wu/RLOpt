"""``wm.fused_adam``, ``wm.actor_opt``, ``wm.wm_opt`` (CPU unless marked)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness

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
