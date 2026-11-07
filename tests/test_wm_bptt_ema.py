"""``wm.actor_ema``: the policy average, its checkpoints and its deployment (CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import wm_bptt_harness as harness
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from rlopt.agent.wm_bptt.networks import Actor
from rlopt.agent.wm_bptt.speed_options import ActorAverage

DECAY = 0.9
RUN = {"frames": harness.frames_for(4)}


def make_actor() -> Actor:
    torch.manual_seed(0)
    actor = Actor(6, 3, [8, 8], 0.5)
    actor.obs_norm.update(torch.randn(16, 6))
    return actor


def perturb(actor: Actor) -> None:
    with torch.no_grad():
        for p in actor.net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    actor.obs_norm.update(3.0 + torch.randn(16, actor.obs_norm.mean.numel()))


def test_first_update_copies_and_later_ones_follow_the_formula() -> None:
    actor = make_actor()
    average = ActorAverage(actor, DECAY)
    perturb(actor)
    average.update()
    assert harness.state_equal(average.module.state_dict(), actor.state_dict())
    old = {k: v.clone() for k, v in average.module.state_dict().items()}
    perturb(actor)
    average.update()
    state, live = average.module.state_dict(), actor.state_dict()
    for name in ("net.0.weight", "net.2.bias", "obs_norm.mean", "obs_norm.var"):
        torch.testing.assert_close(state[name], old[name] + (1 - DECAY) * (live[name] - old[name]), rtol=1e-6, atol=1e-7)
    assert state["obs_norm.count"] == live["obs_norm.count"]
    assert state["obs_norm.frozen"] == live["obs_norm.frozen"]
    assert state["obs_norm.count"].dtype == torch.float64 and state["obs_norm.frozen"].dtype == torch.bool


def test_the_average_equals_torch_averaged_model_bit_for_bit() -> None:
    actor = make_actor()
    average = ActorAverage(actor, DECAY)
    reference = AveragedModel(actor.net, multi_avg_fn=get_ema_multi_avg_fn(DECAY))
    for _ in range(5):
        perturb(actor)
        average.update()
        reference.update_parameters(actor.net)
    for ours, theirs in zip(average.module.net.parameters(), reference.module.parameters(), strict=True):
        assert torch.equal(ours, theirs)


def test_the_average_holds_no_gradients() -> None:
    average = ActorAverage(make_actor(), DECAY)
    assert not any(p.requires_grad for p in average.module.parameters())


def test_the_average_changes_nothing_in_training() -> None:
    """Same seed: the live run with the average on is bit-identical to one with it off."""
    runs = []
    for ema in (0.0, DECAY):
        agent = harness.make_agent(actor_ema=ema, **RUN)
        agent.train()
        runs.append(harness.fingerprint(agent))
    assert runs[0] == runs[1]


def trained(tmp_path: Path, name: str = "ema.pt"):
    agent = harness.make_agent(actor_ema=DECAY, **RUN)
    agent.train()
    assert agent.actor_avg is not None and agent.actor_avg.updates > 1
    path = tmp_path / name
    agent.save_model(path)
    return agent, path


def actor_entries(state: dict) -> dict:
    prefix = "module.actor."
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def test_the_checkpoint_holds_the_average_as_the_deployed_policy(tmp_path: Path) -> None:
    agent, path = trained(tmp_path)
    data = torch.load(path, weights_only=False)
    assert harness.state_equal(actor_entries(data["policy_state_dict"]), agent.actor_avg.module.state_dict())
    assert harness.state_equal(actor_entries(data["policy_live_state_dict"]), agent.actor.state_dict())
    assert not harness.state_equal(data["policy_state_dict"], data["policy_live_state_dict"])
    assert data["actor_ema_updates"] == agent.actor_avg.updates


def test_a_default_evaluator_scores_the_average(tmp_path: Path) -> None:
    agent, path = trained(tmp_path)
    evaluator = harness.make_agent()
    evaluator.load_policy(str(path))
    assert harness.state_equal(evaluator.actor.state_dict(), agent.actor_avg.module.state_dict())
    assert evaluator.collector_policy is evaluator.policy


def test_an_ema_evaluator_scores_the_average_too(tmp_path: Path) -> None:
    agent, path = trained(tmp_path)
    evaluator = harness.make_agent(actor_ema=DECAY)
    evaluator.load_policy(str(path))
    state = actor_entries(evaluator.collector_policy.state_dict())
    assert harness.state_equal(state, agent.actor_avg.module.state_dict())


def test_the_trained_agent_deploys_the_average(tmp_path: Path) -> None:
    agent, _ = trained(tmp_path)
    assert agent.collector_policy is agent.policy_avg
    obs = torch.randn(4, agent.task.policy_obs_dim)
    torch.testing.assert_close(agent.predict(obs), agent.actor_avg.module(obs))


def test_resume_restores_the_live_actor_and_the_average(tmp_path: Path) -> None:
    agent, path = trained(tmp_path)
    resumed = harness.make_agent(actor_ema=DECAY)
    resumed.load_model(str(path))
    assert harness.state_equal(resumed.actor.state_dict(), agent.actor.state_dict())
    assert harness.state_equal(resumed.actor_avg.module.state_dict(), agent.actor_avg.module.state_dict())
    assert resumed.actor_avg.updates == agent.actor_avg.updates
    assert resumed.collector_policy is resumed.policy_avg


def test_a_resumed_average_keeps_lerping(tmp_path: Path) -> None:
    agent, path = trained(tmp_path)
    resumed = harness.make_agent(actor_ema=DECAY)
    resumed.load_model(str(path))
    for a in (agent, resumed):
        torch.manual_seed(3)
        perturb(a.actor)
        a.actor_avg.update()
    assert harness.state_equal(resumed.actor_avg.module.state_dict(), agent.actor_avg.module.state_dict())


@pytest.mark.parametrize(("saved", "resuming"), [(DECAY, 0.0), (0.0, DECAY), (DECAY, 0.5)])
def test_resume_refuses_another_ema(tmp_path: Path, saved: float, resuming: float) -> None:
    agent = harness.make_agent(actor_ema=saved, **RUN)
    agent.train()
    path = tmp_path / "a.pt"
    agent.save_model(path)
    with pytest.raises(ValueError, match="(actor_ema|policy_live_state_dict)"):
        harness.make_agent(actor_ema=resuming).load_model(str(path))


def test_a_checkpoint_before_the_first_update_deploys_the_live_actor(tmp_path: Path) -> None:
    agent = harness.make_agent(actor_ema=DECAY)
    agent.actor.obs_norm.update(torch.randn(8, agent.task.policy_obs_dim))
    agent.ensure_replay()
    path = tmp_path / "early.pt"
    agent.save_model(path)
    data = torch.load(path, weights_only=False)
    assert harness.state_equal(actor_entries(data["policy_state_dict"]), agent.actor.state_dict())
    assert data["actor_ema_updates"] == 0


def test_off_adds_nothing(tmp_path: Path) -> None:
    agent = harness.make_agent(**RUN)
    agent.train()
    path = tmp_path / "off.pt"
    agent.save_model(path)
    data = torch.load(path, weights_only=False)
    assert agent.actor_avg is None and agent.policy_avg is None
    assert "policy_live_state_dict" not in data and "actor_ema_updates" not in data
    assert agent.collector_policy is agent.policy


def test_average_tensor_lists_track_in_place_updates() -> None:
    """The lists are built once, so the optimizer and normalizer must update in place."""
    actor = make_actor()
    average = ActorAverage(actor, DECAY)
    assert isinstance(average.module, nn.Module)
    optim = torch.optim.SGD(actor.net.parameters(), lr=0.1)
    actor(torch.randn(4, 6)).sum().backward()
    optim.step()
    actor.obs_norm.update(torch.randn(16, 6))
    average.update()
    average.update()
    assert harness.state_equal(average.module.state_dict(), actor.state_dict())
