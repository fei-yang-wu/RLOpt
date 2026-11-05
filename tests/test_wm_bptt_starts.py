"""Imagination start eligibility, bounded sampling, and clip-end bootstrap."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rlopt.agent.wm_bptt.agent import WMBPTT
from rlopt.agent.wm_bptt.config import WMBPTTConfig
from rlopt.agent.wm_bptt.replay import SequenceReplay


class _LegacyTask:
    history = 3
    action_dim = 1
    sensor_noise = False
    lengths = torch.tensor([5, 9])

    def proprio(self, states, _actions, _context, _noise):
        return states

    def policy_obs_from_parts(self, _prop, state, _rank, _frame):
        return state

    def wm_input(self, state, _action, _context):
        return state

    def integrate(self, state, increment):
        return state + increment

    def reward_and_termination(self, _nxt, _action, _prev_action, _ctx, _rank, frame):
        return (frame + 1).float(), torch.zeros_like(frame, dtype=torch.float32)

    def clip_end(self, rank, frame):
        return frame >= self.lengths[rank] - 1

    def critic_obs(self, _state, _ctx, _rank, frame):
        return frame.float().unsqueeze(-1)


class _FilteredTask(_LegacyTask):
    def valid_imagination_start(self, rank, frame, horizon):
        return self.lengths[rank] - 1 - frame >= horizon


def _replay(*, steps=14, capacity=8):
    rp = SequenceReplay(2, capacity, 1, 1, 1, torch.device("cpu"))
    rank = torch.tensor([0, 1])
    for step in range(steps):
        frame = (step + torch.arange(2)) % _LegacyTask.lengths
        rp.add(
            frame.float().unsqueeze(-1),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            rank,
            frame,
            torch.full((2,), step),
            torch.zeros(2, dtype=torch.bool),
        )
    return rp


def _agent(rp, *, filtered=True):
    # Exercise the actual rollout without constructing a simulator or networks.
    agent = WMBPTT.__new__(WMBPTT)
    agent.config = SimpleNamespace(
        device="cpu", wm=WMBPTTConfig(horizon=2, gamma=0.9, compile=False)
    )
    agent.task = _FilteredTask() if filtered else _LegacyTask()
    agent.replay = rp
    agent._num_envs = rp.num_envs
    agent.actor = lambda obs, **_kwargs: torch.zeros_like(obs)
    agent.dynamics = lambda inputs, **_kwargs: torch.ones_like(inputs)
    agent.critic = lambda obs, **_kwargs: 10 * obs[:, 0]
    return agent


def _record_starts(monkeypatch, rp):
    seen = {}
    history = rp.history

    def record(tau, env, length):
        seen.update(tau=tau.clone(), env=env.clone())
        return history(tau, env, length)

    monkeypatch.setattr(rp, "history", record)
    return seen


def test_sampled_starts_fit_the_horizon_and_wrapped_history(monkeypatch):
    rp = _replay()
    agent = _agent(rp)
    seen = _record_starts(monkeypatch, rp)
    out = agent._imagine(512, grad=False, generator=torch.Generator().manual_seed(3))
    assert (seen["tau"] >= rp.oldest + agent.task.history - 1).all()
    assert (seen["tau"] < rp.steps).all()
    assert agent.task.valid_imagination_start(out["rank"], out["frame0"], 2).all()
    # Reaching the final frame is allowed; crossing it is not.
    assert ((out["frame0"] + 2) == agent.task.lengths[out["rank"]] - 1).any()
    assert torch.equal(out["cont"], torch.ones(512, 2))


def test_sparse_valid_rows_use_bounded_exhaustive_fallback(monkeypatch):
    rp = _replay(steps=8)
    rp.rank.zero_()
    rp.frame.fill_(4)
    rp.frame[7, 1] = 2  # The only eligible pair, missed by all random proposals.
    agent = _agent(rp)
    proposals = []

    def always_low(low, high, size, **kwargs):
        proposals.append((low, high))
        return torch.full(size, low, device=kwargs["device"], dtype=torch.long)

    monkeypatch.setattr(torch, "randint", always_low)
    tau, env = agent._filter_imagination_starts(
        torch.tensor([2, 7]),
        torch.tensor([0, 1]),
        2,
        resample=True,
        generator=None,
    )
    assert tau.tolist() == [7, 7]
    assert env.tolist() == [1, 1]
    assert len(proposals) == 15  # Seven bounded retries, then one exact draw.


def test_all_invalid_replay_fails_after_bounded_rejection(monkeypatch):
    rp = _replay(steps=8)
    rp.rank.zero_()
    rp.frame.fill_(4)
    agent = _agent(rp)
    predicate = agent.task.valid_imagination_start
    calls = []

    def record(rank, frame, horizon):
        calls.append(frame.shape)
        return predicate(rank, frame, horizon)

    monkeypatch.setattr(agent.task, "valid_imagination_start", record)
    with pytest.raises(RuntimeError, match="No valid imagination starts for horizon 2"):
        agent._imagine(8, grad=False)
    assert len(calls) == 9  # Eight sampled batches plus the exhaustive ring scan.


def test_fallback_excludes_rows_without_retained_history():
    rp = _replay()
    rp.rank.zero_()
    rp.frame.fill_(4)
    # Retained rows 6 and 7 cannot supply a three-row history. Slot 0 is
    # absolute row 8 after ring wrap and must not be treated as absolute 0.
    rp.frame[rp.oldest % rp.capacity, 0] = 1
    agent = _agent(rp)
    with pytest.raises(RuntimeError, match=r"replay rows 8\.\.13"):
        agent._imagine(8, grad=False)


def test_explicit_invalid_starts_fail_without_replacement(monkeypatch):
    rp = _replay(steps=8)
    rp.frame[3, 0] = 3
    agent = _agent(rp)
    seen = _record_starts(monkeypatch, rp)
    generator = torch.Generator().manual_seed(5)
    before = generator.get_state()
    with pytest.raises(
        ValueError, match="Explicit imagination starts are invalid for horizon 2"
    ):
        agent._imagine(
            1,
            grad=False,
            starts=(torch.tensor([3]), torch.tensor([0])),
            generator=generator,
        )
    assert torch.equal(generator.get_state(), before)
    assert seen == {}


@pytest.mark.parametrize("grad", [False, True])
def test_exact_final_frame_preserves_reward_and_critic_bootstrap(monkeypatch, grad):
    rp = _replay(steps=8)
    rp.frame[3, 0] = 2
    rp.state[3, 0, 0] = 2
    agent = _agent(rp)
    seen = _record_starts(monkeypatch, rp)
    starts = (torch.tensor([3]), torch.tensor([0]))
    out = agent._imagine(1, grad=grad, starts=starts)
    assert torch.equal(seen["tau"], starts[0])
    assert torch.equal(seen["env"], starts[1])
    assert out["rewards"].tolist() == [[3.0, 4.0]]
    assert out["states"].squeeze(-1).tolist() == [[2.0, 3.0, 4.0]]
    assert out["alive_H"].item() == 1.0
    assert out["objective"].item() == pytest.approx(3 + 0.9 * 4 + 0.9**2 * 40)


def test_legacy_task_keeps_the_original_start_draws_and_rng(monkeypatch):
    rp = _replay()
    agent = _agent(rp, filtered=False)
    seen = _record_starts(monkeypatch, rp)
    generator = torch.Generator().manual_seed(11)
    reference = torch.Generator().manual_seed(11)
    expected_tau = torch.randint(
        rp.oldest + agent.task.history - 1, rp.steps, (32,), generator=reference
    )
    expected_env = torch.randint(0, rp.num_envs, (32,), generator=reference)
    agent._imagine(32, grad=False, generator=generator)
    assert torch.equal(seen["tau"], expected_tau)
    assert torch.equal(seen["env"], expected_env)
    assert torch.equal(generator.get_state(), reference.get_state())


def test_horizon_override_controls_start_eligibility():
    rp = _replay(steps=8)
    rp.frame[3, 0] = 3
    agent = _agent(rp)
    out = agent._imagine(
        1, grad=False, horizon=1, starts=(torch.tensor([3]), torch.tensor([0]))
    )
    assert out["rewards"].tolist() == [[4.0]]
    assert out["alive_H"].item() == 1.0
