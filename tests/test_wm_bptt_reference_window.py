"""``wm.reference_window``: one gather of each imagined rollout's reference rows.

The rollout reads goals, rewards, terminations and critic inputs through the task's
library by window slot instead of bank rank; every value must equal the plain path's,
bit for bit, and the window must be closed again after the actor-critic update.
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


def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


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


# ------------------------------------------------------------------ validation
def test_the_option_needs_a_library_with_open_window() -> None:
    plain = SimpleNamespace(max_step_reward=0.1)
    with pytest.raises(ValueError, match="open_window"):
        validate_options(config(reference_window=True), plain)
    with_lib = SimpleNamespace(max_step_reward=0.1, library=SimpleNamespace(open_window=lambda *a: None))
    validate_options(config(reference_window=True), with_lib)
    with pytest.raises(ValueError, match="actor_source=real"):
        validate_options(config(reference_window=True, actor_source="real", critic_source="real"), with_lib)
    with pytest.raises(ValueError, match="rebar"):
        validate_options(config(reference_window=True, actor_estimator="rebar"), with_lib)
    validate_options(config(), plain)  # off: no requirement


# ------------------------------------------------------------------ equivalence
def test_windowed_rollouts_equal_plain_rollouts() -> None:
    plain = harness.make_agent(contact_reward=0.15, ref_contact_fix=True)
    windowed = harness.make_agent(contact_reward=0.15, ref_contact_fix=True, reference_window=True)
    # Same seed, same construction order: identical weights, normalizers and replay.
    for a, b in zip(plain.actor.parameters(), windowed.actor.parameters()):
        assert torch.equal(a, b)
    torch.manual_seed(11)  # the fake env and the collection noise draw from torch's RNG
    collected(plain)
    torch.manual_seed(11)
    collected(windowed)
    assert torch.equal(plain.replay.rank, windowed.replay.rank)
    tau, env = starts(plain, 6)
    assert tau.numel() == 6
    gen_a = torch.Generator(device=plain.device).manual_seed(3)
    gen_b = torch.Generator(device=plain.device).manual_seed(3)
    torch.manual_seed(5)  # imagined sensor noise draws from torch's RNG
    a = plain._imagine(6, grad=True, starts=(tau, env), generator=gen_a)
    torch.manual_seed(5)
    b = windowed._imagine(6, grad=True, starts=(tau, env), generator=gen_b)
    lib = windowed.task.library
    assert lib.win_active  # the critic update reads through it next
    assert torch.equal(b["rank"], torch.arange(6, device=plain.device))  # slots, not ranks
    assert torch.equal(a["rank"], plain.replay.rank[plain.replay._slot(tau), env])
    for key in ("objective", "states", "rewards", "dones", "cont", "alive_H", "actions", "frame0"):
        assert torch.equal(a[key].detach(), b[key].detach()), key
    # The critic's inputs over the rollout agree too (window slots in ``b``).
    ca = plain._critic_obs_seq(a)
    cb = windowed._critic_obs_seq(b)
    assert torch.equal(ca.detach(), cb.detach())
    lib.close_window()


def test_the_actor_critic_update_closes_the_window() -> None:
    agent = harness.make_agent(reference_window=True)
    collected(agent)
    metrics: dict[str, list] = defaultdict(list)
    agent._update_actor_critic(metrics)
    lib = agent.task.library
    assert not lib.win_active and lib.win_states is None
    # A second update opens a fresh window and closes it again.
    agent._update_actor_critic(metrics)
    assert not lib.win_active
    # Plain reads by bank rank work after it.
    rank = torch.zeros(3, dtype=torch.long, device=agent.device)
    frame = torch.arange(3, device=agent.device)
    assert lib.ref_state(rank, frame).shape == (3, 71)
