"""Reward-saddle averaging: Polyak target and reservoir replay."""

from __future__ import annotations

import torch
from torch import nn

from rlopt.agent.ipmd.reward_averaging import ReservoirFeatureReplay, polyak_update


def test_polyak_moves_the_target_by_tau() -> None:
    torch.manual_seed(0)
    source, target = nn.Linear(3, 1), nn.Linear(3, 1)
    before = target.weight.detach().clone()
    polyak_update(target, source, 0.25)
    expected = 0.75 * before + 0.25 * source.weight.detach()
    assert torch.allclose(target.weight, expected)
    polyak_update(target, source, 1.0)
    assert torch.allclose(target.weight, source.weight)


def test_reservoir_fills_then_samples_uniformly_over_history() -> None:
    torch.manual_seed(0)
    replay = ReservoirFeatureReplay(capacity=2000, dim=1, device="cpu")
    assert replay.size == 0
    # 20 "rollouts" of 1000 rows each, row value = rollout index
    for k in range(20):
        replay.add(torch.full((1000, 1), float(k)))
    assert replay.size == 2000
    assert int(replay.seen) == 20000
    counts = torch.bincount(replay.rows.reshape(-1).long(), minlength=20).float()
    # uniform over history: each rollout holds about 100 of the 2000 slots,
    # and the earliest rollouts are not flushed out as in a FIFO
    assert counts[0] > 50 and counts[-1] < 150
    assert counts.std() / counts.mean() < 0.25
    sample = replay.sample(500)
    assert sample.shape == (500, 1)


def test_reservoir_partial_fill_keeps_every_row() -> None:
    replay = ReservoirFeatureReplay(capacity=100, dim=2, device="cpu")
    replay.add(torch.arange(60, dtype=torch.float32).reshape(30, 2))
    assert replay.size == 30
    assert torch.equal(replay.rows[:30], torch.arange(60, dtype=torch.float32).reshape(30, 2))
