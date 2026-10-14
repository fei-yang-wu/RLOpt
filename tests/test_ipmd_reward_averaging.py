"""Reward-saddle averaging: Polyak target and reservoir replay."""

from __future__ import annotations

import torch
from torch import nn

from rlopt.agent.ipmd.reward_averaging import (
    PrioritizedFifoFeatureReplay,
    ReservoirFeatureReplay,
    polyak_update,
)


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


def test_fifo_overwrites_the_oldest_rows_first() -> None:
    replay = PrioritizedFifoFeatureReplay(capacity=4, dim=1, device="cpu")
    replay.add(torch.arange(3.0)[:, None])
    assert replay.size == 3
    replay.add(torch.tensor([[10.0], [11.0]]))
    assert replay.size == 4
    # Slots 0..2 held 0,1,2; the 4th and 5th rows land in slots 3 and 0.
    assert replay.rows.reshape(-1).tolist() == [11.0, 1.0, 2.0, 10.0]
    replay.add(torch.arange(100.0, 106.0)[:, None])  # larger than capacity: keep the newest 4
    assert sorted(replay.rows.reshape(-1).tolist()) == [102.0, 103.0, 104.0, 105.0]


def test_prioritized_sampling_follows_priority_and_weights_have_mean_one() -> None:
    torch.manual_seed(0)
    replay = PrioritizedFifoFeatureReplay(capacity=2, dim=1, device="cpu", alpha=1.0, beta=1.0, eps=1e-6)
    replay.add(torch.tensor([[0.0], [1.0]]))
    replay.update_priorities(torch.tensor([0, 1]), torch.tensor([1.0, 3.0]))
    rows, slots, weights = replay.sample(40000)
    share = (slots == 1).float().mean().item()
    assert abs(share - 0.75) < 0.01
    assert abs(weights.mean().item() - 1.0) < 1e-5
    # w ∝ 1/P: the rarer row carries 3x the weight of the common one.
    w0 = weights[slots == 0][0].item()
    w1 = weights[slots == 1][0].item()
    assert abs(w0 / w1 - 3.0) < 1e-4
    # New rows enter at the running max priority.
    replay.add(torch.tensor([[5.0]]))
    assert float(replay.priority[0]) == float(replay.max_priority)


def test_uniform_fifo_has_unit_weights() -> None:
    replay = PrioritizedFifoFeatureReplay(capacity=8, dim=2, device="cpu")
    replay.add(torch.randn(8, 2))
    _, _, weights = replay.sample(16)
    assert torch.all(weights == 1.0)
