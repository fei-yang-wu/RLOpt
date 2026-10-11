"""Bookkeeping of the DiffSR chunk prior: ring across rollouts, episode resets,
evaluation schedule, and the heading anchor."""

from __future__ import annotations

import math

import torch
from torch import nn

from rlopt.agent.ipmd.reward_prior import (
    FRAME_WIDTH,
    PAST,
    SPAN,
    DiffSRChunkPrior,
    heading_anchored_frames,
)


def _stub_prior(num_envs: int, stride: int = 10) -> DiffSRChunkPrior:
    """A prior whose score is the global step stored in frame slot t (PAST)."""
    prior = DiffSRChunkPrior.__new__(DiffSRChunkPrior)
    nn.Module.__init__(prior)
    prior.device = torch.device("cpu")
    prior.weight, prior.kappa, prior.stride = 1.0, 1e9, stride
    prior.draws, prior.batch_size, prior.num_envs = 1, 64, num_envs
    prior.transform, prior.log_lo, prior.log_hi = "exp", 5.0, 1.0e5
    prior.register_buffer("ring", torch.zeros(num_envs, SPAN - 1, FRAME_WIDTH))
    prior.register_buffer("position", torch.zeros(num_envs, dtype=torch.long))
    prior.chunk_scores = lambda windows: windows[:, PAST, 0].clone()  # type: ignore[method-assign]
    return prior


def _frames(num_envs: int, start: int, steps: int) -> torch.Tensor:
    frames = torch.zeros(num_envs, steps, FRAME_WIDTH)
    frames[..., 0] = torch.arange(start, start + steps, dtype=torch.float32)[None, :]
    return frames


def test_schedule_and_ring_across_rollouts() -> None:
    prior = _stub_prior(num_envs=1)
    seen: list[tuple[int, float]] = []
    for k in range(3):  # 72 steps, no reset
        reward, _ = prior.compute(_frames(1, 24 * k, 24), torch.zeros(1, 24, dtype=torch.bool))
        for t in torch.nonzero(reward[0]).reshape(-1).tolist():
            # g = weight * stride * exp(score / kappa) ~= stride; read the chunk
            # origin back from the stub through the frame value instead.
            seen.append((24 * k + t, float(reward[0, t])))
    rows = [row for row, _ in seen]
    # first evaluation at episode position SPAN - 1 = 25, then every 10 steps
    assert rows == [25, 35, 45, 55, 65]
    assert all(math.isclose(value, 10.0, rel_tol=1e-6) for _, value in seen)


def test_window_reads_the_right_frames_across_a_boundary() -> None:
    prior = _stub_prior(num_envs=1, stride=1)
    prior.kappa = 1.0
    prior.weight = 1.0
    captured: list[torch.Tensor] = []

    def capture(windows: torch.Tensor) -> torch.Tensor:
        captured.append(windows[:, :, 0].clone())
        return torch.zeros(windows.shape[0])

    prior.chunk_scores = capture  # type: ignore[method-assign]
    prior.compute(_frames(1, 0, 24), torch.zeros(1, 24, dtype=torch.bool))
    prior.compute(_frames(1, 24, 24), torch.zeros(1, 24, dtype=torch.bool))
    windows = torch.cat(captured)
    # row T's window covers global steps T-25 .. T, contiguous
    first = windows[0]
    assert first.tolist() == [float(v) for v in range(0, SPAN)]
    straddling = windows[1]  # row 26 of the timeline, i.e. global step 26
    assert straddling.tolist() == [float(v) for v in range(1, SPAN + 1)]


def test_reset_restarts_the_counter() -> None:
    prior = _stub_prior(num_envs=2)
    done = torch.zeros(2, 24, dtype=torch.bool)
    done[1, 5] = True  # env 1 resets after row 5
    prior.compute(_frames(2, 0, 24), done)
    reward, _ = prior.compute(_frames(2, 24, 24), torch.zeros(2, 24, dtype=torch.bool))
    env0 = torch.nonzero(reward[0]).reshape(-1).tolist()
    env1 = torch.nonzero(reward[1]).reshape(-1).tolist()
    assert env0 == [1, 11, 21]  # global 25, 35, 45
    # env 1's episode starts at global step 6: position 25 at global 31 = row 7
    assert env1 == [7, 17]


def test_heading_anchor_cancels_yaw_and_translation() -> None:
    torch.manual_seed(0)
    side = torch.zeros(3, SPAN, 36)
    side[..., :29] = torch.randn(3, SPAN, 29)
    side[..., 29:32] = torch.randn(3, SPAN, 3)
    quat = torch.randn(3, SPAN, 4)
    side[..., 32:36] = quat / quat.norm(dim=-1, keepdim=True)
    yaw = torch.tensor([0.0, 0.0, math.sin(0.35), math.cos(0.35)])
    moved = side.clone()
    from rlopt.agent.ipmd.reward_prior import _quat_mul, _quat_rotate

    moved[..., 29:32] = _quat_rotate(yaw.expand(3, SPAN, 4), side[..., 29:32]) + torch.tensor([1.5, -2.0, 0.0])
    moved[..., 32:36] = _quat_mul(yaw.expand(3, SPAN, 4), side[..., 32:36])
    a = heading_anchored_frames(side, PAST)
    b = heading_anchored_frames(moved, PAST)
    assert torch.allclose(a, b, atol=1e-5)


def test_log_transform_is_graded_and_bounded() -> None:
    prior = _stub_prior(num_envs=1)
    prior.transform, prior.log_lo, prior.log_hi = "log", 5.0, 1.0e5
    h = prior.shape(torch.tensor([-1.0, -7.0, -60.0, -300.0, -3.0e4, -1.0e6]))
    assert h[0] == 1.0 and h[-1] == 0.0
    assert torch.all(h[1:-1] < 1.0) and torch.all(h[1:-1] > 0.0)
    assert torch.all(h[:-1] >= h[1:])  # monotone in r_B
