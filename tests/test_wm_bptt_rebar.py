"""Tests of the score-function correction of SVG(H) (CPU)."""

from __future__ import annotations

import pytest
import torch

from rlopt.agent.wm_bptt.rebar import (
    CorrectionAverage,
    clipped_score_surrogate,
    linearized_return,
    masked_center,
    shuffle_rows,
)


def test_linearized_return_matches_the_direct_sum() -> None:
    torch.manual_seed(0)
    q, eta = torch.randn(3, 5, 2), torch.randn(3, 5, 2)
    mask = torch.ones(3, 5)
    mask[1, 3:] = 0
    got = linearized_return(q, eta, mask, 0.9)
    e = (q * eta).sum(-1) * mask
    want = torch.stack([sum(0.9 ** (k - t) * e[:, k] for k in range(t, 5)) for t in range(5)], 1)
    assert torch.allclose(got, want, atol=1e-5)


def test_correction_recovers_the_true_gradient_from_a_wrong_slope() -> None:
    """One step, Q(a) = w . a with a pathwise slope that is wrong: slope plus score
    correction on the residual is unbiased for w, and the slope alone is not."""
    torch.manual_seed(1)
    sigma, n, dim = 0.1, 2_000_000, 3
    true_w = torch.tensor([1.0, -2.0, 0.5])
    wrong = torch.tensor([0.5, -1.0, 2.0])
    mu = torch.zeros(dim, requires_grad=True)
    eta = sigma * torch.randn(n, dim)
    action = mu.detach() + eta
    adv = action @ true_w
    residual = adv - eta @ wrong  # A - q . eta
    surrogate, _ = clipped_score_surrogate(
        mu.expand(n, 1, dim), mu.detach().expand(n, 1, dim), action.unsqueeze(1),
        residual.unsqueeze(1), torch.ones(n, 1), sigma, 1e9,
    )
    (grad,) = torch.autograd.grad(surrogate, mu)
    assert torch.allclose(wrong + grad, true_w, atol=0.02)
    assert (wrong - true_w).abs().max() > 1.0


def test_residual_is_small_when_the_slope_is_right() -> None:
    torch.manual_seed(2)
    w = torch.tensor([1.0, -2.0])
    eta = 0.1 * torch.randn(1000, 1, 2)
    adv = (eta @ w)
    rho = adv - linearized_return(w.expand(1000, 1, 2), eta, torch.ones(1000, 1), 0.9)
    assert rho.abs().max() < 1e-5


def test_clip_stops_the_gradient_outside_the_ratio_range() -> None:
    mu_old = torch.zeros(1, 1, 1)
    action = torch.ones(1, 1, 1)
    mu = torch.full((1, 1, 1), 0.5, requires_grad=True)  # raises the sampled action's density
    out, frac = clipped_score_surrogate(mu, mu_old, action, torch.ones(1, 1), torch.ones(1, 1), 1.0, 0.2)
    (g,) = torch.autograd.grad(out, mu, allow_unused=True)
    assert frac == 1.0
    assert g is None or g.abs().sum() == 0


def test_center_and_shuffle_touch_kept_rows_only() -> None:
    x = torch.tensor([[1.0, 2.0, 100.0], [3.0, 4.0, 200.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    c = masked_center(x, mask)
    assert abs(float((c * mask).sum())) < 1e-6
    torch.manual_seed(3)
    s = shuffle_rows(x, mask)
    assert s[:, 2].tolist() == [100.0, 200.0]
    assert sorted(s[mask > 0].tolist()) == [1.0, 2.0, 3.0, 4.0]


def test_average_is_unbiased_at_the_start_and_capped_against_svg() -> None:
    avg = CorrectionAverage(rate=0.05, cap=0.5)
    svg = torch.tensor([10.0, 0.0])
    out, scale = avg.update(torch.tensor([1.0, 0.0]), svg)
    assert torch.allclose(out, torch.tensor([1.0, 0.0]))
    assert scale == 1.0
    out, scale = avg.update(torch.tensor([1000.0, 0.0]), svg)
    assert abs(float(out.norm()) - 5.0) < 1e-4
    assert scale < 1.0


def test_average_rejects_a_bad_rate() -> None:
    with pytest.raises(ValueError, match="rate"):
        CorrectionAverage(rate=0.0, cap=0.5)
