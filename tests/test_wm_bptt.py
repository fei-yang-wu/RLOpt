"""Unit tests of the world-model BPTT agent's building blocks (CPU)."""

from __future__ import annotations

import pytest
import torch

from rlopt.agent.wm_bptt.agent import lambda_returns
from rlopt.agent.wm_bptt.diffsr_dynamics import DiffSRDynamics
from rlopt.agent.wm_bptt.networks import Actor, Critic, DynamicsModel, RunningNorm
from rlopt.agent.wm_bptt.replay import SequenceReplay


def test_running_norm_matches_batch_statistics() -> None:
    torch.manual_seed(0)
    x = torch.randn(1000, 5) * torch.tensor([1.0, 2.0, 0.5, 3.0, 10.0]) + 4.0
    norm = RunningNorm(5, eps=0.0)
    for chunk in x.split(137):
        norm.update(chunk)
    assert torch.allclose(norm.mean, x.mean(0), atol=1e-4)
    assert torch.allclose(norm.var, x.var(0, unbiased=False), rtol=1e-4)
    norm.frozen.fill_(True)
    norm.update(x + 100)
    assert torch.allclose(norm.mean, x.mean(0), atol=1e-4)


def test_lambda_returns_matches_direct_recursion() -> None:
    torch.manual_seed(0)
    B, H, gamma, lam = 3, 6, 0.9, 0.8
    r = torch.randn(B, H)
    v = torch.randn(B, H + 1)
    c = torch.rand(B, H)
    out = lambda_returns(r, v, c, gamma, lam)
    for b in range(B):
        nxt = v[b, H]
        for t in reversed(range(H)):
            nxt = r[b, t] + gamma * c[b, t] * ((1 - lam) * v[b, t + 1] + lam * nxt)
            assert torch.isclose(out[b, t], nxt, atol=1e-6)
    # lambda = 0 is one-step TD; lambda = 1 is the discounted return plus V_H.
    td0 = lambda_returns(r, v, torch.ones(B, H), gamma, 0.0)
    assert torch.allclose(td0, r + gamma * v[:, 1:], atol=1e-6)
    mc = lambda_returns(r, v, torch.ones(B, H), gamma, 1.0)
    disc = sum(gamma**t * r[:, t] for t in range(H)) + gamma**H * v[:, H]
    assert torch.allclose(mc[:, 0], disc, atol=1e-5)


def _filled_replay() -> SequenceReplay:
    """Two envs; env 0 resets after step 3, env 1 is pushed at step 5."""
    rp = SequenceReplay(2, 16, state_dim=2, action_dim=1, context_dim=1, device=torch.device("cpu"))
    ep = [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    for t in range(10):
        state = torch.tensor([[t, 0.0], [t, 1.0]])
        push = torch.tensor([False, t == 5])
        rp.add(state, torch.zeros(2, 1), torch.full((2, 1), float(t)), torch.zeros(2, 1),
               torch.zeros(2, dtype=torch.long), torch.tensor([t, t]),
               torch.tensor([ep[0][t], ep[1][t]]), push)
    return rp


def test_replay_window_validity_respects_resets_and_pushes() -> None:
    rp = _filled_replay()
    win = rp.window(torch.tensor([2, 2]), torch.tensor([0, 1]), 4)  # rows 2..5
    # env 0: 2->3 valid, 3->4 crosses the reset, 4->5 valid
    assert win["valid"][0].tolist() == [True, False, True]
    # env 1: the push lands on row 5, so 4->5 is invalid
    assert win["valid"][1].tolist() == [True, True, False]


def test_replay_history_backfills_from_episode_start() -> None:
    rp = _filled_replay()
    h = rp.history(torch.tensor([5]), torch.tensor([0]), 4)  # env 0, episode step 1 at row 5
    # rows oldest first; frames before the episode start repeat its first row (4)
    assert h["hist_states"][0, :, 0].tolist() == [4.0, 4.0, 4.0, 5.0]
    assert h["hist_actions"][0, :, 0].tolist() == [4.0, 4.0, 4.0, 5.0]
    h = rp.history(torch.tensor([9]), torch.tensor([1]), 4)
    assert h["hist_states"][0, :, 0].tolist() == [6.0, 7.0, 8.0, 9.0]


@pytest.fixture(autouse=True)
def _few_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prev)


def test_dynamics_model_starts_at_mean_increment_and_fits_kstep() -> None:
    """A linear system: the zero last layer predicts the running mean; an
    autoregressive k-step fit then drives the rollout error down."""
    torch.manual_seed(0)
    A = torch.tensor([[0.0, 0.1], [-0.1, 0.0]])
    model = DynamicsModel(3, 2, [32, 32])
    x = torch.randn(1024, 2)
    u = torch.randn(1024, 1)
    inc = x @ A.T + 0.05 * u
    model.in_norm.update(torch.cat((x, u), -1))
    model.out_norm.update(inc)
    pred0 = model(torch.cat((x, u), -1))
    assert torch.allclose(pred0, inc.mean(0).expand_as(pred0), atol=1e-5)
    opt = torch.optim.Adam(model.net.parameters(), lr=3e-3)

    def rollout_error() -> torch.Tensor:
        s = x[:128]
        s_true = s
        err = 0.0
        for k in range(4):
            a = u[k * 128 : (k + 1) * 128]
            s = s + model(torch.cat((s, a), -1))
            s_true = s_true + s_true @ A.T + 0.05 * a
            err = err + ((s - s_true) / model.out_norm.std()).square().mean()
        return err

    before = float(rollout_error())
    for _ in range(200):
        loss = rollout_error()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(rollout_error()) < 0.05 * before


def test_diffsr_dynamics_readout_and_samples_fit_a_linear_system() -> None:
    """DiffSR on the same linear system: the zero read-out starts at the mean
    increment; after denoising + probe training both the linear read-out of
    phi and the mean of ancestral samples recover the increment."""
    torch.manual_seed(0)
    A = torch.tensor([[0.0, 0.5], [-0.5, 0.0]])
    x = torch.randn(4096, 2)
    u = torch.randn(4096, 1)
    ctx = torch.randn(4096, 1)
    inc = x @ A.T + 0.5 * u
    inp = torch.cat((x, u, ctx), -1)
    model = DiffSRDynamics(
        4, 2, (2, 3), feature_dim=8, embed_dim=16, g_hidden_dims=(64, 64),
        mu_hidden_dims=(64, 64), num_noises=8,
    )
    model.in_norm.update(inp)
    model.out_norm.update(inc)
    assert torch.allclose(model(inp), inc.mean(0).expand_as(inc), atol=1e-5)
    assert model.source_index.tolist() == [0, 1, 3]
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(600):
        idx = torch.randint(0, 4096, (512,))
        diff, phi = model.denoising_loss(inp[idx], inc[idx])
        read = (model.normalize_increment(model.readout_from_phi(phi)) - model.normalize_increment(inc[idx])).square().mean()
        opt.zero_grad()
        (diff + read).backward()
        opt.step()
    test = slice(0, 256)
    scale = model.out_norm.std()
    read_err = ((model(inp[test]) - inc[test]) / scale).square().mean()
    with torch.no_grad():
        draws = torch.stack([model.sample(inp[test]) for _ in range(16)])
    sample_err = ((draws.mean(0) - inc[test]) / scale).square().mean()
    assert float(read_err) < 0.05
    assert float(sample_err) < 0.1
    assert torch.isfinite(draws).all()


def test_actor_and_critic_take_named_activations() -> None:
    actor = Actor(4, 2, [8, 8], out_gain=0.01, act="silu")
    critic = Critic(4, [8], act="SiLU")
    assert all(not isinstance(m, torch.nn.ELU) for m in actor.net)
    assert sum(isinstance(m, torch.nn.SiLU) for m in actor.net) == 2
    assert sum(isinstance(m, torch.nn.SiLU) for m in critic.net) == 1
    assert isinstance(Actor(4, 2, [8], out_gain=0.01).net[1], torch.nn.ELU)
    with pytest.raises(ValueError, match="unknown activation"):
        Critic(4, [8], act="swish")


def test_bptt_through_learned_model_improves_return() -> None:
    """Point mass x' = x + 0.1 a with reward -x'^2: the gradient of an
    imagined return through a fitted model trains a stabilizing actor."""
    torch.manual_seed(0)
    model = DynamicsModel(2, 1, [32])
    xs = torch.randn(1024, 1) * 2
    acts = torch.randn(1024, 1) * 2
    model.in_norm.update(torch.cat((xs, acts), -1))
    model.out_norm.update(0.1 * acts)
    mopt = torch.optim.Adam(model.net.parameters(), lr=3e-3)
    for _ in range(200):
        loss = (model(torch.cat((xs, acts), -1)) - 0.1 * acts).square().mean()
        mopt.zero_grad()
        loss.backward()
        mopt.step()
    model.requires_grad_(False)
    actor = Actor(1, 1, [16], out_gain=0.01)
    actor.obs_norm.update(xs)
    aopt = torch.optim.Adam(actor.net.parameters(), lr=1e-2)

    def imagined_return(x0: torch.Tensor) -> torch.Tensor:
        x, ret = x0, 0.0
        for t in range(16):
            x = x + model(torch.cat((x, actor(x)), -1))
            ret = ret + (0.99**t) * -(x.square().squeeze(-1))
        return ret.mean()

    start = torch.randn(256, 1) * 2
    before = float(imagined_return(start))
    for _ in range(100):
        loss = -imagined_return(torch.randn(256, 1) * 2)
        aopt.zero_grad()
        loss.backward()
        aopt.step()
    after = float(imagined_return(start))
    assert after > 0.5 * before  # returns are negative: closer to zero is better
    # and it transfers to the true dynamics
    x, true_ret = start, 0.0
    with torch.no_grad():
        for t in range(16):
            x = x + 0.1 * actor(x)
            true_ret = true_ret + (0.99**t) * -(x.square().squeeze(-1))
    assert float(true_ret.mean()) > 0.5 * before
