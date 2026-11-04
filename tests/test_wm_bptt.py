"""Unit tests of the world-model BPTT agent's building blocks (CPU)."""

from __future__ import annotations

import pytest
import torch

from rlopt.agent.wm_bptt.agent import lambda_returns, real_window_masks, real_window_targets, score_function_surrogate
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


def test_replay_window_carries_the_env_outcome_of_each_row() -> None:
    rp = _filled_replay()
    rp.add(torch.zeros(2, 2), torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 1),
           torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
           torch.tensor([6, 10]), torch.zeros(2, dtype=torch.bool))
    rp.set_outcome(torch.tensor([0.5, 0.25]), torch.tensor([True, False]))
    win = rp.window(torch.tensor([9]), torch.tensor([0]), 2)  # rows 9, 10
    assert win["env_reward"][0].tolist() == [0.0, 0.5]
    assert win["terminated"][0].tolist() == [False, True]


def test_real_window_masks_and_targets_handle_episode_ends() -> None:
    """Three windows of H = 3: no end, a termination on transition 1, a
    timeout on transition 1 (the state after each end is another episode)."""
    gamma, lam = 0.9, 0.8
    same = torch.tensor([[True, True, True], [True, False, True], [True, False, True]])
    terminated = torch.tensor([[False, False, False], [False, True, False], [False, False, False]])
    inep, term, trunc = real_window_masks(same, terminated)
    assert inep.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]
    assert term.tolist() == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert trunc.tolist() == [[0, 0, 0], [0, 0, 0], [0, 1, 0]]
    r = torch.tensor([[1.0, 2.0, 3.0]] * 3)
    env_r = torch.full((3, 3), 7.0)
    v = torch.tensor([[10.0, 20.0, 30.0, 40.0]] * 3)
    tg = real_window_targets(r, env_r, v, inep, term, trunc, gamma, lam)
    # no end: the ordinary lambda-return
    assert torch.allclose(tg[0], lambda_returns(r[:1], v[:1], torch.ones(1, 3), gamma, lam)[0])
    # termination: the last transition earns the env's reward, then nothing
    assert tg[1, 1].item() == pytest.approx(7.0)
    assert tg[1, 0].item() == pytest.approx(1.0 + gamma * ((1 - lam) * 20.0 + lam * 7.0))
    # timeout: the unknown last transition is replaced by V(s_1)
    assert tg[2, 1].item() == pytest.approx(20.0)
    assert tg[2, 0].item() == pytest.approx(1.0 + gamma * 20.0)


def test_noise_inference_step_replays_data_at_the_collecting_policy() -> None:
    """a = pi_theta(x) + eta, x' = x + f(x, a) + xi with the recorded noise
    eta = a_rec - mu_old(x), xi = (s' - s) - f(s, a_rec): the collecting
    policy reproduces the recording, another policy moves the forward pass,
    and the gradient is the model's and actor's."""
    def f(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:  # model increment
        return -0.1 * x + 2.0 * a

    s, s_next, theta_old = torch.tensor(1.0), torch.tensor(1.7), 0.3
    a_rec = torch.tensor(theta_old * 1.0 + 0.05)  # collector's mean plus noise
    eta = a_rec - theta_old * s
    xi = (s_next - s) - f(s, a_rec)

    def step(theta: torch.Tensor) -> torch.Tensor:
        return s + f(s, theta * s + eta) + xi

    assert step(torch.tensor(theta_old)).item() == pytest.approx(1.7)
    theta = torch.tensor(theta_old + 0.1, requires_grad=True)
    nxt = step(theta)
    assert nxt.item() == pytest.approx(1.7 + 2.0 * 0.1)  # the model's response
    nxt.backward()
    assert theta.grad.item() == pytest.approx(2.0 * 1.0)  # df/da * dpi/dtheta


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


def test_diffsr_readout_carries_the_action_gradient_only_when_phi_is_attached() -> None:
    torch.manual_seed(0)
    gains = {}
    for detach in (True, False):
        model = DiffSRDynamics(6, 3, (2, 4), feature_dim=8, embed_dim=16, g_hidden_dims=(16,), mu_hidden_dims=(16,), num_noises=4, readout_detach_phi=detach)
        torch.nn.init.normal_(model.readout.weight)
        x = torch.randn(5, 6, requires_grad=True)
        y = model(x)
        gains[detach] = torch.autograd.grad(y.sum(), x, allow_unused=True)[0] if y.requires_grad else None
    assert gains[True] is None or gains[True].abs().sum() == 0
    assert gains[False][:, 2:4].abs().sum() > 0


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


def _point_mass_rollout(actor: Actor, x0: torch.Tensor, eps: torch.Tensor, *, detach_action: bool):
    """x' = x + 0.1 a, reward -x'^2, a = mu(x) + eps; returns per-step
    actions, means and rewards (B, H, .) as ``_imagine`` collects them."""
    x = x0
    acts, mus, rews = [], [], []
    for t in range(eps.shape[1]):
        mu = actor(x)
        a = mu + eps[:, t]
        if detach_action:
            a = a.detach()
        x = x + 0.1 * a
        acts.append(a)
        mus.append(mu)
        rews.append(-x.square().squeeze(-1))
        if detach_action:
            x = x.detach()
    return torch.stack(acts, 1), torch.stack(mus, 1), torch.stack(rews, 1)


def test_score_function_surrogate_is_unbiased_for_the_reparameterized_gradient() -> None:
    """On the point mass the likelihood-ratio gradient of the sampled-action
    return agrees in expectation with the reparameterized pathwise gradient of
    the same objective (same samples), at far lower signal-to-noise."""
    torch.manual_seed(0)
    B, H, std, gamma = 4096, 6, 0.3, 0.97
    actor = Actor(1, 1, [16], out_gain=0.3)
    actor.obs_norm.update(torch.randn(1024, 1) * 2)
    params = list(actor.net.parameters())

    def flat(loss):
        return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params)])

    pw, sf = [], []
    for k in range(40):
        g = torch.Generator().manual_seed(100 + k)
        x0 = torch.randn(B, 1, generator=g) * 2
        eps = std * torch.randn(B, H, 1, generator=g)
        _, _, r = _point_mass_rollout(actor, x0, eps, detach_action=False)
        disc = gamma ** torch.arange(H)
        pw.append(flat(-(r * disc).sum(1).mean()))
        a, mu, r = _point_mass_rollout(actor, x0, eps, detach_action=True)
        ones = torch.ones(B, H)
        values = torch.zeros(B, H + 1)
        surrogate, adv = score_function_surrogate(a, mu, r, values, ones, ones, std, gamma, 1.0, normalize=False)
        assert adv.shape == (B, H)
        sf.append(flat(-surrogate.mean()))
    pw, sf = torch.stack(pw), torch.stack(sf)
    cos = torch.nn.functional.cosine_similarity(pw.mean(0), sf.mean(0), dim=0)
    assert float(cos) > 0.95
    rel = (pw.mean(0) - sf.mean(0)).norm() / pw.mean(0).norm()
    assert float(rel) < 0.25

    def snr(g):
        return g.mean(0).norm() / (g - g.mean(0)).square().sum(1).mean().sqrt()

    assert snr(pw) > 5 * snr(sf)
    # actor_steps masks the later steps out of the sum
    short, _ = score_function_surrogate(a, mu, r, values, ones, ones, std, gamma, 1.0, normalize=False, actor_steps=2)
    full, _ = score_function_surrogate(a, mu, r, values, ones, ones, std, gamma, 1.0, normalize=False)
    assert not torch.allclose(short, full)


def test_score_function_actor_update_improves_the_point_mass_return() -> None:
    torch.manual_seed(1)
    B, H, std, gamma = 2048, 8, 0.3, 0.97
    actor = Actor(1, 1, [16], out_gain=0.01)
    actor.obs_norm.update(torch.randn(1024, 1) * 2)
    opt = torch.optim.Adam(actor.net.parameters(), lr=3e-3)

    def true_return(x0):
        with torch.no_grad():
            x, ret = x0, 0.0
            for t in range(H):
                x = x + 0.1 * actor(x)
                ret = ret + (gamma**t) * -(x.square().squeeze(-1))
        return float(ret.mean())

    start = torch.randn(512, 1) * 2
    before = true_return(start)
    ones = torch.ones(B, H)
    for _ in range(300):
        x0 = torch.randn(B, 1) * 2
        eps = std * torch.randn(B, H, 1)
        a, mu, r = _point_mass_rollout(actor, x0, eps, detach_action=True)
        surrogate, _ = score_function_surrogate(a, mu, r, torch.zeros(B, H + 1), ones, ones, std, gamma, 0.95)
        opt.zero_grad()
        (-surrogate.mean()).backward()
        opt.step()
    assert true_return(start) > 0.5 * before
