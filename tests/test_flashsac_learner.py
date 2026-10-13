"""Tests for the FlashSAC learner library (layers, networks, learner, noise).

CPU tests run everywhere. GPU tests are skipped without CUDA. The parity test
needs the MIT reference clone and skips when it is not importable.
"""

# ruff: noqa: ARG001, PT018, C408, PLR0402
from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch

_PKG = "rlopt.agent.flashsac"
_PKG_DIR = Path(__file__).resolve().parents[1] / "rlopt" / "agent" / "flashsac"


def _import_modules():
    """Import the learner modules by package name, or by file path as a fallback."""
    import importlib
    import importlib.util

    try:
        return tuple(
            importlib.import_module(f"{_PKG}.{n}")
            for n in ("layers", "networks", "noise", "learner")
        )
    except ImportError:
        pass
    for parent in ("rlopt", "rlopt.agent", _PKG):
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)
            sys.modules[parent].__path__ = []  # type: ignore[attr-defined]
    mods = []
    for n in ("layers", "networks", "noise", "learner"):
        spec = importlib.util.spec_from_file_location(
            f"{_PKG}.{n}", _PKG_DIR / f"{n}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{_PKG}.{n}"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mods.append(mod)
    return tuple(mods)


layers, networks, noise_mod, learner_mod = _import_modules()
FlashSACLearner = learner_mod.FlashSACLearner
FlashSACLearnerConfig = learner_mod.FlashSACLearnerConfig

HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
DYNAMO_OFF = os.environ.get("TORCHDYNAMO_DISABLE") == "1"

# Small dims for fast tests.
OA, OC, ACT = 24, 40, 6


def _cfg(**kw):
    base = dict(
        actor_dim=32,
        critic_dim=64,
        num_atoms=21,
        batch_size=64,
        amp_dtype="fp32",
        seed=0,
    )
    base.update(kw)
    return FlashSACLearnerConfig(**base)


def make_batch(b: int, device, seed: int = 1, oa=OA, oc=OC, act=ACT):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    c = torch.randn(b, oc, generator=g)
    nc = torch.randn(b, oc, generator=g)
    return {
        "obs": c[:, :oa].contiguous().to(device),
        "critic_obs": c.to(device),
        "action": (torch.rand(b, act, generator=g) * 2 - 1).to(device),
        "reward": (torch.randn(b, generator=g) * 0.5).to(device),
        "next_obs": nc[:, :oa].contiguous().to(device),
        "next_critic_obs": nc.to(device),
        "terminated_n": (torch.rand(b, generator=g) < 0.1).float().to(device),
        "discount_n": torch.full((b,), 0.99**3).to(device),
    }


@pytest.fixture
def fixed_noise(monkeypatch):
    """Replace the Normal sampler by fixed noise keyed by row count."""
    import torch.distributions.normal as normal

    cache: dict[tuple, torch.Tensor] = {}

    def fixed(shape, dtype=None, device=None):
        key = (tuple(shape), str(device))
        if key not in cache:
            g = torch.Generator(device="cpu")
            g.manual_seed(100 + shape[0])
            cache[key] = torch.randn(*shape, generator=g).to(device=device, dtype=dtype)
        return cache[key]

    monkeypatch.setattr(normal, "_standard_normal", fixed)


def _all_tensors(learner):
    out = {}
    for n in ("actor", "critic", "target_critic", "temperature"):
        for k, v in getattr(learner, n).state_dict().items():
            out[f"{n}.{k}"] = v.detach().float().cpu().clone()
    return out


# ------------------------------------------------------------------ shapes


def test_param_counts_at_g1_dims():
    learner = FlashSACLearner(
        FlashSACLearnerConfig(amp_dtype="fp32"), 351, 544, 29, torch.device("cpu")
    )
    counts = learner.num_parameters()
    assert counts["actor"] == 317_944
    assert counts["critic"] == 2_455_486
    assert counts["target_critic"] == 2_455_486
    assert counts["temperature"] == 1
    assert 0.31e6 < counts["actor"] < 0.33e6
    assert 2.4e6 < counts["critic"] < 2.5e6


def test_update_shapes_and_device_scalars():
    learner = FlashSACLearner(_cfg(), OA, OC, ACT, torch.device("cpu"))
    info = learner.update(make_batch(64, "cpu"))
    for key in ("actor/loss", "actor/entropy", "critic/loss", "temperature/value"):
        assert info[key].ndim <= 1
        assert torch.isfinite(info[key]).all()
    info2 = learner.update(make_batch(64, "cpu", seed=2))
    assert "actor/loss" not in info2  # actor_update_period = 2
    assert learner.update_count == 2


def test_act_bound_and_log_prob():
    bound = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    learner = FlashSACLearner(
        _cfg(action_bound=bound), OA, OC, ACT, torch.device("cpu")
    )
    obs = torch.randn(32, OA)
    a_det = learner.act(obs, deterministic=True)
    assert a_det.shape == (32, ACT)
    mean, std = learner.actor.mean_and_std(obs, training=False)
    torch.testing.assert_close(a_det, torch.tensor(bound) * torch.tanh(mean))
    noise = torch.randn(32, ACT)
    a, logp = learner.act(obs, noise=noise, return_log_prob=True)
    torch.testing.assert_close(a, torch.tensor(bound) * torch.tanh(mean + std * noise))
    raw = mean + std * noise
    dist = torch.distributions.Normal(mean, std)
    expect = (dist.log_prob(raw) - layers.safe_tanh_log_det_jacobian(raw)).sum(-1)
    torch.testing.assert_close(logp, expect, rtol=1e-4, atol=1e-4)
    assert (a.abs() <= torch.tensor(bound) + 1e-6).all()


# ------------------------------------------------------------- C51 projection


def _reference_projection(next_lp, reward, term, disc, ent, v_min, v_max):
    """Independent implementation: dim-1 scatter_add as in the FlashSAC reference."""
    b, atoms = next_lp.shape
    support = torch.linspace(v_min, v_max, atoms).view(1, -1)
    tz = reward.view(-1, 1) + disc.view(-1, 1) * (support - ent.view(-1, 1)) * (
        1 - term.view(-1, 1)
    )
    tz = tz.clamp(v_min, v_max)
    pos = (tz - v_min) / ((v_max - v_min) / (atoms - 1))
    lower = pos.floor().long()
    upper = (lower + 1).clamp(0, atoms - 1)
    frac = pos - lower.float()
    p = next_lp.exp()
    out = torch.zeros(b, atoms)
    out.scatter_add_(1, lower.clamp(0, atoms - 1), p * (1 - frac))
    out.scatter_add_(1, upper, p * frac)
    return out


@pytest.mark.parametrize("rows", [64, 200_000])
def test_c51_projection_mass_and_bins(rows):
    g = torch.Generator()
    g.manual_seed(0)
    atoms = 101
    logits = torch.randn(rows, atoms, generator=g)
    next_lp = torch.log_softmax(logits, -1)
    reward = torch.randn(rows, generator=g) * 2.0
    term = (torch.rand(rows, generator=g) < 0.1).float()
    disc = torch.full((rows,), 0.97)
    ent = torch.randn(rows, generator=g) * 0.05
    out = learner_mod.c51_project(next_lp, reward, term, disc, ent, -5.0, 5.0)
    assert out.shape == (rows, atoms)
    # exact mass conservation, row by row
    assert (out.sum(1) - 1).abs().max() < 1e-5
    assert (out >= 0).all()
    expect = _reference_projection(next_lp, reward, term, disc, ent, -5.0, 5.0)
    torch.testing.assert_close(out, expect, rtol=0, atol=1e-6)
    # the rows above the old float32 offset limit (~166k) are correct as well
    if rows > 170_000:
        torch.testing.assert_close(out[-1000:], expect[-1000:], rtol=0, atol=1e-6)
        mean_out = (out[-1000:] * torch.linspace(-5, 5, atoms)).sum(1)
        mean_exp = (expect[-1000:] * torch.linspace(-5, 5, atoms)).sum(1)
        torch.testing.assert_close(mean_out, mean_exp, rtol=0, atol=1e-5)


def test_c51_terminated_row_ignores_next_distribution():
    atoms = 51
    lp = torch.log_softmax(torch.randn(3, atoms), -1)
    out = learner_mod.c51_project(
        lp,
        reward=torch.tensor([0.0, 1.0, -2.0]),
        terminated=torch.ones(3),
        discount=torch.full((3,), 0.9),
        entropy_term=torch.zeros(3),
        v_min=-5.0,
        v_max=5.0,
    )
    support = torch.linspace(-5, 5, atoms)
    mean = (out * support).sum(1)
    torch.testing.assert_close(mean, torch.tensor([0.0, 1.0, -2.0]), atol=1e-5, rtol=0)


# ----------------------------------------------------- entropy, temperature


def test_entropy_target_value():
    assert learner_mod.target_entropy(29, 0.15) == pytest.approx(-13.87, abs=0.01)
    learner = FlashSACLearner(
        FlashSACLearnerConfig(amp_dtype="fp32", actor_dim=32, critic_dim=64),
        OA,
        OC,
        29,
        torch.device("cpu"),
    )
    assert learner.target_entropy == pytest.approx(-13.87, abs=0.01)


def test_physical_entropy_target_subtracts_log_bounds():
    tanh = learner_mod.target_entropy(3, 0.5)
    bounds = [2.0, 4.0, 8.0]
    phys = learner_mod.target_entropy(3, 0.5, "physical", bounds)
    assert phys == pytest.approx(tanh - math.log(2.0 * 4.0 * 8.0))
    # A unit bound and the tanh space agree; a scalar bound broadcasts.
    assert learner_mod.target_entropy(3, 0.5, "physical", 1.0) == pytest.approx(tanh)
    assert learner_mod.target_entropy(3, 0.5, "physical", 2.0) == pytest.approx(
        tanh - 3 * math.log(2.0)
    )
    learner = FlashSACLearner(
        FlashSACLearnerConfig(
            amp_dtype="fp32",
            actor_dim=32,
            critic_dim=64,
            target_entropy_space="physical",
            target_entropy_sigma=0.5,
            action_bound=bounds,
        ),
        OA,
        OC,
        3,
        torch.device("cpu"),
    )
    assert learner.target_entropy == pytest.approx(phys)


def test_entropy_target_rejects_bad_space_and_bounds():
    with pytest.raises(ValueError):
        learner_mod.target_entropy(3, 0.5, "joint")
    with pytest.raises(ValueError):
        learner_mod.target_entropy(3, 0.5, "physical", [1.0, 2.0])
    with pytest.raises(ValueError):
        learner_mod.target_entropy(3, 0.5, "physical", [1.0, 0.0, 2.0])


@pytest.mark.parametrize("sigma", [0.15, 0.001])
def test_temperature_update_direction(sigma, fixed_noise):
    learner = FlashSACLearner(
        _cfg(target_entropy_sigma=sigma), OA, OC, ACT, torch.device("cpu")
    )
    info = learner.update(make_batch(64, "cpu"))
    old = info["temperature/value"].clone()
    new = learner.temperature().detach()
    gap = learner.target_entropy - info["actor/entropy"]
    assert torch.sign(new - old).item() == torch.sign(gap).item() != 0


def test_target_is_in_tanh_space_independent_of_bound(fixed_noise):
    a = FlashSACLearner(_cfg(), OA, OC, ACT, torch.device("cpu"))
    b = FlashSACLearner(
        _cfg(action_bound=[3.0] * ACT), OA, OC, ACT, torch.device("cpu")
    )
    batch = make_batch(64, "cpu")
    ia, ib = a.update(batch), b.update(batch)
    for k in ia:
        torch.testing.assert_close(ia[k], ib[k])
    obs = torch.randn(8, OA)
    noise = torch.randn(8, ACT)
    torch.testing.assert_close(
        b.act(obs, noise=noise), 3.0 * a.act(obs, noise=noise), atol=1e-4, rtol=1e-4
    )


# --------------------------------------------------------- weight norm


def _check_unit_norms(learner, check_target=False):
    modules = [learner.actor, learner.critic] + (
        [learner.target_critic] if check_target else []
    )
    n_checked = 0
    for top in modules:
        for m in top.modules():
            if isinstance(m, layers.UnitLinear):
                rows = m.w.weight.norm(dim=-1)
                torch.testing.assert_close(
                    rows, torch.ones_like(rows), atol=1e-5, rtol=0
                )
            elif isinstance(m, layers.EnsembleUnitLinear):
                rows = m.weight.norm(dim=-1)
                torch.testing.assert_close(
                    rows, torch.ones_like(rows), atol=1e-5, rtol=0
                )
            elif isinstance(m, (layers.UnitBatchNorm, layers.EnsembleUnitBatchNorm)):
                sq = (m.weight**2 + m.bias**2).sum(-1)
                torch.testing.assert_close(
                    sq, torch.full_like(sq, m.weight.shape[-1]), atol=1e-3, rtol=1e-5
                )
            elif isinstance(m, (layers.UnitRMSNorm, layers.EnsembleUnitRMSNorm)):
                sq = (m.weight**2).sum(-1)
                torch.testing.assert_close(
                    sq, torch.full_like(sq, m.weight.shape[-1]), atol=1e-3, rtol=1e-5
                )
            else:
                continue
            n_checked += 1
    assert n_checked > 10


def test_weight_norm_invariants_after_steps():
    learner = FlashSACLearner(_cfg(), OA, OC, ACT, torch.device("cpu"))
    _check_unit_norms(learner)  # at init
    for i in range(4):
        learner.update(make_batch(64, "cpu", seed=i))
        _check_unit_norms(learner)


def test_weight_norm_off_keeps_free_weights(fixed_noise):
    learner = FlashSACLearner(_cfg(weight_norm=False), OA, OC, ACT, torch.device("cpu"))
    for i in range(3):
        learner.update(make_batch(64, "cpu", seed=i))
    rows = learner.critic.encoder[0].w1.weight.norm(dim=-1)
    assert (rows - 1.0).abs().max() > 1e-5


def test_target_ema_over_parameters():
    learner = FlashSACLearner(_cfg(tau=0.5), OA, OC, ACT, torch.device("cpu"))
    before = [p.clone() for p in learner.target_critic.parameters()]
    learner.update(make_batch(64, "cpu"))
    for p0, pt, pc in zip(
        before,
        learner.target_critic.parameters(),
        learner.critic.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(pt, 0.5 * p0 + 0.5 * pc, atol=1e-6, rtol=1e-5)
    assert not any(p.requires_grad for p in learner.target_critic.parameters())


# ------------------------------------------------------- reward scaling


def test_reward_scale_matches_formula():
    n, steps, gamma = 7, 40, 0.99
    scale = learner_mod.ReturnScale(gamma, g_max=5.0)
    g = torch.Generator()
    g.manual_seed(3)
    ret = torch.zeros(n)
    max_abs = 0.0
    hist = []
    for _ in range(steps):
        r = torch.randn(n, generator=g)
        d = (torch.rand(n, generator=g) < 0.1).float()
        ret = gamma * (1 - d) * ret + r
        max_abs = max(max_abs, ret.abs().max().item())
        hist.append(ret.clone())
        scale.observe(r, d)
    allg = torch.stack(hist).double()
    # Pooled mean/var of all per-step cross-env statistics (parallel-moments update).
    var = allg.var(unbiased=False).item()
    expect = max(math.sqrt(var + 1e-8), max_abs / 5.0)
    assert scale.scale().item() == pytest.approx(expect, rel=0.05)
    assert scale.max_abs.item() == pytest.approx(max_abs, rel=1e-5)
    assert scale.count.item() == n * steps


def test_update_divides_reward_by_scale(fixed_noise):
    learner = FlashSACLearner(_cfg(), OA, OC, ACT, torch.device("cpu"))
    learner.observe_rewards(torch.full((16,), 10.0), torch.zeros(16))
    expected = learner.return_scale.scale().item()
    assert expected > 1.0
    info = learner.update(make_batch(64, "cpu"))
    assert info["reward/scale"].item() == pytest.approx(expected)


# --------------------------------------------------------------- noise


def test_noise_repeat_statistics():
    torch.manual_seed(0)
    nr = noise_mod.NoiseRepeat(20_000, 1, "cpu")
    flags = []
    for _ in range(300):
        nr()
        flags.append(nr.last_redrawn.clone())
    flags = torch.stack(flags)  # (T, N)
    assert flags[0].all()
    # run lengths between consecutive redraws, per env (complete runs only)
    t_idx, e_idx = flags.nonzero(as_tuple=True)
    order = torch.argsort(e_idx * 1000 + t_idx)
    t_idx, e_idx = t_idx[order], e_idx[order]
    same = e_idx[1:] == e_idx[:-1]
    runs = (t_idx[1:] - t_idx[:-1])[same].float()
    assert (runs <= 16).all() and (runs >= 1).all()
    assert (runs == 1).float().mean().item() == pytest.approx(0.631, abs=0.01)
    assert runs.mean().item() == pytest.approx(2.13, abs=0.03)


def test_noise_repeat_holds_and_resets_per_env():
    torch.manual_seed(0)
    nr = noise_mod.NoiseRepeat(4, 3, "cpu")
    first = nr().clone()
    nr.remaining[:] = 5  # all envs would hold their noise for 5 more calls
    held = nr()
    torch.testing.assert_close(held, first)
    assert not nr.last_redrawn.any()
    mask = torch.tensor([True, False, False, True])
    new = nr(mask)
    assert nr.last_redrawn.tolist() == [True, False, False, True]
    torch.testing.assert_close(new[1:3], first[1:3])
    assert not torch.allclose(new[0], first[0])
    assert not torch.allclose(new[3], first[3])
    assert nr.remaining[1].item() == 3  # others kept counting down
    assert (nr.remaining >= 0).all()
    # state round trip
    st = nr.state_dict()
    nr2 = noise_mod.NoiseRepeat(4, 3, "cpu")
    nr2.load_state_dict(st)
    torch.testing.assert_close(nr2.noise, nr.noise)


def test_zeta_pmf():
    p = noise_mod.truncated_zeta_pmf(2.0, 16)
    assert p[0].item() == pytest.approx(0.631, abs=1e-3)
    k = torch.arange(1, 17, dtype=torch.float64)
    assert (p * k).sum().item() == pytest.approx(2.13, abs=0.01)


# ------------------------------------------------- checkpoint, publication


def test_state_dict_contents_and_exact_resume(fixed_noise):
    cfg = _cfg(lr_decay_steps=1000)
    a = FlashSACLearner(cfg, OA, OC, ACT, torch.device("cpu"))
    for i in range(3):
        a.observe_rewards(torch.randn(16), (torch.rand(16) < 0.1).float())
        a.update(make_batch(64, "cpu", seed=i))
    sd = a.state_dict()
    for key in (
        "actor",
        "critic",
        "target_critic",
        "temperature",
        "optimizers",
        "scaler",
        "lr_steps",
        "update_count",
        "reward_scale",
        "action_bound",
    ):
        assert key in sd
    for net in ("actor", "critic", "target_critic"):
        names = list(sd[net])
        assert any(n.endswith("running_mean") for n in names)
        assert any(n.endswith("running_var") for n in names)
    assert sd["update_count"] == 3
    assert sd["lr_steps"]["critic"] == 3 and sd["lr_steps"]["actor"] == 2
    assert sd["reward_scale"]["count"].item() == 48
    assert "ret" in sd["reward_scale"]
    # BatchNorm buffers moved away from their init
    assert sd["actor"]["embedder.norm.running_mean"].abs().sum() > 0

    b = FlashSACLearner(
        _cfg(lr_decay_steps=1000, seed=7), OA, OC, ACT, torch.device("cpu")
    )
    b.load_state_dict(sd)
    ta, tb = _all_tensors(a), _all_tensors(b)
    for k in ta:
        assert torch.equal(ta[k], tb[k]), k
    assert b.update_count == 3
    assert b.schedules["actor"].count == 2
    assert b.return_scale.scale().item() == a.return_scale.scale().item()
    for i in range(3, 5):
        batch = make_batch(64, "cpu", seed=i)
        ia, ib = a.update(batch), b.update(batch)
        for k in ia:
            assert torch.equal(ia[k], ib[k]), k
    ta, tb = _all_tensors(a), _all_tensors(b)
    for k in ta:
        assert torch.equal(ta[k], tb[k]), k
    for n in a.optimizers:
        for pa, pb in zip(
            a.optimizers[n].param_groups, b.optimizers[n].param_groups, strict=True
        ):
            assert float(pa["lr"]) == float(pb["lr"])


def test_state_dict_survives_torch_save(tmp_path, fixed_noise):
    a = FlashSACLearner(_cfg(), OA, OC, ACT, torch.device("cpu"))
    a.update(make_batch(64, "cpu"))
    path = tmp_path / "ck.pt"
    torch.save(a.state_dict(), path)
    b = FlashSACLearner(_cfg(seed=3), OA, OC, ACT, torch.device("cpu"))
    b.load_state_dict(torch.load(path, weights_only=False))
    ia, ib = (
        a.update(make_batch(64, "cpu", seed=5)),
        b.update(make_batch(64, "cpu", seed=5)),
    )
    for k in ia:
        assert torch.equal(ia[k], ib[k]), k


def test_publish_actor_state_includes_bn_buffers():
    learner = FlashSACLearner(
        _cfg(action_bound=[2.0] * ACT), OA, OC, ACT, torch.device("cpu")
    )
    pub = noise_mod.publish_actor_state(learner)
    keys = list(pub)
    assert "actor.embedder.norm.running_mean" in keys
    assert "actor.embedder.norm.running_var" in keys
    assert any(k.endswith("encoder.0.norm1.running_mean") for k in keys)
    assert "action_bound" in keys
    n_bn = sum(1 for k in keys if k.endswith(("running_mean", "running_var")))
    assert n_bn == 2 * (1 + 2 * 2)  # embedder + 2 blocks x 2 norms
    # snapshot is independent of later updates
    snap = pub["actor.embedder.norm.running_mean"].clone()
    learner.update(make_batch(64, "cpu"))
    assert torch.equal(pub["actor.embedder.norm.running_mean"], snap)
    assert not torch.equal(learner.actor.embedder.norm.running_mean, snap)
    # a rollout copy built from the published state acts like the learner
    other = FlashSACLearner(_cfg(seed=9), OA, OC, ACT, torch.device("cpu"))
    pub2 = noise_mod.publish_actor_state(learner)
    noise_mod.load_published_actor_state(other.actor, pub2, other.action_bound)
    obs = torch.randn(8, OA)
    torch.testing.assert_close(
        other.act(obs, deterministic=True), learner.act(obs, deterministic=True)
    )
    params_only = {
        k: v for k, v in pub2.items() if not k.endswith(("running_mean", "running_var"))
    }
    with pytest.raises(KeyError):
        noise_mod.load_published_actor_state(other.actor, params_only)


# ------------------------------------------------------------- GPU tests


@needs_cuda
@pytest.mark.parametrize("amp", ["fp32", "fp16"])
def test_graph_equals_eager(amp, fixed_noise):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = _cfg(amp_dtype=amp, use_cuda_graph=True, lr_decay_steps=500)
    eager = FlashSACLearner(cfg, OA, OC, ACT, dev)
    graph = FlashSACLearner(cfg, OA, OC, ACT, dev)
    graph.capture_graphs(make_batch(256, dev))
    # capture must not train
    for k, v in _all_tensors(eager).items():
        assert torch.equal(v, _all_tensors(graph)[k]), k
    assert graph.update_count == 0
    for i in range(3):
        batch = make_batch(256, dev, seed=10 + i)
        ie = eager.update(batch)
        ig = graph.step_graph(batch)
        torch.cuda.synchronize()
        tol = 1e-4 if amp == "fp32" else 5e-3
        for k in ie:
            torch.testing.assert_close(ig[k].float(), ie[k].float(), rtol=tol, atol=tol)
    te, tg = _all_tensors(eager), _all_tensors(graph)
    for k in te:
        torch.testing.assert_close(
            tg[k],
            te[k],
            rtol=2e-3 if amp == "fp16" else 1e-4,
            atol=2e-3 if amp == "fp16" else 1e-5,
        )
    assert graph.update_count == 3
    assert float(graph.optimizers["critic"].param_groups[0]["lr"]) == pytest.approx(
        float(eager.optimizers["critic"].param_groups[0]["lr"])
    )


@needs_cuda
def test_graph_survives_load_state_dict(fixed_noise):
    dev = torch.device("cuda")
    cfg = _cfg(use_cuda_graph=True)
    src = FlashSACLearner(cfg, OA, OC, ACT, dev)
    for i in range(2):
        src.update(make_batch(128, dev, seed=i))
    graph = FlashSACLearner(_cfg(use_cuda_graph=True, seed=5), OA, OC, ACT, dev)
    graph.capture_graphs(make_batch(128, dev))
    graph.load_state_dict(src.state_dict())
    batch = make_batch(128, dev, seed=9)
    isrc, igr = src.update(batch), graph.step_graph(batch)
    torch.cuda.synchronize()
    for k in isrc:
        torch.testing.assert_close(igr[k], isrc[k], rtol=1e-4, atol=1e-4)


@needs_cuda
def test_no_host_sync_in_update(fixed_noise):
    dev = torch.device("cuda")
    learner = FlashSACLearner(_cfg(amp_dtype="fp16"), OA, OC, ACT, dev)
    batch = make_batch(128, dev)
    learner.update(batch)
    learner.update(batch)
    torch.cuda.set_sync_debug_mode("error")
    try:
        learner.update(batch)
        learner.update(batch)
        learner.observe_rewards(
            torch.randn(64, device=dev), torch.zeros(64, device=dev)
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")


@needs_cuda
@pytest.mark.skipif(DYNAMO_OFF, reason="compile is disabled by TORCHDYNAMO_DISABLE")
def test_compile_default_matches_eager(fixed_noise):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    eager = FlashSACLearner(_cfg(), OA, OC, ACT, dev)
    comp = FlashSACLearner(_cfg(compile_mode="default"), OA, OC, ACT, dev)
    for i in range(3):
        batch = make_batch(256, dev, seed=i)
        ie, ic = eager.update(batch), comp.update(batch)
        for k in ie:
            torch.testing.assert_close(ic[k], ie[k], rtol=2e-3, atol=2e-3)


@needs_cuda
@pytest.mark.skipif(DYNAMO_OFF, reason="compile is disabled by TORCHDYNAMO_DISABLE")
def test_compile_plus_graph_matches_eager(fixed_noise):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    eager = FlashSACLearner(_cfg(), OA, OC, ACT, dev)
    cg = FlashSACLearner(
        _cfg(compile_mode="default", use_cuda_graph=True), OA, OC, ACT, dev
    )
    cg.capture_graphs(make_batch(256, dev))
    for i in range(3):
        batch = make_batch(256, dev, seed=i)
        ie, ic = eager.update(batch), cg.step_graph(batch)
        for k in ie:
            torch.testing.assert_close(ic[k], ie[k], rtol=2e-3, atol=2e-3)


@needs_cuda
def test_c51_projection_cuda_large_rows():
    dev = torch.device("cuda")
    rows, atoms = 200_000, 101
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    lp = torch.log_softmax(torch.randn(rows, atoms, device=dev, generator=g), -1)
    reward = torch.randn(rows, device=dev, generator=g) * 2
    z = torch.zeros(rows, device=dev)
    out = learner_mod.c51_project(lp, reward, z, torch.full_like(z, 0.97), z, -5.0, 5.0)
    assert (out.sum(1) - 1).abs().max().item() < 1e-5
    expect = _reference_projection(
        lp.cpu(),
        reward.cpu(),
        z.cpu(),
        torch.full_like(z, 0.97).cpu(),
        z.cpu(),
        -5.0,
        5.0,
    )
    torch.testing.assert_close(out.cpu(), expect, rtol=0, atol=1e-5)


# ------------------------------------------------- parity with reference

_CLONE = Path(
    "/tmp/claude-1169732/-home-fwu91-Documents-SL-IsaacLab-Imitation--claude-worktrees-"
    "flashsac-ipmd-survey-7cc173/7a035945-f161-4e7a-b9d9-552b4c892ae0/scratchpad/survey/repos/FlashSAC"
)


def _import_reference():
    clone = os.environ.get("FLASHSAC_REFERENCE", str(_CLONE))
    if not (Path(clone) / "flash_rl").is_dir():
        return None
    if clone not in sys.path:
        sys.path.insert(0, clone)
    if "jax" not in sys.modules:
        try:
            import jax
        except ModuleNotFoundError:
            jax = types.ModuleType("jax")
            jnp = types.ModuleType("jax.numpy")
            jnp.ndarray = type("ndarray", (), {})
            jax.numpy = jnp
            sys.modules["jax"] = jax
            sys.modules["jax.numpy"] = jnp
    try:
        import flash_rl.agents.flashSAC.agent as agent
        import flash_rl.agents.flashSAC.update as update
    except Exception:
        return None
    return agent, update


@needs_cuda
def test_parity_with_reference_one_update_fp32(fixed_noise):
    ref = _import_reference()
    if ref is None:
        pytest.skip("reference FlashSAC clone not importable")
    agent, update = ref
    for name in ("_select_min_q_log_probs", "_compute_categorical_td_target"):
        fn = getattr(update, name)
        setattr(update, name, getattr(fn, "_torchdynamo_orig_callable", fn))
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    oa, oc, act, b = 351, 544, 29, 256
    ref_cfg = agent.FlashSACConfig(
        seed=0,
        normalize_reward=True,
        normalized_G_max=5.0,
        asymmetric_observation=True,
        device_type="cuda:0",
        buffer_max_length=1,
        buffer_min_length=1,
        buffer_device_type="cuda",
        sample_batch_size=b,
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=1e-6,
        learning_rate_warmup_step=100,
        learning_rate_decay_rate=1.0,
        learning_rate_decay_step=1_000_000,
        actor_num_blocks=2,
        actor_hidden_dim=128,
        actor_bc_alpha=0.0,
        actor_noise_zeta_mu=2.0,
        actor_noise_zeta_max=16,
        actor_update_period=2,
        critic_num_blocks=2,
        critic_hidden_dim=256,
        critic_num_bins=101,
        critic_min_v=-5.0,
        critic_max_v=5.0,
        critic_target_update_tau=0.01,
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        temp_target_entropy=learner_mod.target_entropy(act, 0.15),
        gamma=0.99,
        n_step=1,
        use_compile=False,
        compile_mode="default",
        use_amp=False,
        load_optimizer=False,
        load_reward_normalizer=False,
    )
    torch.manual_seed(0)
    r_actor, r_critic, r_target, r_temp = agent._init_flashsac_networks(
        oa, oc, act, ref_cfg, dev
    )
    mine = FlashSACLearner(
        FlashSACLearnerConfig(amp_dtype="fp32", n_step=1, lr_decay_steps=None),
        oa,
        oc,
        act,
        dev,
    )
    # Same initial weights: copy the reference tensors into ours (names differ, order matches).
    pairs = (
        (r_actor, mine.actor),
        (r_critic, mine.critic),
        (r_target, mine.target_critic),
        (r_temp, mine.temperature),
    )
    with torch.no_grad():
        for rnet, mnet in pairs:
            rmod = getattr(rnet.network, "_orig_mod", rnet.network)
            rt = list(rmod.parameters()) + list(rmod.buffers())
            mt = list(mnet.parameters()) + list(mnet.buffers())
            assert len(rt) == len(mt)
            for x, y in zip(rt, mt, strict=True):
                if x.shape != y.shape:
                    pytest.skip(
                        f"reference layout differs: {tuple(x.shape)} vs {tuple(y.shape)}"
                    )
                y.copy_(x)
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    crit = torch.randn(2, b, oc, generator=g).to(dev)
    batch_ref = {
        "observation": crit[0],
        "next_observation": crit[1],
        "action": (torch.rand(b, act, generator=g) * 2 - 1).to(dev),
        "reward": (torch.randn(b, generator=g) * 0.5).to(dev),
        "terminated": (torch.rand(b, generator=g) < 0.05).float().to(dev),
        "truncated": torch.zeros(b, device=dev),
        "actor_observation": crit[0][:, :oa],
        "actor_next_observation": crit[1][:, :oa],
    }
    batch_mine = {
        "obs": crit[0][:, :oa].contiguous(),
        "critic_obs": crit[0],
        "action": batch_ref["action"],
        "reward": batch_ref["reward"],
        "next_obs": crit[1][:, :oa].contiguous(),
        "next_critic_obs": crit[1],
        "terminated_n": batch_ref["terminated"],
        "discount_n": torch.full((b,), 0.99, device=dev),
    }
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for step in range(2):
        ri = agent._update_networks(
            batch=dict(batch_ref),
            actor=r_actor,
            critic=r_critic,
            target_critic=r_target,
            temperature=r_temp,
            cfg=ref_cfg,
            do_actor_update=(step == 0),
            device=dev,
            grad_scaler=scaler,
        )
        mi = mine.update(batch_mine)
        shared = [k for k in ri if k in mi]
        assert {"critic/loss", "critic/max_entropy_bonus"} <= set(shared)
        if step == 0:
            assert {"actor/loss", "actor/entropy", "temperature/loss"} <= set(shared)
        for key in shared:
            a, c = float(ri[key]), float(mi[key])
            assert abs(a - c) <= 1e-5 * max(1.0, abs(a)), (step, key, a, c)


# ------------------------------------------------- q aggregation, adam beta2


def test_q_aggregation_rejects_unknown_mode():
    with pytest.raises(ValueError):
        FlashSACLearner(_cfg(q_aggregation="max"), OA, OC, ACT, torch.device("cpu"))


def test_adam_beta2_reaches_every_optimizer():
    learner = FlashSACLearner(_cfg(adam_beta2=0.95), OA, OC, ACT, torch.device("cpu"))
    for opt in learner.optimizers.values():
        assert opt.param_groups[0]["betas"] == (0.9, 0.95)


@pytest.mark.parametrize("mode", ["min", "mean"])
def test_q_aggregation_modes_update_finitely(mode, fixed_noise):
    learner = FlashSACLearner(
        _cfg(q_aggregation=mode), OA, OC, ACT, torch.device("cpu")
    )
    for _ in range(3):
        info = learner.update(make_batch(64, "cpu"))
    assert all(torch.isfinite(v).all() for v in info.values())


def test_mean_aggregation_regresses_each_head_on_its_own_target(fixed_noise):
    """With mean aggregation the two heads see different targets, so two heads
    that start identical still drift apart only through their own targets; with
    min aggregation they share one target. Check the critic loss differs."""
    batch = make_batch(64, "cpu")
    losses = {}
    for mode in ("min", "mean"):
        learner = FlashSACLearner(
            _cfg(q_aggregation=mode), OA, OC, ACT, torch.device("cpu")
        )
        losses[mode] = learner.update(batch)["critic/loss"].item()
    assert losses["min"] != pytest.approx(losses["mean"])
