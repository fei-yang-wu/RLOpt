"""Reward-estimation revival (plan 2026-10-07, 5.4 B2/B3): objectives and mixing.

Covers the ``reward_objective`` branches of ``_backward_reward_terms``, the
parameter-free ``linear_dual`` head, the PPO squash / done-mask / constant
bonus / ramp path of ``_prepare_rollout_rewards``, the checkpoint hooks, and
a bitwise regression of the parked ``ipmd_diff`` objective.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rlopt.agent.ipmd.utils import (
    LinearDualRewardHead,
    resolve_ipmd_reward_ppo_squash,
    reward_grad_penalty_from_input,
    reward_saturation_fraction,
    reward_separation_auc,
    squash_reward_for_ppo,
)


@lru_cache(maxsize=1)
def _rlopt() -> SimpleNamespace:
    warnings.filterwarnings(
        "ignore",
        message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
        category=DeprecationWarning,
        append=False,
    )
    from rlopt.agent import IPMD, IPMDRLOptConfig
    from rlopt.env_utils import make_parallel_env

    return SimpleNamespace(
        IPMD=IPMD, IPMDRLOptConfig=IPMDRLOptConfig, make_parallel_env=make_parallel_env
    )


def _base_cfg():
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.reward_input_type = "s"
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.reward_logit_reg_coeff = 0.0
    cfg.ipmd.reward_param_weight_decay_coeff = 0.0
    return cfg


def _make_agent(**ipmd_overrides):
    rlopt = _rlopt()
    cfg = _base_cfg()
    for key, value in ipmd_overrides.items():
        setattr(cfg.ipmd, key, value)
    cfg.ipmd.validate()
    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    agent._use_estimated_rewards_for_ppo = bool(cfg.ipmd.use_estimated_rewards_for_ppo)
    return agent, env


def _obs_td(obs: torch.Tensor) -> TensorDict:
    return TensorDict({"observation": obs}, batch_size=[obs.shape[0]])


def _one_d_sides(n: int, generator: torch.Generator) -> tuple[TensorDict, TensorDict]:
    """1-D mixture on the first observation channel: expert N(+1,1), policy N(-1,1)."""
    x_exp = torch.randn(n, generator=generator) + 1.0
    x_pi = torch.randn(n, generator=generator) - 1.0
    zeros = torch.zeros(n, 2)
    expert = torch.cat([x_exp[:, None], zeros], dim=-1)
    policy = torch.cat([x_pi[:, None], zeros], dim=-1)
    return _obs_td(policy), _obs_td(expert)


def _rollout(
    agent,
    env,
    *,
    env_reward: torch.Tensor,
    done: torch.Tensor,
    truncated: torch.Tensor | None = None,
) -> TensorDict:
    n = env_reward.shape[0]
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    if truncated is None:
        truncated = torch.zeros_like(done)
    return TensorDict(
        {
            "observation": torch.randn(n, obs_dim, device=agent.device),
            "action": torch.randn(n, act_dim, device=agent.device),
            ("next", "observation"): torch.randn(n, obs_dim, device=agent.device),
            ("next", "reward"): env_reward.clone(),
            ("next", "done"): done.clone(),
            ("next", "truncated"): truncated.clone(),
        },
        batch_size=[n],
        device=agent.device,
    )


# --------------------------------------------------------------------------
# Pure helpers.
# --------------------------------------------------------------------------


def test_separation_auc_and_saturation_fraction() -> None:
    expert = torch.tensor([3.0, 4.0, 5.0])
    policy = torch.tensor([0.0, 1.0, 2.0])
    assert reward_separation_auc(expert, policy).item() == pytest.approx(1.0)
    assert reward_separation_auc(policy, expert).item() == pytest.approx(0.0)
    mixed = torch.tensor([0.0, 2.0, 4.0])
    other = torch.tensor([1.0, 3.0, 5.0])
    assert reward_separation_auc(mixed, other).item() == pytest.approx(3.0 / 9.0)
    railed = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 0.0])
    assert reward_saturation_fraction(railed).item() == pytest.approx(5.0 / 6.0)
    # 100 points spaced 1/99: five per side lie within 0.05 of an edge.
    spread = torch.linspace(0.0, 1.0, 100)
    assert reward_saturation_fraction(spread).item() == pytest.approx(10.0 / 100.0)
    assert reward_saturation_fraction(torch.zeros(4)).item() == pytest.approx(1.0)


def test_squash_ranges_per_mode() -> None:
    r = torch.linspace(-4.0, 4.0, 81)
    assert torch.equal(squash_reward_for_ppo(r, "none"), r)
    sig = squash_reward_for_ppo(r, "sigmoid")
    assert sig.min() > 0.0
    assert sig.max() < 1.0
    amp = squash_reward_for_ppo(r, "amp")
    assert amp.min() >= 0.0
    assert amp.max() <= 1.0
    assert squash_reward_for_ppo(torch.tensor(1.0), "amp").item() == pytest.approx(1.0)
    assert squash_reward_for_ppo(torch.tensor(-1.0), "amp").item() == pytest.approx(0.0)
    assert squash_reward_for_ppo(torch.tensor(3.0), "amp").item() == pytest.approx(0.0)
    centre = torch.tensor(0.5)
    scale = torch.tensor(0.25)
    aff = squash_reward_for_ppo(
        r, "affine_frozen", affine_center=centre, affine_scale=scale
    )
    assert aff.min() >= 0.0
    assert aff.max() <= 1.0
    assert squash_reward_for_ppo(
        centre, "affine_frozen", affine_center=centre, affine_scale=scale
    ).item() == pytest.approx(0.5)
    assert squash_reward_for_ppo(
        centre + 3 * scale, "affine_frozen", affine_center=centre, affine_scale=scale
    ).item() == pytest.approx(1.0)
    with pytest.raises(RuntimeError):
        squash_reward_for_ppo(r, "affine_frozen")


def test_auto_squash_resolution_table() -> None:
    assert resolve_ipmd_reward_ppo_squash("ipmd_diff", "auto") == "none"
    assert resolve_ipmd_reward_ppo_squash("logistic_ratio", "auto") == "sigmoid"
    assert resolve_ipmd_reward_ppo_squash("chi2_lsgan", "auto") == "amp"
    assert resolve_ipmd_reward_ppo_squash("linear_dual", "auto") == "affine_frozen"
    assert resolve_ipmd_reward_ppo_squash("chi2_lsgan", "sigmoid") == "sigmoid"
    with pytest.raises(ValueError, match="reward_objective"):
        resolve_ipmd_reward_ppo_squash("bogus", "auto")
    with pytest.raises(ValueError, match="reward_ppo_squash"):
        resolve_ipmd_reward_ppo_squash("chi2_lsgan", "bogus")


# --------------------------------------------------------------------------
# Config validation.
# --------------------------------------------------------------------------


def test_config_rejects_unknown_objective_and_squash() -> None:
    cfg = _base_cfg()
    cfg.ipmd.reward_objective = "bogus"
    with pytest.raises(ValueError, match="reward_objective"):
        cfg.ipmd.validate()
    cfg = _base_cfg()
    cfg.ipmd.reward_ppo_squash = "bogus"
    with pytest.raises(ValueError, match="reward_ppo_squash"):
        cfg.ipmd.validate()


@pytest.mark.parametrize("objective", ["logistic_ratio", "chi2_lsgan", "linear_dual"])
def test_config_new_objectives_require_no_output_activation(objective: str) -> None:
    cfg = _base_cfg()
    cfg.ipmd.reward_objective = objective
    assert cfg.ipmd.reward_output_activation == "tanh"
    with pytest.raises(ValueError, match="reward_output_activation='none'"):
        cfg.ipmd.validate()
    cfg.ipmd.reward_output_activation = "none"
    cfg.ipmd.validate()
    cfg.ipmd.reward_model_type = "grouped"
    with pytest.raises(ValueError, match="reward_model_type='mlp'"):
        cfg.ipmd.validate()


def test_config_defaults_keep_historical_fields() -> None:
    cfg = _rlopt().IPMDRLOptConfig().ipmd
    assert cfg.reward_objective == "ipmd_diff"
    assert cfg.reward_ppo_squash == "auto"
    assert cfg.estimated_reward_scale == 1.0
    assert cfg.reward_ramp_updates == 0
    assert cfg.reward_mask_done_rows is False
    assert cfg.reward_constant_bonus is False
    assert cfg.reward_whiten_ridge == pytest.approx(1e-3)
    assert cfg.reward_linear_weight_decay == 1.0
    assert cfg.reward_output_activation == "tanh"
    bad = _base_cfg()
    bad.ipmd.reward_linear_weight_decay = 0.0
    with pytest.raises(ValueError, match="reward_linear_weight_decay"):
        bad.ipmd.validate()
    bad = _base_cfg()
    bad.ipmd.reward_label_smoothing = 0.5
    with pytest.raises(ValueError, match="reward_label_smoothing"):
        bad.ipmd.validate()


# --------------------------------------------------------------------------
# linear_dual head.
# --------------------------------------------------------------------------


def _reference_whitener(
    rows: torch.Tensor, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    data = rows.to(torch.float64)
    mean = data.mean(dim=0)
    centred = data - mean
    covariance = centred.T @ centred / float(data.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    transform = (
        eigenvectors * (eigenvalues.clamp_min(0.0) + ridge).rsqrt()
    ) @ eigenvectors.T
    return mean.float(), transform.float()


def test_linear_dual_closed_form_equals_analytic_maximiser() -> None:
    generator = torch.Generator().manual_seed(0)
    dim = 6
    policy = torch.randn(4096, dim, generator=generator)
    shift = torch.linspace(0.2, 1.2, dim)
    expert = torch.randn(4096, dim, generator=generator) * 1.5 + shift
    weight_decay = 2.0
    head = LinearDualRewardHead(
        dim, ridge=1e-3, weight_decay=weight_decay, momentum=0.0, closed_form=True
    )
    out = head.dual_step(policy, expert, alpha=0.1)

    mean, transform = _reference_whitener(torch.cat([policy, expert]), 1e-3)
    phi_exp = (expert - mean) @ transform
    phi_pi = (policy - mean) @ transform
    delta = phi_exp.mean(dim=0) - phi_pi.mean(dim=0)
    w_star = delta / weight_decay
    assert torch.allclose(head.feature_mean, mean, atol=1e-5)
    assert torch.allclose(head.whitening, transform, atol=1e-4)
    assert torch.allclose(head.w, w_star, atol=1e-4)
    assert out["delta_norm"].item() == pytest.approx(delta.norm().item(), rel=1e-4)
    assert out["dual_value"].item() == pytest.approx(
        delta.norm().item() ** 2 / weight_decay, rel=1e-4
    )

    # The dual objective w^T Delta - (lambda/2)||w||^2 is maximised at w*.
    def objective(w: torch.Tensor) -> float:
        return float(w @ delta - 0.5 * weight_decay * w @ w)

    for _ in range(8):
        perturbed = w_star + 0.1 * torch.randn(dim, generator=generator)
        assert objective(perturbed) < objective(w_star)
    # forward folds the whitening into one matvec against raw features.
    probe = torch.randn(16, dim, generator=generator)
    expected = ((probe - mean) @ transform) @ w_star
    assert torch.allclose(head(probe).squeeze(-1), expected, atol=1e-4)
    assert len(list(head.parameters())) == 0


def test_linear_dual_iterative_update_converges_to_closed_form() -> None:
    generator = torch.Generator().manual_seed(1)
    dim = 5
    policy = torch.randn(2048, dim, generator=generator)
    expert = torch.randn(2048, dim, generator=generator) + 0.7
    closed = LinearDualRewardHead(dim, weight_decay=1.0, momentum=0.0, closed_form=True)
    closed.dual_step(policy, expert, alpha=0.3)
    iterative = LinearDualRewardHead(
        dim, weight_decay=1.0, momentum=0.0, closed_form=False
    )
    gaps: list[float] = []
    for _ in range(40):
        iterative.dual_step(policy, expert, alpha=0.3)
        gaps.append(float((iterative.w - closed.w).norm()))
    assert gaps[0] > gaps[-1]
    assert gaps[-1] < 1e-4
    # One step from zero is exactly alpha * Delta (the dual update arithmetic).
    single = LinearDualRewardHead(
        dim, weight_decay=1.0, momentum=0.0, closed_form=False
    )
    out = single.dual_step(policy, expert, alpha=0.3)
    assert torch.allclose(single.w, 0.3 * closed.w, atol=1e-6)
    assert out["w_norm"].item() == pytest.approx(0.3 * closed.w.norm().item(), rel=1e-5)


def test_linear_dual_ema_moments_track_the_stream() -> None:
    generator = torch.Generator().manual_seed(2)
    head = LinearDualRewardHead(3, momentum=0.5, closed_form=True)
    first = torch.randn(64, 3, generator=generator)
    head.update_moments(first)
    assert torch.allclose(head.feature_mean, first.mean(dim=0))
    second = torch.randn(64, 3, generator=generator) + 2.0
    head.update_moments(second)
    assert torch.allclose(
        head.feature_mean, 0.5 * first.mean(dim=0) + 0.5 * second.mean(dim=0)
    )
    assert int(head.step.item()) == 2


# --------------------------------------------------------------------------
# Objectives through the agent.
# --------------------------------------------------------------------------


def _train_objective(
    agent, steps: int, batch: int, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    metrics: dict[str, torch.Tensor] = {}
    for idx in range(steps):
        policy_td, expert_td = _one_d_sides(batch, generator)
        metrics = agent._run_reward_updates(policy_td, expert_td, policy_update_idx=idx)
    return metrics


def test_chi2_lsgan_optimum_matches_density_ratio_on_1d_mixture() -> None:
    agent, _ = _make_agent(
        reward_objective="chi2_lsgan",
        reward_output_activation="none",
        reward_num_cells=(64, 64),
        reward_lr=3e-3,
    )
    assert agent.reward_optim is not None
    metrics = _train_objective(agent, steps=400, batch=512, seed=3)
    # p = N(+1,1), q = N(-1,1): (p - q) / (p + q) = tanh(x).
    x = torch.linspace(-2.0, 2.0, 41)
    probe = torch.cat([x[:, None], torch.zeros(41, 2)], dim=-1)
    with torch.no_grad():
        d = agent._reward_from_td(_obs_td(probe), batch_role="rollout").squeeze(-1)
    assert (d - torch.tanh(x)).abs().max().item() < 0.1
    assert d.min() >= -1.1
    assert d.max() <= 1.1
    assert metrics["reward_updates_applied"].item() == pytest.approx(1.0)
    assert metrics["loss_reward_objective"].item() > 0.0
    assert metrics["reward_separation_auc"].item() > 0.85
    assert 0.0 <= metrics["reward_saturation_frac"].item() <= 1.0
    assert (
        metrics["expert_reward_mean"].item() > metrics["estimated_reward_mean"].item()
    )


def test_chi2_lsgan_r1_penalty_is_half_coefficient_on_expert_side() -> None:
    agent, _ = _make_agent(
        reward_objective="chi2_lsgan",
        reward_output_activation="none",
        reward_num_cells=(16, 16),
        reward_grad_penalty_coeff=10.0,
    )
    generator = torch.Generator().manual_seed(4)
    policy_td, expert_td = _one_d_sides(64, generator)
    metrics = agent._backward_reward_terms(policy_td, expert_td)
    assert metrics["loss_reward_grad_penalty_expert"].item() > 0.0
    assert metrics["loss_reward_grad_penalty"].item() == pytest.approx(
        metrics["loss_reward_grad_penalty_expert"].item()
    )
    # Reference: the same loss built by hand gives the same gradients.
    grads = [p.grad.clone() for p in agent.reward_estimator.parameters()]
    agent.reward_estimator.zero_grad(set_to_none=True)
    x_pi = policy_td["observation"].clone()
    x_exp = expert_td["observation"].clone().requires_grad_(True)
    d_pi = agent.reward_estimator(x_pi)
    d_exp = agent.reward_estimator(x_exp)
    loss = (d_exp - 1.0).pow(2).mean() + (d_pi + 1.0).pow(2).mean()
    loss = loss + 0.5 * 10.0 * reward_grad_penalty_from_input(d_exp, x_exp)
    loss.backward()
    for got, param in zip(grads, agent.reward_estimator.parameters(), strict=True):
        assert torch.equal(got, param.grad)


def test_logistic_ratio_reward_is_finite_and_ordered() -> None:
    agent, _ = _make_agent(
        reward_objective="logistic_ratio",
        reward_output_activation="none",
        reward_num_cells=(32, 32),
        reward_lr=3e-3,
        reward_logit_reg_coeff=1e-3,
    )
    metrics = _train_objective(agent, steps=200, batch=256, seed=5)
    assert metrics["loss_reward_objective"].item() > 0.0
    assert metrics["loss_reward_logit_reg"].item() > 0.0
    assert metrics["reward_separation_auc"].item() > 0.8
    x = torch.linspace(-2.0, 2.0, 21)
    probe = torch.cat([x[:, None], torch.zeros(21, 2)], dim=-1)
    with torch.no_grad():
        logit = agent._reward_from_td(_obs_td(probe), batch_role="rollout").squeeze(-1)
    assert torch.isfinite(logit).all()
    assert logit[-1] > logit[0]
    # Smoothed targets bound the population optimum by log((1-eps)/eps) = 2.94
    # on the training support; the finite-step fit sits near it, not far above.
    bound = math.log(0.95 / 0.05) + 0.5
    assert logit.abs().max().item() < bound
    assert (
        metrics["expert_reward_mean"].item() > metrics["estimated_reward_mean"].item()
    )


def test_logistic_ratio_loss_matches_reference_bce() -> None:
    agent, _ = _make_agent(
        reward_objective="logistic_ratio",
        reward_output_activation="none",
        reward_num_cells=(16,),
    )
    generator = torch.Generator().manual_seed(6)
    policy_td, expert_td = _one_d_sides(32, generator)
    metrics = agent._backward_reward_terms(policy_td, expert_td)
    grads = [p.grad.clone() for p in agent.reward_estimator.parameters()]
    agent.reward_estimator.zero_grad(set_to_none=True)
    logits = torch.cat(
        [
            agent.reward_estimator(expert_td["observation"]).reshape(-1),
            agent.reward_estimator(policy_td["observation"]).reshape(-1),
        ]
    )
    targets = torch.cat([torch.full((32,), 0.95), torch.full((32,), 0.05)])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    assert metrics["loss_reward_objective"].item() == pytest.approx(loss.item())
    for got, param in zip(grads, agent.reward_estimator.parameters(), strict=True):
        assert torch.equal(got, param.grad)


def test_linear_dual_agent_runs_without_optimizer() -> None:
    agent, _ = _make_agent(
        reward_objective="linear_dual",
        reward_output_activation="none",
        reward_moment_momentum=0.0,
    )
    assert isinstance(agent.reward_estimator, LinearDualRewardHead)
    assert agent.reward_optim is None
    assert agent._reward_ppo_squash == "affine_frozen"
    generator = torch.Generator().manual_seed(7)
    policy_td, expert_td = _one_d_sides(256, generator)
    metrics = agent._run_reward_updates(policy_td, expert_td, policy_update_idx=0)
    assert metrics["reward_updates_applied"].item() == pytest.approx(1.0)
    assert metrics["reward_grad_norm"].item() == 0.0
    assert metrics["reward_linear_delta_norm"].item() > 0.0
    assert metrics["reward_linear_w_norm"].item() > 0.0
    assert metrics["reward_linear_dual_value"].item() == pytest.approx(
        metrics["reward_linear_delta_norm"].item() ** 2, rel=1e-4
    )
    assert metrics["reward_separation_auc"].item() > 0.8
    assert (
        metrics["expert_reward_mean"].item() > metrics["estimated_reward_mean"].item()
    )
    with pytest.raises(RuntimeError, match="linear_dual has no backward pass"):
        agent._backward_reward_terms(policy_td, expert_td)
    x = torch.linspace(-2.0, 2.0, 9)
    probe = torch.cat([x[:, None], torch.zeros(9, 2)], dim=-1)
    with torch.no_grad():
        r = agent._reward_from_td(_obs_td(probe), batch_role="rollout").squeeze(-1)
    assert r[-1] > r[0]


# --------------------------------------------------------------------------
# PPO mixing path.
# --------------------------------------------------------------------------


def _patch_reward(agent, values: torch.Tensor) -> None:
    def _fixed(*_args, **_kwargs):
        return values.clone()

    agent._reward_from_td = _fixed


def test_prepare_rollout_rewards_logs_metrics_with_driving_off() -> None:
    agent, env = _make_agent(reward_loss_coeff=1.0)
    assert agent._reward_update_enabled
    assert agent._use_estimated_rewards_for_ppo is False
    env_reward = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    done = torch.tensor([[False], [True], [False], [True]])
    rollout = _rollout(agent, env, env_reward=env_reward, done=done)
    metrics = agent._prepare_rollout_rewards(rollout)
    assert torch.equal(rollout.get(("next", "reward")), env_reward)
    assert ("next", "est_reward") in rollout.keys(True)
    assert rollout.get(("next", "est_reward")).shape == env_reward.shape
    assert "train/est_reward_mean" in metrics
    assert "train/est_reward_raw_mean" in metrics
    assert "train/est_reward_saturation_frac" in metrics
    assert "reward/env_vs_est/pearson_corr" in metrics
    assert metrics["train/est_reward_weight_effective"] == 0.0
    assert int(agent._reward_mix_state["driving_rollouts"].item()) == 0


def test_prepare_rollout_rewards_disabled_estimator_is_untouched() -> None:
    agent, env = _make_agent(reward_loss_coeff=0.0)
    assert not agent._reward_update_enabled

    def _unexpected(*_args, **_kwargs):
        msg = "estimator must not run"
        raise AssertionError(msg)

    agent._reward_from_td = _unexpected
    env_reward = torch.tensor([[1.0], [2.0]])
    rollout = _rollout(
        agent, env, env_reward=env_reward, done=torch.zeros(2, 1, dtype=torch.bool)
    )
    metrics = agent._prepare_rollout_rewards(rollout)
    assert torch.equal(rollout.get(("next", "reward")), env_reward)
    assert ("next", "est_reward") not in rollout.keys(True)
    assert "train/est_reward_mean" not in metrics


def test_done_rows_get_zero_estimate_and_env_reward_only() -> None:
    agent, env = _make_agent(
        reward_loss_coeff=0.0,
        use_estimated_rewards_for_ppo=True,
        reward_ppo_squash="none",
        estimated_reward_scale=0.3,
        est_reward_weight=0.5,
        env_reward_weight=1.0,
        reward_mask_done_rows=True,
    )
    env_reward = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    done = torch.tensor([[False], [True], [False], [True]])
    truncated = torch.tensor([[False], [False], [False], [True]])
    rollout = _rollout(
        agent, env, env_reward=env_reward, done=done, truncated=truncated
    )
    _patch_reward(agent, torch.tensor([[2.0], [2.0], [-1.0], [2.0]]))
    metrics = agent._prepare_rollout_rewards(rollout)
    est = rollout.get(("next", "est_reward"))
    expected_est = torch.tensor([[0.6], [0.0], [-0.3], [0.0]])
    assert torch.allclose(est, expected_est)
    mixed = rollout.get(("next", "reward"))
    assert torch.allclose(mixed[done], env_reward[done])
    assert torch.allclose(mixed, env_reward + 0.5 * expected_est)
    assert metrics["train/est_reward_weight_effective"] == pytest.approx(0.5)
    assert metrics["train/est_reward_mean"] == pytest.approx(expected_est.mean().item())


def test_historical_driving_path_is_unchanged_by_defaults() -> None:
    """Squash none, scale 1, no mask: est = clamp(r) - done penalty, as before."""
    agent, env = _make_agent(
        reward_loss_coeff=0.0,
        use_estimated_rewards_for_ppo=True,
        env_reward_weight=0.25,
        est_reward_weight=0.75,
        estimated_reward_done_penalty=0.5,
    )
    assert agent._reward_ppo_squash == "none"
    rollout = _rollout(
        agent,
        env,
        env_reward=torch.ones(2, 1),
        done=torch.ones(2, 1, dtype=torch.bool),
        truncated=torch.tensor([[False], [True]]),
    )
    _patch_reward(agent, torch.full((2, 1), 2.0))
    metrics = agent._prepare_rollout_rewards(rollout)
    expected_est = torch.tensor([[1.5], [2.0]])
    assert torch.equal(rollout.get(("next", "est_reward")), expected_est)
    assert torch.equal(
        rollout.get(("next", "reward")), 0.25 * torch.ones(2, 1) + 0.75 * expected_est
    )
    assert metrics["train/est_reward_mean"] == pytest.approx(1.75)


def test_constant_bonus_pays_frozen_mean_on_alive_rows() -> None:
    agent, env = _make_agent(
        reward_loss_coeff=0.0,
        use_estimated_rewards_for_ppo=True,
        reward_ppo_squash="sigmoid",
        estimated_reward_scale=0.3,
        est_reward_weight=1.0,
        env_reward_weight=0.0,
        reward_constant_bonus=True,
    )
    done = torch.tensor([[False], [True], [False], [False]])
    first_raw = torch.tensor([[0.0], [2.0], [-2.0], [4.0]])
    rollout = _rollout(agent, env, env_reward=torch.ones(4, 1), done=done)
    _patch_reward(agent, first_raw)
    agent._prepare_rollout_rewards(rollout)
    frozen_mean = torch.sigmoid(first_raw).mean()
    est = rollout.get(("next", "est_reward"))
    assert torch.allclose(est[~done], torch.full((3,), 0.3 * frozen_mean.item()))
    assert torch.all(est[done] == 0.0)
    assert torch.allclose(rollout.get(("next", "reward")), est)
    assert bool(agent._reward_mix_state["constant_bonus_frozen"].item())
    # A later rollout with a different estimate pays the same frozen constant.
    second = _rollout(agent, env, env_reward=torch.ones(4, 1), done=done)
    _patch_reward(agent, torch.tensor([[9.0], [9.0], [9.0], [9.0]]))
    agent._prepare_rollout_rewards(second)
    est2 = second.get(("next", "est_reward"))
    assert torch.allclose(est2[~done], torch.full((3,), 0.3 * frozen_mean.item()))
    assert torch.all(est2[done] == 0.0)


def test_affine_frozen_squash_freezes_on_first_driving_rollout() -> None:
    agent, env = _make_agent(
        reward_loss_coeff=0.0,
        use_estimated_rewards_for_ppo=True,
        reward_ppo_squash="affine_frozen",
        env_reward_weight=0.0,
        est_reward_weight=1.0,
    )
    raw = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
    rollout = _rollout(
        agent,
        env,
        env_reward=torch.zeros(4, 1),
        done=torch.zeros(4, 1, dtype=torch.bool),
    )
    _patch_reward(agent, raw)
    agent._prepare_rollout_rewards(rollout)
    state = agent._reward_mix_state
    assert bool(state["affine_frozen"].item())
    assert state["affine_center"].item() == pytest.approx(0.5)
    s0 = raw.std(unbiased=False).item()
    assert state["affine_scale"].item() == pytest.approx(s0)
    expected = (0.5 + (raw - 0.5) / (6.0 * s0)).clamp(0.0, 1.0)
    assert torch.allclose(rollout.get(("next", "est_reward")), expected)
    # Frozen: a shifted second rollout is squashed with the same m0, s0.
    second = _rollout(
        agent,
        env,
        env_reward=torch.zeros(4, 1),
        done=torch.zeros(4, 1, dtype=torch.bool),
    )
    _patch_reward(agent, raw + 10.0)
    agent._prepare_rollout_rewards(second)
    assert state["affine_center"].item() == pytest.approx(0.5)
    assert torch.all(second.get(("next", "est_reward")) == 1.0)


def test_ramp_arithmetic_counts_driving_rollouts() -> None:
    agent, env = _make_agent(
        reward_loss_coeff=0.0,
        use_estimated_rewards_for_ppo=True,
        reward_ppo_squash="none",
        env_reward_weight=0.0,
        est_reward_weight=1.0,
        reward_ramp_updates=4,
    )
    seen: list[float] = []
    for _ in range(6):
        rollout = _rollout(
            agent,
            env,
            env_reward=torch.zeros(2, 1),
            done=torch.zeros(2, 1, dtype=torch.bool),
        )
        _patch_reward(agent, torch.full((2, 1), 2.0))
        metrics = agent._prepare_rollout_rewards(rollout)
        seen.append(metrics["train/est_reward_weight_effective"])
        assert torch.allclose(
            rollout.get(("next", "reward")), torch.full((2, 1), 2.0 * seen[-1])
        )
    assert seen == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    assert int(agent._reward_mix_state["driving_rollouts"].item()) == 6


# --------------------------------------------------------------------------
# Checkpoint round trip.
# --------------------------------------------------------------------------


def test_checkpoint_round_trip_restores_linear_head_and_mix_state(tmp_path) -> None:
    agent, env = _make_agent(
        reward_objective="linear_dual",
        reward_output_activation="none",
        use_estimated_rewards_for_ppo=True,
        reward_constant_bonus=True,
        reward_ramp_updates=3,
    )
    generator = torch.Generator().manual_seed(8)
    policy_td, expert_td = _one_d_sides(128, generator)
    agent._run_reward_updates(policy_td, expert_td, policy_update_idx=0)
    rollout = _rollout(
        agent,
        env,
        env_reward=torch.zeros(4, 1),
        done=torch.zeros(4, 1, dtype=torch.bool),
    )
    agent._prepare_rollout_rewards(rollout)
    state = agent._reward_mix_state
    assert bool(state["affine_frozen"].item())
    assert bool(state["constant_bonus_frozen"].item())
    assert int(state["driving_rollouts"].item()) == 1

    path = tmp_path / "ckpt.pt"
    agent.save_model(path)
    data = torch.load(path, map_location="cpu")
    assert "reward_optimizer_state_dict" not in data
    assert "reward_mix_state_dict" in data
    assert "w" in data["reward_estimator_state_dict"]

    fresh, _ = _make_agent(
        reward_objective="linear_dual",
        reward_output_activation="none",
        use_estimated_rewards_for_ppo=True,
        reward_constant_bonus=True,
        reward_ramp_updates=3,
    )
    assert torch.all(fresh.reward_estimator.w == 0.0)
    fresh.load_model(str(path))
    for key, value in agent.reward_estimator.state_dict().items():
        assert torch.equal(fresh.reward_estimator.state_dict()[key], value), key
    for key, value in state.items():
        assert torch.equal(fresh._reward_mix_state[key], value), key
    probe = torch.randn(8, 3, generator=generator)
    with torch.no_grad():
        assert torch.equal(
            fresh._reward_from_td(_obs_td(probe), batch_role="rollout"),
            agent._reward_from_td(_obs_td(probe), batch_role="rollout"),
        )


def test_load_model_tolerates_checkpoint_without_mix_state(tmp_path) -> None:
    agent, _ = _make_agent(reward_loss_coeff=1.0)
    path = tmp_path / "old.pt"
    agent.save_model(path)
    data = torch.load(path, map_location="cpu")
    data.pop("reward_mix_state_dict")
    torch.save(data, path)
    fresh, _ = _make_agent(reward_loss_coeff=1.0)
    fresh.load_model(str(path))
    assert int(fresh._reward_mix_state["driving_rollouts"].item()) == 0


# --------------------------------------------------------------------------
# ipmd_diff regression: the parked objective is bitwise the historical path.
# --------------------------------------------------------------------------


def _reference_ipmd_diff_backward(
    agent, batch, expert_batch
) -> dict[str, torch.Tensor]:
    """The pre-revival ``_backward_reward_terms`` arithmetic, verbatim."""
    requires_grad = agent._reward_grad_penalty_coeff > 0.0
    r_pi_input = agent._reward_input_from_td(
        batch, detach=agent._reward_detach_features, requires_grad=requires_grad
    )
    r_exp_input = agent._reward_input_from_td(
        expert_batch, detach=agent._reward_detach_features, requires_grad=requires_grad
    )
    r_pi_logits = agent.reward_estimator(r_pi_input)
    r_exp_logits = agent.reward_estimator(r_exp_input)
    r_pi = agent._reward_out_fn(r_pi_logits)
    r_exp = agent._reward_out_fn(r_exp_logits)
    reward_grad_penalty_batch = reward_grad_penalty_from_input(r_pi, r_pi_input)
    reward_grad_penalty_expert = reward_grad_penalty_from_input(r_exp, r_exp_input)
    diff = r_pi.mean() - r_exp.mean()
    l2 = r_pi.pow(2).mean() + r_exp.pow(2).mean()
    reward_grad_penalty = reward_grad_penalty_batch + reward_grad_penalty_expert
    logit_reg = r_pi_logits.pow(2).mean() + r_exp_logits.pow(2).mean()
    param_weight_decay = agent._reward_parameter_weight_decay()
    (
        agent._reward_loss_coeff * diff
        + agent._reward_l2_coeff * l2.pow(0.5)
        + agent._reward_grad_penalty_coeff * reward_grad_penalty
        + agent._reward_logit_reg_coeff * logit_reg
        + agent._reward_param_weight_decay_coeff * param_weight_decay
    ).backward()
    return {
        "loss_reward_diff": diff.detach(),
        "loss_reward_l2": l2.detach(),
        "loss_reward_grad_penalty": reward_grad_penalty.detach(),
        "loss_reward_grad_penalty_batch": reward_grad_penalty_batch.detach(),
        "loss_reward_grad_penalty_expert": reward_grad_penalty_expert.detach(),
        "loss_reward_logit_reg": logit_reg.detach(),
        "loss_reward_param_weight_decay": param_weight_decay.detach(),
        "estimated_reward_mean": r_pi.mean().detach(),
        "estimated_reward_std": r_pi.std().detach(),
        "expert_reward_mean": r_exp.mean().detach(),
        "expert_reward_std": r_exp.std().detach(),
    }


def test_ipmd_diff_objective_is_bitwise_the_historical_path() -> None:
    agent, _ = _make_agent(
        reward_loss_coeff=1.0,
        reward_l2_coeff=0.05,
        reward_grad_penalty_coeff=0.5,
        reward_logit_reg_coeff=0.1,
        reward_param_weight_decay_coeff=0.01,
        reward_num_cells=(16, 16),
    )
    assert agent._reward_objective == "ipmd_diff"
    assert agent.config.ipmd.reward_output_activation == "tanh"
    generator = torch.Generator().manual_seed(9)
    policy_td, expert_td = _one_d_sides(32, generator)

    agent.reward_estimator.zero_grad(set_to_none=True)
    metrics = agent._backward_reward_terms(policy_td, expert_td)
    grads = [p.grad.clone() for p in agent.reward_estimator.parameters()]
    agent.reward_estimator.zero_grad(set_to_none=True)
    reference = _reference_ipmd_diff_backward(agent, policy_td, expert_td)
    for got, param in zip(grads, agent.reward_estimator.parameters(), strict=True):
        assert torch.equal(got, param.grad)
    for key, value in reference.items():
        assert torch.equal(metrics[key], value), key
    # The shared diagnostics stay at their neutral value on this branch.
    assert metrics["reward_separation_auc"].item() == 0.0
    assert metrics["reward_saturation_frac"].item() == 0.0
    assert metrics["loss_reward_objective"].item() == 0.0


def test_current_rows_auc_excludes_replay_rows() -> None:
    """With the replay on, ``reward_separation_auc_current`` reads only the
    current rollout's policy rows, while ``reward_separation_auc`` also sees
    the replay rows. A replay holding expert-like rows lowers the mixed AUC
    but not the current-rows AUC."""
    agent, _ = _make_agent(
        reward_objective="logistic_ratio",
        reward_output_activation="none",
        reward_policy_replay_size=4096,
        reward_policy_replay_fraction=0.5,
    )
    assert agent._reward_replay is not None
    generator = torch.Generator().manual_seed(0)
    expert_like = torch.cat([torch.randn(4096, 1, generator=generator) + 1.0, torch.zeros(4096, 2)], dim=-1)
    agent._reward_replay.add(expert_like)
    metrics = _train_objective(agent, steps=200, batch=512, seed=1)
    current = metrics["reward_separation_auc_current"].item()
    mixed = metrics["reward_separation_auc"].item()
    assert current > 0.8
    assert current - mixed > 0.1
