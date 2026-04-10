"""Tests for IPMD algorithm components."""

from __future__ import annotations

import warnings
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


@lru_cache(maxsize=1)
def _rlopt() -> SimpleNamespace:
    warnings.filterwarnings(
        "ignore",
        message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
        category=DeprecationWarning,
        append=False,
    )
    from rlopt.agent import IPMD, IPMDRLOptConfig
    from rlopt.config_base import NetworkConfig
    from rlopt.env_utils import make_parallel_env

    return SimpleNamespace(
        IPMD=IPMD,
        IPMDRLOptConfig=IPMDRLOptConfig,
        NetworkConfig=NetworkConfig,
        make_parallel_env=make_parallel_env,
    )


def _apply_obs_input_keys(cfg) -> None:
    """Force observation-based keys for smoke tests across torchrl versions."""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.latent_key = "observation"
    cfg.ipmd.latent_dim = 3
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.latent_input_type = "s'"
    cfg.ipmd.diversity_bonus_coeff = 0.0


def _make_test_expert_sampler(expert_data: TensorDict):
    """Return a private expert sampler override matching the new agent contract."""

    def _sample(batch_size: int, required_keys):
        batch = expert_data
        if batch.numel() > batch_size:
            batch = batch[:batch_size]
        return batch.select(*required_keys).clone()

    return _sample


def create_synthetic_expert_data(env, num_transitions: int = 100) -> TensorDict:
    """Create synthetic expert demonstration data by random sampling."""
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]

    return TensorDict(
        {
            "observation": torch.randn(num_transitions, obs_dim),
            "action": torch.randn(num_transitions, act_dim),
            ("next", "observation"): torch.randn(num_transitions, obs_dim),
            "reward": torch.randn(num_transitions),
            "done": torch.zeros(num_transitions, dtype=torch.bool),
            "terminated": torch.zeros(num_transitions, dtype=torch.bool),
            "truncated": torch.zeros(num_transitions, dtype=torch.bool),
        },
        batch_size=[num_transitions],
    )


def _make_env_reward_only_ipmd_agent():
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
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.reward_loss_coeff = 0.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.mi_loss_coeff = 0.0
    cfg.ipmd.mi_reward_weight = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0

    env = rlopt.make_parallel_env(cfg)
    return rlopt.IPMD(env, cfg, logger=None), env


def test_ipmd_prepare_rollout_rewards_ignores_estimator_when_disabled():
    """Disabled estimated PPO rewards should leave env rewards untouched."""
    agent, env = _make_env_reward_only_ipmd_agent()
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    env_reward = torch.tensor([[1.0], [2.0]], device=agent.device)
    rollout = TensorDict(
        {
            "observation": torch.randn(2, obs_dim, device=agent.device),
            "action": torch.randn(2, act_dim, device=agent.device),
            ("next", "observation"): torch.randn(2, obs_dim, device=agent.device),
            ("next", "reward"): env_reward.clone(),
            ("next", "done"): torch.zeros(2, 1, dtype=torch.bool, device=agent.device),
            ("next", "truncated"): torch.zeros(
                2, 1, dtype=torch.bool, device=agent.device
            ),
        },
        batch_size=[2],
        device=agent.device,
    )

    def _unexpected_reward_forward(*_args, **_kwargs):
        msg = "reward estimator should not run"
        raise AssertionError(msg)

    agent._reward_from_batch = _unexpected_reward_forward

    metrics = agent._prepare_rollout_rewards(rollout)

    assert torch.equal(rollout.get(("next", "reward")), env_reward)
    assert ("next", "est_reward") not in rollout.keys(True)
    assert "train/est_reward_mean" not in metrics


def test_ipmd_disabled_reward_update_does_not_sample_expert_batch():
    """Pure-PPO IPMD updates should not require expert data."""
    agent, _ = _make_env_reward_only_ipmd_agent()
    agent._expert_batch_sampler = None
    batch = TensorDict({}, batch_size=[2], device=agent.device)

    expert_batch, has_expert = agent._expert_batch_for_update(batch)

    assert expert_batch.numel() == 2
    assert len(expert_batch.keys(True)) == 0
    assert has_expert.item() == 0.0


def test_ipmd_prepare_rollout_rewards_honors_estimated_reward_gate():
    """Estimated PPO rewards should be mixed only when the gate is enabled."""
    agent, env = _make_env_reward_only_ipmd_agent()
    agent.config.ipmd.use_estimated_rewards_for_ppo = True
    agent._use_estimated_rewards_for_ppo = True
    agent.config.ipmd.env_reward_weight = 0.25
    agent.config.ipmd.est_reward_weight = 0.75
    agent.config.ipmd.estimated_reward_done_penalty = 0.5

    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    rollout = TensorDict(
        {
            "observation": torch.randn(2, obs_dim, device=agent.device),
            "action": torch.randn(2, act_dim, device=agent.device),
            ("next", "observation"): torch.randn(2, obs_dim, device=agent.device),
            ("next", "reward"): torch.ones(2, 1, device=agent.device),
            ("next", "done"): torch.ones(2, 1, dtype=torch.bool, device=agent.device),
            ("next", "truncated"): torch.tensor([[False], [True]], device=agent.device),
        },
        batch_size=[2],
        device=agent.device,
    )

    def _fixed_reward_forward(*_args, **_kwargs):
        return torch.full((2, 1), 2.0, device=agent.device)

    agent._reward_from_batch = _fixed_reward_forward

    metrics = agent._prepare_rollout_rewards(rollout)

    expected_est_reward = torch.tensor([[1.5], [2.0]], device=agent.device)
    expected_reward = (
        0.25 * torch.ones_like(expected_est_reward) + 0.75 * expected_est_reward
    )
    assert torch.allclose(rollout.get(("next", "est_reward")), expected_est_reward)
    assert torch.allclose(rollout.get(("next", "reward")), expected_reward)
    assert metrics["train/est_reward_mean"] == pytest.approx(1.75)


def test_ipmd_initialization():
    """Test IPMD agent initialization."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.ipmd.reward_num_cells = (64, 64)
    _apply_obs_input_keys(cfg)
    cfg.q_function = rlopt.NetworkConfig(
        num_cells=[64, 64],
        activation_fn="relu",
        output_dim=1,
        input_keys=["observation"],
    )

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    assert agent is not None
    assert agent.__class__.__name__ == "IPMD"
    assert hasattr(agent, "reward_estimator")
    assert hasattr(agent, "actor_critic")
    assert hasattr(agent, "loss_module")


def test_ipmd_reward_estimator():
    """Test IPMD reward estimator network."""
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
    cfg.ipmd.reward_num_cells = (32, 32)
    _apply_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    # Create a sample batch
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    batch_size = 8

    td = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, act_dim),
            ("next", "observation"): torch.randn(batch_size, obs_dim),
        },
        batch_size=[batch_size],
    )

    # Compute estimated rewards
    rewards = agent._reward_from_batch(td, batch_role="rollout")
    assert rewards.shape[0] == batch_size
    assert not torch.isnan(rewards).any()
    assert not torch.isinf(rewards).any()


def test_ipmd_private_expert_sampler_integration():
    """Test IPMD integration with the private expert sampler override."""
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
    cfg.ipmd.expert_batch_size = 4
    _apply_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    # Create synthetic expert data
    expert_data = create_synthetic_expert_data(env, num_transitions=50)
    agent._set_test_expert_batch_sampler(_make_test_expert_sampler(expert_data))

    # Sample from expert buffer
    expert_batch = agent._next_expert_batch(batch_size=cfg.ipmd.expert_batch_size)
    assert isinstance(expert_batch, TensorDict)
    assert expert_batch.numel() > 0


def test_ipmd_update_with_expert_data():
    """Test IPMD update step with expert data."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 4
    cfg.compile.compile = False
    cfg.ipmd.reward_num_cells = (32, 32)
    cfg.ipmd.expert_batch_size = 4
    _apply_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    # Create synthetic expert data
    expert_data = create_synthetic_expert_data(env, num_transitions=50)
    agent._set_test_expert_batch_sampler(_make_test_expert_sampler(expert_data))

    # Create policy data batch
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    batch_size = 4

    policy_batch = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, act_dim),
            "action_log_prob": torch.zeros(batch_size),
            ("next", "observation"): torch.randn(batch_size, obs_dim),
            ("next", "reward"): torch.randn(batch_size, 1),
            ("next", "done"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "truncated"): torch.zeros(batch_size, 1, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )
    with torch.no_grad():
        policy_batch = agent.adv_module(policy_batch)

    # Fixed-signature update (torch.compile / CUDA graph)
    num_network_updates = torch.zeros((), dtype=torch.int64, device=agent.device)
    expert_batch = agent._next_expert_batch(batch_size=cfg.ipmd.expert_batch_size)
    has_expert = torch.tensor(1.0, device=agent.device, dtype=torch.float32)
    loss_td, _ = agent.update(
        policy_batch, num_network_updates, expert_batch, has_expert
    )

    # Check loss outputs (PPO-based IPMD)
    assert "loss_critic" in loss_td
    assert "loss_objective" in loss_td
    assert "loss_entropy" in loss_td
    assert "loss_reward_diff" in loss_td
    assert "loss_reward_l2" in loss_td
    # Check that losses are finite
    for key in tuple(loss_td.keys()):
        value = loss_td[key]
        assert not torch.isnan(value).any(), f"NaN in {key}"
        assert not torch.isinf(value).any(), f"Inf in {key}"


def test_ipmd_missing_sampler_raises() -> None:
    """IPMD expert sampling should fail fast when no sampler is available."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 10
    cfg.collector.total_frames = 100  # Divisible by frames_per_batch
    _apply_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    agent._expert_batch_sampler = None

    with pytest.raises(RuntimeError, match="sample_expert_batch"):
        agent._next_expert_batch(batch_size=4)


def test_ipmd_latent_input_requires_exact_keys() -> None:
    """Latent encoder input selection must request the configured transition keys exactly."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    _apply_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    obs_dim = env.observation_spec["observation"].shape[-1]

    current_only = TensorDict(
        {"observation": torch.randn(4, obs_dim)},
        batch_size=[4],
    )
    with pytest.raises(KeyError, match="expert latent batch"):
        agent._latent_encoder_features_from_td(
            current_only,
            detach=False,
            context="expert latent batch",
        )


def test_ipmd_rollout_has_mi_targets() -> None:
    """Prepared latent IPMD rollouts should attach MI value targets."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    agent._set_test_expert_batch_sampler(
        _make_test_expert_sampler(create_synthetic_expert_data(env, num_transitions=20))
    )

    rollout = next(iter(agent.collector))
    agent._prepare_latent_rollout_batch_for_training(rollout)
    agent._prepare_rollout_rewards(rollout)

    prepared = agent.pre_iteration_compute(rollout)
    assert "mi_value" in prepared.keys(True)
    assert "mi_returns" in prepared.keys(True)
    assert "mi_advantage" in prepared.keys(True)


def test_ipmd_rollout_posterior_collector_latents() -> None:
    """Rollout-posterior collection should encode the current rollout batch."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)
    cfg.ipmd.latent_input_type = "s"
    cfg.ipmd.command_source = "rollout_posterior"
    cfg.ipmd.latent_learning.posterior_input_keys = ["observation"]

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    class _NormalizeIdentity(torch.nn.Module):
        def forward(self, obs_features: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(obs_features, dim=-1, eps=1.0e-6)

    agent.mi_encoder = _NormalizeIdentity().to(agent.device)
    rollout_obs = torch.tensor(
        [[0.2, 0.3, 0.4], [0.9, -0.1, 0.1]],
        device=agent.device,
        dtype=torch.float32,
    )
    rollout_td = TensorDict(
        {"observation": rollout_obs.clone()},
        batch_size=[rollout_obs.shape[0]],
        device=agent.device,
    )

    latents = agent._sample_collector_latents(
        batch_size=2,
        device=agent.device,
        dtype=torch.float32,
        td=rollout_td,
    )
    expected = torch.nn.functional.normalize(rollout_obs, dim=-1, eps=1.0e-6)
    assert torch.allclose(latents, expected)


def test_ipmd_patch_autoencoder_uses_only_posterior_expert_keys() -> None:
    """Patch autoencoder latent updates should be reconstruction-only expert batches."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)
    cfg.ipmd.reward_loss_coeff = 0.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.mi_loss_coeff = 0.0
    cfg.ipmd.mi_reward_weight = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0
    cfg.ipmd.latent_learning.method = "patch_autoencoder"
    cfg.ipmd.latent_learning.posterior_input_keys = ["observation"]
    cfg.ipmd.latent_learning.recon_coeff = 1.0
    cfg.ipmd.latent_learning.uniformity_coeff = 0.0
    cfg.ipmd.latent_learning.weight_decay_coeff = 0.0

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    obs_dim = env.observation_spec["observation"].shape[-1]
    expert_data = TensorDict(
        {"observation": torch.randn(4, obs_dim, device=agent.device)},
        batch_size=[4],
        device=agent.device,
    )
    requested_keys = []

    def _sample(batch_size: int, required_keys):
        requested_keys.append(list(required_keys))
        batch = expert_data[:batch_size]
        return batch.select(*required_keys).clone()

    agent._set_test_expert_batch_sampler(_sample)
    assert agent._latent_learner is not None

    metrics = agent._latent_learner.update(
        TensorDict({}, batch_size=[0], device=agent.device)
    )

    assert requested_keys == [["observation"]]
    assert "ipmd/latent_recon_loss" in metrics
    assert metrics["ipmd/latent_recon_loss"] >= 0.0


def test_ipmd_rejects_env_sourced_expert_posterior_command_source() -> None:
    """Collector latents must not bypass rollout observations via live env getters."""
    rlopt = _rlopt()

    with pytest.raises(ValueError, match="Unsupported IPMD command_source"):
        rlopt.IPMD._normalize_command_source("expert_posterior")


def test_ipmd_rejects_non_expert_group_posterior_keys() -> None:
    """Grouped posterior keys must come from expert_state or expert_window."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)
    cfg.ipmd.latent_learning.posterior_input_keys = [("policy", "observation")]

    env = rlopt.make_parallel_env(cfg)
    spec = env.observation_spec.clone()
    spec.set(("policy", "observation"), spec["observation"].clone())
    env.observation_spec = spec

    with pytest.raises(ValueError, match="expert_state or expert_window"):
        rlopt.IPMD(env, cfg, logger=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
