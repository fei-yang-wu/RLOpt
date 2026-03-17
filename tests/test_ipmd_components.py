"""Tests for IPMD algorithm components."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rlopt.agent import IPMD, IPMDRLOptConfig
from rlopt.config_base import NetworkConfig
from rlopt.env_utils import make_parallel_env


def _apply_obs_input_keys(cfg: IPMDRLOptConfig) -> None:
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


def test_ipmd_initialization():
    """Test IPMD agent initialization."""
    cfg = IPMDRLOptConfig()
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
    cfg.q_function = NetworkConfig(
        num_cells=[64, 64],
        activation_fn="relu",
        output_dim=1,
        input_keys=["observation"],
    )

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)

    assert agent is not None
    assert agent.__class__.__name__ == "IPMD"
    assert hasattr(agent, "reward_estimator")
    assert hasattr(agent, "actor_critic")
    assert hasattr(agent, "loss_module")


def test_ipmd_reward_estimator():
    """Test IPMD reward estimator network."""
    cfg = IPMDRLOptConfig()
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

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)

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
    cfg = IPMDRLOptConfig()
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

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)

    # Create synthetic expert data
    expert_data = create_synthetic_expert_data(env, num_transitions=50)
    agent._set_test_expert_batch_sampler(_make_test_expert_sampler(expert_data))

    # Sample from expert buffer
    expert_batch = agent._next_expert_batch(batch_size=cfg.ipmd.expert_batch_size)
    assert isinstance(expert_batch, TensorDict)
    assert expert_batch.numel() > 0


def test_ipmd_update_with_expert_data():
    """Test IPMD update step with expert data."""
    cfg = IPMDRLOptConfig()
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

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)

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
    assert "estimated_reward_mean" in loss_td
    assert "expert_reward_mean" in loss_td

    # Check that losses are finite
    for key in loss_td.keys():
        value = loss_td[key]
        assert not torch.isnan(value).any(), f"NaN in {key}"
        assert not torch.isinf(value).any(), f"Inf in {key}"


def test_ipmd_missing_sampler_raises() -> None:
    """IPMD training should fail fast when no expert sampler is available."""
    cfg = IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 10
    cfg.collector.total_frames = 100  # Divisible by frames_per_batch
    _apply_obs_input_keys(cfg)

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)
    agent._expert_batch_sampler = None

    with pytest.raises(RuntimeError, match="sample_expert_batch"):
        agent.validate_training()


def test_ipmd_latent_input_requires_exact_keys() -> None:
    """Latent encoder input selection must request the configured transition keys exactly."""
    cfg = IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    _apply_obs_input_keys(cfg)

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)
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


def test_ipmd_rollout_has_no_mi_targets() -> None:
    """Prepared IPMD rollouts should not attach MI value targets anymore."""
    cfg = IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)
    agent._set_test_expert_batch_sampler(
        _make_test_expert_sampler(create_synthetic_expert_data(env, num_transitions=20))
    )

    rollout = next(iter(agent.collector))
    agent._prepare_latent_rollout_batch_for_training(rollout)
    agent._prepare_rollout_rewards(rollout)

    prepared = agent.pre_iteration_compute(rollout)
    assert "mi_value" not in prepared.keys(True)
    assert "mi_returns" not in prepared.keys(True)
    assert "mi_advantage" not in prepared.keys(True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
