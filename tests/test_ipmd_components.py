"""Tests for IPMD algorithm components."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer

from rlopt.agent import IPMD, IPMDRLOptConfig
from rlopt.config_base import NetworkConfig
from rlopt.env_utils import make_parallel_env


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
    rewards = agent._reward_from_batch(td)
    assert rewards.shape == (batch_size,)
    assert not torch.isnan(rewards).any()
    assert not torch.isinf(rewards).any()


def test_ipmd_expert_buffer_integration():
    """Test IPMD integration with ExpertReplayBuffer."""
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

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)

    # Create synthetic expert data
    expert_data = create_synthetic_expert_data(env, num_transitions=50)

    # Create expert buffer using agent's helper method
    expert_buffer = agent.create_expert_buffer(expert_data, buffer_size=50)
    assert isinstance(expert_buffer, TensorDictReplayBuffer)

    # Set expert buffer on agent
    agent.set_expert_buffer(expert_buffer)

    # Sample from expert buffer
    expert_batch = agent._next_expert_batch()
    assert expert_batch is not None
    assert isinstance(expert_batch, TensorDict)
    assert expert_batch.batch_size[0] <= cfg.ipmd.expert_batch_size


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

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)

    # Create synthetic expert data
    expert_data = create_synthetic_expert_data(env, num_transitions=50)
    expert_buffer = agent.create_expert_buffer(expert_data, buffer_size=50)
    agent.set_expert_buffer(expert_buffer)

    # Create policy data batch
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    batch_size = 4

    policy_batch = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, act_dim),
            ("next", "observation"): torch.randn(batch_size, obs_dim),
            ("next", "reward"): torch.randn(batch_size, 1),
            ("next", "done"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "truncated"): torch.zeros(batch_size, 1, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )

    # Perform update
    loss_td = agent.update(policy_batch)

    # Check loss outputs
    assert "loss_actor" in loss_td
    assert "loss_qvalue" in loss_td
    assert "loss_alpha" in loss_td
    assert "loss_reward_diff" in loss_td
    assert "loss_reward_l2" in loss_td
    assert "estimated_reward_mean" in loss_td
    assert "expert_reward_mean" in loss_td

    # Check that losses are finite
    for key in loss_td.keys():
        value = loss_td[key]
        assert not torch.isnan(value).any(), f"NaN in {key}"
        assert not torch.isinf(value).any(), f"Inf in {key}"


def test_ipmd_set_expert_source_compatibility():
    """Test IPMD set_expert_source method with different sources."""
    cfg = IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 10
    cfg.collector.total_frames = 100  # Divisible by frames_per_batch

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg, logger=None)

    # Test: Set source with TensorDictReplayBuffer directly
    expert_data = create_synthetic_expert_data(env, num_transitions=20)
    buffer = agent.create_expert_buffer(expert_data, buffer_size=20)
    agent.set_expert_buffer(buffer)
    assert agent._expert_buffer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
