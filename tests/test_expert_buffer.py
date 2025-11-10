"""Tests for the ExpertReplayBuffer wrapper."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from rlopt.imitation import ExpertReplayBuffer


def create_sample_expert_data(
    num_transitions: int = 100, obs_dim: int = 10, act_dim: int = 4
) -> TensorDict:
    """Create sample expert demonstration data."""
    data = TensorDict(
        {
            "observation": torch.randn(num_transitions, obs_dim),
            "action": torch.randn(num_transitions, act_dim),
            ("next", "observation"): torch.randn(num_transitions, obs_dim),
            "reward": torch.randn(num_transitions),
            "done": torch.zeros(num_transitions, dtype=torch.bool),
        },
        batch_size=[num_transitions],
    )
    return data


def test_expert_buffer_with_tensordict_replay_buffer():
    """Test ExpertReplayBuffer with TorchRL TensorDictReplayBuffer."""
    # Create sample data
    expert_data = create_sample_expert_data(num_transitions=100)

    # Create TensorDictReplayBuffer
    storage = LazyTensorStorage(max_size=100)
    buffer = TensorDictReplayBuffer(
        storage=storage,
        batch_size=32,
        sampler=SamplerWithoutReplacement(),
    )
    buffer.extend(expert_data)

    # Wrap in ExpertReplayBuffer
    expert_buffer = ExpertReplayBuffer(buffer)

    # Test sampling
    sample = expert_buffer.sample()
    assert isinstance(sample, TensorDict)
    assert sample.batch_size[0] == 32
    assert "observation" in sample.keys()
    assert "action" in sample.keys()
    assert ("next", "observation") in sample.keys(True)


def test_expert_buffer_with_mock_iltools_manager():
    """Test ExpertReplayBuffer with a mock ImitationLearningTools-style manager."""

    class MockExpertReplayManager:
        """Mock ExpertReplayManager similar to ImitationLearningTools."""

        def __init__(self):
            expert_data = create_sample_expert_data(num_transitions=50)
            storage = LazyTensorStorage(max_size=50)
            self.buffer = TensorDictReplayBuffer(
                storage=storage,
                batch_size=16,
                sampler=SamplerWithoutReplacement(),
            )
            self.buffer.extend(expert_data)

    # Create mock manager
    manager = MockExpertReplayManager()

    # Wrap in ExpertReplayBuffer
    expert_buffer = ExpertReplayBuffer(manager)

    # Test sampling
    sample = expert_buffer.sample()
    assert isinstance(sample, TensorDict)
    assert sample.batch_size[0] == 16
    assert "observation" in sample.keys()
    assert "action" in sample.keys()

    # Test buffer property access
    assert expert_buffer.buffer is not None
    assert isinstance(expert_buffer.buffer, TensorDictReplayBuffer)


def test_expert_buffer_with_generic_sampler():
    """Test ExpertReplayBuffer with a generic object that has sample() method."""

    class GenericSampler:
        """Generic sampler with sample() method."""

        def __init__(self):
            self.data = create_sample_expert_data(num_transitions=20)
            self.batch_size = 8

        def sample(self) -> TensorDict:
            indices = torch.randint(0, self.data.batch_size[0], (self.batch_size,))
            return self.data[indices]

    # Create generic sampler
    sampler = GenericSampler()

    # Wrap in ExpertReplayBuffer
    expert_buffer = ExpertReplayBuffer(sampler)

    # Test sampling
    sample = expert_buffer.sample()
    assert isinstance(sample, TensorDict)
    assert sample.batch_size[0] == 8


def test_expert_buffer_invalid_source():
    """Test ExpertReplayBuffer raises TypeError for invalid source."""
    # Invalid source without sample() method
    invalid_source = {"data": "not a valid source"}

    with pytest.raises(TypeError, match="Expert source must be either"):
        ExpertReplayBuffer(invalid_source)


def test_expert_buffer_repr():
    """Test ExpertReplayBuffer string representation."""
    expert_data = create_sample_expert_data(num_transitions=10)
    storage = LazyTensorStorage(max_size=10)
    buffer = TensorDictReplayBuffer(
        storage=storage,
        batch_size=5,
    )
    buffer.extend(expert_data)

    expert_buffer = ExpertReplayBuffer(buffer)
    repr_str = repr(expert_buffer)
    assert "ExpertReplayBuffer" in repr_str
    assert "TensorDictReplayBuffer" in repr_str
    assert "has_buffer=True" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
