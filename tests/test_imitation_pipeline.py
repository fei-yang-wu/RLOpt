"""Test the full imitation learning pipeline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Add scripts to path and import modules directly
scripts_path = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))


# Import script modules directly
def import_script(script_name):
    """Import a script module by file path."""
    script_file = scripts_path / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, script_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import the script modules
collect_expert_data = import_script("01_collect_expert_data")
train_ipmd = import_script("02_train_ipmd")
train_bc = import_script("03_train_bc")


@pytest.mark.slow
def test_expert_data_collection():
    """Test expert data collection (SAC training and saving)."""
    env_name = "Pendulum-v1"
    total_frames = 1000  # Very short for testing
    save_dir = "test_expert_data"

    # Collect expert data
    buffer_path = collect_expert_data.collect_expert_data(
        env_name=env_name,
        total_frames=total_frames,
        save_dir=save_dir,
        device="cpu",
        seed=42,
    )

    # Verify buffer was created
    assert buffer_path.exists(), f"Expert buffer not created: {buffer_path}"

    # Load and check buffer
    expert_data = torch.load(buffer_path)
    assert "observation" in expert_data.keys()
    assert "action" in expert_data.keys()
    assert len(expert_data) > 0

    # Clean up
    import shutil

    shutil.rmtree(save_dir, ignore_errors=True)

    print(f"✅ Expert data collection test passed!")


@pytest.mark.slow
def test_ipmd_with_expert_data():
    """Test IPMD training with expert data."""
    # First collect minimal expert data
    env_name = "Pendulum-v1"
    save_dir = "test_expert_data"

    buffer_path = collect_expert_data.collect_expert_data(
        env_name=env_name,
        total_frames=1000,  # Minimal
        save_dir=save_dir,
        device="cpu",
        seed=42,
    )

    # Train IPMD
    agent = train_ipmd.train_ipmd(
        expert_data_path=str(buffer_path),
        env_name=env_name,
        total_frames=500,  # Very short
        device="cpu",
        seed=43,
        expert_batch_size=32,
    )

    # Verify agent was created
    assert agent is not None
    assert agent.__class__.__name__ == "IPMD"

    # Clean up
    import shutil

    shutil.rmtree(save_dir, ignore_errors=True)

    print(f"✅ IPMD training test passed!")


@pytest.mark.slow
def test_bc_with_expert_data():
    """Test BC training with expert data."""
    # First collect minimal expert data
    env_name = "Pendulum-v1"
    save_dir = "test_expert_data"

    buffer_path = collect_expert_data.collect_expert_data(
        env_name=env_name,
        total_frames=1000,  # Minimal
        save_dir=save_dir,
        device="cpu",
        seed=42,
    )

    # Train BC
    policy, reward = train_bc.train_bc(
        expert_data_path=str(buffer_path),
        env_name=env_name,
        num_epochs=5,  # Very short
        batch_size=32,
        device="cpu",
        seed=44,
        eval_episodes=2,
    )

    # Verify policy was created
    assert policy is not None
    assert reward is not None

    # Clean up
    import shutil

    shutil.rmtree(save_dir, ignore_errors=True)

    print(f"✅ BC training test passed!")


def test_expert_replay_buffer_creation():
    """Test creating an expert replay buffer from TensorDict."""
    import torch
    from tensordict import TensorDict
    from torchrl.data import TensorDictReplayBuffer
    from torchrl.data.replay_buffers.storages import LazyTensorStorage
    from rlopt.imitation import ExpertReplayBuffer

    # Create mock expert data
    batch_size = 100
    obs_dim = 3
    act_dim = 1

    expert_data = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, act_dim),
            ("next", "observation"): torch.randn(batch_size, obs_dim),
            ("next", "reward"): torch.randn(batch_size, 1),
            ("next", "done"): torch.zeros(batch_size, 1, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )

    # Create replay buffer
    storage = LazyTensorStorage(max_size=batch_size)
    buffer = TensorDictReplayBuffer(storage=storage, batch_size=32)
    buffer.extend(expert_data)

    # Wrap in ExpertReplayBuffer
    expert_buffer = ExpertReplayBuffer(buffer)

    # Test sampling
    sample = expert_buffer.sample()
    assert sample is not None
    assert "observation" in sample.keys()
    assert "action" in sample.keys()
    assert len(sample) == 32  # batch_size

    # Test length
    assert len(expert_buffer) == batch_size

    print("✅ ExpertReplayBuffer creation test passed!")


if __name__ == "__main__":
    # Run fast test
    test_expert_replay_buffer_creation()

    # Slow tests require --slow flag
    print("\nTo run slow tests (full pipeline), use:")
    print(
        "  pytest tests/test_imitation_pipeline.py::test_expert_data_collection -v -s"
    )
    print("  pytest tests/test_imitation_pipeline.py::test_ipmd_with_expert_data -v -s")
    print("  pytest tests/test_imitation_pipeline.py::test_bc_with_expert_data -v -s")
