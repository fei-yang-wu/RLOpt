"""Tests for IPMD with loco-mujoco environments."""

from __future__ import annotations

import pytest
import torch
from rlopt.agent.imitation import IPMD, IPMDRLOptConfig
from rlopt.configs import NetworkConfig
from tensordict import TensorDict
from torchrl.envs import GymEnv, set_gym_backend
from torchrl.envs.transforms import DoubleToFloat, StepCounter

from rlopt.imitation import ExpertReplayBuffer

# Check if loco-mujoco is available
try:
    import gymnasium as gym
    import loco_mujoco  # noqa: F401

    LOCO_MUJOCO_AVAILABLE = True
except ImportError:
    LOCO_MUJOCO_AVAILABLE = False


@pytest.mark.skipif(not LOCO_MUJOCO_AVAILABLE, reason="loco-mujoco not installed")
def test_create_g1_environment():
    """Test creating a UnitreeG1 environment from loco-mujoco."""
    with set_gym_backend("gymnasium"):
        # Create the G1 environment
        gym_env = gym.make("LocoMujoco", env_name="UnitreeG1")

        # Wrap in TorchRL
        env = GymEnv(gym_env, device="cpu")
        env = env.append_transform(DoubleToFloat(in_keys=["observation"]))
        env = env.append_transform(StepCounter(max_steps=1000))

        # Test reset
        td = env.reset()
        assert "observation" in td.keys()

        # Test step
        action = env.action_spec.rand()
        td_next = env.step(td.set("action", action))
        assert ("next", "observation") in td_next.keys(True)

        env.close()


@pytest.mark.skipif(not LOCO_MUJOCO_AVAILABLE, reason="loco-mujoco not installed")
def test_collect_g1_expert_data():
    """Test collecting expert demonstration data from G1 environment."""
    with set_gym_backend("gymnasium"):
        # Create the G1 environment
        gym_env = gym.make("LocoMujoco", env_name="UnitreeG1")
        env = GymEnv(gym_env, device="cpu")
        env = env.append_transform(DoubleToFloat(in_keys=["observation"]))
        env = env.append_transform(StepCounter(max_steps=1000))

        # Collect some random trajectories as "expert" demonstrations
        num_transitions = 50
        transitions = []

        td = env.reset()
        for _ in range(num_transitions):
            action = env.action_spec.rand()
            td_next = env.step(td.set("action", action))

            # Store transition
            transition = TensorDict(
                {
                    "observation": td["observation"],
                    "action": action,
                    ("next", "observation"): td_next["next", "observation"],
                    "reward": td_next["next", "reward"],
                    "done": td_next["next", "done"],
                    "terminated": td_next["next", "terminated"],
                    "truncated": td_next["next", "truncated"],
                },
                batch_size=[],
            )
            transitions.append(transition)

            # Reset if done
            if td_next["next", "done"].item():
                td = env.reset()
            else:
                td = td_next.get("next").exclude(
                    "reward", "done", "terminated", "truncated"
                )

        # Stack transitions
        expert_data = torch.stack(transitions, dim=0)

        assert expert_data.batch_size[0] == num_transitions
        assert "observation" in expert_data.keys()
        assert "action" in expert_data.keys()
        assert ("next", "observation") in expert_data.keys(True)

        env.close()


@pytest.mark.skipif(not LOCO_MUJOCO_AVAILABLE, reason="loco-mujoco not installed")
def test_ipmd_with_g1_smoke():
    """Smoke test for IPMD with G1 environment."""
    # Create configuration
    cfg = IPMDRLOptConfig()
    cfg.env.library = "gymnasium"
    cfg.env.env_name = "LocoMujoco"
    cfg.env.num_envs = 1
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 8
    cfg.collector.total_frames = 16
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 4
    cfg.compile.compile = False
    cfg.ipmd.reward_num_cells = (64, 64)
    cfg.ipmd.expert_batch_size = 4
    cfg.ipmd.utd_ratio = 1.0

    # Create G1 environment through gymnasium
    with set_gym_backend("gymnasium"):
        gym_env = gym.make("LocoMujoco", env_name="UnitreeG1")
        env = GymEnv(gym_env, device="cpu")
        env = env.append_transform(DoubleToFloat(in_keys=["observation"]))
        env = env.append_transform(StepCounter(max_steps=1000))

        # Get observation and action dimensions
        obs_dim = env.observation_spec["observation"].shape[-1]
        act_dim = env.action_spec.shape[-1]

        # Update config with correct dimensions
        cfg.policy.input_dim = obs_dim
        cfg.q_function = NetworkConfig(
            num_cells=[64, 64],
            activation_fn="relu",
            output_dim=1,
            input_keys=["observation"],
            input_dim=obs_dim + act_dim,  # Q-function takes both obs and action
        )

        # Create IPMD agent
        agent = IPMD(env, cfg, logger=None)

        # Create synthetic expert data
        expert_data = TensorDict(
            {
                "observation": torch.randn(50, obs_dim),
                "action": torch.randn(50, act_dim),
                ("next", "observation"): torch.randn(50, obs_dim),
                "reward": torch.randn(50),
                "done": torch.zeros(50, dtype=torch.bool),
                "terminated": torch.zeros(50, dtype=torch.bool),
                "truncated": torch.zeros(50, dtype=torch.bool),
            },
            batch_size=[50],
        )

        # Set expert buffer
        expert_buffer = agent.create_expert_buffer(expert_data, buffer_size=50)
        agent.set_expert_buffer(expert_buffer)

        # Run a short training loop
        agent.train()

        assert agent.__class__.__name__ == "IPMD"

        env.close()


@pytest.mark.skipif(not LOCO_MUJOCO_AVAILABLE, reason="loco-mujoco not installed")
def test_ipmd_with_g1_and_iltools():
    """Test IPMD with G1 environment and ImitationLearningTools replay manager."""
    pytest.importorskip("iltools.datasets")
    from iltools.datasets.replay_manager import (
        EnvAssignment,
        ExpertReplayManager,
        ExpertReplaySpec,
    )

    # Create configuration
    cfg = IPMDRLOptConfig()
    cfg.env.library = "gymnasium"
    cfg.env.env_name = "LocoMujoco"
    cfg.env.num_envs = 2
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False

    # Create G1 environment
    with set_gym_backend("gymnasium"):
        gym_env = gym.make("LocoMujoco", env_name="UnitreeG1")
        env = GymEnv(gym_env, device="cpu")
        env = env.append_transform(DoubleToFloat(in_keys=["observation"]))
        env = env.append_transform(StepCounter(max_steps=1000))

        # Get dimensions
        obs_dim = env.observation_spec["observation"].shape[-1]
        act_dim = env.action_spec.shape[-1]

        # Create synthetic expert trajectories
        traj1 = TensorDict(
            {
                "observation": torch.randn(20, obs_dim),
                "action": torch.randn(20, act_dim),
                ("next", "observation"): torch.randn(20, obs_dim),
                "reward": torch.randn(20),
                "done": torch.zeros(20, dtype=torch.bool),
            },
            batch_size=[20],
        )

        traj2 = TensorDict(
            {
                "observation": torch.randn(25, obs_dim),
                "action": torch.randn(25, act_dim),
                ("next", "observation"): torch.randn(25, obs_dim),
                "reward": torch.randn(25),
                "done": torch.zeros(25, dtype=torch.bool),
            },
            batch_size=[25],
        )

        # Create ExpertReplaySpec with multiple tasks
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = ExpertReplaySpec(
                tasks={
                    0: [traj1],
                    1: [traj2],
                },
                sample_batch_size=8,
                scratch_dir=tmpdir,
                device="cpu",
            )

            # Create ExpertReplayManager
            replay_manager = ExpertReplayManager(spec)

            # Set assignment for sequential sampling
            assignment = [
                EnvAssignment(task_id=0, traj_id=0, step=0),
                EnvAssignment(task_id=1, traj_id=0, step=0),
            ]
            replay_manager.set_assignment(assignment)

            # Wrap in RLOpt's ExpertReplayBuffer
            expert_buffer = ExpertReplayBuffer(replay_manager)

            # Test sampling
            sample = expert_buffer.sample()
            assert isinstance(sample, TensorDict)
            assert sample.batch_size[0] == len(assignment)
            assert "observation" in sample.keys()
            assert "action" in sample.keys()

            # Create IPMD agent and set expert source
            cfg.policy.input_dim = obs_dim
            cfg.q_function = NetworkConfig(
                num_cells=[64, 64],
                activation_fn="relu",
                output_dim=1,
                input_keys=["observation"],
                input_dim=obs_dim + act_dim,
            )

            agent = IPMD(env, cfg, logger=None)
            agent.set_expert_source(replay_manager)

            # Verify expert data can be sampled
            expert_batch = agent._next_expert_batch()
            assert expert_batch is not None
            assert isinstance(expert_batch, TensorDict)

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
