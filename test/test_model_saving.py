import tempfile
import os
import torch
import multiprocessing as mp
import hydra
from omegaconf import DictConfig
import pytest
from rlopt.agent.ppo import PPO
from rlopt.envs.gymlike import make_gym_env
import torchrl.envs.libs.gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def test_model_saving_and_loading(monkeypatch):
    # Force fork for pytest to ensure monkeypatch is inherited by workers
    if mp.get_start_method(allow_none=True) != "fork":
        mp.set_start_method("fork", force=True)

    # This monkeypatch prevents torchrl from trying to import isaaclab,
    # which is not needed for this test and causes an error if isaaclab
    # is not fully installed with its omniverse dependencies.
    monkeypatch.setattr(
        torchrl.envs.libs.gym.GymEnv,
        "_is_batched",
        property(lambda self: isinstance(self._env, VecEnv)),
    )

    with hydra.initialize(config_path=".", version_base=None):
        cfg = hydra.compose(
            config_name="test_config",
            overrides=[
                "collector.frames_per_batch=2040",
                "collector.total_frames=2040",
                "env.env_name=HalfCheetah-v5",
            ],
        )

        env = make_gym_env(
            cfg.env.env_name,
            parallel=True,
            num_workers=cfg.env.num_envs,
            device="cpu",
            from_pixels=False,
        )

        agent = PPO(
            env=env,
            config=cfg,
        )

        # Train for a short period
        agent.train()

        # Get the state dicts before saving
        policy_state_dict_before = agent.policy.state_dict()
        value_state_dict_before = agent.value_function.state_dict()

        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
            saved_model_path = tmpfile.name
            agent.save_model(saved_model_path)

        # Create a new agent and load the saved model
        new_agent = PPO(
            env=env,
            config=cfg,
        )
        new_agent.load_model(saved_model_path)

        # Compare the weights of the two agents
        policy_state_dict_after = new_agent.policy.state_dict()
        value_state_dict_after = new_agent.value_function.state_dict()

        for key in policy_state_dict_before:
            assert torch.allclose(
                policy_state_dict_before[key], policy_state_dict_after[key]
            )

        for key in value_state_dict_before:
            assert torch.allclose(
                value_state_dict_before[key], value_state_dict_after[key]
            )

        # Clean up the temporary file
        os.remove(saved_model_path)
