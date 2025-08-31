from __future__ import annotations

import os
import tempfile

import torch

from rlopt.agent.ppo import PPO


def test_model_saving_and_loading(ppo_cfg_factory, make_env):  # type: ignore
    cfg = ppo_cfg_factory(env_name="Pendulum-v1", num_envs=1, frames_per_batch=128, total_frames=128)  # type: ignore
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore

    agent = PPO(env=env, config=cfg)

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
    new_agent = PPO(env=env, config=cfg)
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
