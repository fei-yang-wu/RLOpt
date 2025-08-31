from __future__ import annotations

import os
import tempfile

import torch

from rlopt.agent.sac import SAC


def test_sac_model_saving_and_loading(sac_cfg_factory, make_env):  # type: ignore
    cfg = sac_cfg_factory(env_name="Pendulum-v1", num_envs=1, frames_per_batch=128, total_frames=128)  # type: ignore
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore

    agent = SAC(env=env, config=cfg)
    agent.train()

    policy_state_before = agent.policy.state_dict()
    q_state_before = agent.q_function.state_dict()  # type: ignore[attr-defined]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
        path = tmpfile.name
        agent.save_model(path)

    new_agent = SAC(env=env, config=cfg)
    new_agent.load_model(path)

    policy_state_after = new_agent.policy.state_dict()
    q_state_after = new_agent.q_function.state_dict()  # type: ignore[attr-defined]

    for k in policy_state_before:
        assert torch.allclose(policy_state_before[k], policy_state_after[k])
    for k in q_state_before:
        assert torch.allclose(q_state_before[k], q_state_after[k])

    os.remove(path)
