from __future__ import annotations

import torch
from tensordict import TensorDict

from rlopt.agent.ipmd import IPMD


def test_ipmd_instantiation_and_predict_shapes(ipmd_cfg_factory, make_env):  # type: ignore
    cfg = ipmd_cfg_factory(env_name="Pendulum-v1", num_envs=4, frames_per_batch=256, total_frames=512)  # type: ignore
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore
    agent = IPMD(env=env, config=cfg)

    # Single-step policy forward via ActorValueOperator ensures feature extractor is applied
    obs = env.reset().get("observation").squeeze(0)
    td_in = TensorDict({"observation": obs.unsqueeze(0)}, batch_size=[1])
    policy_op = agent.actor_critic.get_policy_operator()
    action = policy_op(td_in).get("action").squeeze(0)

    assert isinstance(action, torch.Tensor)
    assert action.shape[-1] == int(env.action_spec_unbatched.shape[-1])

    # Action bounds
    low = env.action_spec_unbatched.space.low
    high = env.action_spec_unbatched.space.high
    assert torch.all(action >= torch.as_tensor(low, device=action.device) - 1e-6)
    assert torch.all(action <= torch.as_tensor(high, device=action.device) + 1e-6)


def test_ipmd_reward_update_smoke(ipmd_cfg_factory, make_env):  # type: ignore
    # Configure small run and allow updates immediately
    cfg = ipmd_cfg_factory(
        env_name="Pendulum-v1",
        num_envs=4,
        frames_per_batch=256,
        total_frames=256,
        utd_ratio=0.5,
        init_random_frames=0,
    )
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore
    agent = IPMD(env=env, config=cfg)

    # Collect one batch from the SyncDataCollector
    collector_iter = iter(agent.collector)
    data = next(collector_iter)

    # Fill replay buffer once
    agent.data_buffer.extend(data.reshape(-1))  # type: ignore[arg-type]

    # Build an expert batch from collected data (just a smoke/integration test)
    expert = data.reshape(-1).select(
        "observation", "action", ("next", "observation")
    )

    # Create an infinite expert iterator yielding the same batch
    def expert_generator():
        while True:
            yield expert.to(agent.device)

    agent.set_expert_iterator(expert_generator())

    # Sample from replay buffer and run one update
    sampled = agent.data_buffer.sample()
    out = agent.update(sampled)

    # Ensure reward-specific metrics are present
    assert "loss_reward_diff" in out.keys(True, True)
    assert "loss_reward_l2" in out.keys(True, True)

