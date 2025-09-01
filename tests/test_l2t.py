from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from rlopt.agent.l2t.l2t import L2T, L2TR
from rlopt.modules import TanhNormalStable


def _action_bounds(env):
    low = env.action_spec_unbatched.space.low
    high = env.action_spec_unbatched.space.high
    return torch.as_tensor(low), torch.as_tensor(high)


def test_l2t_instantiation_and_shapes(l2t_cfg_factory, make_env):  # type: ignore
    cfg = l2t_cfg_factory(env_name="Pendulum-v1", num_envs=4, frames_per_batch=64, total_frames=64)  # type: ignore
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore
    agent = L2T(env=env, config=cfg)

    # Teacher policy action
    obs = env.reset().get("observation").squeeze(0)
    td_in = TensorDict({"observation": obs.unsqueeze(0)}, batch_size=[1])
    teacher_pol = agent.actor_critic.get_policy_operator()
    action = teacher_pol(td_in.clone()).get("action").squeeze(0)
    assert isinstance(action, torch.Tensor)
    assert action.shape[-1] == int(env.action_spec_unbatched.shape[-1])

    low, high = _action_bounds(env)
    assert torch.all(action >= low.to(action) - 1e-6)
    assert torch.all(action <= high.to(action) + 1e-6)

    # Student policy params via student actor-critic
    td_out = agent.student_actor_critic(td_in.clone())
    assert cfg.l2t.student_hidden_key in td_out.keys()
    assert cfg.l2t.student_loc_key in td_out.keys()
    assert cfg.l2t.student_scale_key in td_out.keys()

    student_loc = td_out.get(cfg.l2t.student_loc_key)
    student_scale = td_out.get(cfg.l2t.student_scale_key)
    assert student_loc.shape[-1] == int(env.action_spec_unbatched.shape[-1])
    assert student_scale.shape[-1] == int(env.action_spec_unbatched.shape[-1])

    # Behavior cloning distribution sampling check
    dist = TanhNormalStable(student_loc, student_scale, event_dims=1)
    sample = dist.rsample()
    assert torch.all(sample >= low.to(sample) - 1e-6)
    assert torch.all(sample <= high.to(sample) + 1e-6)

    # Config: mixture coefficient should be settable
    assert np.isclose(cfg.l2t.mixture_coeff, 0.2)


def test_l2t_loss_forward_smoke(l2t_cfg_factory, make_env):  # type: ignore
    cfg = l2t_cfg_factory(env_name="Pendulum-v1", num_envs=4, frames_per_batch=64, total_frames=64, imitation="bc")  # type: ignore
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore
    agent = L2T(env=env, config=cfg)

    # One batch from collector
    data = next(iter(agent.collector))
    # Compute advantages (teacher value)
    with torch.no_grad():
        adv_data = agent.adv_module(data)
    loss_td = agent.loss_module(adv_data)

    # Required keys
    for k in ["loss_objective", "loss_entropy", "loss_critic", "loss_imitation"]:
        assert k in loss_td.keys(True)
        v = loss_td.get(k)
        assert torch.isfinite(v).all()


def test_l2tr_instantiation_and_shapes(l2t_cfg_factory, make_env):  # type: ignore
    cfg = l2t_cfg_factory(
        env_name="Pendulum-v1",
        num_envs=4,
        frames_per_batch=64,
        total_frames=64,
        feature_dim=32,
        student_recurrent=True,
    )  # type: ignore
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore
    # L2TR adds required transforms internally
    agent = L2TR(env=env, config=cfg)

    # Teacher policy action
    obs = env.reset().get("observation").squeeze(0)
    td_in = TensorDict({"observation": obs.unsqueeze(0)}, batch_size=[1])
    teacher_pol = agent.actor_critic.get_policy_operator()
    action = teacher_pol(td_in.clone()).get("action").squeeze(0)
    assert isinstance(action, torch.Tensor)
    assert action.shape[-1] == int(env.action_spec_unbatched.shape[-1])

    # Student LSTM requires recurrent state keys for single-step
    hidden_size = int(cfg.feature_extractor.output_dim)
    td_in.set("recurrent_state_h", torch.zeros(1, 1, hidden_size))
    td_in.set("recurrent_state_c", torch.zeros(1, 1, hidden_size))
    td_in.set("is_init", torch.ones(1, 1, 1, dtype=torch.bool))

    td_out = agent.student_actor_critic(td_in.clone())
    assert cfg.l2t.student_hidden_key in td_out.keys()
    student_loc = td_out.get(cfg.l2t.student_loc_key)
    student_scale = td_out.get(cfg.l2t.student_scale_key)
    assert student_loc.shape[-1] == int(env.action_spec_unbatched.shape[-1])
    assert student_scale.shape[-1] == int(env.action_spec_unbatched.shape[-1])


@pytest.mark.mujoco("HalfCheetah-v5")
def test_l2t_halfcheetah_mujoco_smoke(l2t_cfg_factory, make_env_parallel):  # type: ignore
    cfg = l2t_cfg_factory(
        env_name="HalfCheetah-v5",
        num_envs=8,
        frames_per_batch=1024,
        total_frames=1024,
        feature_dim=64,
        imitation="l2",
        mixture_coeff=0.2,
    )  # type: ignore
    env = make_env_parallel(cfg.env.env_name, num_workers=cfg.env.num_envs, device="cpu")  # type: ignore
    agent = L2T(env=env, config=cfg)

    # Run a tiny training iteration
    start = time.time()
    agent.train()
    duration = time.time() - start
    # Ensure reasonable speed and no crash
    assert duration < 120.0
    assert np.isclose(cfg.l2t.mixture_coeff, 0.2)
