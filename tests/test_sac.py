from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.gym import GymEnv as TorchRLGymEnv

from rlopt.agent.sac import SAC


def evaluate_policy_average_return(
    agent: SAC, env_name: str, num_episodes: int = 3, max_steps: int = 200
) -> float:
    base = TorchRLGymEnv(env_name, device="cpu")
    eval_env = TransformedEnv(base)
    from torchrl.envs import ClipTransform, RewardSum, StepCounter

    eval_env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    eval_env.append_transform(RewardSum())
    eval_env.append_transform(StepCounter())
    # Cast observations to float32 only when needed (e.g., MuJoCo float64)
    try:
        obs_dtype = eval_env.observation_spec["observation"].dtype
    except Exception:
        obs_dtype = None
    if obs_dtype is not None and obs_dtype == torch.float64:
        from torchrl.envs import DoubleToFloat

        eval_env.append_transform(DoubleToFloat(in_keys=["observation"]))

    returns = []
    for _ in range(num_episodes):
        td = eval_env.rollout(
            max_steps=max_steps,
            policy=agent.actor_critic.get_policy_operator(),
            break_when_any_done=True,
        )
        done_mask = td.get(("next", "done"))
        if done_mask.any():
            ep_returns = td.get(("next", "episode_reward"))[done_mask]
            returns.append(ep_returns.mean().item())
        else:
            rewards = td.get(("next", "reward")).sum().item()
            returns.append(rewards)
    return float(np.mean(returns))


# -------------------------
# Structural / shape tests
# -------------------------


def test_sac_instantiation_and_predict_shapes(sac_cfg_factory, make_env):  # type: ignore
    cfg = sac_cfg_factory(env_name="Pendulum-v1", num_envs=4, frames_per_batch=256, total_frames=512)  # type: ignore
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore
    agent = SAC(env=env, config=cfg)

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


# -------------------------
# Training tests (short)
# -------------------------


@pytest.mark.mujoco("HalfCheetah-v5")
def test_sac_halfcheetah_v5_smoke(sac_cfg_factory, make_env_parallel):  # type: ignore
    """Lightweight SAC smoke run on HalfCheetah-v5 to validate integration."""
    # Skip if MuJoCo/Gymnasium env is unavailable in the environment
    try:
        _ = TorchRLGymEnv("HalfCheetah-v5", device="cpu")
    except Exception as e:  # pragma: no cover - env availability varies
        pytest.skip(f"HalfCheetah-v5 unavailable: {e}")

    cfg = sac_cfg_factory(env_name="HalfCheetah-v5", num_envs=8, frames_per_batch=2040, total_frames=2040)  # type: ignore
    env = make_env_parallel(cfg.env.env_name, num_workers=cfg.env.num_envs, device="cpu")  # type: ignore
    agent = SAC(env=env, config=cfg)

    start = time.time()
    agent.train()
    duration = time.time() - start
    assert duration < 180.0


@pytest.mark.slow
@pytest.mark.parametrize("memmap", [False, True])
@pytest.mark.parametrize("utd_ratio", [0.25, 1.0])
def test_sac_training_perf_guardrails(
    sac_cfg_factory, make_env_parallel, tmp_path, memmap, utd_ratio
):  # type: ignore
    cfg = sac_cfg_factory(
        env_name="Pendulum-v1",
        num_envs=4,
        frames_per_batch=1024,
        total_frames=2048,
    )
    cfg.sac.utd_ratio = float(utd_ratio)
    cfg.collector.scratch_dir = str(tmp_path) if memmap else None

    env = make_env_parallel(cfg.env.env_name, num_workers=cfg.env.num_envs, device="cpu")  # type: ignore
    agent = SAC(env=env, config=cfg)

    start = time.time()
    agent.train()
    duration = time.time() - start

    # Guardrails: ensure it finishes quickly and maintains a minimal throughput
    frames = cfg.collector.total_frames
    fps = frames / max(duration, 1e-6)
    assert duration < 120.0
    assert fps > 10.0
