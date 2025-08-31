from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.gym import GymEnv as TorchRLGymEnv
from torchrl.envs.transforms import Compose, InitTracker

from rlopt.agent.ppo import PPO, PPORecurrent


def _configure_recurrent(cfg, hidden_size: int = 64):
    cfg.feature_extractor.lstm = {
        "hidden_size": hidden_size,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
    }
    cfg.feature_extractor.output_dim = hidden_size
    return cfg


def evaluate_policy_average_return(
    agent: PPO | PPORecurrent,
    env_name: str,
    num_episodes: int = 5,
    max_steps: int = 200,
) -> float:
    base = TorchRLGymEnv(env_name, device="cpu")
    eval_env = TransformedEnv(base)
    from torchrl.envs import ClipTransform, RewardSum, StepCounter

    eval_env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    eval_env.append_transform(RewardSum())
    eval_env.append_transform(StepCounter())
    # Cast observations to float32 to match model params
    # Cast to float32 only when observations are float64 (e.g., MuJoCo)
    try:
        obs_dtype = eval_env.observation_spec["observation"].dtype
    except Exception:
        obs_dtype = None
    if obs_dtype is not None and obs_dtype == torch.float64:
        from torchrl.envs import DoubleToFloat

        eval_env.append_transform(DoubleToFloat(in_keys=["observation"]))
    if isinstance(agent, PPORecurrent):
        eval_env = PPORecurrent.add_required_transforms(eval_env)

    # Deterministic evaluation via actor's default setting
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
            # Fallback: sum rewards if no termination was recorded
            rewards = td.get(("next", "reward")).sum().item()
            returns.append(rewards)
    return float(np.mean(returns))


# -------------------------
# Structural / shape tests
# -------------------------


def test_pporecurrent_environment_compatibility():
    env_without_tracker = TransformedEnv(TorchRLGymEnv("Pendulum-v1", device="cpu"))
    env_with_tracker = TransformedEnv(
        TorchRLGymEnv("Pendulum-v1", device="cpu"), InitTracker()
    )

    is_compatible, missing = PPORecurrent.check_environment_compatibility(
        env_without_tracker
    )
    assert not is_compatible
    assert "InitTracker() - needed for episode boundary tracking" in missing

    is_compatible, missing = PPORecurrent.check_environment_compatibility(
        env_with_tracker
    )
    assert is_compatible
    assert len(missing) == 0


def test_pporecurrent_add_required_transforms():
    env = TransformedEnv(TorchRLGymEnv("Pendulum-v1", device="cpu"))

    def has_init_tracker(transform) -> bool:
        if isinstance(transform, InitTracker):
            return True
        if isinstance(transform, Compose):
            return any(has_init_tracker(t) for t in transform.transforms)
        return False

    assert not has_init_tracker(env.transform)
    env = PPORecurrent.add_required_transforms(env)
    assert has_init_tracker(env.transform)


def test_ppo_instantiation_and_predict_shapes(ppo_cfg_factory, make_env):  # type: ignore
    cfg = ppo_cfg_factory(env_name="Pendulum-v1", num_envs=4, frames_per_batch=64, total_frames=64)  # type: ignore
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore
    agent = PPO(env=env, config=cfg)

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


def test_pporecurrent_instantiation_and_predict_shapes(ppo_cfg_factory, make_env):  # type: ignore
    cfg = ppo_cfg_factory(env_name="Pendulum-v1", num_envs=4, frames_per_batch=64, total_frames=64)  # type: ignore
    # Inject recurrent config for feature extractor
    cfg.feature_extractor.lstm = {
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
    }
    cfg.feature_extractor.output_dim = 32
    env = make_env(cfg.env.env_name, device="cpu")  # type: ignore
    env = PPORecurrent.add_required_transforms(env)
    agent = PPORecurrent(env=env, config=cfg)

    obs = env.reset().get("observation").squeeze(0)
    td_in = TensorDict({"observation": obs.unsqueeze(0)}, batch_size=[1])
    # Add required recurrent state keys for a single forward call
    hidden_size = int(cfg.feature_extractor.lstm["hidden_size"])  # type: ignore[index]
    num_layers = int(cfg.feature_extractor.lstm.get("num_layers", 1))  # type: ignore[attr-defined]
    td_in.set("recurrent_state_h", torch.zeros(num_layers, 1, hidden_size))
    td_in.set("recurrent_state_c", torch.zeros(num_layers, 1, hidden_size))
    td_in.set("is_init", torch.ones(1, 1, 1, dtype=torch.bool))
    policy_op = agent.actor_critic.get_policy_operator()
    action = policy_op(td_in).get("action").squeeze(0)
    assert isinstance(action, torch.Tensor)
    assert action.shape[-1] == int(env.action_spec_unbatched.shape[-1])


# -------------------------
# Training tests (short)
# -------------------------


@pytest.mark.slow
def test_training_improves_return_ppo(ppo_cfg_factory, make_env_parallel):  # type: ignore
    cfg = ppo_cfg_factory(env_name="Pendulum-v1", num_envs=10, frames_per_batch=4000, total_frames=100000)  # type: ignore
    env = make_env_parallel(cfg.env.env_name, num_workers=cfg.env.num_envs, device="cpu")  # type: ignore
    agent = PPO(env=env, config=cfg)

    before = evaluate_policy_average_return(
        agent, cfg.env.env_name, num_episodes=5, max_steps=200
    )
    start = time.time()
    agent.train()
    duration = time.time() - start
    after = evaluate_policy_average_return(
        agent, cfg.env.env_name, num_episodes=5, max_steps=200
    )

    # Ensure training runs reasonably fast in CI/local (~1 min target)
    assert duration < 120.0
    # Performance should strictly improve; small tolerance for noise
    assert after > before + 0.01


@pytest.mark.slow
def test_training_improves_return_pporecurrent(ppo_cfg_factory, make_env_parallel):  # type: ignore
    cfg = ppo_cfg_factory(env_name="Pendulum-v1", num_envs=10, frames_per_batch=4000, total_frames=100000)  # type: ignore
    # recurrent
    cfg.feature_extractor.lstm = {
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
    }
    cfg.feature_extractor.output_dim = 64

    env = make_env_parallel(cfg.env.env_name, num_workers=cfg.env.num_envs, device="cpu")  # type: ignore
    env = PPORecurrent.add_required_transforms(env)

    agent = PPORecurrent(env=env, config=cfg)

    before = evaluate_policy_average_return(
        agent, cfg.env.env_name, num_episodes=10, max_steps=200
    )
    start = time.time()
    agent.train()
    duration = time.time() - start
    after = evaluate_policy_average_return(
        agent, cfg.env.env_name, num_episodes=10, max_steps=200
    )

    assert duration < 120.0
    assert after > before + 0.01


@pytest.mark.mujoco("HalfCheetah-v5")
def test_ppo_halfcheetah_v5_smoke(ppo_cfg_factory, make_env_parallel):  # type: ignore
    """Lightweight PPO smoke run on HalfCheetah-v5 to validate integration.

    This keeps frames small to avoid long CI times and only checks that
    training runs without error on a MuJoCo continuous control task.
    """
    # Skip if MuJoCo/Gymnasium env is unavailable in the environment
    try:
        _ = TorchRLGymEnv("HalfCheetah-v5", device="cpu")
    except Exception as e:  # pragma: no cover - env availability varies
        pytest.skip(f"HalfCheetah-v5 unavailable: {e}")

    # Use fixture-injected factories; do not import fixtures directly
    cfg = ppo_cfg_factory(env_name="HalfCheetah-v5", num_envs=8, frames_per_batch=2040, total_frames=2040)  # type: ignore
    env = make_env_parallel(cfg.env.env_name, num_workers=cfg.env.num_envs, device="cpu")  # type: ignore
    agent = PPO(env=env, config=cfg)

    start = time.time()
    agent.train()
    duration = time.time() - start
    assert duration < 180.0
