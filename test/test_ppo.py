from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torchrl.envs.libs.gym
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from tensordict import TensorDict
from torchrl.envs import EnvCreator, ParallelEnv, TransformedEnv
from torchrl.envs.libs.gym import GymEnv as TorchRLGymEnv
from torchrl.envs.transforms import Compose, InitTracker

from rlopt.agent.ppo import PPO, PPORecurrent
from rlopt.common.base_class import BaseAlgorithm

# We avoid using the library helper to skip DoubleToFloat, which expects float64


def make_test_env(env_name: str, device: str = "cpu") -> TransformedEnv:
    base = TorchRLGymEnv(env_name, device=device)
    env = TransformedEnv(base)
    from torchrl.envs import ClipTransform, RewardSum, StepCounter

    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


def make_test_env_parallel(
    env_name: str, num_workers: int, device: str = "cpu"
) -> TransformedEnv:
    def maker():
        return TorchRLGymEnv(env_name, device=device)

    base = ParallelEnv(
        num_workers,
        EnvCreator(maker),
        serial_for_single=True,
        mp_start_method="fork",
    )
    env = TransformedEnv(base)
    from torchrl.envs import ClipTransform, RewardSum, StepCounter

    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


# -------------------------
# Global fixtures/utilities
# -------------------------


# Ensure torchrl's GymEnv treats VecEnv as batched; avoids isaaclab import issues
torchrl.envs.libs.gym.GymEnv._is_batched = property(  # type: ignore[attr-defined]
    lambda self: isinstance(self._env, VecEnv)
)

# Ensure list-like config keys become plain lists for modules expecting list/tuple
BaseAlgorithm.total_input_keys = property(  # type: ignore[assignment]
    lambda self: list(self.config.total_input_keys)
)


def _load_base_cfg() -> DictConfig:
    cfg_path = Path(__file__).with_name("test_config.yaml")
    cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.set_struct(cfg, False)
    # Keep logging off for tests
    cfg.logger.backend = None
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.use_feature_extractor = True
    # Use continuous control env that is light-weight
    cfg.env.env_name = "Pendulum-v1"
    return cfg


def _configure_for_fast_training(
    cfg, num_envs: int = 100, frames_per_batch: int = 4000, total_frames: int = 12000
):
    cfg.env.num_envs = num_envs
    cfg.collector.frames_per_batch = frames_per_batch
    cfg.collector.total_frames = total_frames
    cfg.collector.set_truncated = False
    # Avoid periodic save_model when logger is disabled in tests
    cfg.save_interval = 0
    # Smaller nets for speed
    cfg.policy.num_cells = [64, 64]
    cfg.value_net.num_cells = [64, 64]
    # Larger minibatch to do fewer updates; few epochs
    cfg.loss.mini_batch_size = 1024
    cfg.loss.epochs = 2
    cfg.optim.lr = 3e-4
    # Feature-extractor shape alignment
    cfg.feature_extractor.output_dim = 64
    cfg.policy_in_keys = ["hidden"]
    cfg.value_net_in_keys = ["hidden"]
    cfg.total_input_keys = ["observation"]
    return cfg


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
    eval_env = make_test_env(env_name, device="cpu")
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


def test_ppo_instantiation_and_predict_shapes():
    cfg = _configure_for_fast_training(
        _load_base_cfg(), num_envs=4, frames_per_batch=64, total_frames=64
    )
    env = make_test_env(cfg.env.env_name, device="cpu")
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


def test_pporecurrent_instantiation_and_predict_shapes():
    cfg = _configure_recurrent(
        _configure_for_fast_training(
            _load_base_cfg(), num_envs=4, frames_per_batch=64, total_frames=64
        ),
        hidden_size=32,
    )
    env = make_test_env(cfg.env.env_name, device="cpu")
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


def test_training_improves_return_ppo():
    cfg = _configure_for_fast_training(
        _load_base_cfg(), num_envs=10, frames_per_batch=4000, total_frames=100000
    )
    env = make_test_env_parallel(
        cfg.env.env_name, num_workers=cfg.env.num_envs, device="cpu"
    )
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


def test_training_improves_return_pporecurrent():
    cfg = _configure_recurrent(
        _configure_for_fast_training(
            _load_base_cfg(), num_envs=10, frames_per_batch=4000, total_frames=100000
        ),
        hidden_size=64,
    )

    env = make_test_env_parallel(
        cfg.env.env_name, num_workers=cfg.env.num_envs, device="cpu"
    )
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
