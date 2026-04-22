from __future__ import annotations

import warnings
from contextlib import suppress
from typing import ClassVar

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

warnings.filterwarnings(
    "ignore",
    message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
    category=DeprecationWarning,
    append=False,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script_method` is not supported in Python 3.14\\+.*",
    category=DeprecationWarning,
    append=False,
)

from rlopt.agent import PPO, PPORLOptConfig  # noqa: E402
from rlopt.config_base import NetworkConfig  # noqa: E402
from rlopt.env_utils import make_parallel_env  # noqa: E402


class DummyContinuousEnv(gym.Env):
    """Small deterministic continuous-control env for checkpoint smoke tests."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        return self._obs(), {"options_were_provided": options is not None}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        action_arr = np.asarray(action, dtype=np.float32)
        reward = float(1.0 - np.square(action_arr).sum())
        terminated = self._step_count >= 4
        truncated = False
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                self._step_count / 4.0,
                1.0 - self._step_count / 4.0,
                0.5,
            ],
            dtype=np.float32,
        )


with suppress(gym.error.Error):
    gym.register(
        id="RLOptDummyContinuous-v0",
        entry_point=DummyContinuousEnv,
    )


def _tiny_ppo_cfg(env_name: str = "Pendulum-v1") -> PPORLOptConfig:
    """
    Create a CPU-only config
    """
    cfg = PPORLOptConfig()
    cfg.env.env_name = env_name
    cfg.env.library = "gymnasium"
    cfg.env.device = "cpu"
    cfg.env.num_envs = 1
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.loss.epochs = 1
    cfg.logger.backend = None
    cfg.save_interval = 0
    cfg.policy.input_keys = ["observation"]
    cfg.value_function = NetworkConfig(
        num_cells=[32, 32],
        activation_fn="relu",
        output_dim=1,
        input_keys=["observation"],
    )
    return cfg


def _assert_policy_and_value_match(agent_a: PPO, agent_b: PPO) -> None:
    for key, value in agent_a.policy.state_dict().items():
        assert torch.equal(value.cpu(), agent_b.policy.state_dict()[key].cpu())

    for key, value in agent_a.value_function.state_dict().items():
        assert torch.equal(value.cpu(), agent_b.value_function.state_dict()[key].cpu())


def test_ppo_save_load_roundtrip(tmp_path):
    """
    build PPO agent A
    save agent A checkpoint
    build a fresh PPO agent B with the same config
    load A's checkpoint into B
    compare A and B's network weights
    """
    cfg = _tiny_ppo_cfg()

    env_a = make_parallel_env(cfg)
    agent_a = PPO(env_a, cfg, logger=None)
    path = tmp_path / "ppo_roundtrip.pt"
    agent_a.save_model(path)
    assert path.exists()

    checkpoint = torch.load(path, map_location="cpu")
    assert "policy_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "value_state_dict" in checkpoint

    obs = torch.zeros(agent_a.observation_feature_shape("observation"))
    action_a = agent_a.predict(obs)

    env_b = make_parallel_env(cfg)
    agent_b = PPO(env_b, cfg, logger=None)
    agent_b.load_model(str(path))
    action_b = agent_b.predict(obs)

    _assert_policy_and_value_match(agent_a, agent_b)
    torch.testing.assert_close(action_a.cpu(), action_b.cpu())


def test_ppo_save_load_with_dummy_env(tmp_path):
    """Validate checkpoint roundtrip with a local fake-step environment."""
    cfg = _tiny_ppo_cfg("RLOptDummyContinuous-v0")

    env_a = make_parallel_env(cfg)
    agent_a = PPO(env_a, cfg, logger=None)
    reset_td = env_a.reset()
    action_a = agent_a.predict(reset_td["observation"][0])

    path = tmp_path / "ppo_dummy_env.pt"
    agent_a.save_model(path)

    env_b = make_parallel_env(cfg)
    agent_b = PPO(env_b, cfg, logger=None)
    agent_b.load_model(str(path))
    action_b = agent_b.predict(reset_td["observation"][0])

    _assert_policy_and_value_match(agent_a, agent_b)
    torch.testing.assert_close(action_a.cpu(), action_b.cpu())


def test_ppo_train_save_load_continue_with_dummy_env(tmp_path):
    """Train briefly, save, load into a fresh agent, and continue training."""
    cfg = _tiny_ppo_cfg("RLOptDummyContinuous-v0")

    env_a = make_parallel_env(cfg)
    agent_a = PPO(env_a, cfg, logger=None)
    agent_a.train()

    path = tmp_path / "ppo_resume.pt"
    agent_a.save_model(path)
    assert path.exists()

    env_b = make_parallel_env(cfg)
    agent_b = PPO(env_b, cfg, logger=None)
    agent_b.load_model(str(path))
    _assert_policy_and_value_match(agent_a, agent_b)

    agent_b.train()