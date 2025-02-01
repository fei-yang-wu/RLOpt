import unittest

import gymnasium as gym
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.env_util import make_vec_env
from rlopt.agent.ppo import PPO
from rlopt.envs.gymlike import make_mujoco_env

import hydra
from omegaconf import DictConfig
from torchrl.envs import GymEnv


class TestCustomPPO(unittest.TestCase):
    def test_direct_training(self):

        @hydra.main(config_path=".", config_name="test_config", version_base="1.1")
        def train(cfg: DictConfig) -> None:

            agent = PPO(
                env=make_mujoco_env("HalfCheetah-v4", device="cpu", from_pixels=False),
                config=cfg,
            )

            agent.train()

        train()


if __name__ == "__main__":
    unittest.main()
