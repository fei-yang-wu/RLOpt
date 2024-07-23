import unittest

import gymnasium as gym
from rlopt.agent.torch.ppo import PPO


class TestCustomPPO(unittest.TestCase):
    def test_direct_training(self):
        # TODO: Implement your test logic here
        env = gym.make("CartPole-v1")
        agent = PPO("MlpPolicy", env=env)
        obs = env.reset()
        agent.learn(total_timesteps=1000)

    def test_custombuffer_normal_input(self):
        from rlopt.common.torch.buffer import (
            ReplayBuffer,
            RolloutBuffer,
            DictReplayBuffer,
            DictRolloutBuffer,
        )
        import torch

        env = gym.make("CartPole-v1")
        agent: PPO
        agent = PPO("MlpPolicy", env=env, rollout_buffer_class=RolloutBuffer)
        obs, info = env.reset()
        for _ in range(1000):
            action, _ = agent.predict(torch.Tensor(obs).reshape(1, 4))
            next_obs, reward, truncated, info, done = env.step(action[0])
            if done or truncated:
                obs, info = env.reset()


if __name__ == "__main__":
    unittest.main()
