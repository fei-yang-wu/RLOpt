import unittest
from rlopt.common.torch.buffer import (
    ReplayBuffer,
    RolloutBuffer,
    DictReplayBuffer,
    DictRolloutBuffer,
)
from gymnasium import spaces
import torch as th


class TestBuffer(unittest.TestCase):

    def test_replay_buffer(self):
        buffer_size = 100
        observation_space = spaces.Box(low=0, high=1, shape=(4,))
        action_space = spaces.Discrete(2)
        buffer = ReplayBuffer(buffer_size, observation_space, action_space)

        # Test add method
        for _ in range(buffer_size):
            obs = th.tensor([0.1, 0.2, 0.3, 0.4])
            next_obs = th.tensor([0.2, 0.3, 0.4, 0.5])
            action = th.tensor(1)
            reward = th.tensor(0.5)
            done = th.tensor(0)
            infos = [{"TimeLimit.truncated": True}]
            buffer.add(obs, next_obs, action, reward, done, infos)

        # Test sample method
        batch_size = 32
        samples = buffer.sample(batch_size)
        self.assertEqual(samples.observations.shape, (batch_size, 4))
        self.assertEqual(samples.next_observations.shape, (batch_size, 4))
        self.assertEqual(samples.actions.shape, (batch_size, 1))
        self.assertEqual(samples.rewards.shape, (batch_size, 1))
        self.assertEqual(samples.dones.shape, (batch_size, 1))

    def test_rollout_buffer(self):
        buffer_size = 100
        observation_space = spaces.Box(low=0, high=1, shape=(4,))
        action_space = spaces.Discrete(2)
        device = "cuda:0" if th.cuda.is_available() else "cpu"
        buffer = RolloutBuffer(
            buffer_size, observation_space, action_space, device=device
        )

        # Test add method
        for _ in range(buffer_size):
            obs = th.tensor([0.1, 0.2, 0.3, 0.4], device=device)
            action = th.tensor(1, device=device)
            reward = th.tensor(0.5, device=device)
            episode_start = th.tensor(1, device=device)
            value = th.tensor(0.8, device=device)
            log_prob = th.tensor(0.2, device=device)
            buffer.add(obs, action, reward, episode_start, value, log_prob)

        # Test get method
        samples = buffer.get(batch_size=1)
        for sample in samples:
            self.assertEqual(sample.observations.shape, (1, 4))
            self.assertEqual(sample.actions.shape, (1, 1))
            # self.assertEqual(sample.rewards.shape, ())
            self.assertEqual(sample.advantages.shape, (1,))
            self.assertEqual(sample.returns.shape, (1,))
            self.assertEqual(sample.old_log_prob.shape, (1,))
            self.assertEqual(sample.old_values.shape, (1,))
            self.assertEqual(sample.observations.device, th.device(device))

    def test_dict_replay_buffer(self):
        buffer_size = 100
        observation_space = spaces.Dict(
            {"obs1": spaces.Box(low=0, high=1, shape=(4,)), "obs2": spaces.Discrete(2)}
        )
        action_space = spaces.Discrete(2)
        buffer = DictReplayBuffer(buffer_size, observation_space, action_space)

        for _ in range(buffer_size):
            # Test add method
            obs = {"obs1": th.tensor([0.1, 0.2, 0.3, 0.4]), "obs2": th.tensor(1)}
            next_obs = {"obs1": th.tensor([0.2, 0.3, 0.4, 0.5]), "obs2": th.tensor(0)}
            action = th.tensor(1)
            reward = th.tensor(0.5)
            done = th.tensor(0)
            infos = [{"TimeLimit.truncated": True}]
            buffer.add(obs, next_obs, action, reward, done, infos)

        # Test sample method
        batch_size = 32
        samples = buffer.sample(batch_size)
        self.assertEqual(samples.observations["obs1"].shape, (batch_size, 4))
        self.assertEqual(samples.observations["obs2"].shape, (batch_size, 1))
        self.assertEqual(samples.next_observations["obs1"].shape, (batch_size, 4))
        self.assertEqual(samples.next_observations["obs2"].shape, (batch_size, 1))
        self.assertEqual(samples.actions.shape, (batch_size, 1))
        self.assertEqual(samples.rewards.shape, (batch_size, 1))
        self.assertEqual(samples.dones.shape, (batch_size, 1))

    def test_dict_rollout_buffer(self):
        buffer_size = 100
        observation_space = spaces.Dict(
            {"obs1": spaces.Box(low=0, high=1, shape=(4,)), "obs2": spaces.Discrete(2)}
        )
        action_space = spaces.Discrete(2)
        buffer = DictRolloutBuffer(buffer_size, observation_space, action_space)

        for _ in range(buffer_size):
            # Test add method
            obs = {"obs1": th.tensor([0.1, 0.2, 0.3, 0.4]), "obs2": th.tensor(1)}
            action = th.tensor(1)
            reward = th.tensor(0.5)
            episode_start = th.tensor(1)
            value = th.tensor(0.8)
            log_prob = th.tensor(0.2)
            buffer.add(obs, action, reward, episode_start, value, log_prob)

        # Test get method
        samples = buffer.get()
        for sample in samples:
            self.assertEqual(sample.observations["obs1"].shape, (buffer_size, 4))
            self.assertEqual(sample.observations["obs2"].shape, (buffer_size, 1))
            self.assertEqual(sample.actions.shape, (buffer_size, 1))
            self.assertEqual(sample.rewards.shape, (buffer_size, 1))
            self.assertEqual(sample.advantages.shape, (buffer_size,))
            self.assertEqual(sample.returns.shape, (buffer_size,))
            self.assertEqual(sample.old_log_prob.shape, (buffer_size,))
            self.assertEqual(sample.old_values.shape, (buffer_size,))


if __name__ == "__main__":
    unittest.main()
