from __future__ import annotations

import warnings
from types import SimpleNamespace

import torch


def test_rollout_statistics_are_frozen_for_learning_and_updated_once():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from tensordict import TensorDict

    from rlopt.agent.ppo.ppo import PPO, RunningMeanStdCatInputs

    actor = RunningMeanStdCatInputs(torch.nn.Linear(2, 1), 2)
    critic = RunningMeanStdCatInputs(torch.nn.Linear(3, 1), 3)
    owner = SimpleNamespace(
        policy=actor,
        value_function=critic,
        actor_critic=torch.nn.ModuleList([actor, critic]),
        config=SimpleNamespace(
            policy=SimpleNamespace(get_input_keys=lambda: ["actor_obs"]),
            value_function=SimpleNamespace(get_input_keys=lambda: ["critic_obs"]),
        ),
    )
    actor_obs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    critic_obs = torch.ones(3, 3)
    with PPO._freeze_rollout_normalizer_updates(owner):
        assert not actor.training
        assert actor.module.training
        for _ in range(5):
            actor(actor_obs).sum().backward()
            critic(critic_obs)
        assert float(actor.count) == float(critic.count) == 1
        assert actor.module.weight.grad is not None
    assert actor.training
    assert critic.training
    PPO._update_normalizers_from_rollout(
        owner, TensorDict({"actor_obs": actor_obs, "critic_obs": critic_obs}, [3])
    )
    assert float(actor.count) == float(critic.count) == 4
    torch.testing.assert_close(actor.running_mean, actor_obs.sum(0) / 4)


def test_gae_freeze_restores_a_frozen_normalizer_with_trainable_child_modes():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from rlopt.agent.ppo.ppo import PPO, RunningMeanStdCatInputs

    norm = RunningMeanStdCatInputs(torch.nn.Linear(2, 1), 2)
    norm.training = False
    owner = SimpleNamespace(value_function=norm)
    with PPO._freeze_value_normalizer_updates(owner):
        assert not norm.training
        assert not norm.module.training
    assert not norm.training
    assert norm.module.training


def test_collector_shape_probe_does_not_train_normalization(monkeypatch):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from rlopt.agent.ppo.ppo import PPO, RunningMeanStdCatInputs
    from rlopt.base_class import BaseAlgorithm

    norm = RunningMeanStdCatInputs(torch.nn.Linear(2, 1), 2)

    def construct(_self, _env, policy):
        policy(torch.zeros(8, 2))
        return SimpleNamespace(policy=policy)

    monkeypatch.setattr(BaseAlgorithm, "_construct_collector", construct)
    owner = object.__new__(PPO)
    PPO._construct_collector(owner, None, norm)
    assert norm.training
    assert norm.module.training
    assert float(norm.count) == 1
    torch.testing.assert_close(norm.running_var, torch.ones(2))
