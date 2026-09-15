"""`freeze_normalizers=true` must freeze the running input statistics everywhere."""

import warnings

import torch

with warnings.catch_warnings():
    # torch.jit.script deprecation raised at import by a torchrl dependency.
    warnings.simplefilter("ignore", DeprecationWarning)
    from rlopt.agent.ppo.ppo import PPOConfig, RunningMeanStdCatInputs


def test_frozen_running_stats_do_not_move_in_train_mode():
    live = RunningMeanStdCatInputs(torch.nn.Identity(), feature_dim=2)
    frozen = RunningMeanStdCatInputs(torch.nn.Identity(), feature_dim=2, frozen=True)
    live.train()
    frozen.train()
    batch = torch.tensor([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])

    out_live = live(batch)
    out_frozen = frozen(batch)
    # Both normalize with the prior (zero mean, unit variance) statistics.
    torch.testing.assert_close(out_live, out_frozen)
    # Only the live module moved its statistics.
    assert live.count.item() == 4.0
    assert frozen.count.item() == 1.0
    torch.testing.assert_close(frozen.running_mean, torch.zeros(2))
    torch.testing.assert_close(frozen.running_var, torch.ones(2))

    # The frozen flag is a run setting, not model state: a checkpoint written
    # by a live module restores into a frozen one and stays put afterwards.
    frozen.load_state_dict(live.state_dict())
    before = frozen.running_mean.clone()
    frozen(batch * 10.0)
    torch.testing.assert_close(frozen.running_mean, before)
    assert "frozen" not in frozen.state_dict()


def test_ppo_config_declares_the_freeze_key():
    config = PPOConfig()
    assert config.update_normalizers_after_rollout is True
    config.update_normalizers_after_rollout = False
    assert config.update_normalizers_after_rollout is False


def test_ppo_config_declares_freeze_normalizers():
    config = PPOConfig()
    assert config.freeze_normalizers is False
    assert config.update_normalizers_after_rollout is True
    config.freeze_normalizers = True
    assert config.freeze_normalizers is True


def test_freeze_normalizers_wins_in_both_paths():
    import types

    from rlopt.agent.ppo.ppo import PPO

    class _Probe:
        pass

    probe = _Probe()
    # freeze_normalizers -> forward frozen whatever the after-rollout flag says
    probe.config = types.SimpleNamespace(
        ppo=types.SimpleNamespace(freeze_normalizers=True, update_normalizers_after_rollout=False)
    )
    assert PPO._normalizers_frozen(probe) is True
    probe.config.ppo.update_normalizers_after_rollout = True
    assert PPO._normalizers_frozen(probe) is True
    # without it the after-rollout flag decides
    probe.config.ppo.freeze_normalizers = False
    assert PPO._normalizers_frozen(probe) is True
    probe.config.ppo.update_normalizers_after_rollout = False
    assert PPO._normalizers_frozen(probe) is False
