"""PPO model checkpoints preserve the global budget and optimizer progress."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import torch


def agent(tmp_path):
    warnings.filterwarnings("ignore", category=DeprecationWarning, append=False)
    from rlopt.agent.ppo.ppo import PPO

    value = object.__new__(PPO)
    value.policy = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    value.value_function = torch.nn.Linear(3, 1)
    value.q_function = None
    value.feature_extractor = None
    value.optim = torch.optim.Adam(
        [*value.policy.parameters(), *value.value_function.parameters()], lr=0.01
    )
    value.config = SimpleNamespace(feature_extractor=None, device="cpu")
    value.env = SimpleNamespace(is_closed=True)
    value.log_dir = tmp_path
    return value


def test_model_checkpoint_restores_weights_normalization_optimizer_and_progress(
    tmp_path,
):
    source = agent(tmp_path)
    obs = torch.randn(8, 3)
    loss = (
        source.policy(obs).square().mean() + source.value_function(obs).square().mean()
    )
    loss.backward()
    source.optim.step()
    source._ppo_updates_completed = torch.tensor(20)
    source.save_model(tmp_path, step=153600)
    path = tmp_path / "model_step_153600.pt"
    restored = agent(tmp_path)
    restored.load_model(str(path))
    assert restored._resume_frame_offset == 153600
    assert restored._resume_updates_completed == 20
    for key, value in source.policy.state_dict().items():
        torch.testing.assert_close(value, restored.policy.state_dict()[key])
    for key, value in source.value_function.state_dict().items():
        torch.testing.assert_close(value, restored.value_function.state_dict()[key])
    for original, loaded in zip(
        source.optim.state.values(), restored.optim.state.values(), strict=True
    ):
        for key in original:
            torch.testing.assert_close(original[key], loaded[key])


def test_legacy_checkpoint_does_not_infer_global_frames_from_filename(tmp_path):
    source = agent(tmp_path)
    source.save_model(tmp_path, step=500)
    path = tmp_path / "model_step_500.pt"
    state = torch.load(path, weights_only=True)
    state.pop("cumulative_env_frames")
    state.pop("ppo_updates_completed")
    torch.save(state, path)
    restored = agent(tmp_path)
    restored.load_model(str(path))
    assert not hasattr(restored, "_resume_frame_offset")
    assert restored._resume_updates_completed == 0
