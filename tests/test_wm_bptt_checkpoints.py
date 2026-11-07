"""Inference and resume loads of WMBPTT checkpoints (CPU)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness

RUN = {"frames": harness.frames_for(4)}


def trained_checkpoint(tmp_path: Path, **wm: Any) -> tuple[Any, Path]:
    agent = harness.make_agent(**RUN, **wm)
    agent.train()
    path = tmp_path / "trained.pt"
    agent.save_model(path)
    return agent, path


def assert_same_policy(a: Any, b: Any) -> None:
    assert harness.state_equal(a.actor.state_dict(), b.actor.state_dict())


def test_inference_load_ignores_other_world_model_and_critic_sizes(tmp_path: Path) -> None:
    trained, path = trained_checkpoint(tmp_path, wm_hidden=[48, 48], critic_hidden=[24])
    evaluator = harness.make_agent()
    assert harness.SMALL["wm_hidden"] != [48, 48]
    evaluator.load_policy(str(path))
    assert_same_policy(evaluator, trained)


def test_resume_load_refuses_other_world_model_sizes(tmp_path: Path) -> None:
    _, path = trained_checkpoint(tmp_path, wm_hidden=[48, 48])
    with pytest.raises(RuntimeError, match="size mismatch"):
        harness.make_agent().load_model(str(path))


def test_resume_load_refuses_other_critic_sizes(tmp_path: Path) -> None:
    _, path = trained_checkpoint(tmp_path, critic_hidden=[24])
    with pytest.raises(RuntimeError, match="size mismatch"):
        harness.make_agent().load_model(str(path))


def test_inference_load_reads_only_the_policy_entries(tmp_path: Path) -> None:
    trained, path = trained_checkpoint(tmp_path)
    data = torch.load(path, weights_only=False)
    keep = {key: data[key] for key in ("policy_state_dict", "policy_meta")}
    stripped = tmp_path / "policy_only.pt"
    torch.save(keep, stripped)
    evaluator = harness.make_agent()
    evaluator.load_policy(str(stripped))
    assert_same_policy(evaluator, trained)
    assert evaluator._iteration == 0
    assert evaluator._resume_frame_offset == 0


def test_inference_load_takes_action_semantics_from_the_checkpoint(tmp_path: Path) -> None:
    _, path = trained_checkpoint(tmp_path, bf16=False, sensor_noise=False)
    evaluator = harness.make_agent()
    assert evaluator.policy_net.bf16
    assert evaluator.task.sensor_noise
    evaluator.load_policy(str(path))
    assert not evaluator.policy_net.bf16
    assert not evaluator.task.sensor_noise


def test_inference_load_of_an_older_checkpoint_reads_bf16_from_its_config(tmp_path: Path) -> None:
    _, path = trained_checkpoint(tmp_path, bf16=False)
    data = torch.load(path, weights_only=False)
    del data["policy_meta"]["bf16"]
    torch.save(data, path)
    evaluator = harness.make_agent()
    evaluator.load_policy(str(path))
    assert not evaluator.policy_net.bf16


@pytest.mark.parametrize("load", ["load_policy", "load_model"])
@pytest.mark.parametrize(
    ("key", "value"),
    [("arch", "other"), ("in_keys", [["other"]]), ("action_dim", 3), ("goal_binding", {"task": "other"})],
)
def test_both_loads_refuse_another_policy_interface(tmp_path: Path, load: str, key: str, value: Any) -> None:
    _, path = trained_checkpoint(tmp_path)
    data = torch.load(path, weights_only=False)
    data["policy_meta"][key] = value
    torch.save(data, path)
    with pytest.raises(ValueError, match=key if key != "arch" else "not a wmbptt"):
        getattr(harness.make_agent(), load)(str(path))


def test_resume_load_refuses_another_actor(tmp_path: Path) -> None:
    _, path = trained_checkpoint(tmp_path, actor_hidden=[32, 32])
    with pytest.raises(ValueError, match="actor_hidden"):
        harness.make_agent().load_model(str(path))


def test_inference_load_builds_the_checkpoints_actor(tmp_path: Path) -> None:
    """The actor's shape comes from ``policy_meta``, as its action semantics do, so a
    default evaluator scores an actor of other widths (``wm.actor_arch`` too)."""
    trained, path = trained_checkpoint(tmp_path, actor_hidden=[32, 32], actor_activation="silu")
    evaluator = harness.make_agent()
    evaluator.load_policy(str(path))
    assert_same_policy(evaluator, trained)
    meta = evaluator.policy_meta()
    assert (meta["actor_hidden"], meta["actor_activation"]) == ([32, 32], "silu")
    obs, _ = evaluator._isaac.reset()
    feat = evaluator.task.policy_obs_from_env(*(obs["wm"][key] for _, key in evaluator.task.policy_in_keys))
    torch.testing.assert_close(evaluator.predict(feat), trained.predict(feat), rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["dynamics_state_dict", "critic_state_dict", "optimizers"])
def test_resume_refuses_a_checkpoint_without_training_state(tmp_path: Path, missing: str) -> None:
    _, path = trained_checkpoint(tmp_path)
    data = torch.load(path, weights_only=False)
    del data[missing]
    torch.save(data, path)
    with pytest.raises(ValueError, match="cannot resume"):
        harness.make_agent().load_model(str(path))


def test_an_inference_loaded_agent_refuses_to_train(tmp_path: Path) -> None:
    _, path = trained_checkpoint(tmp_path)
    evaluator = harness.make_agent(**RUN)
    evaluator.load_policy(str(path))
    with pytest.raises(RuntimeError, match="load_model"):
        evaluator.train()


def test_resume_restores_every_component_and_continues_bitwise(tmp_path: Path) -> None:
    """Same replay and RNG on both sides: the resumed agent steps as the original."""
    trained, path = trained_checkpoint(tmp_path)
    resumed = harness.make_agent()
    resumed.load_model(str(path))
    for name in ("actor", "critic", "dynamics", "wm_optim", "actor_optim", "critic_optim"):
        assert harness.state_equal(getattr(trained, name).state_dict(), getattr(resumed, name).state_dict()), name
    assert resumed._iteration == trained._iteration
    assert resumed._resume_frame_offset == trained._frames_processed

    resumed.ensure_replay()
    resumed.replay = trained.replay
    resumed._noise_seed = trained._noise_seed
    after = []
    for agent in (trained, resumed):
        torch.manual_seed(11)
        metrics: dict[str, list] = defaultdict(list)
        agent._set_lrs()
        agent._update_world_model(metrics)
        agent._update_actor_critic(metrics)
        after.append([getattr(agent, n).state_dict() for n in ("actor", "critic", "dynamics", "wm_optim", "actor_optim", "critic_optim")])
    assert harness.state_equal(after[0], after[1])


def test_default_policy_meta_only_adds_bf16_to_the_baseline() -> None:
    meta = harness.make_agent().policy_meta()
    assert set(meta) == {
        "arch", "in_keys", "policy_obs_dim", "action_dim", "actor_hidden", "actor_activation",
        "task_model", "sensor_noise", "bf16", "goal_binding",
    }  # fmt: skip
