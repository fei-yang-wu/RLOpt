"""``wm.ckpt_light`` and ``wm.last_every_min`` (CPU unless marked)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch
import wm_bptt_harness as harness

from rlopt.agent.wm_bptt import agent as agent_module
from rlopt.agent.wm_bptt import speed_options
from rlopt.agent.wm_bptt.agent import LIGHT_CHECKPOINT_KEYS
from rlopt.agent.wm_bptt.speed_options import (
    CheckpointWriter,
    cpu_snapshot,
    full_checkpoint_due,
)

ITERATIONS = 8
FRAMES = harness.frames_for(ITERATIONS)
EVERY = 40  # frames: every second iteration of the small run
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def run(tmp_path: Path, **wm: Any) -> Any:
    agent = harness.make_agent(frames=FRAMES, save_interval=EVERY, log_dir=tmp_path, **wm)
    agent.train()
    return agent


def names(agent: Any, pattern: str) -> list[str]:
    return sorted(p.name for p in agent._checkpoint_dir().glob(pattern))


def steps(agent: Any, pattern: str) -> list[int]:
    return sorted(int(p.stem.rsplit("_", 1)[1]) for p in agent._checkpoint_dir().glob(pattern))


def test_the_run_writes_light_periodic_files_and_one_full_end_file(tmp_path: Path) -> None:
    agent = run(tmp_path, ckpt_light=True)
    assert len(names(agent, "policy_step_*.pt")) >= 2
    assert steps(agent, "model_step_*.pt") == [agent._frames_processed]
    assert not list(agent._checkpoint_dir().glob("*.tmp"))


def test_a_light_file_holds_the_policy_and_the_counters_only(tmp_path: Path) -> None:
    agent = run(tmp_path, ckpt_light=True)
    path = agent._checkpoint_dir() / names(agent, "policy_step_*.pt")[0]
    data = torch.load(path, weights_only=False)
    assert set(data) == set(LIGHT_CHECKPOINT_KEYS)
    assert data["cumulative_env_frames"] == int(path.stem.rsplit("_", 1)[1])
    assert 0 < data["iteration"] < agent._iteration


def test_light_checkpoints_change_nothing_in_training(tmp_path: Path) -> None:
    runs = [run(tmp_path / name, **wm) for name, wm in (("off", {}), ("light", {"ckpt_light": True}))]
    assert harness.fingerprint(runs[0]) == harness.fingerprint(runs[1])


def test_a_light_file_scores_on_a_default_agent(tmp_path: Path) -> None:
    agent = run(tmp_path, ckpt_light=True)
    newest = agent._checkpoint_dir() / f"policy_step_{steps(agent, 'policy_step_*.pt')[-1]}.pt"
    evaluator = harness.make_agent()
    evaluator.load_policy(str(newest))
    state = torch.load(newest, weights_only=False)["policy_state_dict"]
    assert harness.state_equal(evaluator.policy.state_dict(), state)


def test_a_light_file_holds_the_average_with_actor_ema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = harness.make_agent(frames=FRAMES, save_interval=EVERY, log_dir=tmp_path, ckpt_light=True, actor_ema=0.9)
    expected = {}
    submit = CheckpointWriter.submit

    def spy(self: CheckpointWriter, payload: dict[str, Any], target: Path) -> None:
        if agent._deploys_average:  # before the first actor update the live actor is deployed
            expected[target.name] = harness.digest(agent.policy_avg.state_dict())
        submit(self, payload, target)

    monkeypatch.setattr(CheckpointWriter, "submit", spy)
    agent.train()
    assert expected
    for name, digest in expected.items():
        assert harness.digest(torch.load(agent._checkpoint_dir() / name, weights_only=False)["policy_state_dict"]) == digest


def test_resume_refuses_a_light_file(tmp_path: Path) -> None:
    agent = run(tmp_path, ckpt_light=True)
    path = agent._checkpoint_dir() / names(agent, "policy_step_*.pt")[0]
    with pytest.raises(ValueError, match="cannot resume"):
        harness.make_agent(ckpt_light=True, save_interval=EVERY).load_model(str(path))


def test_the_end_file_of_a_light_run_resumes(tmp_path: Path) -> None:
    agent = run(tmp_path, ckpt_light=True)
    resumed = harness.make_agent(ckpt_light=True, save_interval=EVERY)
    resumed.load_model(str(agent._checkpoint_dir() / names(agent, "model_step_*.pt")[0]))
    assert harness.state_equal(resumed.actor_optim.state_dict(), agent.actor_optim.state_dict())
    assert resumed._resume_frame_offset == agent._frames_processed


def test_due_logic() -> None:
    def full(start: float, walls: list[float], every_min: float) -> list[float]:
        last, out = start, []
        for wall in walls:
            if full_checkpoint_due(wall, last, every_min):
                out.append(wall)
                last = wall
        return out

    assert not full_checkpoint_due(119.9, 0.0, 2.0)
    assert full_checkpoint_due(120.0, 0.0, 2.0)
    walls = [50.0 * k for k in range(1, 13)]
    assert full(0.0, walls, 0.0) == []
    assert full(0.0, walls, 2.0) == [150.0, 300.0, 450.0, 600.0]
    assert full(330.0, [330.0 + 50.0 * k for k in range(1, 6)], 2.0) == [480.0]


def test_last_every_min_writes_full_files_at_periodic_checkpoints(tmp_path: Path) -> None:
    agent = run(tmp_path, ckpt_light=True, last_every_min=1e-9)
    assert steps(agent, "model_step_*.pt") == sorted({*steps(agent, "policy_step_*.pt"), agent._frames_processed})


class Writes:
    """Every ``atomic_save``, from the loop and from the writer's thread. The loop's
    write is slow, so a writer that reaches the same file meanwhile is caught."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.lock = threading.Lock()
        self.active: dict[Path, int] = {}
        self.overlaps: list[str] = []
        self.by: dict[str, set[str]] = {}
        real = speed_options.atomic_save

        def save(payload: dict[str, Any], target: Path) -> None:
            thread = "writer" if threading.current_thread().name.startswith("wm-checkpoint") else "loop"
            with self.lock:
                if self.active.get(target):
                    self.overlaps.append(target.name)
                self.active[target] = self.active.get(target, 0) + 1
                self.by.setdefault(target.name, set()).add(thread)
            try:
                time.sleep(0.05 if thread == "writer" else 0.5)
                real(payload, target)
            finally:
                with self.lock:
                    self.active[target] -= 1

        monkeypatch.setattr(speed_options, "atomic_save", save)
        monkeypatch.setattr(agent_module, "atomic_save", save)


def assert_readable_end(agent: Any, writes: Writes) -> None:
    assert writes.overlaps == []
    files = sorted(agent._checkpoint_dir().glob("model_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    assert files, "the run wrote no full file"
    for path in files:
        torch.load(path, weights_only=False)  # a corrupt file raises
    assert not list(agent._checkpoint_dir().glob("*.tmp"))
    resumed = harness.make_agent(ckpt_light=True, last_every_min=1e-9, save_interval=1)
    resumed.load_model(str(files[-1]))
    assert resumed._resume_frame_offset == agent._frames_processed


def test_the_end_file_waits_for_the_writer_at_a_frame_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The last iteration is the only periodic one, and a full file is due there:
    only the loop writes those frames, once."""
    writes = Writes(monkeypatch)
    agent = harness.make_agent(frames=FRAMES, save_interval=FRAMES, log_dir=tmp_path, ckpt_light=True, last_every_min=1e-9)
    agent.train()
    assert steps(agent, "model_step_*.pt") == [agent._frames_processed]
    assert writes.by[f"model_step_{agent._frames_processed}.pt"] == {"loop"}
    assert_readable_end(agent, writes)


def test_the_end_file_waits_for_the_writer_at_a_time_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writes = Writes(monkeypatch)
    agent = harness.make_agent(frames=10**9, save_interval=1, log_dir=tmp_path, ckpt_light=True, last_every_min=1e-9)
    agent._train_time_budget_reached = lambda: agent._iteration >= 4
    agent.train()
    assert agent._iteration == 4
    assert writes.by[f"model_step_{agent._frames_processed}.pt"] == {"loop"}
    assert_readable_end(agent, writes)


def test_an_interrupt_saves_after_the_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Interrupted just after a periodic full file was queued: the interrupt's
    save has the same frames, and starts once the writer is idle."""
    writes = Writes(monkeypatch)
    agent = harness.make_agent(frames=10**9, save_interval=1, log_dir=tmp_path, ckpt_light=True, last_every_min=1e-9)
    write = agent._write_light_checkpoint

    def interrupted(writer: CheckpointWriter) -> None:
        write(writer)
        if agent._iteration == 3:
            raise KeyboardInterrupt

    agent._write_light_checkpoint = interrupted
    with pytest.raises(KeyboardInterrupt):
        agent.train()
    assert writes.by[f"model_step_{agent._frames_processed}.pt"] == {"writer", "loop"}
    assert_readable_end(agent, writes)


def test_a_long_interval_writes_no_full_file_before_the_end(tmp_path: Path) -> None:
    agent = run(tmp_path, ckpt_light=True, last_every_min=60.0)
    assert steps(agent, "model_step_*.pt") == [agent._frames_processed]


def test_the_interval_counts_from_the_resume_point(tmp_path: Path) -> None:
    agent = harness.make_agent(frames=FRAMES, save_interval=EVERY, log_dir=tmp_path, ckpt_light=True, last_every_min=60.0)
    agent._train_seconds_offset = 7200.0  # a resumed run: not due at once
    agent.train()
    assert agent._last_full_seconds == pytest.approx(7200.0, abs=60.0)
    assert steps(agent, "model_step_*.pt") == [agent._frames_processed]


def test_a_background_full_file_resumes_bitwise(tmp_path: Path) -> None:
    """Written mid-run on the thread, it restores what it snapshotted, and the
    resumed agent steps as the original from there."""
    agent = harness.make_agent(frames=FRAMES, save_interval=EVERY, log_dir=tmp_path, ckpt_light=True, last_every_min=1e-9)
    snapshots: dict[str, Any] = {}
    write = agent._write_light_checkpoint

    def spy(writer: CheckpointWriter) -> None:
        write(writer)
        frames = agent._frames_processed
        snapshots[f"model_step_{frames}.pt"] = harness.digest(
            [agent.actor.state_dict(), agent.critic.state_dict(), agent.dynamics.state_dict(), agent.actor_optim.state_dict()]
        )

    agent._write_light_checkpoint = spy
    agent.train()
    assert len(snapshots) >= 2
    name, expected = next(iter(snapshots.items()))
    resumed = harness.make_agent(ckpt_light=True, last_every_min=1e-9, save_interval=EVERY)
    resumed.load_model(str(agent._checkpoint_dir() / name))
    got = [resumed.actor.state_dict(), resumed.critic.state_dict(), resumed.dynamics.state_dict(), resumed.actor_optim.state_dict()]
    assert harness.digest(got) == expected
    assert resumed._iteration < agent._iteration
    assert resumed._resume_frame_offset == int(name[len("model_step_") : -len(".pt")])


def test_a_snapshot_is_not_moved_by_later_updates() -> None:
    weight = torch.ones(3)
    nest = {"a": weight, "b": [weight, (weight, 2)], "c": "text"}
    snapshot = cpu_snapshot(nest)
    weight.add_(1.0)
    assert snapshot["a"].tolist() == [1.0, 1.0, 1.0]
    assert snapshot["b"][1][0].tolist() == [1.0, 1.0, 1.0]
    assert snapshot["b"][1][1] == 2
    assert snapshot["c"] == "text"


def test_the_writer_writes_atomically_and_in_order(tmp_path: Path) -> None:
    writer = CheckpointWriter()
    for i in range(4):
        writer.submit({"i": i}, tmp_path / "deep" / f"{i}.pt")
    writer.close()
    assert [torch.load(tmp_path / "deep" / f"{i}.pt")["i"] for i in range(4)] == [0, 1, 2, 3]
    assert not list((tmp_path / "deep").glob("*.tmp"))


def test_a_failed_write_raises_at_close(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("not a directory")
    writer = CheckpointWriter()
    writer.submit({"i": 0}, tmp_path / "file" / "a.pt")
    with pytest.raises(FileExistsError):
        writer.close()


def test_a_failed_write_raises_in_the_loop(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("not a directory")
    writer = CheckpointWriter()
    writer.submit({"i": 0}, tmp_path / "file" / "a.pt")
    deadline = time.monotonic() + 10
    while not writer._pending[0].done() and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(FileExistsError):
        writer.submit({"i": 1}, tmp_path / "b.pt")
    writer.close()


def test_a_failed_write_stops_the_training_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_payload: dict[str, Any], _target: Path) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(CheckpointWriter, "_write", staticmethod(fail))
    with pytest.raises(OSError, match="disk full"):
        run(tmp_path, ckpt_light=True)


def test_a_slow_disk_holds_the_queue_at_its_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def slow(_payload: dict[str, Any], _target: Path) -> None:
        release.wait(30)

    monkeypatch.setattr(CheckpointWriter, "_write", staticmethod(slow))
    writer = CheckpointWriter()
    limit = CheckpointWriter.MAX_QUEUED
    for i in range(limit):
        writer.submit({}, tmp_path / f"{i}.pt")
    blocked = threading.Thread(target=lambda: [writer.submit({}, tmp_path / f"{i}.pt") for i in range(limit, limit + 2)])
    blocked.start()
    blocked.join(0.3)
    assert blocked.is_alive(), "submit should wait for the queue"
    release.set()
    blocked.join(30)
    assert not blocked.is_alive()
    writer.close()


def test_off_keeps_the_synchronous_full_files(tmp_path: Path) -> None:
    agent = run(tmp_path)
    assert names(agent, "policy_step_*.pt") == []
    assert len(names(agent, "model_step_*.pt")) >= 3


@cuda
def test_light_checkpoints_on_the_gpu_score_on_a_default_agent(tmp_path: Path) -> None:
    agent = harness.make_agent(device="cuda:0", frames=FRAMES, save_interval=EVERY, log_dir=tmp_path, ckpt_light=True)
    agent.train()
    path = agent._checkpoint_dir() / names(agent, "policy_step_*.pt")[0]
    assert all(t.device.type == "cpu" for t in torch.load(path, weights_only=False)["policy_state_dict"].values())
    evaluator = harness.make_agent(device="cuda:0")
    evaluator.load_policy(str(path))
    assert next(evaluator.actor.parameters()).device.type == "cuda"
