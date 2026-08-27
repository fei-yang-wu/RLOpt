from __future__ import annotations

from pathlib import Path

from rlopt import logging_utils
from rlopt.config_base import RLOptConfig
from rlopt.logging_utils import (
    LoggingManager,
    _build_wandb_identity,
    _looks_like_run_dir,
)


def test_timestamped_run_directory_accepts_unique_suffix() -> None:
    assert _looks_like_run_dir(Path("2026-07-22_15-42-35"))
    assert _looks_like_run_dir(Path("2026-07-22_15-42-35_wandb-6vibr8i3"))
    assert _looks_like_run_dir(Path("2026-07-22_15-42-35_slurm-5526702"))


def test_non_run_directory_is_rejected() -> None:
    assert not _looks_like_run_dir(Path("logs"))
    assert not _looks_like_run_dir(Path("2026-07-22"))
    assert not _looks_like_run_dir(Path("2026-07-22_15-42-35_bad suffix"))


def test_wandb_identity_appends_launch_timestamp_and_keeps_logdir_tag(
    monkeypatch,
) -> None:
    monkeypatch.setenv("WANDB_TAGS", "bones-seed,seed0")
    run_dir = Path("2026-08-15_18-00-00_wandb-abc12345")

    run_name, tags = _build_wandb_identity("fsq64-hold10-s0", run_dir)

    assert run_name == "fsq64-hold10-s0-2026-08-15_18-00-00"
    assert tags == ["bones-seed", "seed0", "logdir:2026-08-15_18-00-00_wandb-abc12345"]


def test_wandb_identity_keeps_bare_name_without_a_timestamped_dir() -> None:
    run_name, _tags = _build_wandb_identity("fsq64-hold10-s0", Path("custom-run"))

    assert run_name == "fsq64-hold10-s0"


def test_wandb_metrics_logger_pins_functional_name_and_tags(
    monkeypatch, tmp_path
) -> None:
    run_dir = tmp_path / "2026-08-15_18-00-00_wandb-abc12345"
    run_dir.mkdir()
    config = RLOptConfig()
    config.logger.backend = "wandb"
    config.logger.project_name = "g1-bones-seed"
    config.logger.group_name = "latent-quant-repeats"
    config.logger.exp_name = "fsq64-hold10-s0"
    config.logger.log_dir = str(run_dir)
    config.logger.log_to_file = False
    monkeypatch.setenv("WANDB_TAGS", "bones-seed,seed0")

    captured = {}

    def fake_get_logger(backend, logger_name, experiment_name, **kwargs):  # noqa: ARG001
        captured["backend"] = backend
        captured["wandb_kwargs"] = kwargs["wandb_kwargs"]
        return object()

    monkeypatch.setattr("rlopt.logging_utils.get_logger", fake_get_logger)

    LoggingManager(config=config, component="IPMD")

    assert captured["backend"] == "wandb"
    assert captured["wandb_kwargs"]["name"] == "fsq64-hold10-s0-2026-08-15_18-00-00"
    assert captured["wandb_kwargs"]["tags"] == [
        "bones-seed",
        "seed0",
        "logdir:2026-08-15_18-00-00_wandb-abc12345",
    ]


class _FakeWandbSettings:
    def __init__(self, shared: bool) -> None:
        self._shared = shared


class _FakeWandbRun:
    """Minimal wandb-run double: has ``log``/``define_metric``, no ``add_scalar``."""

    def __init__(self, *, shared: bool) -> None:
        self.settings = _FakeWandbSettings(shared)
        self.defined: list[tuple[str, str | None]] = []
        self.calls: list[tuple[dict, int | None]] = []

    def define_metric(self, name, step_metric=None):
        self.defined.append((name, step_metric))

    def log(self, data, step=None, commit=True):  # noqa: ARG002
        self.calls.append((dict(data), step))


class _FakeMetricsLogger:
    def __init__(self, experiment) -> None:
        self.experiment = experiment


def _report(shared: bool) -> _FakeWandbRun:
    run = _FakeWandbRun(shared=shared)
    reporter = logging_utils.MetricReporter(_FakeMetricsLogger(run), None)
    reporter.log_scalars({"episode": {"length": 120.5}}, step=2_500_067_328)
    reporter.log_scalars({"episode": {"length": 130.5}}, step=2_900_000_000)
    return run


def test_frame_step_is_logged_as_a_metric_and_declared_as_the_x_axis() -> None:
    run = _report(shared=False)

    assert run.defined == [("env_frames", None), ("*", "env_frames")]
    assert [call[0]["env_frames"] for call in run.calls] == [
        2_500_067_328,
        2_900_000_000,
    ]
    # Non-shared runs keep the native wandb step as well.
    assert [call[1] for call in run.calls] == [2_500_067_328, 2_900_000_000]


def test_shared_mode_drops_the_ignored_step_argument_but_keeps_the_frame_metric() -> (
    None
):
    run = _report(shared=True)

    # wandb ignores `step` in shared mode and warns; the frame axis survives
    # only because it also travels as an ordinary metric.
    assert [call[1] for call in run.calls] == [None, None]
    assert [call[0]["env_frames"] for call in run.calls] == [
        2_500_067_328,
        2_900_000_000,
    ]
