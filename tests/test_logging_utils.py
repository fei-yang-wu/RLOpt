from __future__ import annotations

from pathlib import Path

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


def test_wandb_identity_uses_functional_name_and_logdir_tag(
    monkeypatch,
) -> None:
    monkeypatch.setenv("WANDB_TAGS", "bones-seed,seed0")
    run_dir = Path("2026-08-15_18-00-00_wandb-abc12345")

    run_name, tags = _build_wandb_identity("fsq64-hold10-s0", run_dir)

    assert run_name == "fsq64-hold10-s0"
    assert tags == ["bones-seed", "seed0", "logdir:2026-08-15_18-00-00_wandb-abc12345"]


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
    assert captured["wandb_kwargs"]["name"] == "fsq64-hold10-s0"
    assert captured["wandb_kwargs"]["tags"] == [
        "bones-seed",
        "seed0",
        "logdir:2026-08-15_18-00-00_wandb-abc12345",
    ]
