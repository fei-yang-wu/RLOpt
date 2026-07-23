from __future__ import annotations

from pathlib import Path

from rlopt.logging_utils import _looks_like_run_dir


def test_timestamped_run_directory_accepts_unique_suffix() -> None:
    assert _looks_like_run_dir(Path("2026-07-22_15-42-35"))
    assert _looks_like_run_dir(Path("2026-07-22_15-42-35_wandb-6vibr8i3"))
    assert _looks_like_run_dir(Path("2026-07-22_15-42-35_slurm-5526702"))


def test_non_run_directory_is_rejected() -> None:
    assert not _looks_like_run_dir(Path("logs"))
    assert not _looks_like_run_dir(Path("2026-07-22"))
    assert not _looks_like_run_dir(Path("2026-07-22_15-42-35_bad suffix"))
