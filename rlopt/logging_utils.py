from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import datetime, timedelta
from logging import Logger as PyLogger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.record.loggers.common import Logger

from rlopt.configs import RLOptConfig

try:  # pragma: no cover - optional dependency
    from rich.logging import RichHandler
except Exception:  # pragma: no cover
    RichHandler = None  # type: ignore[assignment]


__all__ = [
    "ROOT_LOGGER_NAME",
    "LoggingManager",
    "MetricReporter",
    "resolve_log_level",
]


ROOT_LOGGER_NAME = "rlopt"
_CONSOLE_HANDLER_NAME = "rlopt.console"
_FILE_HANDLER_NAME = "rlopt.file"


def resolve_log_level(level: str | int | None, *, default: int = logging.INFO) -> int:
    """Convert configuration-provided levels to logging module constants."""

    if level is None:
        return default

    if isinstance(level, int):
        return level

    numeric = logging.getLevelName(level.upper())
    if isinstance(numeric, int):
        return numeric
    return default


def _coerce_step(step: Any) -> int:
    """Best-effort conversion of a training step to an ``int``."""

    if step is None:
        return 0

    if isinstance(step, int | np.integer):
        return int(step)

    if isinstance(step, float):
        return int(step)

    if isinstance(step, np.floating):
        return int(float(step))

    if torch.is_tensor(step):
        if step.numel() == 1:
            return int(step.detach().cpu().item())
        return 0

    if hasattr(step, "__int__"):
        try:
            return int(step)
        except Exception:  # pragma: no cover - defensive
            return 0

    return 0


def _flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric mappings using ``/`` as separator."""

    flat: dict[str, Any] = {}

    def _inner(items: Iterable[tuple[str, Any]], base: str) -> None:
        for key, value in items:
            key_str = str(key)
            full_key = f"{base}/{key_str}" if base else key_str
            if isinstance(value, Mapping):
                _inner(value.items(), full_key)
            else:
                flat[full_key] = value

    _inner(metrics.items(), prefix)
    return flat


def _coerce_metric_value(value: Any) -> float | int | None:
    """Convert tensors/arrays/np scalars into Python scalars when possible."""

    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int | float):
        return float(value)

    if isinstance(value, np.integer | np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return None

    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return None

    if isinstance(value, timedelta):
        return value.total_seconds()

    if hasattr(value, "item"):
        try:
            maybe_scalar = value.item()  # type: ignore[misc]
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(maybe_scalar, int | float | bool):
            return float(maybe_scalar)
        return None

    return None


def _slugify(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return fallback
    text = text.lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-_.")
    return text or fallback


def _build_console_handler(cfg: RLOptConfig, level: int) -> logging.Handler | None:
    if not cfg.logger.log_to_console:
        return None

    handler: logging.Handler | None = None
    if cfg.logger.console_use_rich and RichHandler is not None:  # pragma: no branch
        handler = RichHandler(  # type: ignore[call-arg]
            rich_tracebacks=True,
            markup=False,
            show_time=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(cfg.logger.console_format))

    handler.setLevel(level)
    handler.set_name(_CONSOLE_HANDLER_NAME)
    return handler


def _build_file_handler(
    cfg: RLOptConfig, level: int, run_dir: Path
) -> logging.Handler | None:
    if not cfg.logger.log_to_file:
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    file_path = run_dir / cfg.logger.file_name

    if cfg.logger.file_rotation_bytes and cfg.logger.file_rotation_bytes > 0:
        handler: logging.Handler = RotatingFileHandler(
            file_path,
            maxBytes=cfg.logger.file_rotation_bytes,
            backupCount=max(cfg.logger.file_backup_count, 0),
        )
    else:
        handler = logging.FileHandler(file_path)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(cfg.logger.file_format))
    handler.set_name(_FILE_HANDLER_NAME)
    return handler


def _configure_console_handler(
    root_logger: PyLogger, cfg: RLOptConfig, level: int
) -> None:
    existing = next(
        (h for h in root_logger.handlers if h.get_name() == _CONSOLE_HANDLER_NAME),
        None,
    )
    if cfg.logger.log_to_console:
        if existing is None:
            handler = _build_console_handler(cfg, level)
            if handler is not None:
                root_logger.addHandler(handler)
        else:
            existing.setLevel(level)
    elif existing is not None:
        root_logger.removeHandler(existing)
        try:  # noqa: SIM105
            existing.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _configure_file_handler(
    root_logger: PyLogger, cfg: RLOptConfig, level: int, run_dir: Path
) -> None:
    existing = next(
        (h for h in root_logger.handlers if h.get_name() == _FILE_HANDLER_NAME), None
    )

    if not cfg.logger.log_to_file:
        if existing is not None:
            root_logger.removeHandler(existing)
            try:  # noqa: SIM105
                existing.close()
            except Exception:  # pragma: no cover - defensive
                pass
        return

    target_path = (run_dir / cfg.logger.file_name).resolve()

    if existing is not None:
        current_path = Path(getattr(existing, "baseFilename", "")).resolve()
        if current_path != target_path:
            root_logger.removeHandler(existing)
            try:  # noqa: SIM105
                existing.close()
            except Exception:  # pragma: no cover - defensive
                pass
            existing = None

    if existing is None:
        handler = _build_file_handler(cfg, level, run_dir)
        if handler is not None:
            root_logger.addHandler(handler)


class MetricReporter:
    """Thin helper that streams metrics to TorchRL loggers and optional python logs."""

    def __init__(
        self, metrics_logger: Logger | None, python_logger: PyLogger | None
    ) -> None:
        self._metrics_logger = metrics_logger
        self._python_logger = python_logger

    def log_scalars(
        self,
        metrics: Mapping[str, Any],
        *,
        step: Any,
        log_python: bool = False,
        python_level: int = logging.INFO,
    ) -> None:
        if not metrics:
            return

        flattened = _flatten_metrics(metrics)
        sanitized: dict[str, float | int] = {}
        skipped: dict[str, Any] = {}

        for key, value in flattened.items():
            converted = _coerce_metric_value(value)
            if converted is None:
                skipped[key] = value
                continue
            sanitized[key] = converted

        if not sanitized and skipped:
            if self._python_logger is not None:
                skipped_str = ", ".join(
                    f"{k} ({type(v).__name__})" for k, v in skipped.items()
                )
                self._python_logger.debug(
                    "Skipping non-scalar metrics: %s",
                    skipped_str,
                )
            return

        record_step = _coerce_step(step)

        if self._metrics_logger is not None:
            for key, value in sanitized.items():
                self._metrics_logger.log_scalar(key, value, record_step)

        if log_python and self._python_logger is not None:
            message = ", ".join(f"{k}={v}" for k, v in sanitized.items())
            if message:
                if step is not None:
                    message = f"step={record_step} | {message}"
                self._python_logger.log(python_level, message)

        if skipped and self._python_logger is not None:
            skipped_str = ", ".join(
                f"{k} ({type(v).__name__})" for k, v in skipped.items()
            )
            self._python_logger.debug(
                "Metrics were skipped because they are not scalar: %s",
                skipped_str,
            )


class LoggingManager:
    """Centralised control for Python logging and TorchRL metric loggers.

    Creates structured logging directories:
        {log_dir}/{algorithm_name}/{env_or_task_name}/{date_time}/

    Example:
        ./logs/SAC/Pendulum-v1/2025-10-27_19-49-59/
        ./logs/IPMD/UnitreeG1/2025-10-27_20-15-30/

    All logs, metrics, and model checkpoints are saved in the run directory.
    """

    def __init__(
        self,
        *,
        config: RLOptConfig,
        component: str,
        metrics_logger: Logger | None = None,
    ) -> None:
        self._config = config
        self.component = component

        level = resolve_log_level(
            config.logger.python_level or config.log_level,
            default=logging.WARNING,
        )

        # Resolve base directory
        base_dir = Path(config.logger.log_dir).expanduser()
        if not base_dir.is_absolute():
            base_dir = Path.cwd() / base_dir

        # Create hierarchical structure: {base}/{algorithm}/{task}/{timestamp}
        algo_slug = _slugify(component, "algorithm")
        task_name = getattr(config.env, "env_name", None)
        task_slug = _slugify(task_name, "task")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.run_dir = base_dir / algo_slug / task_slug / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        root_logger = logging.getLogger(ROOT_LOGGER_NAME)
        root_logger.setLevel(level)
        # Prevent logs from bubbling up to the Python root logger, which can
        # cause duplicate console outputs when other libraries call basicConfig
        # or attach their own root handlers.
        root_logger.propagate = False
        _configure_console_handler(root_logger, config, level)
        _configure_file_handler(root_logger, config, level, self.run_dir)
        if not root_logger.handlers:
            null_handler = logging.NullHandler()
            null_handler.set_name("rlopt.null")
            root_logger.addHandler(null_handler)

        # Create the component logger as a child of ``rlopt`` root.
        self.logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.{component}")
        self.logger.setLevel(level)

        # Metrics logger setup
        self.metrics_logger = metrics_logger or self._build_metrics_logger()
        self.metric_reporter = MetricReporter(self.metrics_logger, self.logger)
        self.metrics = self.metric_reporter
        self.video_enabled = bool(config.logger.video)

    def _build_metrics_logger(self) -> Logger | None:
        backend = self._config.logger.backend
        if backend in (None, ""):
            return None

        exp_name = generate_exp_name(
            self.component.upper(), f"{self._config.logger.exp_name}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        wandb_kwargs = {
            "project": self._config.logger.project_name,
            "entity": self._config.logger.entity,
            "group": self._config.logger.group_name,
        }

        try:  # noqa: SIM105
            wandb_kwargs["config"] = asdict(self._config)  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive
            pass

        return get_logger(
            backend,
            logger_name=self.component.lower(),
            experiment_name=exp_name,
            log_dir=str(self.run_dir),
            wandb_kwargs=wandb_kwargs,
        )
