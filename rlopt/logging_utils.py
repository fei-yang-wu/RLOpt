from __future__ import annotations

import logging
import os
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

from rlopt.config_base import RLOptConfig

try:  # pragma: no cover - optional dependency
    from rich.logging import RichHandler
except Exception:  # pragma: no cover
    RichHandler = None  # type: ignore[assignment]


__all__ = [
    "ROOT_LOGGER_NAME",
    "LoggingManager",
    "MetricReporter",
    "log_to_file_only",
    "resolve_log_level",
]


ROOT_LOGGER_NAME = "rlopt"
_CONSOLE_HANDLER_NAME = "rlopt.console"
_FILE_HANDLER_NAME = "rlopt.file"
_TIMESTAMP_DIR_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
    r"(?:_[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)


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


# Name of the frame counter published as an ordinary metric alongside every
# scalar, and declared as the wandb x-axis. `MetricReporter.log_scalars` is
# always called with `metadata.frames_processed`, so this is the cumulative
# environment-frame count, continuous across a walltime-segmented resume.
#
# It exists because wandb's SHARED mode discards `wandb.log(step=...)`
# entirely ("In shared mode, the use of `wandb.log` with the step argument is
# not supported and will be ignored", wandb/sdk/wandb_run.py). Without this a
# shared-mode run plots against a bare log-call index, which is not comparable
# with a non-shared run and does not survive a resume offset.
_STEP_METRIC = "env_frames"


def _wandb_is_shared(experiment: Any) -> bool:
    """True when this wandb run was opened in shared (multi-writer) mode."""

    settings = getattr(experiment, "settings", None)
    return bool(getattr(settings, "_shared", False))


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


def _format_metric_for_console(value: float | int) -> str:
    """Render scalar metrics for console logs with 3 significant digits."""

    return f"{float(value):.3g}"


def log_to_file_only(logger: PyLogger, level: int, message: str) -> None:
    """Emit one log record only through RLOpt's file handler."""

    record = logger.makeRecord(
        logger.name,
        level,
        fn="",
        lno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    current: PyLogger | None = logger
    while current is not None:
        for handler in current.handlers:
            if handler.get_name() != _FILE_HANDLER_NAME:
                continue
            if record.levelno >= handler.level:
                handler.handle(record)
        if not current.propagate:
            break
        parent = current.parent
        current = parent if isinstance(parent, logging.Logger) else None


def _slugify(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return fallback
    text = text.lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-_.")
    return text or fallback


def _looks_like_run_dir(path: Path) -> bool:
    """Return True for a timestamped run directory with an optional unique suffix."""
    return bool(_TIMESTAMP_DIR_PATTERN.fullmatch(path.name))


def _run_dir_timestamp(run_dir: Path) -> str | None:
    """The ``YYYY-MM-DD_HH-MM-SS`` prefix of a run directory, when it has one."""
    match = _TIMESTAMP_DIR_PATTERN.fullmatch(run_dir.name)
    if not match:
        return None
    return run_dir.name[:19]


def _build_wandb_identity(exp_name: Any, run_dir: Path) -> tuple[str, list[str]]:
    """Return the W&B run name and tags for one training run.

    The W&B name is the configured functional name followed by the run
    directory's launch timestamp, for example
    ``fsq64-hold10-s0-2026-08-18_16-52-30``. The timestamp makes two launches
    of one arm distinguishable in a W&B list, which the functional name alone
    is not. A directory without a timestamp prefix (an explicit ``log_dir``
    that is already a run directory) keeps the bare name.

    Deliberately the NAME and not the run id: RLOpt also tags the run
    ``logdir:<19-char timestamp>_wandb-<run id>``, and W&B caps a tag at 64
    characters, which leaves 31 for the id. Appending 20 more characters there
    overflows the cap and the run creation fails (job 5580199).

    On a walltime-segmented chain every segment resumes one run id and passes
    its own name, so the displayed name tracks the most recent segment W&B
    accepted a name from; the per-segment ``logdir:`` tags accumulate, so the
    full chain remains traceable to every local run directory.
    """
    tags = [
        tag.strip()
        for tag in os.environ.get("WANDB_TAGS", "").split(",")
        if tag.strip()
    ]
    tags.append(f"logdir:{run_dir.name}")
    timestamp = _run_dir_timestamp(run_dir)
    name = f"{exp_name}-{timestamp}" if timestamp else str(exp_name)
    return name, tags


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
        self._step_metric_declared = False

    def _declare_step_metric(self, experiment: Any) -> None:
        """Make :data:`_STEP_METRIC` the wandb x-axis. Idempotent, once per run."""

        if self._step_metric_declared:
            return
        self._step_metric_declared = True
        define_metric = getattr(experiment, "define_metric", None)
        if not callable(define_metric):
            return
        try:
            define_metric(_STEP_METRIC)
            define_metric("*", step_metric=_STEP_METRIC)
        except Exception:  # pragma: no cover - defensive
            if self._python_logger is not None:
                self._python_logger.debug(
                    "Could not declare %s as the wandb x-axis.", _STEP_METRIC
                )

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
            # wandb >= 0.18 buffers data indefinitely when log_scalar calls
            # experiment.log with commit=False.  Batch all metrics into one
            # experiment.log call with commit=True so every step is committed
            # immediately.  TensorBoard writers expose add_scalar (not log), so
            # the check here safely identifies wandb-backed loggers only.
            experiment = getattr(self._metrics_logger, "experiment", None)
            if (
                experiment is not None
                and sanitized
                and callable(getattr(experiment, "log", None))
                and not callable(getattr(experiment, "add_scalar", None))
            ):
                self._declare_step_metric(experiment)
                # The step travels as a metric so it survives shared mode,
                # where the `step` argument is ignored. Passing `step` as well
                # in shared mode only produces a warning, so it is dropped.
                payload: dict[str, float | int] = dict(sanitized)
                payload[_STEP_METRIC] = record_step
                if _wandb_is_shared(experiment):
                    experiment.log(payload, commit=True)
                else:
                    experiment.log(payload, step=record_step, commit=True)
            else:
                for key, value in sanitized.items():
                    self._metrics_logger.log_scalar(key, value, record_step)

        if log_python and self._python_logger is not None:
            message = " | ".join(
                f"{k}={_format_metric_for_console(v)}" for k, v in sanitized.items()
            )
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

        # Create hierarchical structure: {base}/{algorithm}/{task}/{timestamp}.
        # If a caller already provides a timestamped run directory (common when an
        # outer training script manages run folders), reuse it directly.
        if _looks_like_run_dir(base_dir):
            self.run_dir = base_dir
        else:
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

        # ``generate_exp_name`` is only the TorchRL-internal experiment id.
        # The visible W&B name is set below from ``logger.exp_name`` so it stays
        # stable when a walltime-segmented chain resumes the same W&B run.
        if os.environ.get("WANDB_RUN_ID"):
            exp_name = f"{self.component.upper()}_{self._config.logger.exp_name}"
        else:
            exp_name = generate_exp_name(
                self.component.upper(), f"{self._config.logger.exp_name}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        wandb_kwargs = {
            "project": self._config.logger.project_name,
            "entity": self._config.logger.entity,
            "group": self._config.logger.group_name,
            "dir": str(self.run_dir),
        }
        if backend == "wandb":
            run_name, run_tags = _build_wandb_identity(
                self._config.logger.exp_name, self.run_dir
            )
            wandb_kwargs["name"] = run_name
            wandb_kwargs["tags"] = run_tags
        # Walltime-segmented chains: WANDB_RUN_ID (stable per arm) plus
        # WANDB_RESUME=allow makes every segment append to ONE W&B run instead
        # of opening a fresh run that restarts the x-axis. Safe because resumed
        # segments log at the global frame step (init_metadata seeds
        # frames_processed from the checkpoint), so steps stay monotonic across
        # segments. Left entirely to the environment so interactive runs keep
        # today's one-run-per-launch behavior.
        run_id = os.environ.get("WANDB_RUN_ID")
        if run_id:
            wandb_kwargs["id"] = run_id
            wandb_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")

        try:  # noqa: SIM105
            wandb_kwargs["config"] = asdict(self._config)  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive
            pass

        # TorchRL's get_logger routes the file-backed loggers to ``logger_name``
        # and ignores the ``log_dir`` kwarg entirely, so passing the component
        # name there scattered scalars into a ``./csv``-style directory beside
        # the process CWD instead of the configured run directory. Only wandb
        # reads log_dir/wandb_kwargs, so leave its argument alone.
        file_backed = backend in ("csv", "tensorboard")
        return get_logger(
            backend,
            logger_name=str(self.run_dir) if file_backed else self.component.lower(),
            experiment_name=exp_name,
            log_dir=str(self.run_dir),
            wandb_kwargs=wandb_kwargs,
        )
