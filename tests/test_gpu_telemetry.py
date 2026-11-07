"""NVML telemetry: plausible readings on a GPU, nothing without one."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch

from rlopt import base_class
from rlopt.base_class import BaseAlgorithm
from rlopt.config_base import TrainerConfig
from rlopt.gpu_telemetry import GpuTelemetry


def test_cpu_device_reports_nothing() -> None:
    telemetry = GpuTelemetry("cpu")
    assert not telemetry.available
    assert telemetry.read() == {}


def test_missing_driver_library_reports_nothing(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def missing(name, *args, **kwargs):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", missing)
    assert GpuTelemetry("cuda:0").read() == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_device_reports_plausible_readings() -> None:
    metrics = GpuTelemetry(torch.device("cuda", 0)).read()
    if not metrics:
        pytest.skip("NVML unavailable")
    assert 100 <= metrics["perf/gpu_sm_clock_mhz"] <= 5000
    assert 0 < metrics["perf/gpu_temperature_c"] < 120
    assert 0 < metrics["perf/gpu_power_w"] < 2000
    assert metrics["perf/gpu_slowdown"] in (0.0, 1.0)


@pytest.mark.parametrize("enabled", [False, True])
def test_training_rows_carry_the_readings_only_when_asked(monkeypatch, enabled) -> None:
    class Reader:
        def __init__(self, device) -> None:
            del device

        def read(self) -> dict[str, float]:
            return {"perf/gpu_sm_clock_mhz": 1980.0}

    monkeypatch.setattr(base_class, "GpuTelemetry", Reader)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    agent = SimpleNamespace(
        config=SimpleNamespace(trainer=TrainerConfig(gpu_telemetry=enabled)),
        device=torch.device("cuda", 0),
        train_elapsed_seconds=lambda: 1.0,
    )
    metrics = BaseAlgorithm._train_clock_metrics(agent, 8)
    assert ("perf/gpu_sm_clock_mhz" in metrics) is enabled


def test_telemetry_is_off_by_default() -> None:
    assert TrainerConfig().gpu_telemetry is False


def test_power_cap_is_not_a_fault() -> None:
    from rlopt.gpu_telemetry import FAULT_REASONS_MASK, SLOWDOWN_REASONS

    assert not FAULT_REASONS_MASK & SLOWDOWN_REASONS["sw_power_cap"]
    assert FAULT_REASONS_MASK & SLOWDOWN_REASONS["sw_thermal"]
