"""GPU clock, temperature and power read through NVML (ctypes, no pynvml).

A throttled GPU slows every phase of an iteration, so a speed number is only
interpretable beside the clocks it ran at (2026-10-24: three ICE H200s ran at
SM 345-1260 MHz with the thermal-slowdown flag set). The reader is fail-soft:
without the driver library, or for a device NVML cannot resolve, it returns
no metrics.
"""

from __future__ import annotations

import ctypes

import torch

_NVML_CLOCK_SM = 1
_NVML_TEMPERATURE_GPU = 0
# nvmlClocksEventReasons bits. The software power cap (0x04) is what a healthy
# GPU shows at full load (every sample of a 2-H200 run, 2026-10-24); heat and
# hardware slowdowns are faults.
SLOWDOWN_REASONS = {
    "sw_power_cap": 0x04,
    "hw_slowdown": 0x08,
    "sw_thermal": 0x20,
    "hw_thermal": 0x40,
    "hw_power_brake": 0x80,
}
FAULT_REASONS_MASK = (
    SLOWDOWN_REASONS["hw_slowdown"]
    | SLOWDOWN_REASONS["sw_thermal"]
    | SLOWDOWN_REASONS["hw_thermal"]
    | SLOWDOWN_REASONS["hw_power_brake"]
)


class GpuTelemetry:
    """Reads one CUDA device's SM clock, temperature, power and slowdown flags."""

    def __init__(self, device: torch.device | str | int) -> None:
        self._lib = None
        self._handle = ctypes.c_void_p()
        try:
            device = torch.device(device)
            if device.type != "cuda":
                return
            index = torch.cuda.current_device() if device.index is None else device.index
            uuid = str(torch.cuda.get_device_properties(index).uuid)
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return
            key = uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"
            if lib.nvmlDeviceGetHandleByUUID(key.encode(), ctypes.byref(self._handle)):
                return
            self._reasons = getattr(
                lib,
                "nvmlDeviceGetCurrentClocksEventReasons",
                getattr(lib, "nvmlDeviceGetCurrentClocksThrottleReasons", None),
            )
            self._lib = lib
        except (OSError, RuntimeError, AttributeError, AssertionError):
            self._lib = None

    @property
    def available(self) -> bool:
        return self._lib is not None

    def read(self) -> dict[str, float]:
        """Current readings as ``perf/gpu_*`` metrics; empty when unavailable."""
        if self._lib is None:
            return {}
        lib, handle = self._lib, self._handle
        clock, temperature, power = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        metrics: dict[str, float] = {}
        if lib.nvmlDeviceGetClockInfo(handle, _NVML_CLOCK_SM, ctypes.byref(clock)) == 0:
            metrics["perf/gpu_sm_clock_mhz"] = float(clock.value)
        if (
            lib.nvmlDeviceGetTemperature(
                handle, _NVML_TEMPERATURE_GPU, ctypes.byref(temperature)
            )
            == 0
        ):
            metrics["perf/gpu_temperature_c"] = float(temperature.value)
        if lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power)) == 0:
            metrics["perf/gpu_power_w"] = power.value / 1000.0
        if self._reasons is not None:
            reasons = ctypes.c_ulonglong()
            if self._reasons(handle, ctypes.byref(reasons)) == 0:
                metrics["perf/gpu_slowdown"] = float(
                    bool(reasons.value & FAULT_REASONS_MASK)
                )
                metrics["perf/gpu_clock_event_reasons"] = float(reasons.value)
        return metrics
