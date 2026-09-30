import platform

from sab.monitors.base import Monitor, NullMonitor
from sab.monitors.cpufreq import CpufreqMonitor
from sab.monitors.nvidia import ThrottleMonitor
from sab.runtimes.base import Runtime

NVML_RUNTIMES = frozenset({"tensorrt", "onnxruntime"})


def select_monitor(runtime: type[Runtime], device: str, system: str | None = None) -> Monitor:
    """Pick the throttle monitor for a run. `system` is a platform.system() name, injectable for tests."""
    system = system or platform.system()
    if device == "gpu" and runtime.name in NVML_RUNTIMES:
        return ThrottleMonitor()
    if device == "cpu" and system == "Linux":
        return CpufreqMonitor()
    return NullMonitor()


__all__ = ["CpufreqMonitor", "Monitor", "NullMonitor", "ThrottleMonitor", "select_monitor"]
