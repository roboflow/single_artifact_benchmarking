from dataclasses import dataclass
from typing import Callable

from sab.hosts.base import safe
from sab.monitors.base import Monitor, NullMonitor
from sab.monitors.cpufreq import CpufreqMonitor
from sab.monitors.nvidia import ThrottleMonitor
from sab.runtimes.base import Runtime

CPUINFO_NAME_KEYS = ("model name", "Model")
NVML_RUNTIMES = frozenset({"tensorrt", "onnxruntime"})


def cpu_name_from_cpuinfo(cpuinfo: str) -> str | None:
    """Read the CPU name from /proc/cpuinfo: `model name` on x86, `Model` on Raspberry Pi."""
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in CPUINFO_NAME_KEYS:
            return value.strip()
    return None


def _read_cpuinfo() -> str:
    with open("/proc/cpuinfo") as f:
        return f.read()


def _read_cuda_gpu_name() -> str | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


@dataclass(frozen=True)
class LinuxHost:
    read_cpuinfo: Callable[[], str] = _read_cpuinfo
    read_gpu_name: Callable[[], str | None] = _read_cuda_gpu_name

    def describe(self) -> dict:
        cpuinfo = safe(self.read_cpuinfo)
        return {"cpu": cpu_name_from_cpuinfo(cpuinfo) if cpuinfo else None}

    def accelerator(self, device: str) -> str | None:
        return safe(self.read_gpu_name) if device == "gpu" else None

    def monitor(self, runtime: type[Runtime], device: str) -> Monitor:
        if device == "gpu" and runtime.name in NVML_RUNTIMES:
            return ThrottleMonitor()
        if device == "cpu":
            return CpufreqMonitor()
        return NullMonitor()
