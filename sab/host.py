import platform
import subprocess
from typing import Callable

from sab.runtimes.base import Runtime

CPUINFO_NAME_KEYS = ("model name", "Model")


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


def _read_sysctl(key: str) -> str | None:
    result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
    return result.stdout.strip() or None


def _read_cuda_gpu_name() -> str | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _system() -> tuple[str, str]:
    return platform.system(), platform.release()


def _safe(read: Callable, *args):
    try:
        return read(*args)
    except Exception:
        return None


def host_info(
    runtime: type[Runtime],
    *,
    system: Callable[[], tuple[str, str]] = _system,
    machine: Callable[[], str] = platform.machine,
    read_cpuinfo: Callable[[], str] = _read_cpuinfo,
    read_sysctl: Callable[[str], str | None] = _read_sysctl,
    read_gpu_name: Callable[[], str | None] = _read_cuda_gpu_name,
) -> dict:
    """Describe the host for the results file. A reader that fails gives None, never an error."""
    os_name, os_release = _safe(system) or (None, None)
    on_macos = os_name == "Darwin"

    if on_macos:
        cpu = _safe(read_sysctl, "machdep.cpu.brand_string")
        accelerator = cpu
    else:
        cpuinfo = _safe(read_cpuinfo)
        cpu = cpu_name_from_cpuinfo(cpuinfo) if cpuinfo else None
        accelerator = _safe(read_gpu_name)

    return {
        "os": f"{os_name} {os_release}" if os_name else None,
        "arch": _safe(machine),
        "cpu": cpu,
        "accelerator": accelerator,
        "runtime_version": _safe(runtime.version),
    }
