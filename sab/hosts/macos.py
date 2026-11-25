import subprocess
from dataclasses import dataclass
from typing import Callable

from sab.hosts.base import safe
from sab.monitors.base import Monitor
from sab.monitors.macos import ThermalStateMonitor
from sab.runtimes.base import Runtime


def _read_sysctl(key: str) -> str | None:
    result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
    return result.stdout.strip() or None


@dataclass(frozen=True)
class MacHost:
    read_sysctl: Callable[[str], str | None] = _read_sysctl

    def describe(self) -> dict:
        chip = safe(self.read_sysctl, "machdep.cpu.brand_string")
        return {"cpu": chip, "accelerator": chip}

    def monitor(self, runtime: type[Runtime], device: str) -> Monitor:
        return ThermalStateMonitor()
