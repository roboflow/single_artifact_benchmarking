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

    def _chip(self) -> str | None:
        return safe(self.read_sysctl, "machdep.cpu.brand_string")

    def describe(self) -> dict:
        return {"cpu": self._chip()}

    def accelerator(self, device: str) -> str | None:
        # The GPU and the Neural Engine are part of the Apple silicon chip.
        return self._chip() if device in ("gpu", "npu") else None

    def monitor(self, runtime: type[Runtime], device: str) -> Monitor:
        return ThermalStateMonitor()
