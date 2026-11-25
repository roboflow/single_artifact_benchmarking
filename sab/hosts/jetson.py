import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable

from sab.hosts.base import safe
from sab.hosts.linux import LinuxHost, board_name, read_device_tree_model
from sab.monitors.base import Monitor
from sab.monitors.cpufreq import CpufreqMonitor, FrequencyReader
from sab.monitors.jetson import read_gpu_frequencies
from sab.runtimes.base import Runtime

L4T_RELEASE_PATH = "/etc/nv_tegra_release"
L4T_RELEASE_PATTERN = re.compile(r"R(\d+) \(release\), REVISION: ([\d.]+)")
POWER_MODE_PATTERN = re.compile(r"NV Power Mode: (\S+)")


def l4t_release_from_nv_tegra_release(text: str) -> str | None:
    match = L4T_RELEASE_PATTERN.search(text)
    return f"R{match.group(1)}.{match.group(2)}" if match else None


def power_mode_from_nvpmodel(output: str) -> str | None:
    match = POWER_MODE_PATTERN.search(output)
    return match.group(1) if match else None


def _read_l4t_release() -> str:
    with open(L4T_RELEASE_PATH) as f:
        return f.read()


def _read_power_mode() -> str:
    return subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, check=True, timeout=5).stdout


def on_jetson(
    read_board: Callable[[], str] = read_device_tree_model,
    read_l4t_release: Callable[[], str] = _read_l4t_release,
) -> bool:
    """True on a Jetson: the board name says so, or the L4T release file exists.

    A container hides the device tree, but the NVIDIA container runtime mounts the L4T release file.
    """
    board = board_name(read_board)
    if board is not None and "Jetson" in board:
        return True
    l4t_release = safe(read_l4t_release)
    return bool(l4t_release) and l4t_release_from_nv_tegra_release(l4t_release) is not None


@dataclass(frozen=True)
class JetsonHost:
    linux: LinuxHost = field(default_factory=LinuxHost)
    read_l4t_release: Callable[[], str] = _read_l4t_release
    read_power_mode: Callable[[], str] = _read_power_mode
    read_gpu_frequencies: FrequencyReader = read_gpu_frequencies

    def describe(self) -> dict:
        l4t_release = safe(self.read_l4t_release)
        power_mode = safe(self.read_power_mode)
        return {
            **self.linux.describe(),
            "l4t": l4t_release_from_nv_tegra_release(l4t_release) if l4t_release else None,
            "power_mode": power_mode_from_nvpmodel(power_mode) if power_mode else None,
        }

    def accelerator(self, device: str) -> str | None:
        return self.linux.accelerator(device)

    def monitor(self, runtime: type[Runtime], device: str) -> Monitor:
        if device == "gpu":
            return CpufreqMonitor(read_frequencies=self.read_gpu_frequencies)
        return self.linux.monitor(runtime, device)
