import platform
from typing import Callable

from sab.hosts.base import Host, safe
from sab.hosts.linux import LinuxHost
from sab.hosts.macos import MacHost
from sab.monitors.base import Monitor
from sab.runtimes.base import Runtime


def detect_host(system: str | None = None) -> Host:
    """The platform of this machine. `system` is a platform.system() name, injectable for tests.

    An OS with no platform of its own gets LinuxHost. Its Linux-only sources give no CPU name and no CPU throttle verdict there.
    """
    system = system or platform.system()
    if system == "Darwin":
        return MacHost()
    return LinuxHost()


def select_monitor(runtime: type[Runtime], device: str, host: Host | None = None) -> Monitor:
    return (host or detect_host()).monitor(runtime, device)


def _system() -> tuple[str, str]:
    return platform.system(), platform.release()


def host_info(
    runtime: type[Runtime],
    device: str,
    *,
    host: Host | None = None,
    system: Callable[[], tuple[str, str]] = _system,
    machine: Callable[[], str] = platform.machine,
) -> dict:
    """Describe the host of a `device` row for the results file. A reader that fails gives None, never an error."""
    os_name, os_release = safe(system) or (None, None)
    host = host or detect_host(os_name)
    return {
        "os": f"{os_name} {os_release}" if os_name else None,
        "arch": safe(machine),
        **host.describe(),
        "accelerator": host.accelerator(device),
        "runtime_version": safe(runtime.version),
    }


__all__ = ["Host", "LinuxHost", "MacHost", "detect_host", "host_info", "select_monitor"]
