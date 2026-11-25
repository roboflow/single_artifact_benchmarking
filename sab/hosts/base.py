from typing import Callable, Protocol

from sab.monitors.base import Monitor
from sab.runtimes.base import Runtime


class Host(Protocol):
    """One hardware platform. It knows its own description and its own throttle signals."""

    def describe(self) -> dict:
        """The platform fields of the host record. A reader that fails gives None, never an error."""
        ...

    def accelerator(self, device: str) -> str | None:
        """The hardware that runs a `device` row. A failed read gives None, never an error.

        Also None for a CPU row, because `cpu` already names it, and for a device with no reader on this platform.
        """
        ...

    def monitor(self, runtime: type[Runtime], device: str) -> Monitor:
        """The throttle monitor for a run of `runtime` on `device`."""
        ...


def safe(read: Callable, *args):
    try:
        return read(*args)
    except Exception:
        return None
