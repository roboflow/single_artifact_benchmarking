from typing import Protocol, Self


class Monitor(Protocol):
    """Watches for throttling while it is open. Read the verdict after __exit__."""

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    def did_throttle(self) -> bool | None:
        """True or False when the signal was readable. None when this host gives no readable signal."""
        ...

    def summary(self) -> dict: ...


class NullMonitor:
    """For hosts and devices with no throttle signal. The verdict is always unknown."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def did_throttle(self) -> bool | None:
        return None

    def summary(self) -> dict:
        return {}
