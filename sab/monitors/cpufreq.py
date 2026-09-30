import threading
from collections.abc import Iterator
from contextlib import contextmanager
from glob import glob
from typing import Callable, Self

CPUFREQ_CURRENT_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"

FrequencyReader = Callable[[], list[float]]


def read_sysfs_frequencies() -> list[float]:
    """Current frequency of every core in MHz. Raises OSError when a file is unreadable."""
    paths = sorted(glob(CPUFREQ_CURRENT_GLOB))
    frequencies = []
    for path in paths:
        with open(path) as file:
            frequencies.append(int(file.read()) / 1000.0)  # sysfs reports kHz
    return frequencies


class CpufreqMonitor:
    """Polls the core frequencies while the model runs. A core that falls below its baseline is throttling.

    Only polls inside `busy()` count. Between images, a scaling governor slows the idle cores,
    and that is not throttling. The baseline is the first busy poll, because the read at enter
    comes right after the warm-up, when the cores are at their boost frequency.

    The verdict is None when the frequency files are not readable at enter, for example on macOS,
    or when no poll happened inside `busy()`.
    """

    def __init__(
        self,
        poll_interval_s: float = 0.5,
        tolerance_mhz: float = 50.0,
        read_frequencies: FrequencyReader = read_sysfs_frequencies,
    ):
        self._poll_interval_s = poll_interval_s
        self._tolerance_mhz = tolerance_mhz
        self._read_frequencies = read_frequencies
        self._readable = False
        self._busy = threading.Event()
        self._baseline: list[float] | None = None
        self._min_mhz = 0.0
        self._max_drop_mhz = 0.0
        self._samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        try:
            self._readable = bool(self._read_frequencies())
        except OSError:
            self._readable = False
        if not self._readable:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    @contextmanager
    def busy(self) -> Iterator[None]:
        self._busy.set()
        try:
            yield
        finally:
            self._busy.clear()

    def poll_once(self) -> None:
        if not self._readable or not self._busy.is_set():
            return
        try:
            current = self._read_frequencies()
        except OSError:
            return  # one failed read must not end the watch
        if self._baseline is None:
            self._baseline = current
            self._min_mhz = min(current)
        self._samples += 1
        self._min_mhz = min([self._min_mhz, *current])
        drops = [before - now for before, now in zip(self._baseline, current)]
        self._max_drop_mhz = max([self._max_drop_mhz, *drops])

    def _poll_until_stopped(self) -> None:
        while not self._stop.wait(self._poll_interval_s):
            self.poll_once()

    def did_throttle(self) -> bool | None:
        if self._baseline is None:
            return None
        return self._max_drop_mhz > self._tolerance_mhz

    def summary(self) -> dict:
        if self._baseline is None:
            return {}
        return {
            "baseline_mean_mhz": sum(self._baseline) / len(self._baseline),
            "min_mhz": self._min_mhz,
            "max_drift_mhz": self._max_drop_mhz,
            "samples": self._samples,
        }
