import threading
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
    """Polls the core frequencies during the run. A core that falls below its start frequency is throttling.

    The verdict is None when the frequency files are not readable at enter, for example on macOS.
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
        self._baseline: list[float] | None = None
        self._min_mhz = 0.0
        self._max_drop_mhz = 0.0
        self._samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        try:
            baseline = self._read_frequencies()
        except OSError:
            baseline = []
        if not baseline:
            return self
        self._baseline = baseline
        self._min_mhz = min(baseline)
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

    def poll_once(self) -> None:
        if self._baseline is None:
            return
        try:
            current = self._read_frequencies()
        except OSError:
            return  # one failed read must not end the watch
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
