"""Did the GPU throttle during the run, on a Jetson?

A Jetson carries no NVML, so `sab.clock_watch.ThrottleMonitor` cannot answer
there. The kernel publishes the same facts in sysfs instead: the GPU frequency
under /sys/class/devfreq, and the thermal zones with their trip points under
/sys/class/thermal.

Two rules make this watch comparable with the NVML one.

A dead watch never certifies a clean pass. Every reading that fails raises, and
`did_throttle()` re-raises that failure, exactly as `ThrottleMonitor` does. A
run that swallowed the error would publish a `throttled: false` that nothing
observed.

A governor downclock is not throttling. AI1 cannot lock its clocks, so the
devfreq node drops to an idle step between passes and during the buffer time.
That is the governor doing its job, not a limit. A downclock counts only when a
thermal trip or an overcurrent event happened with it.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

DEVFREQ_ROOT = Path("/sys/class/devfreq")
THERMAL_ROOT = Path("/sys/class/thermal")
TEGRA_RELEASE_FILE = Path("/etc/nv_tegra_release")

# devfreq node names follow the SoC, not the product: Orin (Orin NX included)
# names the GPU node after the GA10B die, Xavier after the GV11B, TX2 after the
# GP10B. Some kernels expose a `gpu` alias or a `gpc` clock domain beside them.
GPU_DEVFREQ_MARKERS = ("ga10b", "gv11b", "gp10b", "gpc", "gpu")

# A thermal zone slows the SoC at a `passive` trip and cuts it at a `critical`
# one. Every other trip type reports a temperature and changes nothing.
THROTTLING_TRIP_TYPES = ("passive", "critical")

# A trip point at or below 0 mC is a disabled slot, not a limit.
DISABLED_TRIP_TEMP = 0

# Jetson counts overcurrent events in the soctherm debugfs nodes. That path is
# root-only and no L4T build promises it, so a missing source states no reason.
# A source that exists and does not read raises, like every other reading here.
OVERCURRENT_ROOT = Path("/sys/kernel/debug/bpmp/debug/soctherm")
OVERCURRENT_EVENT_GLOBS = ("oc*/event_cnt", "oc*_event_cnt")

DOWNCLOCKED_REASON = "gpu_downclocked"

# sysfs reads are cheap and the watch runs beside the benchmark, so the interval
# is short enough to catch a passing thermal event and long enough to stay off
# the measured path.
POLL_INTERVAL_S = 0.5


class TegraClockReadError(RuntimeError):
    """The box cannot say whether its GPU throttled."""


def is_tegra() -> bool:
    """True on a Jetson, which needs this watch instead of the NVML one.

    The docker images state their hardware in SAB_HARDWARE, and a stated
    hardware wins over a probe. Outside a SAB image the tegra release file is
    the answer.
    """
    if os.environ.get("SAB_HARDWARE", "").upper() == "AI1":
        return True
    return TEGRA_RELEASE_FILE.exists()


def is_throttled(reasons: tuple[str, ...]) -> bool:
    """Whether these reasons mean the pass was slowed.

    A downclock with no limit beside it is the governor at idle. A limit with
    the GPU still at full speed did not cost this pass anything yet.
    """
    limits = [reason for reason in reasons if reason != DOWNCLOCKED_REASON]
    return DOWNCLOCKED_REASON in reasons and bool(limits)


def find_gpu_devfreq_node(root: Path | None = None) -> Path:
    """The devfreq node of the GPU, found by name (`GPU_DEVFREQ_MARKERS`)."""
    directory = root if root is not None else DEVFREQ_ROOT
    candidates = sorted(entry for entry in directory.glob("*") if entry.is_dir()) if directory.is_dir() else []
    for node in candidates:
        if _names_a_gpu(node):
            return node
    raise TegraClockReadError(
        f"No GPU devfreq node under {directory}, which holds {[node.name for node in candidates]}. The tegra "
        f"clock watch reads cur_freq and max_freq from that node, so this box cannot say whether the GPU "
        f"throttled. Add this SoC's node name to GPU_DEVFREQ_MARKERS in sab/tegra_clock_watch.py."
    )


def _names_a_gpu(node: Path) -> bool:
    """True when this devfreq node is the GPU's.

    The directory name carries the device (`17000000.ga10b`), and some kernels
    also publish a `name` file. Either one may hold the marker.
    """
    labels = [node.name.lower()]
    name_file = node / "name"
    if name_file.is_file():
        labels.append(name_file.read_text(errors="replace").strip().lower())
    return any(marker in label for label in labels for marker in GPU_DEVFREQ_MARKERS)


@dataclass(frozen=True)
class TegraClockSample:
    sm_hz: int
    sm_max_hz: int
    reasons: tuple[str, ...]


class TegraClockReader:
    """A Jetson's GPU frequency and thermal state, read from sysfs."""

    def __init__(
        self,
        gpu_node: Path,
        thermal_root: Path = THERMAL_ROOT,
        overcurrent_root: Path = OVERCURRENT_ROOT,
    ) -> None:
        self._gpu_node = gpu_node
        self._thermal_root = thermal_root
        self._overcurrent_root = overcurrent_root
        self._overcurrent_baseline: dict[str, int] | None = None

    def sample(self) -> TegraClockSample:
        current, maximum = self._frequencies()
        reasons = [DOWNCLOCKED_REASON] if current < maximum else []
        reasons.extend(self._thermal_reasons())
        reasons.extend(self._overcurrent_reasons())
        return TegraClockSample(sm_hz=current, sm_max_hz=maximum, reasons=tuple(reasons))

    def close(self) -> None:
        """sysfs holds nothing open, so there is nothing to release."""

    def _frequencies(self) -> tuple[int, int]:
        return self._read_int(self._gpu_node / "cur_freq"), self._read_int(self._gpu_node / "max_freq")

    def _thermal_reasons(self) -> list[str]:
        """Every zone that sits at or above a trip that slows the machine."""
        reasons = []
        for zone in sorted(self._thermal_root.glob("thermal_zone*")):
            temperature_file = zone / "temp"
            if not temperature_file.is_file():
                continue
            temperature = self._read_int(temperature_file)
            reasons.extend(self._passed_trips(zone, temperature))
        return reasons

    def _passed_trips(self, zone: Path, temperature: int) -> list[str]:
        name = self._zone_name(zone)
        passed = []
        for type_file in sorted(zone.glob("trip_point_*_type")):
            trip_type = type_file.read_text(errors="replace").strip().lower()
            if trip_type not in THROTTLING_TRIP_TYPES:
                continue
            limit_file = zone / type_file.name.replace("_type", "_temp")
            if not limit_file.is_file():
                continue
            limit = self._read_int(limit_file)
            if limit > DISABLED_TRIP_TEMP and temperature >= limit:
                passed.append(f"thermal_{trip_type}:{name}")
        return passed

    def _overcurrent_reasons(self) -> list[str]:
        """Every overcurrent counter that moved since the first reading.

        The counters are cumulative and count the life of the boot, so only a
        rise during the pass says anything about the pass.
        """
        counters = self._overcurrent_counters()
        if self._overcurrent_baseline is None:
            self._overcurrent_baseline = counters
            return []
        return [
            f"overcurrent:{name}"
            for name, count in counters.items()
            if count > self._overcurrent_baseline.get(name, count)
        ]

    def _overcurrent_counters(self) -> dict[str, int]:
        counters = {}
        for pattern in OVERCURRENT_EVENT_GLOBS:
            for counter_file in sorted(self._overcurrent_root.glob(pattern)):
                counters[counter_file.parent.name] = self._read_int(counter_file)
        return counters

    def _zone_name(self, zone: Path) -> str:
        type_file = zone / "type"
        if type_file.is_file():
            return type_file.read_text(errors="replace").strip() or zone.name
        return zone.name

    def _read_int(self, path: Path) -> int:
        try:
            return int(path.read_text(errors="replace").strip())
        except (OSError, ValueError) as error:
            raise TegraClockReadError(
                f"{path} does not read as a number: {error}. The tegra clock watch reads this file to say "
                f"whether the run throttled, so the run cannot state that either way."
            ) from error


def open_tegra_clock_reader(
    devfreq_root: Path | None = None,
    thermal_root: Path | None = None,
    overcurrent_root: Path | None = None,
) -> TegraClockReader:
    """A reader on this Jetson's GPU frequency and thermal zones.

    It raises when the box publishes no GPU devfreq node, because a watch built
    on nothing would report every run as clean.
    """
    return TegraClockReader(
        gpu_node=find_gpu_devfreq_node(devfreq_root),
        thermal_root=thermal_root if thermal_root is not None else THERMAL_ROOT,
        overcurrent_root=overcurrent_root if overcurrent_root is not None else OVERCURRENT_ROOT,
    )


class TegraClockWatch:
    """The `ThrottleMonitor` verdict, from tegra sysfs.

    It takes the same shape as `sab.clock_watch.ThrottleMonitor`: a context
    manager whose `did_throttle()` is clean, throttled, or a raise. Jetson has
    no clock lock, so this watch records what happened and locks nothing.
    """

    def __init__(self, reader: TegraClockReader | None = None, poll_interval_s: float = POLL_INTERVAL_S):
        self._reader = reader
        self._owns_reader = reader is None
        self._poll_interval_s = poll_interval_s
        self._throttle_detected = False
        self._reasons: tuple[str, ...] = ()
        self._stop_event = threading.Event()
        self._thread = None
        self._error: BaseException | None = None

    def _watch(self):
        try:
            while not self._stop_event.is_set():
                sample = self._reader.sample()
                if is_throttled(sample.reasons):
                    self._throttle_detected = True
                    self._reasons = sample.reasons
                    print(
                        f"🔴  GPU throttled: {', '.join(sample.reasons)}, "
                        f"SM={sample.sm_hz // 1_000_000} MHz of {sample.sm_max_hz // 1_000_000} MHz"
                    )
                    return
                self._stop_event.wait(self._poll_interval_s)
        except BaseException as e:  # a dead watcher must never read as a clean verdict
            self._error = e
            print(f"Error watching tegra clocks: {e}")

    def monitor_throttling(self, target_freq: int | None = None):
        """Start the watch. `target_freq` exists for ThrottleMonitor parity and is unused.

        Jetson states its own maximum in the devfreq node, so nothing outside
        needs to pass one.
        """
        if self._thread is not None:
            return

        if self._reader is None:
            # Open on the caller's thread: a box that cannot read its clocks
            # must fail the run here, not silently on the worker.
            self._reader = open_tegra_clock_reader()

        self._stop_event.clear()
        worker = threading.Thread(target=self._watch)
        worker.daemon = True
        worker.start()
        self._thread = worker

    def did_throttle(self) -> bool:
        """True when the pass throttled. Raises when the watcher itself failed."""
        if self._error is not None:
            raise RuntimeError(f"The tegra clock watch failed mid-pass: {self._error}") from self._error
        return self._throttle_detected

    def reasons(self) -> tuple[str, ...]:
        return self._reasons

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self._poll_interval_s + 1)
            self._thread = None
        if self._owns_reader and self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        self.monitor_throttling()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        # The join in stop() is what makes a late worker failure visible, so
        # surface it here - but never mask an exception already escaping the
        # benchmark body.
        if exc_type is None and self._error is not None:
            raise RuntimeError(f"The tegra clock watch failed during the pass: {self._error}") from self._error
