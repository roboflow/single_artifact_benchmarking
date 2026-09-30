import time
from contextlib import nullcontext
from typing import ClassVar

import pytest

from sab.monitors import NullMonitor, select_monitor
from sab.monitors.cpufreq import CpufreqMonitor
from sab.monitors.nvidia import ThrottleMonitor
from tests.fakes import FakeRuntime


def scripted_reader(*readings):
    """Gives each reading in turn, then repeats the last. An Exception item is raised."""
    queue = list(readings)

    def read() -> list[float]:
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    return read


def make_monitor(*readings, poll_interval_s: float = 3600.0) -> CpufreqMonitor:
    """The default interval keeps the background thread idle, so the tests poll by hand."""
    return CpufreqMonitor(poll_interval_s=poll_interval_s, tolerance_mhz=50.0, read_frequencies=scripted_reader(*readings))


def verdict_after_polls(*readings, busy: list[bool]) -> bool | None:
    """Enter the monitor (one reading), then poll once for each item of `busy`, inside busy() when True."""
    monitor = make_monitor(*readings)
    with monitor:
        for is_busy in busy:
            with monitor.busy() if is_busy else nullcontext():
                monitor.poll_once()
    return monitor.did_throttle()


class TestCpufreqMonitor:
    @pytest.mark.parametrize("unreadable", [FileNotFoundError("no cpufreq"), []])
    def test_no_signal_at_enter_gives_an_unknown_verdict(self, unreadable):
        assert verdict_after_polls(unreadable, busy=[True]) is None

    def test_no_busy_poll_gives_an_unknown_verdict(self):
        assert verdict_after_polls([3000.0], [1000.0], busy=[False, False]) is None

    def test_busy_drop_below_the_first_busy_poll_by_more_than_tolerance_throttles(self):
        assert verdict_after_polls([3000.0, 3000.0], [3000.0, 3000.0], [3000.0, 2900.0], busy=[True, True]) is True

    def test_busy_drop_within_tolerance_does_not_throttle(self):
        assert verdict_after_polls([3000.0], [3000.0], [2960.0], busy=[True, True]) is False

    def test_the_baseline_is_the_first_busy_poll_not_the_boosted_read_at_enter(self):
        assert verdict_after_polls([3500.0], [3000.0], [3000.0], busy=[True, True]) is False

    def test_a_poll_between_images_reads_nothing(self):
        reads = []
        monitor = CpufreqMonitor(poll_interval_s=3600.0, read_frequencies=lambda: reads.append(1) or [1000.0])
        with monitor:
            monitor.poll_once()
        assert len(reads) == 1  # only the readability check at enter

    def test_a_busy_dip_that_recovers_still_throttles(self):
        assert verdict_after_polls([3000.0], [3000.0], [2000.0], [3000.0], busy=[True, True, True]) is True

    def test_failed_busy_poll_is_skipped(self):
        assert verdict_after_polls([3000.0], [3000.0], OSError("gone"), [3000.0], busy=[True, True, True]) is False

    def test_background_thread_polls_while_busy_and_stops_on_exit(self):
        monitor = make_monitor([3000.0], [3000.0], [1000.0], poll_interval_s=0.001)
        deadline = time.monotonic() + 5.0
        with monitor:
            with monitor.busy():
                while monitor.summary().get("samples", 0) < 2:
                    assert time.monotonic() < deadline, "the thread did not poll"
        samples_at_exit = monitor.summary()["samples"]
        time.sleep(0.01)
        assert monitor.did_throttle() is True
        assert monitor.summary()["samples"] == samples_at_exit


def test_nvml_did_throttle_raises_when_the_watcher_failed():
    monitor = ThrottleMonitor(target_freq=1590)
    monitor._error = RuntimeError("nvml died")
    with pytest.raises(RuntimeError, match="watcher failed"):
        monitor.did_throttle()


def runtime_named(runtime_name: str) -> type[FakeRuntime]:
    class Named(FakeRuntime):
        name: ClassVar[str] = runtime_name

    return Named


@pytest.mark.parametrize(
    "runtime_name, device, system, expected",
    [
        ("tensorrt", "gpu", "Linux", ThrottleMonitor),
        ("onnxruntime", "gpu", "Linux", ThrottleMonitor),
        ("onnxruntime", "cpu", "Linux", CpufreqMonitor),
        ("onnxruntime", "cpu", "Darwin", NullMonitor),
        ("coreml", "gpu", "Linux", NullMonitor),
    ],
)
def test_select_monitor(runtime_name, device, system, expected):
    assert type(select_monitor(runtime_named(runtime_name), device, system=system)) is expected
