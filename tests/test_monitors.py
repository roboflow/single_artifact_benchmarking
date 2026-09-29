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


def make_monitor(*readings) -> CpufreqMonitor:
    return CpufreqMonitor(poll_interval_s=0.001, tolerance_mhz=50.0, read_frequencies=scripted_reader(*readings))


def verdict_after_polls(*readings, polls: int) -> bool | None:
    monitor = make_monitor(*readings)
    with monitor:
        for _ in range(polls):
            monitor.poll_once()
    return monitor.did_throttle()


class TestCpufreqMonitor:
    @pytest.mark.parametrize("unreadable", [FileNotFoundError("no cpufreq"), []])
    def test_no_signal_at_enter_gives_an_unknown_verdict(self, unreadable):
        assert verdict_after_polls(unreadable, polls=1) is None

    def test_drop_below_baseline_by_more_than_tolerance_throttles(self):
        assert verdict_after_polls([3000.0, 3000.0], [3000.0, 2900.0], polls=1) is True

    def test_drop_within_tolerance_does_not_throttle(self):
        assert verdict_after_polls([3000.0], [2960.0], polls=1) is False

    def test_a_dip_that_recovers_still_throttles(self):
        assert verdict_after_polls([3000.0], [2000.0], [3000.0], polls=2) is True

    def test_failed_poll_during_the_run_is_skipped(self):
        assert verdict_after_polls([3000.0], OSError("gone"), [3000.0], polls=2) is False

    def test_background_thread_polls_and_stops_on_exit(self):
        monitor = make_monitor([3000.0], [1000.0])
        with monitor:
            while monitor.summary()["samples"] == 0:
                pass
        samples_at_exit = monitor.summary()["samples"]
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
