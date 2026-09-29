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


def make_monitor(*readings, tolerance_mhz=50.0) -> CpufreqMonitor:
    return CpufreqMonitor(
        poll_interval_s=0.001, tolerance_mhz=tolerance_mhz, read_frequencies=scripted_reader(*readings)
    )


class TestCpufreqMonitor:
    def test_steady_frequencies_do_not_throttle(self):
        monitor = make_monitor([3000.0, 3000.0])
        with monitor:
            monitor.poll_once()
            monitor.poll_once()
        assert monitor.did_throttle() is False

    def test_no_signal_when_files_are_unreadable_at_enter(self):
        monitor = make_monitor(FileNotFoundError("no cpufreq"))
        with monitor:
            monitor.poll_once()
        assert monitor.did_throttle() is None
        assert monitor.summary() == {}

    def test_no_signal_when_no_core_reports_a_frequency(self):
        monitor = make_monitor([])
        with monitor:
            pass
        assert monitor.did_throttle() is None

    def test_drop_below_baseline_by_more_than_tolerance_throttles(self):
        monitor = make_monitor([3000.0, 3000.0], [3000.0, 2900.0])
        with monitor:
            monitor.poll_once()
        assert monitor.did_throttle() is True

    def test_drop_within_tolerance_does_not_throttle(self):
        monitor = make_monitor([3000.0], [2960.0])
        with monitor:
            monitor.poll_once()
        assert monitor.did_throttle() is False

    def test_rise_above_baseline_does_not_throttle(self):
        monitor = make_monitor([1000.0], [3000.0])
        with monitor:
            monitor.poll_once()
        assert monitor.did_throttle() is False

    def test_a_dip_that_recovers_still_throttles(self):
        monitor = make_monitor([3000.0], [2000.0], [3000.0])
        with monitor:
            monitor.poll_once()
            monitor.poll_once()
        assert monitor.did_throttle() is True

    def test_failed_poll_during_the_run_is_skipped(self):
        monitor = make_monitor([3000.0], OSError("gone"), [3000.0])
        with monitor:
            monitor.poll_once()
            monitor.poll_once()
        assert monitor.did_throttle() is False

    def test_summary_reports_baseline_min_drift_and_samples(self):
        monitor = make_monitor([3000.0, 2000.0], [2500.0, 2000.0], [3000.0, 1800.0])
        with monitor:
            monitor.poll_once()
            monitor.poll_once()
        assert monitor.summary() == {
            "baseline_mean_mhz": 2500.0,
            "min_mhz": 1800.0,
            "max_drift_mhz": 500.0,
            "samples": 2,
        }

    def test_background_thread_polls_and_stops_on_exit(self):
        monitor = make_monitor([3000.0], [1000.0])
        with monitor:
            while monitor.summary()["samples"] == 0:
                pass
        samples_at_exit = monitor.summary()["samples"]
        assert monitor.did_throttle() is True
        assert monitor.summary()["samples"] == samples_at_exit


class TestThrottleMonitorVerdicts:
    def test_summary_reports_verdict_and_target(self):
        monitor = ThrottleMonitor(target_freq=1590)
        assert monitor.summary() == {"throttled": False, "target_mhz": 1590}

    def test_did_throttle_raises_when_the_watcher_failed(self):
        monitor = ThrottleMonitor(target_freq=1590)
        monitor._error = RuntimeError("nvml died")
        with pytest.raises(RuntimeError, match="watcher failed"):
            monitor.did_throttle()


def runtime_named(runtime_name: str) -> type[FakeRuntime]:
    class Named(FakeRuntime):
        name: ClassVar[str] = runtime_name

    return Named


class TestSelectMonitor:
    @pytest.mark.parametrize("runtime_name", ["tensorrt", "onnxruntime"])
    def test_gpu_on_nvml_runtimes_gets_throttle_monitor(self, runtime_name):
        assert isinstance(select_monitor(runtime_named(runtime_name), "gpu", system="Linux"), ThrottleMonitor)

    def test_cpu_on_linux_gets_cpufreq_monitor(self):
        assert isinstance(select_monitor(runtime_named("onnxruntime"), "cpu", system="Linux"), CpufreqMonitor)

    def test_cpu_on_mac_gets_null_monitor(self):
        assert isinstance(select_monitor(runtime_named("onnxruntime"), "cpu", system="Darwin"), NullMonitor)

    def test_gpu_on_other_runtimes_gets_null_monitor(self):
        assert isinstance(select_monitor(runtime_named("coreml"), "gpu", system="Linux"), NullMonitor)

    def test_npu_gets_null_monitor(self):
        assert isinstance(select_monitor(runtime_named("hailort"), "npu", system="Linux"), NullMonitor)
