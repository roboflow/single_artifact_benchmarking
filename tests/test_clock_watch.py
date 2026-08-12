"""NVML reason labels, the pinned throttling-bit set, and the lazy dlopen."""

import ctypes
import ctypes.util
import importlib

from sab import clock_watch


# The nvml.h macro that owns each bit, per the NVIDIA NVML reference.
NVML_H_MACROS = {
    0x00000001: "nvmlClocksEventReasonGpuIdle",
    0x00000002: "nvmlClocksEventReasonApplicationsClocksSetting",
    0x00000004: "nvmlClocksEventReasonSwPowerCap",
    0x00000008: "nvmlClocksThrottleReasonHwSlowdown",
    0x00000010: "nvmlClocksEventReasonSyncBoost",
    0x00000020: "nvmlClocksEventReasonSwThermalSlowdown",
    0x00000040: "nvmlClocksThrottleReasonHwThermalSlowdown",
    0x00000080: "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
}


def test_import_does_not_dlopen_nvml(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("NVML must not load at import time")

    monkeypatch.setattr(ctypes.util, "find_library", fail)
    monkeypatch.setattr(ctypes, "CDLL", fail)

    importlib.reload(clock_watch)


def test_throttling_bit_set_is_unchanged():
    """Pinned: these bits decide the `throttled` column of every published row.

    Adding or removing a bit re-baselines the benchmark. Relabeling does not.
    """
    assert set(clock_watch.REASONS) == {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80}
    assert clock_watch.THROTTLE_REASON_BITS == 0xFF


def test_decoded_bits_match_nvml_h():
    assert set(clock_watch.REASONS) == set(NVML_H_MACROS)


def test_applications_clocks_bit_is_no_longer_labelled_thermal():
    assert clock_watch.REASONS[0x00000002] == "Applications clocks setting"


def test_thermal_labels_sit_on_the_thermal_bits():
    assert clock_watch.REASONS[0x00000020] == "SW thermal slowdown"
    assert clock_watch.REASONS[0x00000040] == "HW thermal slowdown"


def test_every_decoded_bit_counts_as_throttling():
    for bit in clock_watch.REASONS:
        assert clock_watch.is_throttled(bit)


def test_undecoded_and_empty_masks_are_benign():
    assert not clock_watch.is_throttled(0x00000000)
    assert not clock_watch.is_throttled(0x00000100)  # DisplayClockSetting


def run_monitor_over(monkeypatch, events, target_freq=1590):
    monkeypatch.setattr(clock_watch, "emit_clock_changes", lambda: iter(events))
    monitor = clock_watch.ThrottleMonitor(target_freq=target_freq)
    monitor._check_for_throttling()
    return monitor


def test_monitor_decides_on_bits_not_on_label_text(monkeypatch):
    monkeypatch.setitem(clock_watch.REASONS, 0x00000002, "some future relabel")
    monitor = run_monitor_over(
        monkeypatch, [(1000, 5000, "some future relabel", 0x00000002)]
    )
    assert monitor.did_throttle()


def test_monitor_ignores_undecoded_reason_bits(monkeypatch):
    monitor = run_monitor_over(
        monkeypatch, [(1000, 5000, clock_watch.NO_THROTTLE_TEXT, 0x00000100)]
    )
    assert not monitor.did_throttle()


def test_monitor_ignores_events_at_the_target_frequency(monkeypatch):
    monitor = run_monitor_over(monkeypatch, [(1590, 5000, "GPU idle", 0x00000001)])
    assert not monitor.did_throttle()
