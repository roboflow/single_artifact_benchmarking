#!/usr/bin/env python3
"""
clock_watch.py  –  prints a line whenever any SM or memory-clock change occurs.
Tested on Tesla T4 / driver R555, but works on any dGPU that supports NVML events.
"""

import ctypes as ct
import ctypes.util
import datetime
import os
import shutil
import signal
import sys
import threading
from functools import reduce
from operator import or_
from subprocess import run
from contextlib import contextmanager


NVML_TIMEOUT = 1000  # ms

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load NVML and declare the handful of functions we need
# ──────────────────────────────────────────────────────────────────────────────
_nvml = None


class NvmlUnavailableError(RuntimeError):
    """NVML is not present or failed to load, so clocks cannot be watched."""


def _load_nvml():
    """Load NVML on first use and cache it.

    Loading at import time makes every transitive import of this module fail on a
    machine without libnvidia-ml, so the dlopen waits until a caller needs NVML.
    """
    global _nvml

    if _nvml is not None:
        return _nvml

    lib_path = ctypes.util.find_library("nvidia-ml")
    if not lib_path:
        # A normal exception, never sys.exit(): the first load can happen on the
        # monitor worker thread, and a SystemExit there dies silently — the
        # monitor would then report "did not throttle" for a pass it never
        # watched.
        raise NvmlUnavailableError("NVML library not found - is the NVIDIA driver installed?")
    lib = ct.CDLL(lib_path)

    for name in (
        "nvmlInit_v2", "nvmlShutdown",
        "nvmlDeviceGetHandleByIndex_v2",
        "nvmlEventSetCreate", "nvmlEventSetFree",
        "nvmlDeviceRegisterEvents", "nvmlEventSetWait",
        "nvmlDeviceGetCurrentClocksThrottleReasons",
        "nvmlDeviceGetClockInfo",
    ):
        getattr(lib, name).restype = ct.c_int  # all return nvmlReturn_t

    _nvml = lib
    return _nvml

# ──────────────────────────────────────────────────────────────────────────────
# 2.  NVML constants we need (from the public headers)
# ──────────────────────────────────────────────────────────────────────────────
NVML_SUCCESS                = 0
NVML_EVENT_TYPE_CLOCK       = 0x10                         # any clock change  [oai_citation:0‡docs.nvidia.com](https://docs.nvidia.com/deploy/archive/R525/nvml-api/group__nvmlEventType.html?utm_source=chatgpt.com)
NVML_CLOCK_GRAPHICS         = 0                            # SM core clock domain
NVML_CLOCK_MEM              = 1                            # Memory clock domain

# Clock-event ("throttle") reason bitmask – report them all. Each label is checked
# against the nvml.h macro named beside it.
# The KEYS of this map are the throttle decision: is_throttled() reports any bit in
# here as throttling. Do not add or remove keys without a benchmark re-baseline.
# nvmlClocksEventReasonDisplayClockSetting (0x0100) is left out on purpose, so a
# display clock change stays benign, as it has always been here.
REASONS = {
    0x00000001: "GPU idle",                    # nvmlClocksEventReasonGpuIdle
    0x00000002: "Applications clocks setting",  # nvmlClocksEventReasonApplicationsClocksSetting
    0x00000004: "SW power-cap",                # nvmlClocksEventReasonSwPowerCap
    0x00000008: "HW slowdown",                 # nvmlClocksThrottleReasonHwSlowdown
    0x00000010: "Sync-boost",                  # nvmlClocksEventReasonSyncBoost
    0x00000020: "SW thermal slowdown",         # nvmlClocksEventReasonSwThermalSlowdown
    0x00000040: "HW thermal slowdown",         # nvmlClocksThrottleReasonHwThermalSlowdown
    0x00000080: "HW power-brake",              # nvmlClocksThrottleReasonHwPowerBrakeSlowdown
}                                                # constants list  [oai_citation:1‡docs.nvidia.com](https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html?utm_source=chatgpt.com)

NO_THROTTLE_TEXT = "No throttle (max clocks)"

# Every bit this module decodes, as one mask.
THROTTLE_REASON_BITS = reduce(or_, REASONS)


def is_throttled(reason_mask: int) -> bool:
    """Report whether NVML gave any reason this module counts as throttling.

    The decision is on the BITS, never on the label text, so a label correction
    cannot change which runs are marked throttled.
    """
    return bool(reason_mask & THROTTLE_REASON_BITS)

class Event(ct.Structure):        # minimal nvmlEventData_t
    _fields_ = [("device", ct.c_void_p),
                ("eventType", ct.c_ulonglong),
                ("eventData", ct.c_ulonglong),
                ("timestamp", ct.c_longlong)]

def chk(ret, func):
    if ret != NVML_SUCCESS:
        _load_nvml().nvmlShutdown()
        sys.exit(f"{func} failed with code {ret}")

def emit_clock_changes():
    nvml = _load_nvml()

    chk(nvml.nvmlInit_v2(), "nvmlInit")

    dev = ct.c_void_p()
    chk(nvml.nvmlDeviceGetHandleByIndex_v2(0, ct.byref(dev)),
        "getHandle(0)")

    evset = ct.c_void_p()
    chk(nvml.nvmlEventSetCreate(ct.byref(evset)), "eventSetCreate")
    chk(nvml.nvmlDeviceRegisterEvents(dev, NVML_EVENT_TYPE_CLOCK, evset),
        "register CLOCK")

    ev = Event()
    while True:
        rc = nvml.nvmlEventSetWait(evset, ct.byref(ev), NVML_TIMEOUT)  # ms
        if rc != NVML_SUCCESS:                              # timeout → loop
            continue

        # Get current SM & MEM clocks
        sm = ct.c_uint(); mem = ct.c_uint()
        nvml.nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, ct.byref(sm))
        nvml.nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM,      ct.byref(mem))

        # Decode throttle reasons
        mask = ct.c_ulonglong()
        nvml.nvmlDeviceGetCurrentClocksThrottleReasons(dev, ct.byref(mask))
        reasons = [name for bit, name in REASONS.items() if mask.value & bit]
        reason_txt = ", ".join(reasons) or NO_THROTTLE_TEXT

        yield sm.value, mem.value, reason_txt, mask.value

class ThrottleMonitor:
    def __init__(self, target_freq: int|None=None):
        self._throttle_detected = False
        self._target_freq = target_freq
        self._stop_thread = False
        self._thread = None
        self._error = None
    
    def _check_for_throttling(self):
        try:
            clock_generator = emit_clock_changes()

            while not self._stop_thread:
                try:
                    # Get the next clock reading (this will timeout every 5 seconds)
                    sm, mem, reason_txt, reason_mask = next(clock_generator)
                except StopIteration:
                    break

                if sm != self._target_freq and is_throttled(reason_mask):
                    self._throttle_detected = True
                    print(f"🔴  GPU throttled: {reason_txt}, SM={sm} MHz, MEM={mem} MHz")
                    break
        except BaseException as e:  # a dead watcher must never read as a clean verdict
            self._error = e
            print(f"Error monitoring clocks: {e}")
    
    def monitor_throttling(self, target_freq: int|None=None):
        if target_freq is None and self._target_freq is None:
            raise ValueError("Target frequency is not set")
        
        if target_freq is not None:
            self._target_freq = target_freq
        
        if self._thread is not None:
            return  # Already monitoring

        # Load NVML on the caller's thread, so a missing library fails the run
        # here instead of killing the worker silently.
        _load_nvml()

        self._stop_thread = False
        # Assign only after start() succeeds: stop() joins every non-None
        # thread, and joining a never-started thread raises.
        worker = threading.Thread(target=self._check_for_throttling)
        worker.daemon = True
        worker.start()
        self._thread = worker
    
    def did_throttle(self) -> bool:
        """True when the pass throttled. Raises when the watcher itself failed.

        The verdict is tri-state in effect: clean, throttled, or failed. A
        watcher that died must never certify a pass as clean.
        """
        if self._error is not None:
            raise RuntimeError(f"The throttle watcher failed mid-pass: {self._error}") from self._error
        return self._throttle_detected

    def stop(self):
        if self._thread is not None:
            self._stop_thread = True
            self._thread.join(timeout=NVML_TIMEOUT/1000 + 1)  # Wait up to 6 seconds (longer than nvml timeout)
            self._thread = None
    
    # @contextmanager
    # def __call__(self):
    #     gpu_clock, mem_clock = get_max_clocks()
    #     enable_persistence(True)
    #     lock_clocks(gpu_clock, mem_clock)
    #     try:
    #         self.monitor_throttling(gpu_clock)
    #         yield
    #     finally:
    #         self.stop()
    #         enable_persistence(False)
    #         unlock_clocks()
    def __enter__(self):
        # Load NVML before touching any hardware state: when __enter__ raises,
        # __exit__ never runs, so nothing may fail after a side effect without
        # a rollback.
        _load_nvml()
        gpu_clock, mem_clock = get_max_clocks()
        enable_persistence(True)
        try:
            lock_clocks(gpu_clock, mem_clock)
            self.monitor_throttling(gpu_clock)
        except BaseException:
            try:
                self.stop()
            finally:
                enable_persistence(False)
                unlock_clocks()
            raise
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.stop()
        finally:
            enable_persistence(False)
            unlock_clocks()
        # The join in stop() is what makes a late worker failure visible, so
        # surface it here - but never mask an exception already escaping the
        # benchmark body.
        if exc_type is None and self._error is not None:
            raise RuntimeError(f"The throttle watcher failed during the pass: {self._error}") from self._error


def open_throttle_monitor(target_freq: int|None = None):
    """The throttle watch that can read this box.

    A Jetson has no NVML, so ThrottleMonitor answers nothing there. The tegra
    watch reads the GPU devfreq node and the thermal trips instead. Both give
    the same verdict shape, so the caller does not branch.
    """
    from sab.tegra_clock_watch import TegraClockWatch, is_tegra

    if is_tegra():
        return TegraClockWatch()
    return ThrottleMonitor(target_freq)


def _nvidia_smi(*args: str, capture_output: bool = False):
    # Clock locking needs root. Containers run as root and have no sudo.
    # Bare-metal dev boxes are not root and need it.
    command = ["nvidia-smi", *args]
    if os.geteuid() != 0 and shutil.which("sudo"):
        command = ["sudo", *command]
    return run(command, capture_output=capture_output)


def get_max_clocks() -> tuple[int, int]:
    """Get the maximum GPU and memory clocks."""
    res = _nvidia_smi("--query-gpu=clocks.max.graphics,clocks.max.memory", "--format=csv,noheader", capture_output=True)
    output = res.stdout.decode("utf-8").splitlines()[0].strip()
    # Parse comma-separated values and remove "MHz" suffix
    gpu_clock_str, mem_clock_str = output.split(',')
    gpu_clock = int(gpu_clock_str.strip().replace(' MHz', ''))
    mem_clock = int(mem_clock_str.strip().replace(' MHz', ''))
    return gpu_clock, mem_clock


def lock_clocks(gpu_mhz: int, mem_mhz: int|None = None) -> None:
    """Lock GPU and memory clocks (requires root)."""
    _nvidia_smi("--lock-gpu-clocks", str(gpu_mhz))
    if mem_mhz:
        _nvidia_smi("--lock-memory-clocks", str(mem_mhz))


def unlock_clocks() -> None:
    """Reset GPU and memory clocks (requires root)."""
    _nvidia_smi("--reset-gpu-clocks")
    _nvidia_smi("--reset-memory-clocks")


def enable_persistence(enable: bool) -> None:
    _nvidia_smi("-pm", "1" if enable else "0")


class CPUFrequencyMonitor:
    """Monitors CPU frequency drift during a benchmark run via /proc/cpuinfo."""

    def __init__(self, tolerance_mhz: float = 50.0):
        self._tolerance_mhz = tolerance_mhz
        self._baseline_freqs: list[float] | None = None
        self._end_freqs: list[float] | None = None
        self._drifted = False

    @staticmethod
    def _read_cpu_frequencies() -> list[float]:
        freqs = []
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    freqs.append(float(line.split(":")[1].strip()))
        return freqs

    def __enter__(self):
        try:
            self._baseline_freqs = self._read_cpu_frequencies()
        except OSError:
            self._baseline_freqs = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._baseline_freqs is None:
            return
        try:
            self._end_freqs = self._read_cpu_frequencies()
        except OSError:
            return
        for before, after in zip(self._baseline_freqs, self._end_freqs):
            if abs(before - after) > self._tolerance_mhz:
                self._drifted = True
                return

    def did_drift(self) -> bool:
        return self._drifted

    def get_summary(self) -> dict | None:
        if self._baseline_freqs is None or self._end_freqs is None:
            return None
        return {
            "baseline_mean_mhz": sum(self._baseline_freqs) / len(self._baseline_freqs),
            "end_mean_mhz": sum(self._end_freqs) / len(self._end_freqs),
            "max_drift_mhz": max(abs(b - e) for b, e in zip(self._baseline_freqs, self._end_freqs)),
        }


def main():
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))   # clean Ctrl-C
    print("🟢  Watching for any GPU clock changes (press Ctrl-C to quit)")
    for sm, mem, reason_txt, _reason_mask in emit_clock_changes():
        print(f"{datetime.datetime.now():%H:%M:%S}  "
              f"SM={sm} MHz  MEM={mem} MHz  |  {reason_txt}")

if __name__ == "__main__":
    main()