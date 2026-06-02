#!/usr/bin/env python3
"""
clock_watch.py  –  prints a line whenever any SM or memory-clock change occurs.
Tested on Tesla T4 / driver R555, but works on any dGPU that supports NVML events.
"""

import ctypes as ct
import ctypes.util
import datetime
import signal
import sys
import threading
from subprocess import run
from contextlib import contextmanager


NVML_TIMEOUT = 1000  # ms

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load NVML and declare the handful of functions we need
# ──────────────────────────────────────────────────────────────────────────────
# NVML lives in the NVIDIA driver (libnvidia-ml.so.1), which is injected by GPU
# nodes but ABSENT on CPU-only nodes. Load it lazily so importing this module
# (and therefore sab.models.utils) does not crash on a CPU-only machine -- the
# CPU benchmarking path uses CPUFrequencyMonitor, not the NVML ThrottleMonitor.
lib_path = ctypes.util.find_library("nvidia-ml")
try:
    nvml = ct.CDLL(lib_path) if lib_path else None
except OSError:
    nvml = None

if nvml is not None:
    for name in (
        "nvmlInit_v2", "nvmlShutdown",
        "nvmlDeviceGetHandleByIndex_v2",
        "nvmlEventSetCreate", "nvmlEventSetFree",
        "nvmlDeviceRegisterEvents", "nvmlEventSetWait",
        "nvmlDeviceGetCurrentClocksThrottleReasons",
        "nvmlDeviceGetClockInfo",
    ):
        getattr(nvml, name).restype = ct.c_int  # all return nvmlReturn_t

# ──────────────────────────────────────────────────────────────────────────────
# 2.  NVML constants we need (from the public headers)
# ──────────────────────────────────────────────────────────────────────────────
NVML_SUCCESS                = 0
NVML_EVENT_TYPE_CLOCK       = 0x10                         # any clock change  [oai_citation:0‡docs.nvidia.com](https://docs.nvidia.com/deploy/archive/R525/nvml-api/group__nvmlEventType.html?utm_source=chatgpt.com)
NVML_CLOCK_GRAPHICS         = 0                            # SM core clock domain
NVML_CLOCK_MEM              = 1                            # Memory clock domain

# Throttle-reason bitmask – report them all
REASONS = {
    0x00000001: "GPU idle",
    0x00000002: "Thermal",
    0x00000004: "SW power-cap",
    0x00000008: "HW slowdown",
    0x00000010: "Sync-boost",
    0x00000020: "SW thermal slowdown",
    0x00000040: "HW thermal slowdown",
    0x00000080: "HW power-brake",
}                                                # constants list  [oai_citation:1‡docs.nvidia.com](https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html?utm_source=chatgpt.com)

class Event(ct.Structure):        # minimal nvmlEventData_t
    _fields_ = [("device", ct.c_void_p),
                ("eventType", ct.c_ulonglong),
                ("eventData", ct.c_ulonglong),
                ("timestamp", ct.c_longlong)]

def chk(ret, func):
    if ret != NVML_SUCCESS:
        nvml.nvmlShutdown()
        sys.exit(f"{func} failed with code {ret}")

def emit_clock_changes():
    if nvml is None:
        raise RuntimeError(
            "NVML unavailable (no NVIDIA driver) -- ThrottleMonitor needs a GPU node; "
            "use CPUFrequencyMonitor for CPU benchmarking."
        )
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
        reason_txt = ", ".join(reasons) or "No throttle (max clocks)"

        yield sm.value, mem.value, reason_txt

class ThrottleMonitor:
    def __init__(self, target_freq: int|None=None):
        self._throttle_detected = False
        self._target_freq = target_freq
        self._stop_thread = False
        self._thread = None
    
    def _check_for_throttling(self):
        # Get the generator
        clock_generator = emit_clock_changes()
        
        while not self._stop_thread:
            try:
                # Get the next clock reading (this will timeout every 5 seconds)
                sm, mem, reason_txt = next(clock_generator)
                
                if sm != self._target_freq and reason_txt != "No throttle (max clocks)":
                    self._throttle_detected = True
                    print(f"🔴  GPU throttled: {reason_txt}, SM={sm} MHz, MEM={mem} MHz")
                    break
                    
            except StopIteration:
                break
            except Exception as e:
                print(f"Error monitoring clocks: {e}")
                break
    
    def monitor_throttling(self, target_freq: int|None=None):
        if target_freq is None and self._target_freq is None:
            raise ValueError("Target frequency is not set")
        
        if target_freq is not None:
            self._target_freq = target_freq
        
        if self._thread is not None:
            return  # Already monitoring
            
        self._stop_thread = False
        self._thread = threading.Thread(target=self._check_for_throttling)
        self._thread.daemon = True
        self._thread.start()
    
    def did_throttle(self) -> bool:
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
        gpu_clock, mem_clock = get_max_clocks()
        enable_persistence(True)
        lock_clocks(gpu_clock, mem_clock)
        self.monitor_throttling(gpu_clock)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        enable_persistence(False)
        unlock_clocks()


def get_max_clocks() -> tuple[int, int]:
    """Get the maximum GPU and memory clocks."""
    res = run(["sudo", "nvidia-smi", "--query-gpu=clocks.max.graphics,clocks.max.memory", "--format=csv,noheader"], capture_output=True)
    output = res.stdout.decode("utf-8").splitlines()[0].strip()
    # Parse comma-separated values and remove "MHz" suffix
    gpu_clock_str, mem_clock_str = output.split(',')
    gpu_clock = int(gpu_clock_str.strip().replace(' MHz', ''))
    mem_clock = int(mem_clock_str.strip().replace(' MHz', ''))
    return gpu_clock, mem_clock


def lock_clocks(gpu_mhz: int, mem_mhz: int|None = None) -> None:
    """Lock GPU and memory clocks (requires root)."""
    run(["sudo", "nvidia-smi", "--lock-gpu-clocks", str(gpu_mhz)])
    if mem_mhz:
        run(["sudo", "nvidia-smi", "--lock-memory-clocks", str(mem_mhz)])


def unlock_clocks() -> None:
    """Reset GPU and memory clocks (requires root)."""
    run(["sudo", "nvidia-smi", "--reset-gpu-clocks"])
    run(["sudo", "nvidia-smi", "--reset-memory-clocks"])


def enable_persistence(enable: bool) -> None:
    run(["sudo", "nvidia-smi", "-pm", "1" if enable else "0"])


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
    for sm, mem, reason_txt in emit_clock_changes():
        print(f"{datetime.datetime.now():%H:%M:%S}  "
              f"SM={sm} MHz  MEM={mem} MHz  |  {reason_txt}")

if __name__ == "__main__":
    main()