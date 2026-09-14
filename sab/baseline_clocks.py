"""Checked maximum-clock policy and sampled telemetry for SAB campaigns.

Locking clocks is not a promise that a power-limited GPU can sustain them.
Preserve actual readings throughout builds and timing, including throttling.
"""

import ctypes as ct
import subprocess
import threading
import time

from sab import clock_watch


class MaximumClocks:
    def __init__(self, interval=.1):
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = None
        self.initialized = False
        self.started = time.monotonic()

    @staticmethod
    def command(*args):
        return subprocess.check_output(['sudo', '-n', 'nvidia-smi', '-i', '0', *args], text=True)

    def reading(self):
        values = {}
        for key, kind in [('sm_mhz', 1), ('memory_mhz', 2)]:
            value = ct.c_uint()
            rc = clock_watch.nvml.nvmlDeviceGetClockInfo(self.device, kind, ct.byref(value))
            if rc:
                raise RuntimeError(f'NVML clock query failed: {rc}')
            values[key] = value.value
        reason = ct.c_ulonglong()
        rc = clock_watch.nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.device, ct.byref(reason))
        if rc:
            raise RuntimeError(f'NVML throttle query failed: {rc}')
        values.update(seconds=time.monotonic() - self.started, throttle_mask=reason.value)
        return values

    def poll(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.samples.append(self.reading())
            except Exception as error:
                self.errors.append(str(error))
                return

    def __enter__(self):
        raw = self.command('--query-gpu=clocks.max.sm,clocks.max.memory,persistence_mode',
                           '--format=csv,noheader,nounits').strip().split(', ')
        self.sm_mhz, self.memory_mhz = int(raw[0]), int(raw[1])
        self.persistence_was_enabled = raw[2] == 'Enabled'
        self.started = time.monotonic()
        try:
            self.command('-pm', '1')
            self.command('--lock-gpu-clocks', f'{self.sm_mhz},{self.sm_mhz}')
            self.command('--lock-memory-clocks', f'{self.memory_mhz},{self.memory_mhz}')
            if clock_watch.nvml.nvmlInit_v2():
                raise RuntimeError('NVML initialization failed')
            self.initialized = True
            self.device = ct.c_void_p()
            if clock_watch.nvml.nvmlDeviceGetHandleByIndex_v2(0, ct.byref(self.device)):
                raise RuntimeError('NVML device lookup failed')
            for _ in range(50):
                sample = self.reading()
                self.samples.append(sample)
                # T4 reports memory at 5000 despite a requested 5001 MHz.
                if sample['sm_mhz'] == self.sm_mhz and sample['memory_mhz'] >= self.memory_mhz - 2:
                    break
                time.sleep(.1)
            else:
                raise RuntimeError(f'maximum clocks did not take effect: {sample}')
            self.thread = threading.Thread(target=self.poll, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.initialized:
            clock_watch.nvml.nvmlShutdown()
        # Attempt every restoration even if one command fails.
        failures = []
        for command in [('--reset-gpu-clocks',), ('--reset-memory-clocks',),
                        ('-pm', '1' if self.persistence_was_enabled else '0')]:
            try:
                self.command(*command)
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError('clock restoration failed: ' + '; '.join(failures))

    def report(self):
        active = [s for s in self.samples if not s['throttle_mask'] & 1]
        below = [s for s in active if s['sm_mhz'] < self.sm_mhz]
        return dict(requested_sm_mhz=self.sm_mhz, requested_memory_mhz=self.memory_mhz,
                    interval_seconds=self.interval, samples=self.samples, errors=self.errors,
                    nonidle_samples=len(active), nonidle_below_target=len(below),
                    all_sampled_nonidle_at_target=bool(active) and not below,
                    policy='maximum clocks locked; power/thermal throttling not hidden')
