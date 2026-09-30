import ctypes
import ctypes.util
import threading
from contextlib import AbstractContextManager, nullcontext
from functools import cache
from typing import Callable, Self

FOUNDATION_PATH = "/System/Library/Frameworks/Foundation.framework/Foundation"
STATE_NAMES = ("nominal", "fair", "serious", "critical")
FAIR = 1

StateReader = Callable[[], int]


@cache
def _thermal_state_getter() -> Callable[[], int]:
    objc_path = ctypes.util.find_library("objc")
    if objc_path is None:
        raise OSError("libobjc is not available")
    objc = ctypes.CDLL(objc_path)
    ctypes.CDLL(FOUNDATION_PATH)  # loading Foundation registers NSProcessInfo

    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    process_info_class = objc.objc_getClass(b"NSProcessInfo")
    if not process_info_class:
        raise OSError("NSProcessInfo is not available")
    process_info_selector = objc.sel_registerName(b"processInfo")
    thermal_state_selector = objc.sel_registerName(b"thermalState")

    # On arm64, each objc_msgSend call needs a function type that matches its signature.
    msg_send_address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    send_returning_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(msg_send_address)
    # NSInteger is `long`: 64-bit on both arm64 and x86_64 macOS.
    send_returning_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(msg_send_address)

    def get_thermal_state() -> int:
        process_info = send_returning_object(process_info_class, process_info_selector)
        return send_returning_long(process_info, thermal_state_selector)

    return get_thermal_state


def read_thermal_state() -> int:
    """NSProcessInfo thermalState: 0 nominal, 1 fair, 2 serious, 3 critical. Raises OSError when unavailable."""
    return _thermal_state_getter()()


class ThermalStateMonitor:
    """Polls the system thermal state during the run. Fair or higher counts as throttling.

    macOS derives thermalState from the system thermal pressure level. At moderate pressure (fair), the system
    already limits the CPU and GPU clocks. Apple Silicon has no per-core frequency that a normal user can read.

    The verdict is None when the state is not readable at enter, for example on Linux.
    """

    def __init__(self, poll_interval_s: float = 0.5, read_state: StateReader = read_thermal_state):
        self._poll_interval_s = poll_interval_s
        self._read_state = read_state
        self._start_state: int | None = None
        self._max_state = 0
        self._samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        try:
            start_state = self._read_state()
        except OSError:
            return self
        self._start_state = start_state
        self._max_state = start_state
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_until_stopped, daemon=True)
        self._thread.start()
        return self

    def busy(self) -> AbstractContextManager:
        # The thermal state belongs to the whole system, so the idle time between images needs no filter.
        return nullcontext()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def poll_once(self) -> None:
        if self._start_state is None:
            return
        try:
            state = self._read_state()
        except OSError:
            return  # one failed read must not end the watch
        self._samples += 1
        self._max_state = max(self._max_state, state)

    def _poll_until_stopped(self) -> None:
        while not self._stop.wait(self._poll_interval_s):
            self.poll_once()

    def did_throttle(self) -> bool | None:
        if self._start_state is None:
            return None
        return self._max_state >= FAIR

    def summary(self) -> dict:
        if self._start_state is None:
            return {}
        return {
            "start_state": STATE_NAMES[self._start_state],
            "max_state": STATE_NAMES[self._max_state],
            "samples": self._samples,
        }
