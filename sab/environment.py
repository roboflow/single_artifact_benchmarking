"""Environment metadata for benchmark artifacts.

Latency numbers are only comparable within a known software/hardware stack.
This module captures that stack so results files are self-describing. Every
probe degrades to None rather than raising: the module must work on GPU-less
dev machines, CPU-only nodes, and Jetson devices alike.
"""

import datetime
import hashlib
import os
import platform
import socket
import subprocess


def _probe(fn):
    try:
        return fn()
    except Exception:
        return None


def _git_commit() -> str | None:
    sha = os.environ.get("SAB_GIT_SHA")
    if sha and sha != "unknown":
        return sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _tensorrt_version() -> str | None:
    import tensorrt

    return tensorrt.__version__


def _torch_version() -> str | None:
    import torch

    return torch.__version__


def _torch_cuda_version() -> str | None:
    import torch

    return torch.version.cuda


def _onnxruntime_version() -> str | None:
    import onnxruntime

    return onnxruntime.__version__


def _gpu_name_and_driver() -> tuple[str | None, str | None]:
    import ctypes as ct
    import ctypes.util

    lib_path = ctypes.util.find_library("nvidia-ml")
    if not lib_path:
        return None, None
    nvml = ct.CDLL(lib_path)
    if nvml.nvmlInit_v2() != 0:
        return None, None
    try:
        name_buf = ct.create_string_buffer(96)
        driver_buf = ct.create_string_buffer(96)
        gpu_name = None
        driver_version = None
        device = ct.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(0, ct.byref(device)) == 0:
            if nvml.nvmlDeviceGetName(device, name_buf, ct.c_uint(96)) == 0:
                gpu_name = name_buf.value.decode("utf-8", errors="replace")
        if nvml.nvmlSystemGetDriverVersion(driver_buf, ct.c_uint(96)) == 0:
            driver_version = driver_buf.value.decode("utf-8", errors="replace")
        return gpu_name, driver_version
    finally:
        nvml.nvmlShutdown()


def _l4t_release() -> str | None:
    with open("/etc/nv_tegra_release") as f:
        return f.readline().strip()


def _jetson_device_model() -> str | None:
    with open("/proc/device-tree/model", "rb") as f:
        return f.read().decode("utf-8", errors="replace").rstrip("\x00").strip()


def collect_environment() -> dict:
    gpu_info = _probe(_gpu_name_and_driver) or (None, None)
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": _probe(socket.gethostname),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "sab_commit": _probe(_git_commit),
        "base_image": os.environ.get("SAB_BASE_IMAGE"),
        "hardware_label": os.environ.get("SAB_HARDWARE"),
        "gpu_name": gpu_info[0],
        "driver_version": gpu_info[1],
        "tensorrt_version": _probe(_tensorrt_version),
        "torch_version": _probe(_torch_version),
        "torch_cuda_version": _probe(_torch_cuda_version),
        "onnxruntime_version": _probe(_onnxruntime_version),
        "l4t_release": _probe(_l4t_release),
        "jetson_device_model": _probe(_jetson_device_model),
    }


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
