from typing import ClassVar

import pytest

from sab.hosts import LinuxHost, MacHost, detect_host, host_info, select_monitor
from sab.hosts.linux import cpu_name_from_cpuinfo
from sab.monitors import CpufreqMonitor, NullMonitor, ThermalStateMonitor, ThrottleMonitor
from tests.fakes import FakeRuntime

CPUINFO_X86 = "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Gold 6248\ncpu MHz\t: 2500.0\n"
CPUINFO_PI = "processor\t: 0\n\nHardware\t: BCM2712\nModel\t\t: Raspberry Pi 5 Model B Rev 1.1\n"


def test_cpu_name_reads_model_name_on_x86_and_model_on_a_pi():
    assert cpu_name_from_cpuinfo(CPUINFO_X86) == "Intel(R) Xeon(R) Gold 6248"
    assert cpu_name_from_cpuinfo(CPUINFO_PI) == "Raspberry Pi 5 Model B Rev 1.1"
    assert cpu_name_from_cpuinfo("processor\t: 0\n") is None


def test_host_info_on_linux_with_a_gpu():
    info = host_info(
        FakeRuntime,
        host=LinuxHost(read_cpuinfo=lambda: CPUINFO_X86, read_gpu_name=lambda: "NVIDIA RTX 4090"),
        system=lambda: ("Linux", "6.12.0"),
        machine=lambda: "x86_64",
    )
    assert info == {
        "os": "Linux 6.12.0",
        "arch": "x86_64",
        "cpu": "Intel(R) Xeon(R) Gold 6248",
        "accelerator": "NVIDIA RTX 4090",
        "runtime_version": "fake 0.0",
    }


def test_host_info_on_macos_uses_sysctl_for_cpu_and_accelerator():
    info = host_info(
        FakeRuntime,
        host=MacHost(read_sysctl=lambda key: "Apple M4 Pro"),
        system=lambda: ("Darwin", "27.0.0"),
        machine=lambda: "arm64",
    )
    assert info["cpu"] == "Apple M4 Pro"
    assert info["accelerator"] == "Apple M4 Pro"


def test_a_failed_read_gives_none_and_never_raises():
    def fail(*_):
        raise RuntimeError("no access")

    class BrokenVersionRuntime(FakeRuntime):
        @classmethod
        def version(cls):
            raise RuntimeError("not installed")

    info = host_info(
        BrokenVersionRuntime,
        host=LinuxHost(read_cpuinfo=fail, read_gpu_name=fail),
        system=fail,
        machine=fail,
    )
    assert info == {"os": None, "arch": None, "cpu": None, "accelerator": None, "runtime_version": None}


def test_a_failed_sysctl_read_on_macos_gives_none():
    def fail(*_):
        raise RuntimeError("no access")

    assert MacHost(read_sysctl=fail).describe() == {"cpu": None, "accelerator": None}


@pytest.mark.parametrize("system, expected", [("Darwin", MacHost), ("Linux", LinuxHost), ("Windows", LinuxHost)])
def test_detect_host(system, expected):
    assert type(detect_host(system)) is expected


def runtime_named(runtime_name: str) -> type[FakeRuntime]:
    class Named(FakeRuntime):
        name: ClassVar[str] = runtime_name

    return Named


@pytest.mark.parametrize(
    "runtime_name, device, expected",
    [
        ("tensorrt", "gpu", ThrottleMonitor),
        ("onnxruntime", "gpu", ThrottleMonitor),
        ("onnxruntime", "cpu", CpufreqMonitor),
        ("coreml", "gpu", NullMonitor),
    ],
)
def test_linux_selects_the_monitor_from_the_runtime_and_the_device(runtime_name, device, expected):
    assert type(select_monitor(runtime_named(runtime_name), device, host=LinuxHost())) is expected


@pytest.mark.parametrize("device", ["cpu", "gpu", "npu"])
def test_macos_selects_the_thermal_state_monitor_for_every_device(device):
    assert type(select_monitor(runtime_named("coreml"), device, host=MacHost())) is ThermalStateMonitor
