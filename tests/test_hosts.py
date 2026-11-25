from typing import ClassVar

import pytest

from sab import hosts
from sab.hosts import JetsonHost, LinuxHost, MacHost, detect_host, host_info, select_monitor
from sab.hosts.jetson import on_jetson
from sab.hosts.linux import cpu_name_from_cpuinfo
from sab.monitors import CpufreqMonitor, NullMonitor, ThermalStateMonitor, ThrottleMonitor
from tests.fakes import FakeRuntime

CPUINFO_X86 = "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Gold 6248\ncpu MHz\t: 2500.0\n"
CPUINFO_PI = "processor\t: 0\n\nHardware\t: BCM2712\nModel\t\t: Raspberry Pi 5 Model B Rev 1.1\n"


JETSON_BOARD = "NVIDIA Jetson AGX Orin Developer Kit\x00"
PI_BOARD = "Raspberry Pi 5 Model B Rev 1.1\x00"


L4T_RELEASE = "# R36 (release), REVISION: 4.0, GCID: 37537400, BOARD: generic, EABI: aarch64\n"


def no_board() -> str:
    raise FileNotFoundError("no device tree")


def no_l4t_release() -> str:
    raise FileNotFoundError("no L4T release file")


@pytest.fixture(autouse=True)
def not_on_a_jetson(monkeypatch):
    """Detection must not read the machine that runs the tests."""
    monkeypatch.setattr(hosts, "on_jetson", lambda: False)


def test_cpu_name_reads_model_name_on_x86_and_model_on_a_pi():
    assert cpu_name_from_cpuinfo(CPUINFO_X86) == "Intel(R) Xeon(R) Gold 6248"
    assert cpu_name_from_cpuinfo(CPUINFO_PI) == "Raspberry Pi 5 Model B Rev 1.1"
    assert cpu_name_from_cpuinfo("processor\t: 0\n") is None


def test_host_info_on_linux_with_a_gpu():
    info = host_info(
        FakeRuntime,
        "gpu",
        host=LinuxHost(read_cpuinfo=lambda: CPUINFO_X86, read_gpu_name=lambda: "NVIDIA RTX 4090", read_board=no_board),
        system=lambda: ("Linux", "6.12.0"),
        machine=lambda: "x86_64",
    )
    assert info == {
        "os": "Linux 6.12.0",
        "arch": "x86_64",
        "cpu": "Intel(R) Xeon(R) Gold 6248",
        "accelerator": "NVIDIA RTX 4090",
        "board": None,
        "runtime_version": "fake 0.0",
    }


def test_host_info_on_macos_uses_sysctl_for_cpu_and_accelerator():
    info = host_info(
        FakeRuntime,
        "npu",
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
        "gpu",
        host=LinuxHost(read_cpuinfo=fail, read_gpu_name=fail, read_board=fail),
        system=fail,
        machine=fail,
    )
    assert info == {
        "os": None,
        "arch": None,
        "cpu": None,
        "accelerator": None,
        "board": None,
        "runtime_version": None,
    }


def test_a_failed_sysctl_read_on_macos_gives_none():
    def fail(*_):
        raise RuntimeError("no access")

    info = host_info(FakeRuntime, "gpu", host=MacHost(read_sysctl=fail))
    assert info["cpu"] is None
    assert info["accelerator"] is None


@pytest.mark.parametrize("device, expected", [("gpu", "NVIDIA RTX 4090"), ("cpu", None), ("npu", None)])
def test_the_accelerator_on_linux_is_the_cuda_gpu_only_for_a_gpu_row(device, expected):
    host = LinuxHost(read_cpuinfo=lambda: CPUINFO_X86, read_gpu_name=lambda: "NVIDIA RTX 4090", read_board=no_board)
    assert host_info(FakeRuntime, device, host=host)["accelerator"] == expected


@pytest.mark.parametrize("device, expected", [("gpu", "Apple M4 Pro"), ("npu", "Apple M4 Pro"), ("cpu", None)])
def test_the_accelerator_on_macos_is_the_chip_only_for_a_gpu_or_npu_row(device, expected):
    host = MacHost(read_sysctl=lambda key: "Apple M4 Pro")
    info = host_info(FakeRuntime, device, host=host)
    assert info["accelerator"] == expected
    assert info["cpu"] == "Apple M4 Pro"


@pytest.mark.parametrize("system, expected_cpu", [("Darwin", "Apple M4 Pro"), ("Linux", "Intel(R) Xeon(R) Gold 6248")])
def test_host_info_detects_the_platform_from_the_system_reader(monkeypatch, system, expected_cpu):
    monkeypatch.setattr(hosts, "MacHost", lambda: MacHost(read_sysctl=lambda key: "Apple M4 Pro"))
    monkeypatch.setattr(hosts, "LinuxHost", lambda: LinuxHost(read_cpuinfo=lambda: CPUINFO_X86, read_board=no_board))
    info = host_info(FakeRuntime, "cpu", system=lambda: (system, "1.0"))
    assert info["cpu"] == expected_cpu


@pytest.mark.parametrize("system, expected", [("Darwin", MacHost), ("Linux", LinuxHost), ("Windows", LinuxHost)])
def test_detect_host(system, expected):
    assert type(detect_host(system)) is expected


def test_detect_host_gives_the_jetson_platform_only_on_linux(monkeypatch):
    monkeypatch.setattr(hosts, "on_jetson", lambda: True)
    assert type(detect_host("Linux")) is JetsonHost
    assert type(detect_host("Darwin")) is MacHost
    assert type(detect_host("Windows")) is LinuxHost


@pytest.mark.parametrize(
    "read_board, read_l4t_release, environ, expected",
    [
        (lambda: JETSON_BOARD, no_l4t_release, {}, True),
        (lambda: PI_BOARD, no_l4t_release, {}, False),
        (no_board, no_l4t_release, {}, False),
        # A container hides the device tree. The NVIDIA runtime can mount the L4T release file.
        (no_board, lambda: L4T_RELEASE, {}, True),
        (lambda: "Vendor carrier board\x00", lambda: L4T_RELEASE, {}, True),
        (no_board, lambda: "not a release line", {}, False),
        # The Roboflow Inference images for a Jetson set this variable.
        (no_board, no_l4t_release, {"RUNNING_ON_JETSON": "True"}, True),
        (no_board, no_l4t_release, {"RUNNING_ON_JETSON": "False"}, False),
    ],
)
def test_a_jetson_is_known_by_its_board_name_its_l4t_release_or_its_image(read_board, read_l4t_release, environ, expected):
    assert on_jetson(read_board=read_board, read_l4t_release=read_l4t_release, environ=environ) is expected


def test_a_linux_board_with_a_device_tree_records_its_board_name():
    host = LinuxHost(read_cpuinfo=lambda: CPUINFO_PI, read_gpu_name=lambda: None, read_board=lambda: PI_BOARD)
    assert host.describe()["board"] == "Raspberry Pi 5 Model B Rev 1.1"


def test_a_jetson_adds_the_l4t_release_and_the_power_mode_to_the_linux_fields():
    host = JetsonHost(
        linux=LinuxHost(
            read_cpuinfo=lambda: "processor\t: 0\nmodel name\t: ARMv8 Processor rev 1 (v8l)\n",
            read_gpu_name=lambda: "Orin",
            read_board=lambda: JETSON_BOARD,
        ),
        read_l4t_release=lambda: L4T_RELEASE,
        read_power_mode=lambda: "NV Power Mode: MAXN\n0\n",
    )
    assert host.describe() == {
        "cpu": "ARMv8 Processor rev 1 (v8l)",
        "board": "NVIDIA Jetson AGX Orin Developer Kit",
        "l4t": "R36.4.0",
        "power_mode": "MAXN",
    }
    assert host.accelerator("gpu") == "Orin"
    assert host.accelerator("cpu") is None


def test_failed_jetson_reads_give_none():
    def fail():
        raise RuntimeError("no access")

    linux = LinuxHost(read_cpuinfo=fail, read_gpu_name=fail, read_board=fail)
    info = JetsonHost(linux=linux, read_l4t_release=fail, read_power_mode=fail).describe()
    assert info["l4t"] is None
    assert info["power_mode"] is None


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
        ("executorch", "npu", NullMonitor),
    ],
)
def test_linux_selects_the_monitor_from_the_runtime_and_the_device(runtime_name, device, expected):
    assert type(select_monitor(runtime_named(runtime_name), device, host=LinuxHost())) is expected


@pytest.mark.parametrize("device", ["cpu", "gpu", "npu"])
def test_macos_selects_the_thermal_state_monitor_for_every_device(device):
    assert type(select_monitor(runtime_named("coreml"), device, host=MacHost())) is ThermalStateMonitor


@pytest.mark.parametrize("runtime_name", ["tensorrt", "onnxruntime"])
def test_a_jetson_gpu_row_watches_the_gpu_frequency_and_not_nvml(runtime_name):
    reads = []
    host = JetsonHost(read_gpu_frequencies=lambda: reads.append(1) or [1267.2])
    monitor = select_monitor(runtime_named(runtime_name), "gpu", host=host)
    with monitor:
        pass
    assert type(monitor) is CpufreqMonitor
    assert reads, "the monitor of a GPU row must read the GPU frequency"


@pytest.mark.parametrize("device, expected", [("cpu", CpufreqMonitor), ("npu", NullMonitor)])
def test_a_jetson_row_off_the_gpu_selects_the_linux_monitor(device, expected):
    def must_not_run():
        raise AssertionError("the GPU reader ran for a row off the GPU")

    host = JetsonHost(read_gpu_frequencies=must_not_run)
    monitor = select_monitor(runtime_named("onnxruntime"), device, host=host)
    with monitor:
        pass
    assert type(monitor) is expected


@pytest.mark.parametrize("system, expected", [("Darwin", ThermalStateMonitor), ("Linux", CpufreqMonitor)])
def test_select_monitor_detects_the_platform(monkeypatch, system, expected):
    monkeypatch.setattr(hosts.platform, "system", lambda: system)
    assert type(select_monitor(runtime_named("onnxruntime"), "cpu")) is expected
