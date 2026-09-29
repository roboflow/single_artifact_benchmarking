from sab.host import cpu_name_from_cpuinfo, host_info
from tests.fakes import FakeRuntime

CPUINFO_X86 = "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Gold 6248\ncpu MHz\t: 2500.0\n"
CPUINFO_PI = "processor\t: 0\n\nHardware\t: BCM2712\nModel\t\t: Raspberry Pi 5 Model B Rev 1.1\n"


def test_cpu_name_from_x86_cpuinfo():
    assert cpu_name_from_cpuinfo(CPUINFO_X86) == "Intel(R) Xeon(R) Gold 6248"


def test_cpu_name_from_pi_cpuinfo():
    assert cpu_name_from_cpuinfo(CPUINFO_PI) == "Raspberry Pi 5 Model B Rev 1.1"


def test_cpu_name_is_none_without_a_known_line():
    assert cpu_name_from_cpuinfo("processor\t: 0\n") is None


def test_host_info_on_linux_with_a_gpu():
    info = host_info(
        FakeRuntime,
        system=lambda: ("Linux", "6.12.0"),
        machine=lambda: "x86_64",
        read_cpuinfo=lambda: CPUINFO_X86,
        read_sysctl=lambda key: None,
        read_gpu_name=lambda: "NVIDIA RTX 4090",
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
        system=lambda: ("Darwin", "27.0.0"),
        machine=lambda: "arm64",
        read_cpuinfo=lambda: (_ for _ in ()).throw(FileNotFoundError()),
        read_sysctl=lambda key: "Apple M4 Pro",
        read_gpu_name=lambda: None,
    )
    assert info["os"] == "Darwin 27.0.0"
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
        system=lambda: ("Linux", "6"),
        machine=fail,
        read_cpuinfo=fail,
        read_sysctl=fail,
        read_gpu_name=fail,
    )
    assert info["arch"] is None
    assert info["cpu"] is None
    assert info["accelerator"] is None
    assert info["runtime_version"] is None
    assert info["os"] == "Linux 6"
