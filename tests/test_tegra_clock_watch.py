"""The tegra watch over fake sysfs trees: clean, throttled, and failed."""

import pytest

from sab import tegra_clock_watch
from sab.tegra_clock_watch import (
    DOWNCLOCKED_REASON,
    TegraClockReadError,
    TegraClockWatch,
    find_gpu_devfreq_node,
    is_throttled,
    is_tegra,
    open_tegra_clock_reader,
)

MAX_HZ = 1_300_500_000
IDLE_HZ = 306_000_000


def write_devfreq(root, cur_hz=MAX_HZ, max_hz=MAX_HZ, node_name="17000000.ga10b"):
    node = root / "devfreq" / node_name
    node.mkdir(parents=True)
    (node / "cur_freq").write_text(f"{cur_hz}\n")
    (node / "max_freq").write_text(f"{max_hz}\n")
    return root / "devfreq"


def write_thermal_zone(root, index, temp_mc, trips=(("passive", 97000), ("critical", 103000)), name="GPU-therm"):
    zone = root / "thermal" / f"thermal_zone{index}"
    zone.mkdir(parents=True, exist_ok=True)
    (zone / "type").write_text(f"{name}\n")
    (zone / "temp").write_text(f"{temp_mc}\n")
    for trip_index, (trip_type, limit) in enumerate(trips):
        (zone / f"trip_point_{trip_index}_type").write_text(f"{trip_type}\n")
        (zone / f"trip_point_{trip_index}_temp").write_text(f"{limit}\n")
    return root / "thermal"


def reader_over(tmp_path, cur_hz=MAX_HZ, temp_mc=45000, **thermal):
    devfreq_root = write_devfreq(tmp_path, cur_hz=cur_hz)
    thermal_root = write_thermal_zone(tmp_path, 0, temp_mc, **thermal)
    return open_tegra_clock_reader(
        devfreq_root=devfreq_root,
        thermal_root=thermal_root,
        overcurrent_root=tmp_path / "no-soctherm",
    )


def test_full_speed_and_cool_is_clean(tmp_path):
    assert reader_over(tmp_path).sample().reasons == ()


def test_governor_downclock_alone_is_not_throttling(tmp_path):
    # AI1 cannot lock its clocks, so the GPU drops to an idle step between
    # passes. That is the governor, not a limit.
    reasons = reader_over(tmp_path, cur_hz=IDLE_HZ).sample().reasons
    assert reasons == (DOWNCLOCKED_REASON,)
    assert not is_throttled(reasons)


def test_thermal_trip_alone_is_not_throttling(tmp_path):
    reasons = reader_over(tmp_path, temp_mc=99000).sample().reasons
    assert reasons == ("thermal_passive:GPU-therm",)
    assert not is_throttled(reasons)


def test_downclock_at_a_thermal_trip_is_throttling(tmp_path):
    reasons = reader_over(tmp_path, cur_hz=IDLE_HZ, temp_mc=99000).sample().reasons
    assert reasons == (DOWNCLOCKED_REASON, "thermal_passive:GPU-therm")
    assert is_throttled(reasons)


def test_critical_trip_is_reported_with_its_type(tmp_path):
    reasons = reader_over(tmp_path, cur_hz=IDLE_HZ, temp_mc=104000).sample().reasons
    assert reasons == (DOWNCLOCKED_REASON, "thermal_passive:GPU-therm", "thermal_critical:GPU-therm")


def test_disabled_trip_slots_are_ignored(tmp_path):
    reasons = reader_over(tmp_path, cur_hz=IDLE_HZ, temp_mc=45000, trips=(("passive", 0),)).sample().reasons
    assert reasons == (DOWNCLOCKED_REASON,)


def test_unreadable_frequency_raises(tmp_path):
    devfreq_root = write_devfreq(tmp_path)
    (devfreq_root / "17000000.ga10b" / "cur_freq").write_text("not a number\n")
    reader = open_tegra_clock_reader(
        devfreq_root=devfreq_root,
        thermal_root=tmp_path / "no-thermal",
        overcurrent_root=tmp_path / "no-soctherm",
    )
    with pytest.raises(TegraClockReadError):
        reader.sample()


def test_missing_gpu_node_raises(tmp_path):
    (tmp_path / "devfreq").mkdir()
    with pytest.raises(TegraClockReadError):
        find_gpu_devfreq_node(tmp_path / "devfreq")


def test_gpu_node_found_by_its_name_file(tmp_path):
    node = tmp_path / "devfreq" / "13e10000.host1x"
    node.mkdir(parents=True)
    (node / "name").write_text("gpu\n")
    assert find_gpu_devfreq_node(tmp_path / "devfreq") == node


def test_overcurrent_counter_rise_is_a_reason(tmp_path):
    soctherm = tmp_path / "soctherm" / "oc1"
    soctherm.mkdir(parents=True)
    (soctherm / "event_cnt").write_text("3\n")
    devfreq_root = write_devfreq(tmp_path, cur_hz=IDLE_HZ)
    reader = open_tegra_clock_reader(
        devfreq_root=devfreq_root,
        thermal_root=tmp_path / "no-thermal",
        overcurrent_root=tmp_path / "soctherm",
    )

    # The counter counts the life of the boot, so the first reading is only a
    # baseline.
    assert reader.sample().reasons == (DOWNCLOCKED_REASON,)

    (soctherm / "event_cnt").write_text("4\n")
    reasons = reader.sample().reasons
    assert reasons == (DOWNCLOCKED_REASON, "overcurrent:oc1")
    assert is_throttled(reasons)


class FakeReader:
    def __init__(self, samples):
        self._samples = list(samples)
        self.closed = False

    def sample(self):
        if not self._samples:
            return tegra_clock_watch.TegraClockSample(MAX_HZ, MAX_HZ, ())
        return self._samples.pop(0)

    def close(self):
        self.closed = True


def sample(reasons, sm_hz=IDLE_HZ):
    return tegra_clock_watch.TegraClockSample(sm_hz=sm_hz, sm_max_hz=MAX_HZ, reasons=reasons)


def test_watch_reports_a_clean_pass():
    with TegraClockWatch(reader=FakeReader([]), poll_interval_s=0.01) as watch:
        pass
    assert not watch.did_throttle()


def test_watch_reports_a_throttled_pass():
    reader = FakeReader([sample((DOWNCLOCKED_REASON, "thermal_passive:GPU-therm"))])
    with TegraClockWatch(reader=reader, poll_interval_s=0.01) as watch:
        pass
    assert watch.did_throttle()
    assert watch.reasons() == (DOWNCLOCKED_REASON, "thermal_passive:GPU-therm")


class FailingReader:
    def sample(self):
        raise TegraClockReadError("sysfs went away")

    def close(self):
        pass


def test_a_dead_watch_never_certifies_a_clean_pass():
    watch = TegraClockWatch(reader=FailingReader(), poll_interval_s=0.01)
    with pytest.raises(RuntimeError, match="tegra clock watch failed"):
        with watch:
            pass


def test_did_throttle_raises_after_a_watcher_failure():
    watch = TegraClockWatch(reader=FailingReader(), poll_interval_s=0.01)
    watch.monitor_throttling()
    watch.stop()
    with pytest.raises(RuntimeError, match="tegra clock watch failed"):
        watch.did_throttle()


def test_watch_failure_never_masks_the_benchmark_error():
    watch = TegraClockWatch(reader=FailingReader(), poll_interval_s=0.01)
    with pytest.raises(ValueError, match="the benchmark blew up"):
        with watch:
            raise ValueError("the benchmark blew up")


def test_is_tegra_follows_the_stated_hardware(monkeypatch, tmp_path):
    monkeypatch.setattr(tegra_clock_watch, "TEGRA_RELEASE_FILE", tmp_path / "absent")
    monkeypatch.setenv("SAB_HARDWARE", "AI1")
    assert is_tegra()
    monkeypatch.setenv("SAB_HARDWARE", "T4")
    assert not is_tegra()


def test_is_tegra_falls_back_to_the_l4t_release_file(monkeypatch, tmp_path):
    monkeypatch.delenv("SAB_HARDWARE", raising=False)
    release_file = tmp_path / "nv_tegra_release"
    monkeypatch.setattr(tegra_clock_watch, "TEGRA_RELEASE_FILE", release_file)
    assert not is_tegra()
    release_file.write_text("# R36 (release), REVISION: 4.3\n")
    assert is_tegra()
