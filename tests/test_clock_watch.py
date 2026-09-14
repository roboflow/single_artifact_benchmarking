from sab import clock_watch


class FakeNVML:
    def __init__(self):
        self.clocks = []
        self.freed = False
        self.stopped = False

    def __getattr__(self, name):
        return lambda *args: 0

    def nvmlDeviceGetClockInfo(self, device, kind, value):
        self.clocks.append(kind)
        value._obj.value = {0: 1585, 1: 1590, 2: 5001}[kind]
        return 0

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, device, mask):
        mask._obj.value = 2
        return 0

    def nvmlEventSetFree(self, event_set):
        self.freed = True
        return 0

    def nvmlShutdown(self):
        self.stopped = True
        return 0


def test_clock_domains_reason_labels_and_cleanup(monkeypatch):
    nvml = FakeNVML()
    monkeypatch.setattr(clock_watch, 'nvml', nvml)
    generator = clock_watch.emit_clock_changes()
    assert next(generator) == (1590, 5001, 'Application clocks setting')
    assert nvml.clocks == [1, 2]
    generator.close()
    assert nvml.freed and nvml.stopped


def test_monitor_can_stop_without_another_clock_event(monkeypatch):
    nvml = FakeNVML()
    monkeypatch.setattr(clock_watch, 'nvml', nvml)
    assert list(clock_watch.emit_clock_changes(lambda: True)) == []
    assert nvml.clocks == []
    assert nvml.freed and nvml.stopped
