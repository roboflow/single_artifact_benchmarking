"""The nvidia-smi wrapper drops sudo when it already runs as root."""

from sab import clock_watch


def record_commands(monkeypatch) -> list[list[str]]:
    commands: list[list[str]] = []
    monkeypatch.setattr(clock_watch, "run", lambda command, **kwargs: commands.append(command))
    return commands


def test_root_calls_nvidia_smi_directly(monkeypatch):
    # The benchmark containers run as root and carry no sudo.
    monkeypatch.setattr(clock_watch.os, "geteuid", lambda: 0)
    monkeypatch.setattr(clock_watch.shutil, "which", lambda name: None)
    commands = record_commands(monkeypatch)

    clock_watch.lock_clocks(1590)

    assert commands == [["nvidia-smi", "--lock-gpu-clocks", "1590"]]


def test_non_root_with_sudo_keeps_sudo(monkeypatch):
    monkeypatch.setattr(clock_watch.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(clock_watch.shutil, "which", lambda name: "/usr/bin/sudo")
    commands = record_commands(monkeypatch)

    clock_watch.enable_persistence(True)

    assert commands == [["sudo", "nvidia-smi", "-pm", "1"]]


def test_non_root_without_sudo_still_tries(monkeypatch):
    # Without sudo the call is the only thing left to try, and its failure is
    # visible. A silent skip would leave the clocks unlocked.
    monkeypatch.setattr(clock_watch.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(clock_watch.shutil, "which", lambda name: None)
    commands = record_commands(monkeypatch)

    clock_watch.unlock_clocks()

    assert commands == [["nvidia-smi", "--reset-gpu-clocks"], ["nvidia-smi", "--reset-memory-clocks"]]
