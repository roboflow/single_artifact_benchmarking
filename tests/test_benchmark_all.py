import json
import subprocess
import sys
from pathlib import Path

import pytest

from sab import benchmark_all
from sab.results import save_results
from tests.test_results import make_row


class ScriptRecorder:
    def __init__(self):
        self.commands: list[list[str]] = []
        self.fail_for: set[str] = set()

    def run(self, command, check):
        assert check is True
        self.commands.append(command)
        if Path(command[1]).name in self.fail_for:
            raise subprocess.CalledProcessError(1, command)
        save_results(command[5], [make_row()])


@pytest.fixture
def recorder(monkeypatch):
    recorder = ScriptRecorder()
    monkeypatch.setattr(benchmark_all.subprocess, "run", recorder.run)
    return recorder


@pytest.fixture
def models_dir(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    for name in ("benchmark_b.py", "benchmark_a.py", "utils.py"):
        (directory / name).write_text("")
    return directory


def run_main(tmp_path, models_dir, **kwargs):
    benchmark_all.main("images", "ann.json", models_dir=str(models_dir), output_dir=str(tmp_path / "out"), **kwargs)


def combined_rows(tmp_path) -> list[dict]:
    return json.loads((tmp_path / "out" / "combined_results.json").read_text())


def test_runs_each_script_with_the_current_interpreter_and_no_flags_by_default(tmp_path, models_dir, recorder):
    run_main(tmp_path, models_dir, buffer_time=1.5)
    out = tmp_path / "out"
    assert recorder.commands == [
        [sys.executable, str(models_dir / "benchmark_a.py"), "images", "ann.json", "1.5", str(out / "benchmark_a_results.txt")],
        [sys.executable, str(models_dir / "benchmark_b.py"), "images", "ann.json", "1.5", str(out / "benchmark_b_results.txt")],
    ]
    assert len(combined_rows(tmp_path)) == 2


def test_the_yololite_script_is_not_run(tmp_path, models_dir, recorder):
    (models_dir / "benchmark_yololite.py").write_text("")
    run_main(tmp_path, models_dir)
    assert [Path(command[1]).name for command in recorder.commands] == ["benchmark_a.py", "benchmark_b.py"]


def test_flags_are_passed_when_set(tmp_path, models_dir, recorder):
    run_main(tmp_path, models_dir, runtimes="fake,trt", devices=("cpu", "gpu"), max_images=5, rerun=True)
    assert recorder.commands[0][6:] == ["--runtimes=fake,trt", "--devices=cpu,gpu", "--max_images=5", "--rerun"]


def test_a_failed_script_is_reported_and_the_others_still_run(tmp_path, models_dir, recorder, capsys):
    recorder.fail_for = {"benchmark_a.py"}
    run_main(tmp_path, models_dir)
    assert len(recorder.commands) == 2
    assert "Failed to run" in capsys.readouterr().out
    assert len(combined_rows(tmp_path)) == 1
