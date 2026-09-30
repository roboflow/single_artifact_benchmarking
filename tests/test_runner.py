from typing import ClassVar

import json

import pytest
import torch

from sab import runner
from sab.request import ArtifactBenchmarkRequest
from sab.results import load_results
from sab.runner import parse_filter, run_benchmark_on_artifact, run_benchmark_on_artifacts
from sab.runtimes.base import UnavailableOnHost
from tests.fakes import FakeProcessor, FakeRuntime

STATS = [0.5] * 12


class FakeMonitor:
    def __init__(self, throttled):
        self.throttled = throttled
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    def did_throttle(self):
        assert self.exited, "the runner must read the verdict only after the monitor stops"
        return self.throttled


class Harness:
    """Replaces the slow collaborators of the runner and records what runs."""

    def __init__(self):
        self.evaluated = []  # (artifact path, max_images) for each evaluate() call
        self.ensured = []
        self.fail_on: str | None = None
        self.throttled = False
        self.monitors: list[FakeMonitor] = []
        self.evaluate_kwargs: list[dict] = []
        self.class_mappings: list[dict | None] = []
        self.normalize_flags: list[bool] = []

    def evaluate(self, pipeline, images_dir, annotations, class_mapping=None, **kwargs):
        path = pipeline.runtime.artifact_path
        if path == self.fail_on:
            raise RuntimeError("boom")
        monitor = kwargs.pop("monitor")
        assert monitor is self.monitors[-1]
        with monitor:
            pipeline.infer(torch.zeros(3, 8, 8))
        self.evaluated.append((path, kwargs["max_images"]))
        self.evaluate_kwargs.append(kwargs)
        self.class_mappings.append(class_mapping)
        self.normalize_flags.append(pipeline.processor.normalize)
        return STATS

    def select_monitor(self, runtime, device):
        self.monitors.append(FakeMonitor(self.throttled))
        return self.monitors[-1]

    def ensure_artifact(self, path):
        self.ensured.append(path)
        return path


@pytest.fixture
def harness(monkeypatch):
    harness = Harness()
    monkeypatch.setattr(runner, "evaluate", harness.evaluate)
    monkeypatch.setattr(runner, "select_monitor", harness.select_monitor)
    monkeypatch.setattr(runner, "ensure_artifact", harness.ensure_artifact)
    monkeypatch.setattr(runner, "get_coco_class_index_mapping", lambda path: {0: 1, 1: 2})
    monkeypatch.setattr(FakeRuntime, "available", True)
    return harness


class OtherRuntime(FakeRuntime):
    name: ClassVar[str] = "other"


class BrokenAvailabilityRuntime(FakeRuntime):
    name: ClassVar[str] = "broken"

    @classmethod
    def is_available(cls, device: str) -> bool:
        raise OSError("driver exploded")


def make_request(path="a.onnx", **overrides) -> ArtifactBenchmarkRequest:
    fields = dict(artifact_path=path, runtime=FakeRuntime, processor=FakeProcessor, device="cpu")
    return ArtifactBenchmarkRequest(**{**fields, **overrides})


def run_many(requests, output_file, **kwargs):
    return run_benchmark_on_artifacts(requests, "images", "ann.json", output_file=str(output_file), **kwargs)


def saved_paths(output) -> list[str]:
    return [row["artifact_request"]["artifact_path"] for row in load_results(str(output))]


def test_single_row_builds_the_pipeline_from_the_request(harness):
    request = make_request(normalized_in_graph=True, needs_class_remapping=True, max_images=3, max_dets=50, buffer_time=0.5)

    row = run_benchmark_on_artifact(request, "images", "ann.json")

    assert harness.evaluated == [("a.onnx", 3)]
    assert harness.class_mappings == [{1: 0, 2: 1}]
    assert harness.normalize_flags == [False]
    kwargs = harness.evaluate_kwargs[0]
    assert (kwargs["buffer_time"], kwargs["max_dets"]) == (0.5, 50)
    assert row["artifact_request"] == request.dump()
    assert row["accuracy_stats"] == STATS
    assert row["latency_stats"]["count"] == 1
    assert row["throttled"] is False


def test_single_row_without_class_remapping_normalizes_and_passes_no_mapping(harness):
    run_benchmark_on_artifact(make_request(), "images", "ann.json")
    assert harness.class_mappings == [None]
    assert harness.normalize_flags == [True]


def test_single_row_applies_graph_surgery_after_ensure(harness):
    request = make_request(graph_surgery_func=lambda path: path.replace(".onnx", ".cut.onnx"))
    run_benchmark_on_artifact(request, "images", "ann.json")
    assert harness.ensured == ["a.onnx"]
    assert harness.evaluated == [("a.cut.onnx", None)]
    assert request.artifact_path == "a.onnx"


def test_single_row_records_and_shows_a_throttle(harness, capsys):
    harness.throttled = True
    row = run_benchmark_on_artifact(make_request(), "images", "ann.json")
    assert row["throttled"] is True
    assert "Throttled during evaluation" in capsys.readouterr().out


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", None), (" a , b ", {"a", "b"}), (("a", "b"), {"a", "b"})],
)
def test_parse_filter(value, expected):
    assert parse_filter(value) == expected


def test_runs_every_request_and_saves_the_rows(harness, tmp_path):
    output = tmp_path / "out.json"
    rows = run_many([make_request("a.onnx"), make_request("b.onnx")], output)
    assert [r["artifact_request"]["artifact_path"] for r in rows] == ["a.onnx", "b.onnx"]
    assert load_results(str(output)) == rows


def test_runtime_filter_matches_by_runtime_name(harness, tmp_path):
    requests = [make_request("a.onnx"), make_request("b.onnx", runtime=OtherRuntime)]
    rows = run_many(requests, tmp_path / "out.json", runtimes="other,trt")
    assert [r["artifact_request"]["artifact_path"] for r in rows] == ["b.onnx"]
    assert harness.evaluated == [("b.onnx", None)]


def test_device_filter(harness, tmp_path):
    requests = [make_request("a.onnx", device="cpu"), make_request("b.onnx", device="gpu")]
    rows = run_many(requests, tmp_path / "out.json", devices="gpu")
    assert [r["artifact_request"]["device"] for r in rows] == ["gpu"]


def test_unsupported_request_writes_a_row_without_numbers_even_when_the_runtime_is_unavailable(harness, tmp_path, monkeypatch):
    monkeypatch.setattr(FakeRuntime, "available", False)
    output = tmp_path / "out.json"

    (row,) = run_many([make_request(unsupported="no NMS")], output)

    assert harness.evaluated == [] and harness.ensured == []
    assert row["accuracy_stats"] is None and row["latency_stats"] is None and row["throttled"] is None
    assert row["artifact_request"]["unsupported"] == "no NMS"
    assert row["host"]["runtime_version"] == "fake 0.0"
    assert load_results(str(output)) == [row]


def test_unavailable_runtime_prints_one_skip_line_and_writes_no_row(harness, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(FakeRuntime, "available", False)
    rows = run_many([make_request()], tmp_path / "out.json")
    assert rows == [] and harness.evaluated == []
    assert capsys.readouterr().out.count("Skipping") == 1


def test_an_availability_check_that_raises_skips_the_request_and_the_run_continues(harness, tmp_path, capsys):
    requests = [make_request("a.onnx", runtime=BrokenAvailabilityRuntime), make_request("b.onnx")]
    rows = run_many(requests, tmp_path / "out.json")
    assert [r["artifact_request"]["artifact_path"] for r in rows] == ["b.onnx"]
    out = capsys.readouterr().out
    assert out.count("Skipping a.onnx") == 1 and "driver exploded" in out


class RuntimeThatCannotRunHere(FakeRuntime):
    def __init__(self, artifact_path, device, precision):
        raise UnavailableOnHost("this CPU compiles fp16 as fp32")


def test_unavailable_on_host_at_load_skips_the_row_and_runs_the_next(harness, tmp_path, capsys):
    requests = [make_request("a.xml", runtime=RuntimeThatCannotRunHere), make_request("b.onnx")]

    rows = run_many(requests, tmp_path / "out.json")

    out = capsys.readouterr().out
    assert [row["artifact_request"]["artifact_path"] for row in rows] == ["b.onnx"]
    assert "Skipping a.xml" in out and "this CPU compiles fp16 as fp32" in out
    assert [row["artifact_request"]["artifact_path"] for row in load_results(str(tmp_path / "out.json"))] == ["b.onnx"]


def test_resume_keeps_the_old_row_and_does_not_run(harness, tmp_path):
    output = tmp_path / "out.json"
    first = run_many([make_request()], output)
    harness.evaluated.clear()
    second = run_many([make_request()], output)
    assert harness.evaluated == []
    assert second == first


def test_rerun_replaces_the_old_row(harness, tmp_path):
    output = tmp_path / "out.json"
    run_many([make_request()], output)
    rows = run_many([make_request()], output, rerun=True)
    assert len(harness.evaluated) == 2
    assert load_results(str(output)) == rows


def test_max_images_is_part_of_the_key(harness, tmp_path):
    output = tmp_path / "out.json"
    request = make_request()
    run_many([request], output)
    rows = run_many([request], output, max_images=5)
    assert harness.evaluated == [("a.onnx", None), ("a.onnx", 5)]
    assert rows[0]["artifact_request"]["max_images"] == 5
    assert len(load_results(str(output))) == 2
    assert request.max_images is None


def test_buffer_time_is_part_of_the_key(harness, tmp_path):
    output = tmp_path / "out.json"
    run_many([make_request(buffer_time=0.0)], output)
    run_many([make_request(buffer_time=5.0)], output)
    assert len(harness.evaluated) == 2
    assert [r["artifact_request"]["buffer_time"] for r in load_results(str(output))] == [0.0, 5.0]


def test_rows_with_the_old_schema_are_dropped_and_the_run_goes_on(harness, tmp_path, capsys):
    output = tmp_path / "out.json"
    old_rows = [{"artifact_request": {"onnx_path": "old.onnx"}}, {"artifact_request": {"onnx_path": "older.onnx"}}]
    output.write_text(json.dumps(old_rows))

    run_many([make_request()], output)

    assert saved_paths(output) == ["a.onnx"]
    printed = capsys.readouterr().out
    assert str(output) in printed and "2" in printed


def test_rows_of_filtered_out_requests_stay_in_the_file(harness, tmp_path):
    output = tmp_path / "out.json"
    run_many([make_request("a.onnx", device="gpu")], output)
    run_many([make_request("a.onnx", device="gpu"), make_request("b.onnx")], output, devices="cpu")
    devices = [r["artifact_request"]["device"] for r in load_results(str(output))]
    assert devices == ["gpu", "cpu"]


def test_a_crash_keeps_the_saved_rows_and_the_next_run_resumes_from_them(harness, tmp_path):
    output = tmp_path / "out.json"
    requests = [make_request("a.onnx"), make_request("b.onnx")]
    harness.fail_on = "b.onnx"
    with pytest.raises(RuntimeError, match="boom"):
        run_many(requests, output)
    assert saved_paths(output) == ["a.onnx"]

    harness.fail_on = None
    harness.evaluated.clear()
    rows = run_many(requests, output)
    assert harness.evaluated == [("b.onnx", None)]
    assert len(rows) == 2 and saved_paths(output) == ["a.onnx", "b.onnx"]
