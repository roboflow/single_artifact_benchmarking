import pytest
import torch

from sab import runner
from sab.request import ArtifactBenchmarkRequest
from sab.results import load_results, save_results
from sab.runner import parse_filter, run_benchmark_on_artifact, run_benchmark_on_artifacts
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
        return self.throttled

    def summary(self):
        return {}


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

    def evaluate(self, pipeline, images_dir, annotations, class_mapping=None, **kwargs):
        path = pipeline.runtime.artifact_path
        if path == self.fail_on:
            raise RuntimeError("boom")
        assert self.monitors[-1].entered and not self.monitors[-1].exited
        pipeline.infer(torch.zeros(3, 8, 8))
        self.evaluated.append((path, kwargs["max_images"]))
        self.evaluate_kwargs.append(kwargs)
        self.class_mappings.append(class_mapping)
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


def make_request(path="a.onnx", **overrides) -> ArtifactBenchmarkRequest:
    fields = dict(artifact_path=path, runtime=FakeRuntime, processor=FakeProcessor, device="cpu")
    return ArtifactBenchmarkRequest(**{**fields, **overrides})


def run_many(requests, output_file, **kwargs):
    return run_benchmark_on_artifacts(requests, "images", "ann.json", output_file=str(output_file), **kwargs)


def test_single_row_has_the_new_schema(harness):
    row = run_benchmark_on_artifact(make_request(), "images", "ann.json")
    assert set(row) == {"artifact_request", "host", "accuracy_stats", "latency_stats", "throttled"}
    assert row["artifact_request"] == make_request().dump()
    assert row["accuracy_stats"] == STATS
    assert "median" in row["latency_stats"]
    assert row["throttled"] is False
    assert row["host"]["runtime_version"] == "fake 0.0"


def test_single_row_builds_the_pipeline_from_the_request(harness):
    request = make_request(normalized_in_graph=True, needs_class_remapping=True, max_images=3, max_dets=50, buffer_time=0.5)
    run_benchmark_on_artifact(request, "images", "ann.json")
    assert harness.evaluated == [("a.onnx", 3)]
    assert harness.class_mappings == [{1: 0, 2: 1}]
    kwargs = harness.evaluate_kwargs[0]
    assert (kwargs["buffer_time"], kwargs["max_dets"]) == (0.5, 50)


def test_single_row_without_class_remapping_passes_none(harness):
    run_benchmark_on_artifact(make_request(), "images", "ann.json")
    assert harness.class_mappings == [None]


def test_single_row_applies_graph_surgery_after_ensure(harness):
    request = make_request(graph_surgery_func=lambda path: path.replace(".onnx", ".cut.onnx"))
    run_benchmark_on_artifact(request, "images", "ann.json")
    assert harness.ensured == ["a.onnx"]
    assert harness.evaluated == [("a.cut.onnx", None)]
    assert request.artifact_path == "a.onnx"


@pytest.mark.parametrize("throttled, message", [(True, "Throttled during evaluation"), (None, "Throttle state unknown")])
def test_single_row_reports_throttle_state(harness, capsys, throttled, message):
    harness.throttled = throttled
    row = run_benchmark_on_artifact(make_request(), "images", "ann.json")
    assert row["throttled"] is throttled
    assert message in capsys.readouterr().out


def test_monitor_closes_before_the_verdict_is_read(harness):
    run_benchmark_on_artifact(make_request(), "images", "ann.json")
    assert harness.monitors[0].exited


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", None), ("a,b", {"a", "b"}), (" a , b ", {"a", "b"}), (("a", "b"), {"a", "b"}), (["a"], {"a"}), ("a", {"a"})],
)
def test_parse_filter(value, expected):
    assert parse_filter(value) == expected


def test_runs_every_request_and_saves_the_rows(harness, tmp_path):
    output = tmp_path / "out.json"
    rows = run_many([make_request("a.onnx"), make_request("b.onnx")], output)
    assert [r["artifact_request"]["artifact_path"] for r in rows] == ["a.onnx", "b.onnx"]
    assert load_results(str(output)) == rows


def test_runtime_filter_skips_silently(harness, tmp_path):
    rows = run_many([make_request()], tmp_path / "out.json", runtimes="onnxruntime,trt")
    assert rows == [] and harness.evaluated == []
    assert load_results(str(tmp_path / "out.json")) == []


def test_runtime_filter_matches_by_runtime_name(harness, tmp_path):
    rows = run_many([make_request()], tmp_path / "out.json", runtimes=("fake",))
    assert len(rows) == 1


def test_device_filter(harness, tmp_path):
    requests = [make_request("a.onnx", device="cpu"), make_request("b.onnx", device="gpu")]
    rows = run_many(requests, tmp_path / "out.json", devices="gpu")
    assert [r["artifact_request"]["device"] for r in rows] == ["gpu"]


def test_unsupported_request_writes_a_row_without_numbers_and_does_not_run(harness, tmp_path):
    output = tmp_path / "out.json"
    rows = run_many([make_request(unsupported="no NMS")], output)
    assert harness.evaluated == [] and harness.ensured == []
    (row,) = rows
    assert row["accuracy_stats"] is None and row["latency_stats"] is None and row["throttled"] is None
    assert row["artifact_request"]["unsupported"] == "no NMS"
    assert row["host"]["runtime_version"] == "fake 0.0"
    assert load_results(str(output)) == [row]


def test_unsupported_row_is_written_even_when_the_runtime_is_unavailable(harness, tmp_path, monkeypatch):
    monkeypatch.setattr(FakeRuntime, "available", False)
    rows = run_many([make_request(unsupported="no NMS")], tmp_path / "out.json")
    assert len(rows) == 1


def test_unavailable_runtime_prints_one_skip_line_and_writes_no_row(harness, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(FakeRuntime, "available", False)
    rows = run_many([make_request()], tmp_path / "out.json")
    out = capsys.readouterr().out
    assert rows == [] and harness.evaluated == []
    assert out.count("Skipping") == 1 and "fake" in out
    assert load_results(str(tmp_path / "out.json")) == []


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
    run_many([make_request()], output)
    rows = run_many([make_request()], output, max_images=5)
    assert harness.evaluated == [("a.onnx", None), ("a.onnx", 5)]
    assert rows[0]["artifact_request"]["max_images"] == 5
    assert len(load_results(str(output))) == 2


def test_max_images_override_leaves_the_caller_request_unchanged(harness, tmp_path):
    request = make_request()
    run_many([request], tmp_path / "out.json", max_images=5)
    assert request.max_images is None


def test_rows_for_other_keys_in_the_file_are_preserved(harness, tmp_path):
    output = tmp_path / "out.json"
    run_many([make_request("other.onnx")], output)
    run_many([make_request("a.onnx")], output)
    paths = [r["artifact_request"]["artifact_path"] for r in load_results(str(output))]
    assert paths == ["other.onnx", "a.onnx"]


def test_rows_of_filtered_out_requests_stay_in_the_file(harness, tmp_path):
    output = tmp_path / "out.json"
    run_many([make_request("a.onnx", device="gpu")], output)
    run_many([make_request("a.onnx", device="gpu"), make_request("b.onnx")], output, devices="cpu")
    devices = [r["artifact_request"]["device"] for r in load_results(str(output))]
    assert devices == ["gpu", "cpu"]


def test_the_file_is_saved_after_every_row(harness, tmp_path):
    output = tmp_path / "out.json"
    harness.fail_on = "b.onnx"
    with pytest.raises(RuntimeError, match="boom"):
        run_many([make_request("a.onnx"), make_request("b.onnx")], output)
    saved = load_results(str(output))
    assert [r["artifact_request"]["artifact_path"] for r in saved] == ["a.onnx"]


def test_a_crashed_run_resumes_from_the_saved_rows(harness, tmp_path):
    output = tmp_path / "out.json"
    requests = [make_request("a.onnx"), make_request("b.onnx")]
    harness.fail_on = "b.onnx"
    with pytest.raises(RuntimeError):
        run_many(requests, output)
    harness.fail_on = None
    harness.evaluated.clear()
    rows = run_many(requests, output)
    assert harness.evaluated == [("b.onnx", None)]
    assert len(rows) == 2 and len(load_results(str(output))) == 2


def test_unrelated_rows_survive_a_crash(harness, tmp_path):
    output = tmp_path / "out.json"
    other = {"artifact_request": make_request("other.onnx").dump(), "host": {}, "accuracy_stats": STATS,
             "latency_stats": {"median": 1.0}, "throttled": False}
    save_results(str(output), [other])
    harness.fail_on = "a.onnx"
    with pytest.raises(RuntimeError):
        run_many([make_request("a.onnx")], output)
    assert load_results(str(output)) == [other]
