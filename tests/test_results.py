import json
import os

import pytest

from sab.request import ArtifactBenchmarkRequest
from sab.results import load_results, pretty_print_results, result_key, save_results
from tests.fakes import FakeProcessor, FakeRuntime

STATS = [0.40, 0.60, 0.45, 0.1, 0.2, 0.3, 0.31, 0.32, 0.33, 0.11, 0.22, 0.33]


def make_request(**overrides) -> ArtifactBenchmarkRequest:
    fields = dict(artifact_path="a.onnx", runtime=FakeRuntime, processor=FakeProcessor, device="cpu")
    return ArtifactBenchmarkRequest(**{**fields, **overrides})


def make_row(request=None, **overrides) -> dict:
    row = {
        "artifact_request": (request or make_request()).dump(),
        "host": {"os": "Linux 6"},
        "accuracy_stats": STATS,
        "latency_stats": {"median": 12.345},
        "throttled": False,
    }
    return {**row, **overrides}


def test_load_results_of_a_missing_file_is_empty(tmp_path):
    assert load_results(str(tmp_path / "missing.json")) == []


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "out.json")
    rows = [make_row()]
    save_results(path, rows)
    assert load_results(path) == rows


def test_save_creates_the_parent_directory(tmp_path):
    path = str(tmp_path / "nested" / "out.json")
    save_results(path, [make_row()])
    assert os.path.exists(path)


def test_save_leaves_no_temp_file_and_keeps_the_old_file_on_failure(tmp_path):
    path = str(tmp_path / "out.json")
    save_results(path, [make_row()])
    with pytest.raises(TypeError):
        save_results(path, [{"bad": object()}])
    assert load_results(path) == [make_row()]
    assert os.listdir(tmp_path) == ["out.json"]


def test_save_accepts_numpy_numbers(tmp_path):
    import numpy as np

    path = str(tmp_path / "out.json")
    save_results(path, [make_row(latency_stats={"median": np.float32(1.5), "count": np.int64(3)})])
    assert json.load(open(path))[0]["latency_stats"] == {"median": 1.5, "count": 3}


@pytest.mark.parametrize("overrides", [{}, {"max_images": 5}, {"precision": "fp16"}, {"device": "gpu"}])
def test_result_key_matches_the_request_key(overrides):
    request = make_request(**overrides)
    assert result_key(make_row(request)) == request.key()


def render(capsys, rows) -> str:
    pretty_print_results(rows)
    return capsys.readouterr().out


def test_summary_table_has_the_new_columns_and_values(capsys):
    out = render(capsys, [make_row()])
    header = out.splitlines()[0].split()
    assert header == ["Artifact", "Runtime", "Device", "Precision", "mAP50", "mAP50-95", "AP75", "Latency", "Throttled"]
    row = out.splitlines()[2].split()
    assert row == ["a.onnx", "fake", "cpu", "fp32", "60.0", "40.0", "45.0", "12.35", "no"]


@pytest.mark.parametrize("throttled, shown", [(True, "yes"), (False, "no"), (None, "?")])
def test_throttled_column(capsys, throttled, shown):
    out = render(capsys, [make_row(throttled=throttled)])
    assert out.splitlines()[2].split()[-1] == shown


def unsupported_row():
    request = make_request(artifact_path="u.hef", unsupported="grid_sample not supported")
    return make_row(request, accuracy_stats=None, latency_stats=None, throttled=None)


def test_unsupported_row_shows_the_reason_and_is_skipped_in_breakdowns(capsys):
    out = render(capsys, [make_row(), unsupported_row()])
    summary_line = next(line for line in out.splitlines() if line.startswith("u.hef"))
    assert "unsupported: grid_sample not supported" in summary_line
    assert out.count("u.hef") == 1


def test_partial_row_gets_a_star_and_one_footnote_for_each_distinct_count(capsys):
    rows = [
        make_row(make_request(max_images=5)),
        make_row(make_request(artifact_path="b.onnx", max_images=5)),
        make_row(make_request(artifact_path="c.onnx", max_images=10)),
        make_row(make_request(artifact_path="d.onnx")),
    ]
    out = render(capsys, rows)
    lines = out.splitlines()
    assert "40.0*" in next(line for line in lines if line.startswith("a.onnx"))
    assert "*" not in next(line for line in lines if line.startswith("d.onnx"))
    assert "* evaluated on the first 5 images" in out
    assert "* evaluated on the first 10 images" in out
    assert out.count("* evaluated on the first 5 images") == 1


def test_no_footnote_without_partial_rows(capsys):
    assert "evaluated on the first" not in render(capsys, [make_row()])


def test_ap_and_ar_breakdowns_are_keyed_by_artifact_path(capsys):
    out = render(capsys, [make_row()])
    assert "AP breakdown (COCO):" in out
    assert "AR breakdown (COCO):" in out
    ap_line = next(line for line in out.splitlines() if line.startswith("a.onnx") and "10.0" in line)
    assert ap_line.split()[1:] == ["10.0", "20.0", "30.0"]
    ar_line = next(line for line in out.splitlines() if line.startswith("a.onnx") and "31.0" in line)
    assert ar_line.split()[1:] == ["31.0", "32.0", "33.0", "11.0", "22.0", "33.0"]
