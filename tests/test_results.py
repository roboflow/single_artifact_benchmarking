import json
import os

import numpy as np
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


def test_save_creates_the_directory_and_keeps_the_old_file_on_failure(tmp_path):
    directory = tmp_path / "nested"
    path = str(directory / "out.json")
    save_results(path, [make_row()])
    with pytest.raises(TypeError):
        save_results(path, [{"bad": object()}])
    assert load_results(path) == [make_row()]
    assert os.listdir(directory) == ["out.json"]


def test_save_accepts_numpy_numbers(tmp_path):
    path = str(tmp_path / "out.json")
    save_results(path, [make_row(latency_stats={"median": np.float32(1.5), "count": np.int64(3)})])
    assert json.load(open(path))[0]["latency_stats"] == {"median": 1.5, "count": 3}


def test_result_key_matches_the_request_key():
    request = make_request(max_images=5, precision="fp16", device="gpu", buffer_time=2.0, max_dets=50)
    assert result_key(make_row(request)) == request.key()


def render(capsys, rows) -> str:
    pretty_print_results(rows)
    return capsys.readouterr().out


def line_starting_with(out: str, prefix: str, contains: str = "") -> str:
    return next(line for line in out.splitlines() if line.startswith(prefix) and contains in line)


def test_summary_row_shows_percentages_latency_and_throttle(capsys):
    row = render(capsys, [make_row()]).splitlines()[2].split()
    assert row == ["a.onnx", "fake", "cpu", "fp32", "60.0", "40.0", "45.0", "12.35", "no"]


@pytest.mark.parametrize("throttled, shown", [(True, "yes"), (None, "?")])
def test_throttled_column(capsys, throttled, shown):
    out = render(capsys, [make_row(throttled=throttled)])
    assert out.splitlines()[2].split()[-1] == shown


def test_unsupported_row_shows_the_reason_and_is_skipped_in_breakdowns(capsys):
    request = make_request(artifact_path="u.hef", unsupported="grid_sample not supported")
    unsupported = make_row(request, accuracy_stats=None, latency_stats=None, throttled=None)
    out = render(capsys, [make_row(), unsupported])
    assert "unsupported: grid_sample not supported" in line_starting_with(out, "u.hef")
    assert out.count("u.hef") == 1


def test_partial_row_gets_a_star_and_one_footnote_for_each_distinct_count(capsys):
    rows = [
        make_row(make_request(max_images=5)),
        make_row(make_request(artifact_path="b.onnx", max_images=5)),
        make_row(make_request(artifact_path="c.onnx", max_images=10)),
        make_row(make_request(artifact_path="d.onnx")),
    ]
    out = render(capsys, rows)
    assert "40.0*" in line_starting_with(out, "a.onnx")
    assert "*" not in line_starting_with(out, "d.onnx")
    assert out.count("* evaluated on the first 5 images") == 1
    assert out.count("* evaluated on the first 10 images") == 1


def test_ap_and_ar_breakdowns_read_the_right_coco_stats(capsys):
    out = render(capsys, [make_row()])
    assert line_starting_with(out, "a.onnx", "10.0").split()[1:] == ["10.0", "20.0", "30.0"]
    assert line_starting_with(out, "a.onnx", "31.0").split()[1:] == ["31.0", "32.0", "33.0", "11.0", "22.0", "33.0"]
