"""The annotation-free timed loop: call count, buffer sleeps, max_images, monitor."""

import importlib
import sys

import pytest
import torch
from PIL import Image

import sab.evaluation
from sab.evaluation import run_timed_pass
from sab.onnx_inference import ONNXInferenceCPU


class FakeProfiler:
    def __init__(self):
        self.timings = []

    def get_stats(self):
        return {"count": len(self.timings), "median": 1.0}


class FakeInference(ONNXInferenceCPU):
    """Subclasses ONNXInferenceCPU only so the loop keeps the tensors on the CPU."""

    def __init__(self, prediction_type="bbox"):
        self.prediction_type = prediction_type
        self.profiler = FakeProfiler()
        self.calls = []

    def infer(self, image):
        self.calls.append(tuple(image.shape))
        self.profiler.timings.append(1.0)
        empty = torch.zeros(1, 0, 4), torch.zeros(1, 0), torch.zeros(1, 0)
        if self.prediction_type == "segm":
            return (*empty, torch.zeros(1, 0, 4, 4))
        return empty


class RecordingMonitor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self.log.append("monitor-enter")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log.append("monitor-exit")


@pytest.fixture
def image_paths(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (32 + index, 16 + index)).save(path)
        paths.append(str(path))
    return paths


def test_runs_every_image_once(image_paths):
    inference = FakeInference()

    run_timed_pass(inference, image_paths, sleep_fn=lambda seconds: None)

    assert len(inference.calls) == len(image_paths)


def test_max_images_is_respected(image_paths):
    inference = FakeInference()

    run_timed_pass(inference, image_paths, max_images=2, sleep_fn=lambda seconds: None)

    assert len(inference.calls) == 2


def test_buffer_sleeps_once_per_image_after_the_result(image_paths):
    inference = FakeInference()
    log = []

    run_timed_pass(
        inference,
        image_paths,
        buffer_time=0.25,
        max_images=2,
        on_result=lambda index, shape, outputs: log.append(f"result-{index}"),
        sleep_fn=lambda seconds: log.append(f"sleep-{seconds}"),
    )

    assert log == ["result-0", "sleep-0.25", "result-1", "sleep-0.25"]


def test_on_result_reports_index_and_original_image_size(image_paths):
    inference = FakeInference()
    seen = []

    run_timed_pass(
        inference,
        image_paths,
        max_images=2,
        on_result=lambda index, shape, outputs: seen.append((index, shape, len(outputs))),
        sleep_fn=lambda seconds: None,
    )

    assert seen == [(0, (32, 16), 4), (1, (33, 17), 4)]


def test_monitor_is_held_open_across_the_whole_pass(image_paths):
    log = []
    inference = FakeInference()
    inference.infer = lambda image: (
        log.append("infer"),
        (torch.zeros(1, 0, 4), torch.zeros(1, 0), torch.zeros(1, 0)),
    )[1]

    run_timed_pass(
        inference,
        image_paths,
        max_images=2,
        monitor=RecordingMonitor(log),
        sleep_fn=lambda seconds: None,
    )

    assert log == ["monitor-enter", "infer", "infer", "monitor-exit"]


def test_returns_the_profiler_stats(image_paths):
    inference = FakeInference()

    stats = run_timed_pass(inference, image_paths, sleep_fn=lambda seconds: None)

    assert stats == {"count": 4, "median": 1.0}


def test_segmentation_models_yield_masks(image_paths):
    inference = FakeInference(prediction_type="segm")
    seen = []

    run_timed_pass(
        inference,
        image_paths,
        max_images=1,
        on_result=lambda index, shape, outputs: seen.append(outputs[3]),
        sleep_fn=lambda seconds: None,
    )

    assert seen[0] is not None


def test_unknown_prediction_type_is_rejected(image_paths):
    inference = FakeInference(prediction_type="panoptic")

    with pytest.raises(ValueError, match="Invalid prediction type"):
        run_timed_pass(inference, image_paths, sleep_fn=lambda seconds: None)


def test_module_imports_without_the_coco_stack(monkeypatch):
    """The timed path must import on a box that has no faster_coco_eval."""
    for blocked in ("faster_coco_eval", "pycocotools", "pycocotools.coco",
                    "pycocotools.cocoeval", "pycocotools.mask"):
        monkeypatch.setitem(sys.modules, blocked, None)

    reloaded = importlib.reload(sab.evaluation)

    assert callable(reloaded.run_timed_pass)
