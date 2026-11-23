from contextlib import contextmanager

import pytest
from PIL import Image

from sab.evaluation import run_timed_pass
from sab.processors import Pipeline
from tests.fakes import FakeProcessor, FakeRuntime


@pytest.fixture
def image_paths(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (8, 8), color=(index * 40, 10, 10)).save(path)
        paths.append(str(path))
    return paths


def make_pipeline(input_device="cpu"):
    runtime = FakeRuntime("artifact.onnx", "cpu", "fp32", shape=(1, 3, 8, 8))
    runtime.input_device = input_device
    return Pipeline(runtime, FakeProcessor(runtime.input_spec)), runtime


def no_sleep(_seconds):
    pass


def test_run_timed_pass_records_one_timing_per_image_up_to_max_images(image_paths):
    pipeline, _ = make_pipeline()

    assert run_timed_pass(pipeline, image_paths, sleep_fn=no_sleep)["count"] == 3
    assert run_timed_pass(pipeline, image_paths, max_images=2, sleep_fn=no_sleep)["count"] == 2


def test_run_timed_pass_moves_images_to_the_input_device_of_the_runtime(image_paths):
    # The meta device stands in for cuda: it is not cpu, and it needs no GPU.
    pipeline, runtime = make_pipeline(input_device="meta")

    run_timed_pass(pipeline, image_paths, sleep_fn=no_sleep)

    assert [call["images"].device.type for call in runtime.calls] == ["meta"] * 3


class BusyRecordingMonitor:
    def __init__(self):
        self.inside_busy = False
        self.busy_entries = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    @contextmanager
    def busy(self):
        self.inside_busy = True
        self.busy_entries += 1
        yield
        self.inside_busy = False


def test_run_timed_pass_marks_each_inference_as_busy(image_paths):
    pipeline, runtime = make_pipeline()
    monitor = BusyRecordingMonitor()
    run_inside_busy = []
    original_run = runtime.run
    runtime.run = lambda inputs: run_inside_busy.append(monitor.inside_busy) or original_run(inputs)

    run_timed_pass(pipeline, image_paths, monitor=monitor, sleep_fn=lambda _: run_inside_busy.append(monitor.inside_busy))

    assert monitor.busy_entries == 3
    assert run_inside_busy == [True, False] * 3


def test_run_timed_pass_reports_each_prediction_to_on_result(image_paths):
    pipeline, _ = make_pipeline()
    seen = []

    run_timed_pass(
        pipeline,
        image_paths,
        on_result=lambda index, shape, outputs: seen.append((index, shape, len(outputs))),
        sleep_fn=no_sleep,
    )

    assert seen == [(0, (8, 8), 4), (1, (8, 8), 4), (2, (8, 8), 4)]


def test_run_timed_pass_resets_the_profiler_at_entry(image_paths):
    pipeline, runtime = make_pipeline()
    runtime.profiler.timings.append(999.0)

    stats = run_timed_pass(pipeline, image_paths, sleep_fn=no_sleep)

    assert stats["count"] == 3
    assert stats["max"] < 999.0


def test_run_timed_pass_sleeps_the_buffer_time_after_each_image(image_paths):
    pipeline, _ = make_pipeline()
    sleeps = []

    run_timed_pass(pipeline, image_paths, buffer_time=0.25, sleep_fn=sleeps.append)

    assert sleeps == [0.25] * 3


def test_run_timed_pass_rejects_an_unknown_prediction_type(image_paths):
    pipeline, _ = make_pipeline()
    pipeline.processor.prediction_type = "keypoints"

    with pytest.raises(ValueError, match="keypoints"):
        run_timed_pass(pipeline, image_paths, sleep_fn=no_sleep)
