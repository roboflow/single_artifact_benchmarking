import pytest
import torch

from sab.processors import Pipeline
from tests.fakes import FakeProcessor, FakeRuntime


class RecordingProcessor(FakeProcessor):
    """Adds an extra input and keeps the outputs that postprocess receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postprocessed: list[dict[str, torch.Tensor]] = []

    def extra_inputs(self, image, metadata):
        return {"orig_target_sizes": torch.ones((1, 2), dtype=torch.int64)}

    def postprocess(self, outputs, metadata):
        self.postprocessed.append(outputs)
        return super().postprocess(outputs, metadata)


def make_pipeline(output_names=None, input_device="cpu"):
    runtime = FakeRuntime("artifact.onnx", "cpu", "fp32")
    runtime.input_device = input_device
    processor = RecordingProcessor(runtime.input_spec)
    return Pipeline(runtime, processor, output_names), runtime, processor


def test_pipeline_sends_the_image_under_the_input_spec_name():
    pipeline, runtime, _ = make_pipeline()
    image = torch.rand(3, 8, 8)

    pipeline.infer(image)

    (call,) = runtime.calls
    assert call["images"].shape == (1, 3, 8, 8)
    torch.testing.assert_close(call["images"], image.unsqueeze(0))


def test_pipeline_passes_extra_inputs_to_the_runtime():
    pipeline, runtime, _ = make_pipeline()

    pipeline.infer(torch.rand(3, 8, 8))

    (call,) = runtime.calls
    assert set(call) == {"images", "orig_target_sizes"}
    assert call["orig_target_sizes"].dtype == torch.int64
    assert call["orig_target_sizes"].tolist() == [[1, 1]]


def test_pipeline_renames_outputs_before_postprocess():
    pipeline, _, processor = make_pipeline(output_names={"images": "dets"})

    pipeline.infer(torch.rand(3, 8, 8))

    (outputs,) = processor.postprocessed
    assert set(outputs) == {"dets", "orig_target_sizes"}


def test_pipeline_keeps_outputs_that_have_no_new_name():
    pipeline, _, processor = make_pipeline(output_names=None)

    pipeline.infer(torch.rand(3, 8, 8))

    (outputs,) = processor.postprocessed
    assert set(outputs) == {"images", "orig_target_sizes"}


def test_pipeline_returns_the_postprocessed_prediction():
    pipeline, _, _ = make_pipeline()

    xyxy, class_id, score = pipeline.infer(torch.rand(3, 8, 8))

    assert xyxy.shape == (1, 1, 4)
    assert class_id.tolist() == [[1]]
    assert score.tolist()[0] == pytest.approx([0.9])


def test_pipeline_uses_the_profiler_of_the_runtime():
    pipeline, runtime, _ = make_pipeline()
    assert pipeline.profiler is runtime.profiler

    pipeline.infer(torch.rand(3, 8, 8))
    pipeline.infer(torch.rand(3, 8, 8))

    assert len(pipeline.profiler.timings) == 2


def test_pipeline_takes_the_input_device_from_the_runtime():
    pipeline, _, _ = make_pipeline(input_device="cuda")
    assert pipeline.input_device == "cuda"


def test_pipeline_takes_the_prediction_type_from_the_processor():
    pipeline, _, processor = make_pipeline()
    assert pipeline.prediction_type == processor.prediction_type == "bbox"


def test_normalize_false_reaches_the_runtime_as_0_to_255_values():
    runtime = FakeRuntime("artifact.onnx", "cpu", "fp32")
    pipeline = Pipeline(runtime, FakeProcessor(runtime.input_spec, normalize=False))

    pipeline.infer(torch.ones(3, 8, 8))

    assert runtime.calls[0]["images"].eq(255.0).all()
