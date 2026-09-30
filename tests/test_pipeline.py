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


def make_pipeline(output_names=None):
    runtime = FakeRuntime("artifact.onnx", "cpu", "fp32")
    processor = RecordingProcessor(runtime.input_spec)
    return Pipeline(runtime, processor, output_names), runtime, processor


def test_pipeline_sends_the_image_under_the_input_spec_name_plus_the_extra_inputs():
    pipeline, runtime, _ = make_pipeline()
    image = torch.rand(3, 8, 8)

    pipeline.infer(image)

    (call,) = runtime.calls
    assert set(call) == {"images", "orig_target_sizes"}
    torch.testing.assert_close(call["images"], image.unsqueeze(0))
    assert call["orig_target_sizes"].dtype == torch.int64


def test_pipeline_renames_outputs_before_postprocess_and_keeps_the_others():
    pipeline, _, processor = make_pipeline(output_names={"images": "dets"})

    pipeline.infer(torch.rand(3, 8, 8))

    (outputs,) = processor.postprocessed
    assert set(outputs) == {"dets", "orig_target_sizes"}
