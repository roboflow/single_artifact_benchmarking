import importlib
from functools import partial

import pytest
import torch
import torchvision.transforms.functional as TF

from sab.request import ArtifactBenchmarkRequest
from sab.runtimes.base import InputSpec, runtime_class

INPUT_SPEC = InputSpec(name="images", shape=(1, 3, 32, 32))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(family: str):
    return importlib.import_module(f"sab.models.benchmark_{family}")


def describe(request: ArtifactBenchmarkRequest) -> tuple:
    """(artifact, runtime name, device, precision, use_cuda_graph for TensorRT rows)"""
    use_cuda_graph = None
    if runtime_class(request.runtime).name == "tensorrt":
        use_cuda_graph = request.runtime.keywords.get("use_cuda_graph", True) if isinstance(request.runtime, partial) else True
    return (request.artifact_path, request.runtime_name, request.device, request.precision, use_cuda_graph)


def tensorrt_rows(path: str, use_cuda_graph: bool, precisions=("fp32", "fp16")) -> list[tuple]:
    return [(path, "tensorrt", "gpu", precision, use_cuda_graph) for precision in precisions]


def cpu_row(path: str) -> tuple:
    return (path, "onnxruntime", "cpu", "fp32", None)


def rfdetr_rows() -> list[tuple]:
    rows = []
    for size in ("nano", "small", "medium"):
        path = f"rf-detr-{size}.onnx"
        rows += tensorrt_rows(path, True) + [cpu_row(path)]
    return rows


def yolov11_rows() -> list[tuple]:
    rows = []
    for size in "nsmlx":
        path = f"yolo11{size}_nms_conf_0.01.onnx"
        rows += tensorrt_rows(path, False) + [cpu_row(path)]
    return rows


def yolov8_rows() -> list[tuple]:
    return [
        row
        for size in "nsm"
        for row in tensorrt_rows(f"yolov8{size}_nms_conf_0.01.onnx", False, precisions=("fp16",))
    ]


def yolo26_rows() -> list[tuple]:
    rows = []
    for size in "nsmlx":
        path = f"yolo26{size}.onnx"
        rows += tensorrt_rows(path, True) + [cpu_row(path)]
    return rows


def lwdetr_rows() -> list[tuple]:
    rows = []
    for size in ("tiny", "small", "medium", "large", "xlarge"):
        rows += tensorrt_rows(f"lw-detr-{size}.onnx", True)
    return rows


def yololite_rows() -> list[tuple]:
    return [cpu_row("model.onnx")] + tensorrt_rows("model.onnx", True)


def build(family: str, buffer_time: float = 0.0) -> list[ArtifactBenchmarkRequest]:
    module = load(family)
    if family == "yololite":
        return module.build_requests("model.onnx", buffer_time)
    return module.build_requests(buffer_time)


EXPECTED_ROWS = {
    "rfdetr": rfdetr_rows,
    "yolov11": yolov11_rows,
    "yolov8": yolov8_rows,
    "yolo26": yolo26_rows,
    "lwdetr": lwdetr_rows,
    "yololite": yololite_rows,
}
PROCESSOR_NAMES = {
    "rfdetr": "RFDETRProcessor",
    "yolov11": "YOLOv11Processor",
    "yolov8": "YOLOv8Processor",
    "yolo26": "YOLO26Processor",
    "lwdetr": "LWDETRProcessor",
    "yololite": "YoloLiteProcessor",
}
REMAPPED = {"yolov11", "yolov8", "yolo26"}


@pytest.mark.parametrize("family", EXPECTED_ROWS)
def test_build_requests_reproduces_the_old_rows(family):
    requests = build(family, buffer_time=0.25)

    expected = EXPECTED_ROWS[family]()

    assert [describe(request) for request in requests[: len(expected)]] == expected
    processor = getattr(load(family), PROCESSOR_NAMES[family])
    max_dets = 500 if family == "yololite" else 100
    assert {
        (request.processor, request.needs_class_remapping, request.max_dets, request.buffer_time, request.normalized_in_graph)
        for request in requests
    } == {(processor, family in REMAPPED, max_dets, 0.25, False)}


def test_yololite_onnx_row_uses_dynamic_output_shapes():
    onnx_row = build("yololite")[0]
    assert onnx_row.runtime.keywords == {"dynamic_output_shapes": True}


@pytest.fixture
def image():
    generator = torch.Generator().manual_seed(0)
    return torch.rand(3, 48, 64, generator=generator)


@pytest.mark.parametrize("family", ["rfdetr", "lwdetr"])
def test_detr_preprocess_default_matches_module_function(family, image):
    module = load(family)
    processor = getattr(module, PROCESSOR_NAMES[family])(INPUT_SPEC, normalize=True)

    tensor, metadata = processor.preprocess(image)
    expected, expected_metadata = module.preprocess_image(image, INPUT_SPEC.shape)

    assert torch.equal(tensor, expected)
    assert metadata == expected_metadata
    assert torch.equal(tensor, TF.resize(TF.normalize(image.unsqueeze(0), MEAN, STD), [32, 32]))


@pytest.mark.parametrize("family", ["rfdetr", "lwdetr"])
def test_detr_preprocess_without_normalization_scales_to_255(family, image):
    processor = getattr(load(family), PROCESSOR_NAMES[family])(INPUT_SPEC, normalize=False)

    tensor, _ = processor.preprocess(image)

    assert torch.equal(tensor, TF.resize(image.unsqueeze(0) * 255.0, [32, 32]))


@pytest.mark.parametrize("family", ["yolov11", "yolov8", "yolo26"])
def test_yolo_preprocess_matches_module_function(family, image):
    module = load("yolov11")
    processor = getattr(load(family), PROCESSOR_NAMES[family])(INPUT_SPEC, normalize=True)

    tensor, metadata = processor.preprocess(image)
    expected, expected_metadata = module.preprocess_image(image, INPUT_SPEC.shape)

    assert torch.equal(tensor, expected)
    assert metadata == expected_metadata


def test_yolo_preprocess_without_normalization_scales_to_255(image):
    # YOLOv8 and YOLO26 inherit this preprocess unchanged.
    processor_class = load("yolov11").YOLOv11Processor
    normalized, normalized_metadata = processor_class(INPUT_SPEC, normalize=True).preprocess(image)

    tensor, metadata = processor_class(INPUT_SPEC, normalize=False).preprocess(image)

    assert torch.equal(tensor, normalized * 255.0)
    assert metadata["padding"] == normalized_metadata["padding"]


def test_yololite_preprocess_default_matches_module_function(image):
    module = load("yololite")
    processor = module.YoloLiteProcessor(INPUT_SPEC, normalize=True)

    tensor, metadata = processor.preprocess(image)
    expected, expected_metadata = module.preprocess_image(image, INPUT_SPEC.shape)

    assert torch.equal(tensor, expected)
    assert metadata == expected_metadata


def test_yololite_preprocess_without_normalization_scales_to_255(image):
    module = load("yololite")
    tensor, metadata = module.YoloLiteProcessor(INPUT_SPEC, normalize=False).preprocess(image)

    letterboxed = TF.pad(
        TF.resize(image.unsqueeze(0), (24, 32), antialias=False),
        metadata["padding"],
        fill=module._PAD_VALUE,
    )
    assert torch.equal(tensor, letterboxed * 255.0)
