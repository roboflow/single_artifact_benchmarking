import importlib
from functools import partial

import pytest
import torch

from sab.models.graph_surgery import fuse_yolo_mask_postprocessing_into_onnx
from sab.runtimes.base import InputSpec, runtime_class
from sab.runtimes.tensorrt import TRTRuntime

MODULE_NAMES = [
    "benchmark_dfine",
    "benchmark_rtdetr",
    "benchmark_rfdetr_seg",
    "benchmark_yolov8_seg",
    "benchmark_yolov11_seg",
]


def load(module_name: str):
    return importlib.import_module(f"sab.models.{module_name}")


def load_functions(module_name: str):
    """The module that holds preprocess_image and postprocess_output. YOLOv8 seg reuses those of YOLOv11 seg."""
    return load("benchmark_yolov11_seg" if module_name == "benchmark_yolov8_seg" else module_name)


def trt_row(artifact_path, precision, use_cuda_graph, graph_surgery_func=None):
    return (artifact_path, "tensorrt", "gpu", precision, use_cuda_graph, graph_surgery_func)


def fp32_and_fp16_rows(artifact_paths, use_cuda_graph):
    return [
        trt_row(path, precision, use_cuda_graph)
        for path in artifact_paths
        for precision in ("fp32", "fp16")
    ]


def fp16_rows(artifact_paths, use_cuda_graph, graph_surgery_func):
    return [trt_row(path, "fp16", use_cuda_graph, graph_surgery_func) for path in artifact_paths]


# Each row is (artifact_path, runtime name, device, precision, use_cuda_graph, graph_surgery_func).
# The values come from the request lists of the old scripts.
EXPECTED_ROWS = {
    "benchmark_dfine": fp32_and_fp16_rows(
        [
            "dfine_n_coco.opset17.onnx",
            "dfine_s_obj2coco.opset17.onnx",
            "dfine_m_obj2coco.opset17.onnx",
            "dfine_l_obj2coco_e25.opset17.onnx",
            "dfine_x_obj2coco.opset17.onnx",
        ],
        use_cuda_graph=False,
    ),
    "benchmark_rtdetr": fp32_and_fp16_rows(
        ["rtdetr_r18_coco.onnx", "rtdetr_r50_coco.onnx", "rtdetr_r101_coco.onnx"],
        use_cuda_graph=True,
    ),
    "benchmark_rfdetr_seg": [],
    "benchmark_yolov8_seg": fp16_rows(
        [
            "yolov8n_seg_nms_conf_0.01.onnx",
            "yolov8s_seg_nms_conf_0.01.onnx",
            "yolov8m_seg_nms_conf_0.01.onnx",
            "yolov8l_seg_nms_conf_0.01.onnx",
            "yolov8x_seg_nms_conf_0.01.onnx",
        ],
        use_cuda_graph=False,
        graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
    ),
    "benchmark_yolov11_seg": fp16_rows(
        [
            "yolo11n_seg_nms_conf_0.01.onnx",
            "yolo11s_seg_nms_conf_0.01.onnx",
            "yolo11m_seg_nms_conf_0.01.onnx",
            "yolo11l_seg_nms_conf_0.01.onnx",
            "yolo11x_seg_nms_conf_0.01.onnx",
        ],
        use_cuda_graph=False,
        graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
    ),
}

PROCESSOR_NAMES = {
    "benchmark_dfine": "DFINEProcessor",
    "benchmark_rtdetr": "RTDETRProcessor",
    "benchmark_rfdetr_seg": "RFDETRSegProcessor",
    "benchmark_yolov8_seg": "YOLOv8SegProcessor",
    "benchmark_yolov11_seg": "YOLOv11SegProcessor",
}

def describe(request):
    factory = request.runtime
    # A bare TRTRuntime uses its default, which is CUDA graphs on.
    use_cuda_graph = factory.keywords.get("use_cuda_graph", True) if isinstance(factory, partial) else True
    return (
        request.artifact_path,
        request.runtime_name,
        request.device,
        request.precision,
        use_cuda_graph,
        request.graph_surgery_func,
    )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_build_requests_reproduces_the_old_rows(module_name):
    module = load(module_name)
    requests = module.build_requests(buffer_time=2.5)

    assert [describe(request) for request in requests] == EXPECTED_ROWS[module_name]
    for request in requests:
        assert request.processor is getattr(module, PROCESSOR_NAMES[module_name])
        assert request.needs_class_remapping is True
        assert request.max_dets == 100
        assert request.buffer_time == 2.5
        assert runtime_class(request.runtime) is TRTRuntime


INPUT_SPEC = InputSpec(name="images", shape=(1, 3, 64, 96))


@pytest.fixture
def image():
    generator = torch.Generator().manual_seed(0)
    return torch.rand(3, 48, 80, generator=generator)


def make_processor(module_name, normalize=True):
    return getattr(load(module_name), PROCESSOR_NAMES[module_name])(INPUT_SPEC, normalize=normalize)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_preprocess_matches_the_module_function(module_name, image):
    expected, expected_metadata = load_functions(module_name).preprocess_image(image, INPUT_SPEC.shape)

    actual, actual_metadata = make_processor(module_name).preprocess(image)

    assert torch.equal(actual, expected)
    assert actual_metadata.keys() == expected_metadata.keys()


@pytest.mark.parametrize("module_name", ["benchmark_dfine", "benchmark_rtdetr"])
def test_preprocess_without_normalization_scales_to_0_255(module_name, image):
    expected, _ = load_functions(module_name).preprocess_image(image, INPUT_SPEC.shape)

    actual, _ = make_processor(module_name, normalize=False).preprocess(image)

    torch.testing.assert_close(actual, expected * 255.0)


def test_yolo_seg_preprocess_without_normalization_scales_to_0_255(image):
    # YOLOv8 seg inherits this preprocess unchanged.
    expected, _ = load_functions("benchmark_yolov11_seg").preprocess_image(image, INPUT_SPEC.shape)

    actual, _ = make_processor("benchmark_yolov11_seg", normalize=False).preprocess(image)

    torch.testing.assert_close(actual, expected * 255.0)


def test_rfdetr_seg_preprocess_without_normalization_skips_mean_and_std(image):
    import torchvision.transforms.functional as TF

    expected = TF.resize(image.unsqueeze(0) * 255.0, INPUT_SPEC.shape[2:])

    actual, metadata = make_processor("benchmark_rfdetr_seg", normalize=False).preprocess(image)

    torch.testing.assert_close(actual, expected)
    assert metadata["orig_target_sizes"].tolist() == [48, 80]


@pytest.mark.parametrize("module_name", ["benchmark_dfine", "benchmark_rtdetr"])
def test_extra_inputs_spoof_target_sizes_with_ones(module_name, image):
    processor = make_processor(module_name)
    tensor, metadata = processor.preprocess(image)

    extra = processor.extra_inputs(tensor, metadata)

    assert set(extra) == {"orig_target_sizes"}
    assert extra["orig_target_sizes"].dtype == torch.int64
    assert extra["orig_target_sizes"].shape == (1, 2)
    assert torch.equal(extra["orig_target_sizes"], torch.ones((1, 2), dtype=torch.int64))
