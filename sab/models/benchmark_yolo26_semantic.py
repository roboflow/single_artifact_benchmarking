"""YOLO26 native semantic segmentation on ADEChallengeData2016 validation.

Use yolo26{n,s,m,l,x}-sem-ade20k ONNX exports (150 classes). Both floating
NCHW logits and the current export's NHW integer class maps are supported.
"""

import json
from pathlib import Path

import fire
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results
from sab.onnx_inference import ONNXInferenceCPU, ONNXInferenceCUDA
from sab.semantic_evaluation import ADE20K_CONFIG, YOLO26_ADE20K_CONFIG, letterbox_geometry, semantic_image_pairs
from sab.trt_inference import TRTInference


def preprocess_image(image: torch.Tensor, image_input_shape):
    """Match Ultralytics validation: OpenCV uint8 resize, then 114 letterbox."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError("YOLO26 semantic benchmarking requires one RGB image at a time")
    if len(image_input_shape) != 4 or tuple(image_input_shape[:2]) != (1, 3):
        raise ValueError("Expected a fixed [1,3,H,W] ONNX input with batch=1")
    input_h, input_w = image_input_shape[2:]
    if not all(isinstance(d, int) and d > 0 for d in (input_h, input_w)):
        raise ValueError("YOLO26 semantic benchmarking requires static input dimensions")

    original_h, original_w = image.shape[-2:]
    resized_h, resized_w, top, left = letterbox_geometry((original_h, original_w), (input_h, input_w))
    # The upstream validator resizes uint8 images before normalization. Resizing
    # floating tensors instead changes rounding and can shift benchmark accuracy.
    pixels = image[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    pixels = cv2.resize(pixels, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).unsqueeze(0).to(image.device).float() / 255
    pad_h, pad_w = input_h - resized_h, input_w - resized_w
    image = F.pad(image, (left, pad_w - left, top, pad_h - top), value=114 / 255)
    return image.contiguous(), {
        "original_shape": (original_h, original_w),
        "input_shape": (input_h, input_w),
        "crop": (top, left, resized_h, resized_w),
    }


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict, protocol: str = "native"):
    if protocol not in {"native", "ultralytics"}:
        raise ValueError("protocol must be native or ultralytics")
    if len(outputs) != 1:
        raise ValueError("Expected one semantic output; use a YOLO26 -sem-ade20k export")
    output = next(iter(outputs.values()))
    if output.ndim not in (3, 4) or output.shape[0] != 1:
        raise ValueError(f"Expected [1,H,W] class IDs or [1,150,H,W] logits, got {tuple(output.shape)}")
    class_map = output.ndim == 3
    if output.is_floating_point() and not torch.isfinite(output).all():
        raise ValueError("Semantic output contains NaN/Inf; check runtime precision")
    if not class_map and output.shape[1] != ADE20K_CONFIG.num_classes:
        raise ValueError("Expected 150 ADE20K class channels; use the -sem-ade20k model")
    if class_map and output.is_floating_point() and not torch.equal(output, output.round()):
        raise ValueError("A semantic class-map output must contain integer IDs")
    values = (output.unsqueeze(1) if class_map else output).float()

    def resize(tensor, size):
        if tensor.shape[-2:] == tuple(size):
            return tensor
        if class_map:
            return F.interpolate(tensor, size=size, mode="nearest")
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    # The upstream validation metric uses this input grid, with padding ignored
    # in the ground truth. ArgMax must follow interpolation for logits exports.
    values = resize(values, metadata["input_shape"])
    if protocol == "native":
        top, left, height, width = metadata["crop"]
        values = values[:, :, top:top + height, left:left + width]
        values = resize(values, metadata["original_shape"])
    return values[0, 0].long() if class_map else values[0].argmax(dim=0)


class _YOLO26SemanticAdapter:
    def __init__(self, *args, protocol="native", **kwargs):
        if protocol not in {"native", "ultralytics"}:
            raise ValueError("protocol must be native or ultralytics")
        self.protocol = protocol
        super().__init__(*args, **kwargs)

    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata, self.protocol)


class YOLO26SemanticTRTInference(_YOLO26SemanticAdapter, TRTInference):
    def __init__(self, model_path, image_input_name=None, protocol="native"):
        super().__init__(model_path, image_input_name, protocol=protocol, use_cuda_graph=True, prediction_type="semantic")


class YOLO26SemanticONNXInference(_YOLO26SemanticAdapter, ONNXInferenceCUDA):
    def __init__(self, model_path, image_input_name=None, protocol="native"):
        super().__init__(model_path, image_input_name, protocol=protocol, prediction_type="semantic")


class YOLO26SemanticONNXCPUInference(_YOLO26SemanticAdapter, ONNXInferenceCPU):
    def __init__(self, model_path, image_input_name=None, protocol="native"):
        super().__init__(model_path, image_input_name, protocol=protocol, prediction_type="semantic")


def main(image_dir: str, mask_dir: str, buffer_time: float = 0.0,
         output_file_name: str = "yolo26_semantic_results.json", onnx_path: str | None = None,
         sizes: str = "n,s,m,l,x", runtime: str = "trt", fp16: bool | None = None,
         max_images: int | None = None, protocol: str = "native"):
    """Benchmark local YOLO26 semantic ONNX files against ADE20K mask PNGs.

    image_dir: ADEChallengeData2016/images/validation
    mask_dir: ADEChallengeData2016/annotations/validation (raw labels 0..150)
    onnx_path: one exported artifact; otherwise benchmark the selected sizes
    runtime: trt, onnx-cuda, or onnx-cpu; FP16 defaults on for TRT only
    """
    runtimes = {
        "trt": YOLO26SemanticTRTInference,
        "onnx-cuda": YOLO26SemanticONNXInference,
        "onnx-cpu": YOLO26SemanticONNXCPUInference,
    }
    if runtime not in runtimes:
        raise ValueError(f"runtime must be one of {', '.join(runtimes)}")
    if protocol not in {"native", "ultralytics"}:
        raise ValueError("protocol must be native or ultralytics")
    if fp16 is None:
        fp16 = runtime == "trt"
    if fp16 and runtime != "trt":
        raise ValueError("FP16 engine compilation requires runtime=trt")
    if buffer_time < 0:
        raise ValueError("buffer_time must be nonnegative")
    if onnx_path is not None:
        paths = [onnx_path]
    else:
        selected_sizes = sizes.split(",") if isinstance(sizes, str) else list(sizes)
        selected_sizes = [s.strip() for s in selected_sizes]
        if not selected_sizes or any(s not in {"n", "s", "m", "l", "x"} for s in selected_sizes):
            raise ValueError("sizes must be a comma-separated selection of n,s,m,l,x")
        paths = [f"yolo26{s}-sem-ade20k.onnx" for s in selected_sizes]
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Semantic ONNX artifact not found: {path}")
    # Fail on missing data before doing an expensive TensorRT build.
    semantic_image_pairs(image_dir, mask_dir, max_images)
    requests = [ArtifactBenchmarkRequest(
        onnx_path=path, inference_class=runtimes[runtime], needs_fp16=fp16,
        buffer_time=buffer_time, max_images=max_images,
        semantic_config=ADE20K_CONFIG if protocol == "native" else YOLO26_ADE20K_CONFIG,
        inference_kwargs={"protocol": protocol},
    ) for path in paths]
    results = run_benchmark_on_artifacts(requests, image_dir, mask_dir)
    with open(output_file_name, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
