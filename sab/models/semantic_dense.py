"""Fixed-shape ADE20K adapters for SeaFormer and EfficientViT logits."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sab.models.utils import ArtifactBenchmarkRequest, pretty_print_results, run_benchmark_on_artifacts
from sab.semantic_evaluation import ADE20K_CONFIG, semantic_image_pairs


def preprocess_image(image, family, image_input_shape):
    """Resize RGB pixels to the artifact's fixed input, then normalize."""
    if family not in {"seaformer", "efficientvit"}:
        raise ValueError("Unknown dense semantic model family")
    if (len(image_input_shape) != 4 or tuple(image_input_shape[:2]) != (1, 3)
            or any(not isinstance(d, int) or d <= 0 for d in image_input_shape)):
        raise ValueError("Semantic ONNX models require fixed [1,3,H,W] input dimensions")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected one RGB image")
    original_shape = tuple(image.shape[-2:])
    input_h, input_w = image_input_shape[-2:]
    pixels = image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    interpolation = cv2.INTER_CUBIC if family == "efficientvit" else cv2.INTER_LINEAR
    pixels = cv2.resize(pixels, (input_w, input_h), interpolation=interpolation)
    if family == "efficientvit":
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float().div(255)
        tensor = (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    else:
        # Match MMCV's RGB normalization after the uint8 resize.
        pixels = pixels.astype(np.float32)
        mean = np.float64(np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, -1))
        stdinv = 1 / np.float64(np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, -1))
        cv2.subtract(pixels, mean, pixels)
        cv2.multiply(pixels, stdinv, pixels)
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
    return tensor[None].to(image.device).contiguous(), {
        "original_shape": original_shape, "input_shape": (input_h, input_w),
    }


def postprocess_output(outputs, metadata, family):
    if len(outputs) != 1:
        raise ValueError("Expected one raw semantic logits output")
    logits = next(iter(outputs.values()))
    if logits.ndim != 4 or tuple(logits.shape[:2]) != (1, 150) or not logits.is_floating_point():
        raise ValueError("Expected floating-point [1,150,H,W] ADE20K logits")
    if not torch.isfinite(logits).all():
        raise ValueError("Semantic logits contain NaN/Inf; check runtime precision")
    logits = logits.float()
    if family == "seaformer":
        logits = F.interpolate(logits, size=metadata["input_shape"], mode="bilinear", align_corners=False)
    mode = "bicubic" if family == "efficientvit" else "bilinear"
    logits = F.interpolate(logits, size=metadata["original_shape"], mode=mode, align_corners=False)
    return logits[0].argmax(0)


class DenseSemanticAdapter:
    family = None

    def preprocess(self, image):
        return preprocess_image(image, self.family, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata, self.family)


def run_dense_benchmark(image_dir, mask_dir, onnx_path, runtimes, runtime="trt", fp16=None,
                        buffer_time=0.0, output_file_name="semantic_results.json", max_images=None,
                        graph_surgery_func=None):
    if runtime not in runtimes:
        raise ValueError(f"runtime must be one of {', '.join(runtimes)}")
    if fp16 is None:
        fp16 = runtime == "trt"
    if fp16 and runtime != "trt":
        raise ValueError("FP16 compilation requires runtime=trt")
    if buffer_time < 0:
        raise ValueError("buffer_time must be nonnegative")
    if not Path(onnx_path).is_file():
        raise FileNotFoundError(f"Semantic ONNX artifact not found: {onnx_path}")
    semantic_image_pairs(image_dir, mask_dir, max_images)
    requests = [ArtifactBenchmarkRequest(
        onnx_path, runtimes[runtime], needs_fp16=fp16, buffer_time=buffer_time,
        max_images=max_images, semantic_config=ADE20K_CONFIG,
        graph_surgery_func=graph_surgery_func if fp16 else None)]
    results = run_benchmark_on_artifacts(requests, image_dir, mask_dir)
    with open(output_file_name, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    pretty_print_results(results)
