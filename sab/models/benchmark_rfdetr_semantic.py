"""RF-DETR semantic logits on ADE20K, matching the standalone training evaluator."""

import json
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from sab.clock_watch import ThrottleMonitor
from sab.evaluation import run_timed_pass
from sab.models.utils import ArtifactBenchmarkRequest, pretty_print_results, run_benchmark_on_artifacts
from sab.onnx_inference import ONNXInferenceCPU, ONNXInferenceCUDA
from sab.semantic_evaluation import ADE20K_CONFIG, semantic_image_pairs
from sab.trt_inference import TRTInference, build_engine


def preprocess_image(image: torch.Tensor, image_input_shape):
    """PIL bilinear square resize on uint8 RGB, then ImageNet normalization.

    RF-DETR validation resizes PIL images before converting to floating tensors.
    Resizing normalized float tensors changes uint8 rounding and the score.
    """
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("RF-DETR semantic benchmarking requires one RGB image")
    if (len(image_input_shape) != 4 or tuple(image_input_shape[:2]) != (1, 3)
            or not all(isinstance(d, int) and d > 0 for d in image_input_shape)):
        raise ValueError("Export RF-DETR semantic with static [1,3,H,W] input dimensions")
    input_h, input_w = image_input_shape[-2:]
    original_shape = tuple(image.shape[-2:])
    pixels = image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    resized = Image.fromarray(pixels).resize((input_w, input_h), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor.unsqueeze(0).to(image.device).contiguous(), {
        "original_shape": original_shape, "input_shape": (input_h, input_w),
    }


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict, background_channel: bool = False):
    """Match upstream's two bilinear resizes before argmax, without resizing GT.

    ``background_channel``: the artifact has an extra background logit at channel 0 (rf-detr-internal exports
    [1,151,H,W]: background + the 150 ADE20K classes). ADE20K has no background label, so the channel is dropped
    and the argmax runs over the 150 classes.
    """
    if len(outputs) != 1:
        raise ValueError("Expected one final-stage RF-DETR semantic logits output")
    logits = next(iter(outputs.values()))
    expected_channels = ADE20K_CONFIG.num_classes + int(background_channel)
    if logits.ndim != 4 or logits.shape[:2] != (1, expected_channels):
        raise ValueError(f"Expected [1,{expected_channels},H,W] semantic logits, got {tuple(logits.shape)}")
    if background_channel:
        logits = logits[:, 1:]
    if not logits.is_floating_point():
        raise ValueError("RF-DETR semantic output must contain floating-point logits")
    if not torch.isfinite(logits).all():
        raise ValueError("RF-DETR semantic logits contain NaN/Inf; check runtime precision")
    # The training evaluator explicitly uses both stages. Direct low-resolution
    # -> original interpolation is not numerically equivalent.
    scores = F.interpolate(logits.float(), size=metadata["input_shape"], mode="bilinear", align_corners=False)
    scores = F.interpolate(scores, size=metadata["original_shape"], mode="bilinear", align_corners=False)
    return scores[0].argmax(0)


class _RFDETRSemanticAdapter:
    background_channel = False

    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata, self.background_channel)


class RFDETRSemanticTRTInference(_RFDETRSemanticAdapter, TRTInference):
    def __init__(self, model_path, image_input_name=None):
        super().__init__(model_path, image_input_name, use_cuda_graph=True, prediction_type="semantic")


class RFDETRSemanticONNXInference(_RFDETRSemanticAdapter, ONNXInferenceCUDA):
    def __init__(self, model_path, image_input_name=None):
        super().__init__(model_path, image_input_name, prediction_type="semantic")


class RFDETRSemanticONNXCPUInference(_RFDETRSemanticAdapter, ONNXInferenceCPU):
    def __init__(self, model_path, image_input_name=None):
        super().__init__(model_path, image_input_name, prediction_type="semantic")


class RFDETRInternalSemanticTRTInference(RFDETRSemanticTRTInference):
    """rf-detr-internal semantic export: [1,151,H,W] logits with background at channel 0."""

    background_channel = True


class RFDETRInternalSemanticONNXInference(RFDETRSemanticONNXInference):
    background_channel = True


class RFDETRInternalSemanticONNXCPUInference(RFDETRSemanticONNXCPUInference):
    background_channel = True


class RFDETRInternalSemanticLatencyTRTInference(RFDETRInternalSemanticTRTInference):
    """Latency-only runs: postprocess returns the raw logits instead of resizing them and taking the argmax.

    The two full-resolution resizes of 151-channel logits and the argmax run on the GPU between timed calls. Timing
    excludes them, but on a T4 they keep the GPU at its 70 W power cap, which throttles the clocks for the timed
    calls too. Without accuracy there is no reason to run them.
    """

    def postprocess(self, outputs, metadata):
        return next(iter(outputs.values()))


def run_latency_only(onnx_path: str, image_dir: str, buffer_time: float, max_images: int | None) -> dict:
    """Time an rf-detr-internal semantic artifact (TensorRT FP16, CUDA graphs) on images, without scoring them."""
    engine_path = onnx_path.replace(".onnx", ".fp16.engine")
    with ThrottleMonitor() as build_monitor:
        build_engine(onnx_path, engine_path, use_fp16=True)
    if build_monitor.did_throttle():
        print("GPU throttled during engine build. This is expected and is a limitation of TensorRT.")
    inference = RFDETRInternalSemanticLatencyTRTInference(engine_path)
    images = [str(path) for path, _ in semantic_image_pairs(image_dir, image_dir.replace("images", "annotations"), max_images)]
    with ThrottleMonitor() as monitor:
        latency_stats = run_timed_pass(inference, images, buffer_time=buffer_time)
    throttled = monitor.did_throttle()
    print("🔴  GPU throttled, latency results are unreliable." if throttled
          else "GPU did not throttle during evaluation. Latency numbers should be reliable.")
    request = ArtifactBenchmarkRequest(
        onnx_path=onnx_path, inference_class=RFDETRInternalSemanticLatencyTRTInference, needs_fp16=True,
        buffer_time=buffer_time, max_images=max_images,
    )
    return {"artifact_request": request.dump(), "accuracy_stats": None, "latency_stats": latency_stats,
            "throttled": throttled}


def main(image_dir: str, mask_dir: str, buffer_time: float = 0.2,
         output_file_name: str = "rfdetr_semantic_results.json",
         onnx_path: str = "rf-detr-semseg-nano-ade20k-best-ema.onnx",
         runtime: str = "trt", fp16: bool | None = None, max_images: int | None = None,
         background_channel: bool = False, latency_only: bool = False):
    """Benchmark a static RF-DETR semantic ONNX export on raw ADE20K masks.

    Uses original-resolution labels, a 150-class global confusion matrix and
    the upstream two-stage logit resize. Omit max_images for the full split.
    Latency measures artifact execution; external resize and argmax are excluded.
    Pass background_channel for rf-detr-internal exports ([1,151,H,W], background first).
    latency_only (rf-detr-internal exports, TensorRT FP16) times the artifact without scoring it; see
    RFDETRInternalSemanticLatencyTRTInference.
    """
    if latency_only:
        if not (background_channel and runtime == "trt" and fp16 is not False):
            raise ValueError("latency_only supports rf-detr-internal exports (background_channel) on TensorRT FP16")
        if not Path(onnx_path).is_file():
            raise FileNotFoundError(f"RF-DETR semantic ONNX artifact not found: {onnx_path}")
        results = [run_latency_only(onnx_path, image_dir, buffer_time, max_images)]
        with open(output_file_name, "w") as f:
            json.dump(results, f, indent=2, allow_nan=False)
        print(results[0]["latency_stats"], "throttled:", results[0]["throttled"])
        return
    if background_channel:
        runtimes = {"trt": RFDETRInternalSemanticTRTInference, "onnx-cuda": RFDETRInternalSemanticONNXInference,
                    "onnx-cpu": RFDETRInternalSemanticONNXCPUInference}
    else:
        runtimes = {"trt": RFDETRSemanticTRTInference, "onnx-cuda": RFDETRSemanticONNXInference,
                    "onnx-cpu": RFDETRSemanticONNXCPUInference}
    if runtime not in runtimes:
        raise ValueError(f"runtime must be one of {', '.join(runtimes)}")
    if fp16 is None:
        fp16 = runtime == "trt"
    if fp16 and runtime != "trt":
        raise ValueError("FP16 engine compilation requires runtime=trt")
    if buffer_time < 0:
        raise ValueError("buffer_time must be nonnegative")
    if not Path(onnx_path).is_file():
        raise FileNotFoundError(f"RF-DETR semantic ONNX artifact not found: {onnx_path}")
    semantic_image_pairs(image_dir, mask_dir, max_images)
    requests = [ArtifactBenchmarkRequest(
        onnx_path=onnx_path, inference_class=runtimes[runtime], needs_fp16=fp16,
        buffer_time=buffer_time, max_images=max_images, semantic_config=ADE20K_CONFIG,
    )]
    results = run_benchmark_on_artifacts(requests, image_dir, mask_dir)
    with open(output_file_name, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
