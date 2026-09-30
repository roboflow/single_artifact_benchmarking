"""Benchmark adapter for yololite decoded ONNX models.

Yololite's decoded ONNX export produces three outputs (no NMS):
  - boxes_xyxy:  (batch, N, 4)  pixel coordinates [0, img_size-1]
  - obj_logits:  (batch, N, 1)  raw objectness logits
  - cls_logits:  (batch, N, C)  raw per-class logits

This adapter reproduces the preprocessing and postprocessing from
yololite's own inference pipeline so that benchmarking results are
comparable to native evaluation.

Preprocessing reference:
  yololite/scripts/data/augment.py:153-171  (get_val_transform)
  yololite/export/infer_onnx_decoded.py:19-30, 99-100, 111-126

Postprocessing reference:
  yololite/export/infer_onnx_decoded.py:128-162
  yololite/scripts/helpers/helpers.py:86-153  (_decode_batch_to_coco_dets)
"""

from functools import partial

import fire
import torch
import torchvision.transforms.functional as TF
from torchvision.ops import batched_nms

from sab.processors import Processor
from sab.request import ArtifactBenchmarkRequest
from sab.results import pretty_print_results
from sab.runner import run_benchmark_on_artifacts
from sab.runtimes.onnxruntime import ONNXRuntime
from sab.runtimes.tensorrt import TRTRuntime


# ── Preprocessing ────────────────────────────────────────────────────────────
# Reproduces yololite/export/infer_onnx_decoded.py:19-30 (letterbox) and
# lines 99-100, 124-126 (ImageNet normalize, CHW transpose).
#
# The target resolution comes from the ONNX model's input shape
# (image_input_shape) rather than being hardcoded, so this works for any
# img_size the model was exported with.

# ImageNet normalization constants used by yololite
# Reference: yololite/scripts/data/augment.py:159-160
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Letterbox pad value: 114 in uint8 ≈ 0.447 in [0,1] float
# Reference: yololite/scripts/data/augment.py:27-30 (make_pad, value=114)
# Reference: yololite/export/infer_onnx_decoded.py:19 (letterbox color=(114,114,114))
_PAD_VALUE = 114.0 / 255.0


def preprocess_image(
    image: torch.Tensor, image_input_shape: tuple, normalize: bool = True
) -> tuple[torch.Tensor, dict]:
    """Letterbox + ImageNet normalize, matching yololite's val transform.

    Reference: yololite/export/infer_onnx_decoded.py:111-126
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    _, _, orig_h, orig_w = image.shape
    input_h, input_w = image_input_shape[2], image_input_shape[3]

    # Letterbox: uniform scale to fit, then center-pad to target size.
    # Reference: yololite/export/infer_onnx_decoded.py:21-29
    scale = min(input_h / orig_h, input_w / orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    pad_h = input_h - new_h
    pad_w = input_w - new_w
    top = pad_h // 2
    left = pad_w // 2

    image = TF.resize(image, (new_h, new_w), antialias=False)
    padding = (left, top, pad_w - left, pad_h - top)  # (left, top, right, bottom)
    image = TF.pad(image, padding, fill=_PAD_VALUE)

    # ImageNet normalization (applied AFTER letterbox padding).
    # Reference: yololite/export/infer_onnx_decoded.py:124-125
    if normalize:
        means = _MEAN.to(device=image.device, dtype=image.dtype)
        stds = _STD.to(device=image.device, dtype=image.dtype)
        image = (image - means) / stds
    else:
        image = image * 255.0

    metadata = {
        "original_shape": (orig_h, orig_w),
        "image_input_shape": image_input_shape,
        "scale": scale,
        "padding": padding,
    }
    return image, metadata


# ── Postprocessing ───────────────────────────────────────────────────────────
# Reproduces yololite/export/infer_onnx_decoded.py:128-171 and
# yololite/scripts/helpers/helpers.py:118-143 (confidence filter + per-class NMS).

# NMS parameters matching yololite's internal COCO evaluation.
# Reference: yololite/scripts/helpers/helpers.py:87 (conf_th=0.001, iou_th=0.65)
_CONF_THRESHOLD = 0.001
_IOU_THRESHOLD = 0.65
_MAX_DETECTIONS = 300


def postprocess_output(
    outputs: dict[str, torch.Tensor], metadata: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode logits, apply NMS, un-letterbox to normalized [0,1] coordinates.

    Reference: yololite/export/infer_onnx_decoded.py:128-171
    """
    # Raw model outputs (no NMS in the ONNX graph).
    boxes_xyxy = outputs["boxes_xyxy"]  # (1, N, 4) pixel coords
    obj_logits = outputs["obj_logits"]  # (1, N, 1) raw logits
    cls_logits = outputs["cls_logits"]  # (1, N, C) raw logits

    # Squeeze batch dimension.
    boxes = boxes_xyxy[0]  # (N, 4)

    # Score = sigmoid(obj) * max(sigmoid(cls)).
    # Reference: yololite/export/infer_onnx_decoded.py:130-135
    obj_scores = obj_logits[0, :, 0].sigmoid()  # (N,)
    cls_scores = cls_logits[0].sigmoid()  # (N, C)
    max_cls_scores, class_ids = cls_scores.max(dim=1)  # (N,), (N,)
    scores = obj_scores * max_cls_scores  # (N,)

    # Confidence filter.
    # Reference: yololite/scripts/helpers/helpers.py:118
    keep = scores > _CONF_THRESHOLD
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    # Per-class NMS.
    # Reference: yololite/scripts/helpers/helpers.py:125-143
    # Reference: yololite/export/infer_onnx_decoded.py:147-149
    nms_idx = batched_nms(boxes, scores, class_ids, _IOU_THRESHOLD)
    nms_idx = nms_idx[:_MAX_DETECTIONS]
    boxes = boxes[nms_idx]
    scores = scores[nms_idx]
    class_ids = class_ids[nms_idx]

    # Un-letterbox: pixel coords in padded image → normalized [0,1] in original.
    # Reference: yololite/export/infer_onnx_decoded.py:165-171 (adapted to [0,1])
    padding = metadata["padding"]  # (left, top, right, bottom)
    image_input_shape = metadata["image_input_shape"]
    unpadded_w = image_input_shape[3] - padding[0] - padding[2]
    unpadded_h = image_input_shape[2] - padding[1] - padding[3]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / unpadded_w
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / unpadded_h
    boxes = torch.clamp(boxes, 0.0, 1.0)

    # SAB evaluation.py expects (xyxy, class_id, score) each with batch dim.
    return (
        boxes.unsqueeze(0),
        class_ids.float().unsqueeze(0),
        scores.unsqueeze(0),
    )


# ── Processor ────────────────────────────────────────────────────────────────

class YoloLiteProcessor(Processor):
    """Letterbox, ImageNet normalization and NMS for yololite decoded models."""

    def preprocess(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(image, self.input_spec.shape, self.normalize)

    def postprocess(
        self, outputs: dict[str, torch.Tensor], metadata: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


# ── CLI entry point ──────────────────────────────────────────────────────────

def build_requests(onnx_path: str, buffer_time: float = 0.0) -> list[ArtifactBenchmarkRequest]:
    # The decoded ONNX has a dynamic batch dimension in its outputs, so ONNX Runtime allocates them.
    # The decoded ONNX has fixed output shapes (no NMS in the graph), so CUDA graphs work for TensorRT.
    return [
        ArtifactBenchmarkRequest(
            artifact_path=onnx_path,
            runtime=partial(ONNXRuntime, dynamic_output_shapes=True),
            processor=YoloLiteProcessor,
            device="cpu",
            precision="fp32",
            max_dets=500,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            artifact_path=onnx_path,
            runtime=TRTRuntime,
            processor=YoloLiteProcessor,
            device="gpu",
            precision="fp32",
            max_dets=500,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            artifact_path=onnx_path,
            runtime=TRTRuntime,
            processor=YoloLiteProcessor,
            device="gpu",
            precision="fp16",
            max_dets=500,
            buffer_time=buffer_time,
        ),
    ]


def main(
    onnx_path: str,
    image_dir: str,
    annotations_file_path: str,
    buffer_time: float = 0.0,
    output_file_name: str = "yololite_results.json",
    runtimes=None,
    devices=None,
    max_images: int | None = None,
    rerun: bool = False,
):
    """Benchmark a yololite ONNX model with ONNX-CPU, TRT-fp32, and TRT-fp16.

    Args:
        onnx_path: Path to the decoded yololite .onnx file.
        image_dir: Directory containing evaluation images.
        annotations_file_path: Path to COCO-format annotations JSON.
        buffer_time: Seconds to wait between inferences (GPU cooling).
        output_file_name: Where to save the results JSON.
        runtimes: Run only these runtimes, for example "tensorrt".
        devices: Run only these devices, for example "cpu".
        max_images: Evaluate only the first N images.
        rerun: Run rows again that the output file already holds.
    """
    results = run_benchmark_on_artifacts(
        build_requests(onnx_path, buffer_time),
        image_dir,
        annotations_file_path,
        output_file=output_file_name,
        runtimes=runtimes,
        devices=devices,
        max_images=max_images,
        rerun=rerun,
    )
    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
