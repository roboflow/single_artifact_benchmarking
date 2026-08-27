"""SAB benchmark adapter for YOLO26 instance-segmentation ONNX models.

YOLO26 is NMS-free, so the seg export needs no fused-NMS graph surgery and no
`fuse_yolo_mask_postprocessing_into_onnx` pass. The plain ultralytics export is
benchmarked as it comes out of the bucket:

  - Input:   images  (1, 3, 640, 640)
  - Output0: (1, 300, 38)      [x1, y1, x2, y2, score, class_id, 32 mask coeffs]
  - Output1: (1, 32, 160, 160) mask prototypes

That differs from `benchmark_yolov11_seg`, whose engine carries the mask matmul,
the upsample, and the box crop inside the graph (`_det_meta`/`_masks_cropped`).
Here the mask assembly happens in torch, after inference, so it sits outside the
timed region exactly like every other SAB postprocess.

Preprocessing is the YOLOv11 letterbox, unchanged.
"""

import json
from typing import Dict, Tuple

import fire
import torch
import torch.nn.functional as F

from sab.models.benchmark_yolov11_seg import preprocess_image
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results
from sab.onnx_inference import ONNXInferenceCUDA
from sab.trt_inference import TRTInference


def postprocess_output(
    outputs: Dict[str, torch.Tensor],
    metadata: Dict,
    score_threshold: float = 0.0,
    mask_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post-process the plain YOLO26 seg export.

    The artifact contract (verified against the bucket exports, ultralytics
    8.4.0):
      - outputs["output0"]: (1, 300, 6 + C) rows of
            [x1, y1, x2, y2, score, class_id, C mask coeffs]
        with xyxy in model-input pixels
      - outputs["output1"]: (1, C, mh, mw) mask prototypes

    Anything else raises: the models this adapter runs are known exactly, so a
    shape mismatch means a wrong artifact, not a layout to adapt to.

    Returns:
      bboxes: (N, 4) float32 xyxy, normalized to [0, 1] against the original image
      labels: (N,) int64
      scores: (N,) float32
      masks:  (N, orig_h, orig_w) bool, thresholded at mask_threshold
    """
    det = outputs["output0"].squeeze(0)
    proto = outputs["output1"].squeeze(0)
    device = det.device

    if proto.ndim != 3:
        raise ValueError(f"Unexpected output1 shape after squeeze: {tuple(proto.shape)}")
    c, mh, mw = proto.shape
    if det.ndim != 2 or det.shape[1] != 6 + c:
        raise ValueError(
            f"Unexpected output0 shape after squeeze: {tuple(det.shape)}; "
            f"expected (N, {6 + c}) for {c} mask prototypes"
        )

    orig_h, orig_w = (int(x) for x in metadata["original_shape"][-2:])
    input_h, input_w = (int(x) for x in metadata["image_input_shape"][-2:])
    left, top, right, bottom = (int(x) for x in metadata["padding"])

    scores = det[:, 4]
    keep = scores > score_threshold
    det = det[keep]
    scores = scores[keep]

    if det.numel() == 0:
        return (
            torch.empty((0, 4), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.int64),
            torch.empty((0,), device=device, dtype=torch.float32),
            torch.empty((0, orig_h, orig_w), device=device, dtype=torch.bool),
        )

    boxes = det[:, :4].clone()
    labels = det[:, 5].to(torch.int64)
    mask_coeff = det[:, 6:]

    # Undo the letterbox, mirroring benchmark_yolov11.postprocess_output.
    # Separate gains match the integer rounding in preprocess_image.
    new_w = input_w - left - right
    new_h = input_h - top - bottom
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) * (orig_w / new_w)).clamp(0, orig_w)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) * (orig_h / new_h)).clamp(0, orig_h)

    # Masks: coeffs @ protos, upsample to model input, drop the padding, upsample
    # to the original image. The values stay raw logits until the final cut, so a
    # 0.0 threshold means the same cut as sigmoid > 0.5.
    masks = (mask_coeff @ proto.reshape(c, -1)).reshape(-1, mh, mw)
    masks = F.interpolate(masks.unsqueeze(1), size=(input_h, input_w), mode="bilinear", align_corners=False).squeeze(1)
    masks = masks[:, top : input_h - bottom, left : input_w - right]
    masks = F.interpolate(masks.unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)

    # Crop each mask to its box (vectorized).
    x1, y1, x2, y2 = boxes.unbind(1)
    cols = torch.arange(orig_w, device=device)[None, None, :]
    rows = torch.arange(orig_h, device=device)[None, :, None]
    crop = (
        (cols >= x1[:, None, None])
        & (cols < x2[:, None, None])
        & (rows >= y1[:, None, None])
        & (rows < y2[:, None, None])
    )
    masks = (masks * crop.to(masks.dtype)) > mask_threshold

    # SAB's contract: postprocess returns xyxy normalized to [0, 1]. sab.evaluation
    # multiplies the box back up by the source image size. Masks stay at original
    # resolution, which is what the RLE encoder reads.
    boxes[:, [0, 2]] /= orig_w
    boxes[:, [1, 3]] /= orig_h

    return boxes.to(torch.float32), labels, scores.to(torch.float32), masks


class YOLO26SegONNXInference(ONNXInferenceCUDA):
    def __init__(self, model_path: str, image_input_name: str | None = None):
        super().__init__(model_path, image_input_name, prediction_type="segm")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


class YOLO26SegTRTInference(TRTInference):
    def __init__(self, model_path: str, image_input_name: str | None = None):
        super().__init__(model_path, image_input_name, use_cuda_graph=True, prediction_type="segm")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


def main(image_dir: str, annotations_file_path: str, buffer_time: float = 0.0, output_file_name: str = "yolo26_seg_results.json"):
    requests = [
        ArtifactBenchmarkRequest(
            onnx_path=f"yolo26{size}-seg.onnx",
            inference_class=YOLO26SegTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        )
        for size in ("n", "s", "m", "l", "x")
    ]

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)

    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
