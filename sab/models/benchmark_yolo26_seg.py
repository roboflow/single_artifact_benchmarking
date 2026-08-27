"""SAB benchmark adapter for YOLO26 instance-segmentation ONNX models.

The plain ultralytics export emits detections (1, 300, 38) and mask prototypes
(1, 32, 160, 160). `fuse_yolo26_mask_postprocessing_into_onnx` — the upstream
variant behind the published yolo26-seg rows — folds the mask matmul, the
resize to the model-input grid, and the box crop into the graph. Masks leave
the engine as raw logits at input resolution; the postprocess removes the
letterbox padding, resizes to the original image, and binarizes at logit 0
(the sigmoid-0.5 cut). Boxes follow the yolov11-seg letterbox undo exactly.
CUDA-graph replay stays on, the NMS-free family convention.
"""

import json

import fire
import torch
import torch.nn.functional as F

from sab.models.benchmark_yolov11_seg import preprocess_image
from sab.models.graph_surgery import fuse_yolo26_mask_postprocessing_into_onnx
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results
from sab.onnx_inference import ONNXInferenceCUDA
from sab.trt_inference import TRTInference


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["_det_meta"][0, :, :4]
    scores = outputs["_det_meta"][0, :, 4]
    labels = outputs["_det_meta"][0, :, 5]

    input_h, input_w = metadata["image_input_shape"][2:]
    left, top, right_pad, bottom_pad = metadata["padding"]
    orig_h, orig_w = metadata["original_shape"][2:]

    # Boxes arrive in model-input pixels; undo the letterbox, normalize, clamp —
    # the same math as benchmark_yolov11_seg.postprocess_output.
    bboxes = bboxes.clone()
    bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - left) / (input_w - left - right_pad)
    bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - top) / (input_h - top - bottom_pad)
    bboxes = torch.clamp(bboxes, 0, 1)

    # Masks arrive as raw logits at input resolution, already cropped to their
    # box in-graph. Drop the padding, resize to the original image, cut at 0.
    masks = outputs["_masks_cropped"]
    if masks.dim() == 3:
        masks = masks.unsqueeze(0)
    masks = masks[:, :, top : input_h - bottom_pad, left : input_w - right_pad]
    masks = F.interpolate(masks.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    masks = masks.squeeze(0) > 0

    return bboxes, labels, scores, masks


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
            graph_surgery_func=fuse_yolo26_mask_postprocessing_into_onnx,
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
