"""SAB benchmark adapter for YOLO26 instance-segmentation ONNX models.

The plain ultralytics export emits detections (1, 300, 38) and mask prototypes
(1, 32, 160, 160). `fuse_yolo_mask_postprocessing_into_onnx` folds the mask
matmul, sigmoid, and box crop into the graph — the same surgery, and therefore
the same timed region, as the yolov8-seg and yolov11-seg rows behind the
published numbers. Preprocessing and postprocessing are identical to YOLOv11
seg; the only YOLO26 difference is CUDA-graph replay, the NMS-free family
convention.
"""

import json

import fire
import torch

from sab.models.benchmark_yolov11_seg import postprocess_output, preprocess_image
from sab.models.graph_surgery import fuse_yolo_mask_postprocessing_into_onnx
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results
from sab.onnx_inference import ONNXInferenceCUDA
from sab.trt_inference import TRTInference


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
            graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
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
