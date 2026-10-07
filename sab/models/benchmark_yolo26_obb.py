"""SAB benchmark adapter for YOLO26 oriented-bounding-box (OBB) ONNX models, evaluated on YOLO-OBB tiles (e.g. DOTA).

Export (Ultralytics, as for the other YOLO models; ``nms=True`` reproduces ``model.predict`` exactly):

    yolo export model=yolo26n-obb.pt format=onnx imgsz=1024 nms=True conf=0.001

  - Input:  images  (1, 3, 1024, 1024)
  - Output: output0 (1, 300, 7)  [cx, cy, w, h, conf, cls, theta] in input pixels, theta in radians

Evaluate on DOTA v1.0 val tiles (1024 px, Ultralytics ``split_dota`` layout) with the YOLO-OBB labels directory in
place of the COCO annotations file:

    python -m sab.models.benchmark_yolo26_obb <tiles>/images/val <tiles>/labels/val
"""

import json

import fire
import torch

from sab.models.benchmark_yolo26_semantic import preprocess_image
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results
from sab.onnx_inference import ONNXInferenceCPU
from sab.trt_inference import TRTInference


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows ``[cx, cy, w, h, conf, cls, theta]`` in letterboxed input pixels -> oriented boxes in original pixels.

    Empty rows (score 0) are dropped. Letterboxing keeps the aspect ratio, so one scale (the mean of both axes,
    which differ only by rounding) maps centre offsets and sides back.
    """
    rows = outputs["output0"][0].float()
    rows = rows[rows[:, 4] > 0]
    top, left, resized_h, resized_w = metadata["crop"]
    original_h, original_w = metadata["original_shape"]
    scale_x, scale_y = original_w / resized_w, original_h / resized_h
    scale = (scale_x + scale_y) / 2
    obboxes = torch.stack([
        (rows[:, 0] - left) * scale_x,
        (rows[:, 1] - top) * scale_y,
        rows[:, 2] * scale,
        rows[:, 3] * scale,
        rows[:, 6],
    ], dim=1)
    return obboxes, rows[:, 5].long(), rows[:, 4]


class YOLO26OBBONNXCPUInference(ONNXInferenceCPU):
    def __init__(self, model_path, image_input_name=None):
        super().__init__(model_path, image_input_name, prediction_type="obb")

    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata)


class YOLO26OBBTRTInference(TRTInference):
    def __init__(self, model_path, image_input_name=None):
        super().__init__(model_path, image_input_name, use_cuda_graph=True, prediction_type="obb")

    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata)


def main(image_dir: str, labels_dir: str, buffer_time: float = 0.0, output_file_name: str = "yolo26_obb_results.json",
         sizes: str = "nsmlx", include_cpu: bool = False, max_images: int | None = None, max_dets: int = 300):
    """Benchmark yolo26{size}-obb.onnx for each size: TensorRT FP32 and FP16, plus ONNX CPU with ``include_cpu``.

    ``max_dets`` defaults to the model's 300 outputs: DOTA tiles can hold more than COCO's usual 100 objects.
    """
    requests = []
    for size in sizes:
        onnx_path = f"yolo26{size}-obb.onnx"
        for fp16 in (False, True):
            requests.append(ArtifactBenchmarkRequest(onnx_path=onnx_path, inference_class=YOLO26OBBTRTInference, needs_fp16=fp16,
                                                     buffer_time=buffer_time, max_images=max_images, max_dets=max_dets))
        if include_cpu:
            requests.append(ArtifactBenchmarkRequest(onnx_path=onnx_path, inference_class=YOLO26OBBONNXCPUInference,
                                                     buffer_time=buffer_time, max_images=max_images, max_dets=max_dets))

    results = run_benchmark_on_artifacts(requests, image_dir, labels_dir)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)

    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
