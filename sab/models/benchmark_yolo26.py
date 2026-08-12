"""SAB benchmark adapter for YOLO26 ONNX models.

YOLO26 uses the same ultralytics export format as YOLOv11:
  - Input:  images  (1, 3, 640, 640)
  - Output: output0 (1, 300, 6)  [x1, y1, x2, y2, conf, cls]

Preprocessing and postprocessing are identical to YOLOv11.
"""

import json

import fire

from sab.models.benchmark_yolov11 import preprocess_image, postprocess_output
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results
from sab.onnx_inference import ONNXInferenceCPU, ONNXInferenceCUDA
from sab.trt_inference import TRTInference


class YOLO26ONNXCPUInference(ONNXInferenceCPU):
    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata)


class YOLO26ONNXInference(ONNXInferenceCUDA):
    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata)


class YOLO26TRTInference(TRTInference):
    def __init__(self, model_path, image_input_name=None):
        super().__init__(model_path, image_input_name, use_cuda_graph=True)

    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata)


def main(image_dir: str, annotations_file_path: str, buffer_time: float = 0.0, output_file_name: str = "yolo26_results.json"):
    requests = [
        request
        for size in ("n", "s", "m", "l", "x")
        for request in (
            ArtifactBenchmarkRequest(
                onnx_path=f"yolo26{size}.onnx",
                inference_class=YOLO26TRTInference,
                needs_fp16=False,
                buffer_time=buffer_time,
                needs_class_remapping=True,
            ),
            ArtifactBenchmarkRequest(
                onnx_path=f"yolo26{size}.onnx",
                inference_class=YOLO26TRTInference,
                needs_fp16=True,
                buffer_time=buffer_time,
                needs_class_remapping=True,
            ),
            ArtifactBenchmarkRequest(
                onnx_path=f"yolo26{size}.onnx",
                inference_class=YOLO26ONNXCPUInference,
                buffer_time=buffer_time,
                needs_class_remapping=True,
            ),
        )
    ]

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)

    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
