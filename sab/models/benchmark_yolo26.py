"""SAB benchmark adapter for YOLO26 ONNX models.

YOLO26 uses the same ultralytics export format as YOLOv11:
  - Input:  images  (1, 3, 640, 640)
  - Output: output0 (1, 300, 6)  [x1, y1, x2, y2, conf, cls]

Preprocessing and postprocessing are identical to YOLOv11.
"""

import fire

from sab.models.benchmark_yolov11 import YOLOv11Processor
from sab.request import ArtifactBenchmarkRequest
from sab.results import pretty_print_results
from sab.runner import run_benchmark_on_artifacts
from sab.runtimes.onnxruntime import ONNXRuntime
from sab.runtimes.openvino import OpenVINORuntime
from sab.runtimes.tensorrt import TRTRuntime


class YOLO26Processor(YOLOv11Processor):
    pass


def build_requests(buffer_time: float = 0.0) -> list[ArtifactBenchmarkRequest]:
    onnx_rows = [
        request
        for size in ("n", "s", "m", "l", "x")
        for request in (
            ArtifactBenchmarkRequest(
                artifact_path=f"yolo26{size}.onnx",
                runtime=TRTRuntime,
                processor=YOLO26Processor,
                device="gpu",
                precision="fp32",
                buffer_time=buffer_time,
                needs_class_remapping=True,
            ),
            ArtifactBenchmarkRequest(
                artifact_path=f"yolo26{size}.onnx",
                runtime=TRTRuntime,
                processor=YOLO26Processor,
                device="gpu",
                precision="fp16",
                buffer_time=buffer_time,
                needs_class_remapping=True,
            ),
            ArtifactBenchmarkRequest(
                artifact_path=f"yolo26{size}.onnx",
                runtime=ONNXRuntime,
                processor=YOLO26Processor,
                device="cpu",
                precision="fp32",
                buffer_time=buffer_time,
                needs_class_remapping=True,
            ),
        )
    ]
    openvino_rows = [
        ArtifactBenchmarkRequest(
            artifact_path=f"yolo26{size}.onnx",
            runtime=OpenVINORuntime,
            processor=YOLO26Processor,
            device="cpu",
            precision=precision,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        )
        for size in ("n", "s", "m", "l", "x")
        for precision in ("fp32", "fp16")
    ]
    return onnx_rows + openvino_rows


def main(
    image_dir: str,
    annotations_file_path: str,
    buffer_time: float = 0.0,
    output_file_name: str = "yolo26_results.json",
    runtimes=None,
    devices=None,
    max_images: int | None = None,
    rerun: bool = False,
):
    results = run_benchmark_on_artifacts(
        build_requests(buffer_time),
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
