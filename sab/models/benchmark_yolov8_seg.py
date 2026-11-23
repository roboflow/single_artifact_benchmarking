from functools import partial

import fire

from sab.models.benchmark_yolov11_seg import YOLOv11SegProcessor
from sab.models.graph_surgery import fuse_yolo_mask_postprocessing_into_onnx
from sab.request import ArtifactBenchmarkRequest
from sab.results import pretty_print_results
from sab.runner import run_benchmark_on_artifacts
from sab.runtimes.tensorrt import TRTRuntime


class YOLOv8SegProcessor(YOLOv11SegProcessor):
    """YOLOv8 seg shares the letterbox preprocess and the mask postprocess of YOLOv11 seg."""


def build_requests(buffer_time: float = 0.0) -> list[ArtifactBenchmarkRequest]:
    artifact_paths = [
        "yolov8n_seg_nms_conf_0.01.onnx",
        "yolov8s_seg_nms_conf_0.01.onnx",
        "yolov8m_seg_nms_conf_0.01.onnx",
        "yolov8l_seg_nms_conf_0.01.onnx",
        "yolov8x_seg_nms_conf_0.01.onnx",
    ]
    return [
        ArtifactBenchmarkRequest(
            artifact_path=artifact_path,
            graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
            runtime=partial(TRTRuntime, use_cuda_graph=False),
            processor=YOLOv8SegProcessor,
            device="gpu",
            precision="fp16",
            buffer_time=buffer_time,
            needs_class_remapping=True,
        )
        for artifact_path in artifact_paths
    ]

def main(
    image_dir: str,
    annotations_file_path: str,
    buffer_time: float = 0.0,
    output_file_name: str = "yolov8_results.json",
    runtimes: str | None = None,
    devices: str | None = None,
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
