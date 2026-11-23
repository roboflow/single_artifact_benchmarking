import fire

from sab.models.benchmark_yolov11 import TRT_WITHOUT_CUDA_GRAPH, YOLOv11Processor
from sab.request import ArtifactBenchmarkRequest
from sab.results import pretty_print_results
from sab.runner import run_benchmark_on_artifacts


class YOLOv8Processor(YOLOv11Processor):
    """YOLOv8 exports use the same letterbox and NMS output layout as YOLOv11."""


def build_requests(buffer_time: float = 0.0) -> list[ArtifactBenchmarkRequest]:
    return [
        ArtifactBenchmarkRequest(
            artifact_path=f"yolov8{size}_nms_conf_0.01.onnx",
            runtime=TRT_WITHOUT_CUDA_GRAPH,
            processor=YOLOv8Processor,
            device="gpu",
            precision="fp16",
            buffer_time=buffer_time,
            needs_class_remapping=True,
        )
        for size in ("n", "s", "m")
    ]


def main(
    image_dir: str,
    annotations_file_path: str,
    buffer_time: float = 0.0,
    output_file_name: str = "yolov8_results.json",
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
