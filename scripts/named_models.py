"""Named trainable model registry for standardized latency benchmarking.

The registry contains the models that users can train on the platform with
roboflow-train, in every trainable size:

- rfdetr: nano, small, medium, large, xlarge, 2xlarge (base is legacy and
  pico is not trainable, so both are excluded)
- rfdetr-seg: nano, small, medium, large, xlarge, 2xlarge
- yolov8 and yolov8-seg: n, s, m, l, x
- yolov11 and yolov11-seg: n, s, m, l, x (keys use the artifact spelling
  "yolo11*"; the platform id spelling is "yolov11*")
- yolo26 and yolo26-seg: n, s, m, l, x
- yololite (GPU family) and yololite-edge (CPU family): n, s, m, l, xl

Every entry benchmarks a TensorRT FP16 engine at batch 1. This is the same
protocol that the NAS timing tables use. As a result, named models and NAS
children are directly comparable on a latency vs accuracy pareto chart.

Some entries name artifacts that are not in the GCS bucket yet. The runner
checks availability before each run and reports the missing artifacts. Run
`get_sab_latencies.py --list-missing` for the current gap list.
"""

from dataclasses import dataclass
from typing import Callable

from sab.models.benchmark_rfdetr import RFDETRTRTInference
from sab.models.benchmark_rfdetr_seg import RFDETRSegTRTInference
from sab.models.benchmark_yolo26 import YOLO26TRTInference
from sab.models.benchmark_yololite import YoloLiteTRTInference
from sab.models.benchmark_yolov8 import YOLOv8TRTInference
from sab.models.benchmark_yolov8_seg import YOLOv8SegTRTInference
from sab.models.benchmark_yolov11 import YOLOv11TRTInference
from sab.models.benchmark_yolov11_seg import YOLOv11SegTRTInference
from sab.models.graph_surgery import fuse_yolo_mask_postprocessing_into_onnx
from sab.models.utils import ArtifactBenchmarkRequest


@dataclass(frozen=True)
class NamedModel:
    key: str
    family: str
    inference_class: type
    onnx_artifact: str
    needs_class_remapping: bool = False
    graph_surgery: Callable[[str], str] | None = None
    max_dets: int = 100

    def to_request(self, buffer_time: float, max_images: int | None, is_jetson: bool) -> ArtifactBenchmarkRequest:
        return ArtifactBenchmarkRequest(
            onnx_path=self.onnx_artifact,
            inference_class=self.inference_class,
            needs_class_remapping=self.needs_class_remapping,
            needs_fp16=True,
            buffer_time=buffer_time,
            max_images=max_images,
            graph_surgery_func=self.graph_surgery,
            max_dets=self.max_dets,
            is_jetson=is_jetson,
        )


def _family(
    family: str,
    inference_class: type,
    artifacts: dict[str, str],
    needs_class_remapping: bool = False,
    graph_surgery: Callable[[str], str] | None = None,
    max_dets: int = 100,
) -> list[NamedModel]:
    return [
        NamedModel(
            key=key,
            family=family,
            onnx_artifact=artifact,
            inference_class=inference_class,
            needs_class_remapping=needs_class_remapping,
            graph_surgery=graph_surgery,
            max_dets=max_dets,
        )
        for key, artifact in artifacts.items()
    ]


_RFDETR_SIZES = ["nano", "small", "medium", "large", "xlarge", "2xlarge"]
_YOLO_SIZES = ["n", "s", "m", "l", "x"]
_YOLOLITE_SIZES = ["n", "s", "m", "l", "xl"]

_ALL_MODELS: list[NamedModel] = [
    *_family("rfdetr", RFDETRTRTInference, {
        f"rfdetr-{size}": f"rf-detr-{size}.onnx" for size in _RFDETR_SIZES
    }),
    *_family("rfdetr-seg", RFDETRSegTRTInference, {
        f"rfdetr-seg-{size}": f"rf-detr-seg-{size}.onnx" for size in _RFDETR_SIZES
    }),
    *_family("yolov8", YOLOv8TRTInference, {
        f"yolov8{size}": f"yolov8{size}_nms_conf_0.01.onnx" for size in _YOLO_SIZES
    }, needs_class_remapping=True),
    *_family("yolov8-seg", YOLOv8SegTRTInference, {
        f"yolov8{size}-seg": f"yolov8{size}_seg_nms_conf_0.01.onnx" for size in _YOLO_SIZES
    }, needs_class_remapping=True, graph_surgery=fuse_yolo_mask_postprocessing_into_onnx),
    *_family("yolov11", YOLOv11TRTInference, {
        f"yolo11{size}": f"yolo11{size}_nms_conf_0.01.onnx" for size in _YOLO_SIZES
    }, needs_class_remapping=True),
    *_family("yolov11-seg", YOLOv11SegTRTInference, {
        f"yolo11{size}-seg": f"yolo11{size}_seg_nms_conf_0.01.onnx" for size in _YOLO_SIZES
    }, needs_class_remapping=True, graph_surgery=fuse_yolo_mask_postprocessing_into_onnx),
    *_family("yolo26", YOLO26TRTInference, {
        f"yolo26{size}": f"yolo26{size}.onnx" for size in _YOLO_SIZES
    }, needs_class_remapping=True),
    # yolo26-seg exports masks directly, so it uses the yolov11-seg adapter
    # without graph surgery (see the yolo26_seg branch).
    *_family("yolo26-seg", YOLOv11SegTRTInference, {
        f"yolo26{size}-seg": f"yolo26{size}-seg.onnx" for size in _YOLO_SIZES
    }, needs_class_remapping=True),
    *_family("yololite", YoloLiteTRTInference, {
        f"yololite-{size}": f"yololite-{size}.onnx" for size in _YOLOLITE_SIZES
    }, max_dets=500),
    *_family("yololite-edge", YoloLiteTRTInference, {
        f"yololite-edge-{size}": f"yololite-edge-{size}.onnx" for size in _YOLOLITE_SIZES
    }, max_dets=500),
]

assert len({m.key for m in _ALL_MODELS}) == len(_ALL_MODELS), "duplicate model keys"

NAMED_MODELS: dict[str, NamedModel] = {model.key: model for model in _ALL_MODELS}
