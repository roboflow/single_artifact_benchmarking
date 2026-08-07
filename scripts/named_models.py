"""Named trainable model registry for standardized latency benchmarking.

The registry contains only models that users can train on the platform with
roboflow-train: rfdetr, yolov8, yolov11, yolo26 (each with its seg variant
where artifacts exist), and yololite.

Every entry benchmarks a TensorRT FP16 engine at batch 1. This is the same
protocol that the NAS timing tables use. As a result, named models and NAS
children are directly comparable on a latency vs accuracy pareto chart.

Known gaps:
- rfdetr-seg: no benchmark ONNX artifacts are in the GCS bucket yet
  (sab/models/benchmark_rfdetr_seg.py has classes but an empty artifact
  list). Add entries here when the artifacts are uploaded.
- yolo26-seg: only the nano artifact (yolo26n-seg.onnx) was exercised on the
  yolo26_seg branch. Add the other sizes after a validation run.
- yololite: there is no canonical ONNX in the bucket. The model needs a
  decoded local file. Pass it with --yololite-onnx (or SAB_YOLOLITE_ONNX).
"""

from dataclasses import dataclass
from typing import Callable

from sab.models.benchmark_rfdetr import RFDETRTRTInference
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
    onnx_artifact: str | None = None  # None: the model needs a local ONNX path
    needs_class_remapping: bool = False
    graph_surgery: Callable[[str], str] | None = None
    max_dets: int = 100

    def to_request(
        self,
        buffer_time: float,
        max_images: int | None,
        is_jetson: bool,
        local_onnx: str | None = None,
    ) -> ArtifactBenchmarkRequest:
        onnx_path = self.onnx_artifact or local_onnx
        if onnx_path is None:
            raise ValueError(f"{self.key} needs a local ONNX path")
        return ArtifactBenchmarkRequest(
            onnx_path=onnx_path,
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
) -> list[NamedModel]:
    return [
        NamedModel(
            key=key,
            family=family,
            onnx_artifact=artifact,
            inference_class=inference_class,
            needs_class_remapping=needs_class_remapping,
            graph_surgery=graph_surgery,
        )
        for key, artifact in artifacts.items()
    ]


_ALL_MODELS: list[NamedModel] = [
        *_family("rfdetr", RFDETRTRTInference, {
            "rfdetr-nano": "rf-detr-nano.onnx",
            "rfdetr-small": "rf-detr-small.onnx",
            "rfdetr-medium": "rf-detr-medium.onnx",
        }),
        *_family("yolov8", YOLOv8TRTInference, {
            "yolov8n": "yolov8n_nms_conf_0.01.onnx",
            "yolov8s": "yolov8s_nms_conf_0.01.onnx",
            "yolov8m": "yolov8m_nms_conf_0.01.onnx",
        }, needs_class_remapping=True),
        *_family("yolov8-seg", YOLOv8SegTRTInference, {
            "yolov8n-seg": "yolov8n_seg_nms_conf_0.01.onnx",
            "yolov8s-seg": "yolov8s_seg_nms_conf_0.01.onnx",
            "yolov8m-seg": "yolov8m_seg_nms_conf_0.01.onnx",
            "yolov8l-seg": "yolov8l_seg_nms_conf_0.01.onnx",
            "yolov8x-seg": "yolov8x_seg_nms_conf_0.01.onnx",
        }, needs_class_remapping=True, graph_surgery=fuse_yolo_mask_postprocessing_into_onnx),
        *_family("yolov11", YOLOv11TRTInference, {
            "yolo11n": "yolo11n_nms_conf_0.01.onnx",
            "yolo11s": "yolo11s_nms_conf_0.01.onnx",
            "yolo11m": "yolo11m_nms_conf_0.01.onnx",
            "yolo11l": "yolo11l_nms_conf_0.01.onnx",
            "yolo11x": "yolo11x_nms_conf_0.01.onnx",
        }, needs_class_remapping=True),
        *_family("yolov11-seg", YOLOv11SegTRTInference, {
            "yolo11n-seg": "yolo11n_seg_nms_conf_0.01.onnx",
            "yolo11s-seg": "yolo11s_seg_nms_conf_0.01.onnx",
            "yolo11m-seg": "yolo11m_seg_nms_conf_0.01.onnx",
            "yolo11l-seg": "yolo11l_seg_nms_conf_0.01.onnx",
            "yolo11x-seg": "yolo11x_seg_nms_conf_0.01.onnx",
        }, needs_class_remapping=True, graph_surgery=fuse_yolo_mask_postprocessing_into_onnx),
        *_family("yolo26", YOLO26TRTInference, {
            "yolo26n": "yolo26n.onnx",
            "yolo26s": "yolo26s.onnx",
            "yolo26m": "yolo26m.onnx",
            "yolo26l": "yolo26l.onnx",
            "yolo26x": "yolo26x.onnx",
        }, needs_class_remapping=True),
        # yolo26-seg exports masks directly, so it uses the yolov11-seg adapter
        # without graph surgery (see the yolo26_seg branch).
        *_family("yolo26-seg", YOLOv11SegTRTInference, {
            "yolo26n-seg": "yolo26n-seg.onnx",
        }, needs_class_remapping=True),
        NamedModel(
            key="yololite",
            family="yololite",
            inference_class=YoloLiteTRTInference,
            onnx_artifact=None,
            max_dets=500,
        ),
]

assert len({m.key for m in _ALL_MODELS}) == len(_ALL_MODELS), "duplicate model keys"

NAMED_MODELS: dict[str, NamedModel] = {model.key: model for model in _ALL_MODELS}
