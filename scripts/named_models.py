"""Named trainable model registry for standardized latency benchmarking.

Every entry benchmarks a TensorRT FP16 engine at batch 1 -- the same protocol
the NAS timing tables use -- so named models and NAS children are directly
comparable on a latency vs accuracy pareto chart.
"""

from dataclasses import dataclass

from sab.models.benchmark_lwdetr import LWDETRTRTInference
from sab.models.benchmark_rfdetr import RFDETRTRTInference
from sab.models.benchmark_rtdetr import RTDETRTRTInference
from sab.models.benchmark_yolov8 import YOLOv8TRTInference
from sab.models.benchmark_yolov11 import YOLOv11TRTInference
from sab.models.utils import ArtifactBenchmarkRequest


@dataclass(frozen=True)
class NamedModel:
    key: str
    family: str
    onnx_artifact: str
    inference_class: type
    needs_class_remapping: bool = False

    def to_request(self, buffer_time: float, max_images: int | None, is_jetson: bool) -> ArtifactBenchmarkRequest:
        return ArtifactBenchmarkRequest(
            onnx_path=self.onnx_artifact,
            inference_class=self.inference_class,
            needs_class_remapping=self.needs_class_remapping,
            needs_fp16=True,
            buffer_time=buffer_time,
            max_images=max_images,
            is_jetson=is_jetson,
        )


def _family(family: str, inference_class: type, artifacts: dict[str, str], needs_class_remapping: bool = False) -> list[NamedModel]:
    return [
        NamedModel(
            key=key,
            family=family,
            onnx_artifact=artifact,
            inference_class=inference_class,
            needs_class_remapping=needs_class_remapping,
        )
        for key, artifact in artifacts.items()
    ]


NAMED_MODELS: dict[str, NamedModel] = {
    model.key: model
    for model in [
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
        *_family("yolov11", YOLOv11TRTInference, {
            "yolo11n": "yolo11n_nms_conf_0.01.onnx",
            "yolo11s": "yolo11s_nms_conf_0.01.onnx",
            "yolo11m": "yolo11m_nms_conf_0.01.onnx",
            "yolo11l": "yolo11l_nms_conf_0.01.onnx",
            "yolo11x": "yolo11x_nms_conf_0.01.onnx",
        }, needs_class_remapping=True),
        *_family("lwdetr", LWDETRTRTInference, {
            "lwdetr-tiny": "lw-detr-tiny.onnx",
            "lwdetr-small": "lw-detr-small.onnx",
            "lwdetr-medium": "lw-detr-medium.onnx",
            "lwdetr-large": "lw-detr-large.onnx",
            "lwdetr-xlarge": "lw-detr-xlarge.onnx",
        }),
        *_family("rtdetr", RTDETRTRTInference, {
            "rtdetr-r18": "rtdetr_r18_coco.onnx",
            "rtdetr-r50": "rtdetr_r50_coco.onnx",
            "rtdetr-r101": "rtdetr_r101_coco.onnx",
        }),
    ]
}
