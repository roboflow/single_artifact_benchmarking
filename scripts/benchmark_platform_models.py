"""Benchmark every named model the platform trains, TRT-FP16 at batch 1.

This is the named half of the platform latency table. The NAS half is
`scripts/benchmark_platform_nas_models.py`, and both use one protocol, so a
named model and a NAS child sit on one latency/accuracy chart.

`model_id` is the platform registry key, verbatim, not the artifact name. The
two differ often: the platform key is `rfdetr-large` and the artifact is
`rf-detr-large-new.onnx`; the platform key is `yolov11n` and the artifact is
`yolo11n_nms_conf_0.01.onnx`.

Every entry below states its artifact. Entries whose artifact is not in
gs://single_artifact_benchmarking are commented out with the reason, so the
gap is visible here instead of in a failed run. The inventory was read on
2026-08-25. The one exception is yolo26*-seg: those rows are enabled ahead of
their exports, which land in the bucket shortly.

Example:
    python scripts/benchmark_platform_models.py \\
        --image_dir /data \\
        --annotations_file_path /data/_annotations.coco.json \\
        --output_file_name /results/platform_models.json
"""

import json
import os
from dataclasses import dataclass
from typing import Callable

import fire

from sab.environment import collect_environment
from sab.models.benchmark_rfdetr import RFDETRTRTInference
from sab.models.benchmark_rfdetr_seg import RFDETRSegTRTInference
from sab.models.benchmark_yolo26 import YOLO26TRTInference
from sab.models.benchmark_yolo26_seg import YOLO26SegTRTInference
from sab.models.benchmark_yolov8 import YOLOv8TRTInference
from sab.models.benchmark_yolov8_seg import YOLOv8SegTRTInference
from sab.models.benchmark_yolov11 import YOLOv11TRTInference
from sab.models.benchmark_yolov11_seg import YOLOv11SegTRTInference
from sab.models.graph_surgery import (
    fuse_yolo_mask_postprocessing_into_onnx,
    fuse_yolo26_mask_postprocessing_into_onnx,
)
from sab.models.utils import (
    ArtifactBenchmarkRequest,
    nest_latency_by_hardware,
    pretty_print_results,
    run_benchmark_on_artifacts,
)

# Pinned methodology constant. The production NAS tables and the published
# RF-DETR benchmarks put 200 ms between passes. Every standard run uses the
# same value, so new numbers stay comparable with those tables. An entry that
# throttles in spite of the buffer keeps "throttled": true.
BUFFER_TIME_S = 0.2

DEFAULT_HARDWARE = "T4"


@dataclass(frozen=True)
class PlatformModel:
    """One named model: its platform key, its artifact, and how SAB runs it."""

    model_id: str
    onnx_artifact: str
    inference_class: type
    needs_class_remapping: bool = False
    graph_surgery: Callable[[str], str] | None = None
    max_dets: int = 100

    def to_request(self, buffer_time: float, max_images: int | None) -> ArtifactBenchmarkRequest:
        return ArtifactBenchmarkRequest(
            onnx_path=self.onnx_artifact,
            inference_class=self.inference_class,
            needs_class_remapping=self.needs_class_remapping,
            needs_fp16=True,
            buffer_time=buffer_time,
            max_images=max_images,
            graph_surgery_func=self.graph_surgery,
            max_dets=self.max_dets,
            model_id=self.model_id,
        )


# --- RF-DETR (CUDA-graph replay, no class remapping) -------------------------
#
# rfdetr-large takes rf-detr-large-new.onnx. rf-detr-large.onnx is a stale
# 560x560 graph and is not this model.
RFDETR_MODELS = [
    PlatformModel("rfdetr-nano", "rf-detr-nano.onnx", RFDETRTRTInference),
    PlatformModel("rfdetr-small", "rf-detr-small.onnx", RFDETRTRTInference),
    PlatformModel("rfdetr-medium", "rf-detr-medium.onnx", RFDETRTRTInference),
    PlatformModel("rfdetr-large", "rf-detr-large-new.onnx", RFDETRTRTInference),
    PlatformModel("rfdetr-xlarge", "rf-detr-xlarge.onnx", RFDETRTRTInference),
    PlatformModel("rfdetr-2xlarge", "rf-detr-xxlarge.onnx", RFDETRTRTInference),
]

# --- RF-DETR segmentation ----------------------------------------------------
#
# Exports of the pretrained gs://rfdetr checkpoints through the platform export
# path (rf-detr-internal@31743fa, class-default resolutions). The bucket also
# holds rf-detr-seg-nano-finetune.onnx, which is NOT the named artifact.
RFDETR_SEG_MODELS: list[PlatformModel] = [
    PlatformModel("rfdetr-seg-nano", "rf-detr-seg-nano.onnx", RFDETRSegTRTInference),
    PlatformModel("rfdetr-seg-small", "rf-detr-seg-small.onnx", RFDETRSegTRTInference),
    PlatformModel("rfdetr-seg-medium", "rf-detr-seg-medium.onnx", RFDETRSegTRTInference),
    PlatformModel("rfdetr-seg-large", "rf-detr-seg-large.onnx", RFDETRSegTRTInference),
    PlatformModel("rfdetr-seg-xlarge", "rf-detr-seg-xlarge.onnx", RFDETRSegTRTInference),
    PlatformModel("rfdetr-seg-2xlarge", "rf-detr-seg-xxlarge.onnx", RFDETRSegTRTInference),
]

# --- YOLOv8 (fused NMS in the graph, no CUDA graph, COCO class remapping) ----
YOLOV8_MODELS = [
    PlatformModel(f"yolov8{size}", f"yolov8{size}_nms_conf_0.01.onnx", YOLOv8TRTInference, needs_class_remapping=True)
    for size in ("n", "s", "m", "l", "x")
]

YOLOV8_SEG_MODELS = [
    PlatformModel(
        f"yolov8{size}-seg",
        f"yolov8{size}_seg_nms_conf_0.01.onnx",
        YOLOv8SegTRTInference,
        needs_class_remapping=True,
        graph_surgery=fuse_yolo_mask_postprocessing_into_onnx,
    )
    for size in ("n", "s", "m", "l", "x")
]

# --- YOLOv11 -----------------------------------------------------------------
#
# The platform key is "yolov11n" and the artifact is "yolo11n...". The spelling
# differs on purpose: the key is what a user trains, the file is what
# ultralytics exported.
YOLOV11_MODELS = [
    PlatformModel(f"yolov11{size}", f"yolo11{size}_nms_conf_0.01.onnx", YOLOv11TRTInference, needs_class_remapping=True)
    for size in ("n", "s", "m", "l", "x")
]

YOLOV11_SEG_MODELS = [
    PlatformModel(
        f"yolov11{size}-seg",
        f"yolo11{size}_seg_nms_conf_0.01.onnx",
        YOLOv11SegTRTInference,
        needs_class_remapping=True,
        graph_surgery=fuse_yolo_mask_postprocessing_into_onnx,
    )
    for size in ("n", "s", "m", "l", "x")
]

# --- YOLO26 (CUDA-graph replay, COCO class remapping) ------------------------
YOLO26_MODELS = [
    PlatformModel(f"yolo26{size}", f"yolo26{size}.onnx", YOLO26TRTInference, needs_class_remapping=True)
    for size in ("n", "s", "m", "l", "x")
]

# --- YOLO26 segmentation (CUDA-graph replay, COCO class remapping) -----------
#
# The same mask graph surgery as the yolov8/yolov11 seg rows: the published seg
# numbers time the mask matmul, sigmoid, and box crop inside the engine, so the
# yolo26 seg rows fold them in too. The only family difference is CUDA-graph
# replay (NMS-free convention).
#
# Note that this script has no artifact preflight: the HEAD check lives in
# benchmark_platform_nas_models.missing_artifacts, and a run against a missing
# artifact downloads a 404 XML page and fails inside the TensorRT parser.
YOLO26_SEG_MODELS = [
    PlatformModel(
        f"yolo26{size}-seg",
        f"yolo26{size}-seg.onnx",
        YOLO26SegTRTInference,
        needs_class_remapping=True,
        graph_surgery=fuse_yolo26_mask_postprocessing_into_onnx,
    )
    for size in ("n", "s", "m", "l", "x")
]

PLATFORM_MODELS: list[PlatformModel] = [
    *RFDETR_MODELS,
    *RFDETR_SEG_MODELS,
    *YOLOV8_MODELS,
    *YOLOV8_SEG_MODELS,
    *YOLOV11_MODELS,
    *YOLOV11_SEG_MODELS,
    *YOLO26_MODELS,
    *YOLO26_SEG_MODELS,
]


def main(
    image_dir: str,
    annotations_file_path: str,
    buffer_time: float = BUFFER_TIME_S,
    output_file_name: str = "platform_models_results.json",
    hardware: str | None = None,
    max_images: int | None = 100,
):
    hardware = hardware or os.environ.get("SAB_HARDWARE") or DEFAULT_HARDWARE
    environment = collect_environment()

    requests = [model.to_request(buffer_time, max_images) for model in PLATFORM_MODELS]
    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump({"environment": environment, "results": nest_latency_by_hardware(results, hardware)}, f, indent=2)

    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
