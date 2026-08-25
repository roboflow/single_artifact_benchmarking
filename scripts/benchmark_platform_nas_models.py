"""Benchmark RF-DETR NAS children, TRT-FP16 at batch 1.

This is the NAS half of the platform latency table. The named half is
`scripts/benchmark_platform_models.py`, and both use one protocol, so a NAS
child and a named model sit on one latency/accuracy chart.

A NAS child belongs to one of four families. The platform registers each NAS
parent as its own model, so the family name is a platform name, not a label
this script invented (roboflow-train `train/src/rfdetr/nas.py`). The family
states the backbone and the task; the five-field tuple states the rest:

| family | task | backbone |
|---|---|---|
| `rfdetr-nas-parent` | object detection | dinov2_windowed_small |
| `rfdetr-nas-base-parent` | object detection | dinov2_windowed_base |
| `rfdetr-nas-pecoret-parent` | object detection | PE-Core-T |
| `rfdetr-nas-seg-parent` | instance segmentation | dinov2_windowed_small |

One tuple means the same architecture in every family, and two families can
hold one tuple and mean two different models. `model_id` is therefore
`<family>/<resolution>_<patch>_<windows>_<dec_layers>_<queries>`.

The children start as the tuples that reproduce a named model. A child that
times like its named twin proves the two halves measure the same thing.

Artifacts: Lee copies the NAS ONNX files into
gs://single_artifact_benchmarking. The layout is not fixed yet, so each family
carries one path template. When the files land, change the template of that
family and nothing else.

Example:
    python scripts/benchmark_platform_nas_models.py \\
        --image_dir /data \\
        --annotations_file_path /data/_annotations.coco.json \\
        --output_file_name /results/platform_nas_models.json
"""

import json
import os
import urllib.request
from dataclasses import dataclass

import fire

from sab.environment import collect_environment
from sab.models.benchmark_rfdetr import RFDETRTRTInference
from sab.models.benchmark_rfdetr_seg import RFDETRSegTRTInference
from sab.models.utils import (
    ArtifactBenchmarkRequest,
    nest_latency_by_hardware,
    pretty_print_results,
    run_benchmark_on_artifacts,
)

# Pinned methodology constant, the same one the named models use. See
# scripts/benchmark_platform_models.py.
BUFFER_TIME_S = 0.2

DEFAULT_HARDWARE = "T4"

ARTIFACT_URL_BASE = "https://storage.googleapis.com/single_artifact_benchmarking"

NasTuple = tuple[int, int, int, int, int]


def config_id(config: NasTuple) -> str:
    """The tuple as the NAS tables key it: `<res>_<patch>_<windows>_<dec>_<queries>`."""
    return "_".join(str(field_value) for field_value in config)


@dataclass(frozen=True)
class NasFamily:
    """One NAS family: where its children live, and which ones to measure."""

    family: str
    inference_class: type
    onnx_path_template: str
    """Where a child sits in gs://single_artifact_benchmarking, with `{config_id}`.

    This is the one line to change per family when the artifacts land. It is
    also the local path SAB downloads to.
    """

    configs: tuple[NasTuple, ...] = ()

    def onnx_path(self, config: NasTuple) -> str:
        return self.onnx_path_template.format(config_id=config_id(config))

    def model_id(self, config: NasTuple) -> str:
        return f"{self.family}/{config_id(config)}"

    def to_requests(self, buffer_time: float, max_images: int | None) -> list[ArtifactBenchmarkRequest]:
        return [
            ArtifactBenchmarkRequest(
                onnx_path=self.onnx_path(config),
                inference_class=self.inference_class,
                needs_fp16=True,
                buffer_time=buffer_time,
                max_images=max_images,
                model_id=self.model_id(config),
            )
            for config in self.configs
        ]


# The tuples that reproduce a named RF-DETR model, per family. They are the
# class defaults of rf-detr-internal `rfdetr_internal/config.py` on develop,
# which is the repo the platform trains from. The comment beside each tuple is
# the named model it must reproduce.
NAS_FAMILIES: list[NasFamily] = [
    NasFamily(
        family="rfdetr-nas-parent",
        inference_class=RFDETRTRTInference,
        onnx_path_template="nas/rfdetr-nas-parent/{config_id}/inference_model.onnx",
        configs=(
            (384, 16, 2, 2, 300),  # rfdetr-nano
            (512, 16, 2, 3, 300),  # rfdetr-small
            (576, 16, 2, 4, 300),  # rfdetr-medium
            (704, 16, 2, 4, 300),  # rfdetr-large
        ),
    ),
    NasFamily(
        family="rfdetr-nas-base-parent",
        inference_class=RFDETRTRTInference,
        onnx_path_template="nas/rfdetr-nas-base-parent/{config_id}/inference_model.onnx",
        configs=(
            (700, 20, 1, 5, 300),  # rfdetr-xlarge
            (880, 20, 2, 5, 300),  # rfdetr-2xlarge
        ),
    ),
    NasFamily(
        family="rfdetr-nas-pecoret-parent",
        inference_class=RFDETRTRTInference,
        onnx_path_template="nas/rfdetr-nas-pecoret-parent/{config_id}/inference_model.onnx",
        # No named model uses the PE-Core-T backbone, so this family has no
        # named-equivalent tuple to start from. Add the tuples this sweep needs.
        configs=(),
    ),
    NasFamily(
        family="rfdetr-nas-seg-parent",
        inference_class=RFDETRSegTRTInference,
        onnx_path_template="nas/rfdetr-nas-seg-parent/{config_id}/inference_model.onnx",
        configs=(
            (312, 12, 1, 4, 100),  # rfdetr-seg-nano
            (384, 12, 2, 4, 100),  # rfdetr-seg-small
            (432, 12, 2, 5, 200),  # rfdetr-seg-medium
            (504, 12, 2, 5, 200),  # rfdetr-seg-large
            (624, 12, 2, 6, 300),  # rfdetr-seg-xlarge
            (768, 12, 2, 6, 300),  # rfdetr-seg-2xlarge
        ),
    ),
]


def artifact_in_bucket(artifact: str) -> bool:
    request = urllib.request.Request(f"{ARTIFACT_URL_BASE}/{artifact}", method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=10):  # noqa: S310 -- fixed https base
            return True
    except Exception:
        return False


def missing_artifacts(requests: list[ArtifactBenchmarkRequest]) -> list[str]:
    return [
        request.onnx_path
        for request in requests
        if not os.path.exists(request.onnx_path) and not artifact_in_bucket(request.onnx_path)
    ]


def main(
    image_dir: str,
    annotations_file_path: str,
    buffer_time: float = BUFFER_TIME_S,
    output_file_name: str = "platform_nas_models_results.json",
    hardware: str | None = None,
    max_images: int | None = None,
):
    hardware = hardware or os.environ.get("SAB_HARDWARE") or DEFAULT_HARDWARE
    environment = collect_environment()

    requests = [
        request for family in NAS_FAMILIES for request in family.to_requests(buffer_time, max_images)
    ]
    if not requests:
        raise SystemExit("No NAS children are configured. Add tuples to NAS_FAMILIES.")

    # The artifact layout is still moving, so a wrong template must stop the run
    # here. A 404 downloads an XML error page and fails hours later, inside the
    # TensorRT parser.
    missing = missing_artifacts(requests)
    if missing:
        raise SystemExit(
            "These NAS artifacts are not in the bucket:\n  "
            + "\n  ".join(missing)
            + f"\nCopy them to {ARTIFACT_URL_BASE}, or correct onnx_path_template in NAS_FAMILIES."
        )

    for request in requests:
        os.makedirs(os.path.dirname(request.onnx_path), exist_ok=True)

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)
    rows = with_nas_config(nest_latency_by_hardware(results, hardware))

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump({"environment": environment, "results": rows}, f, indent=2)

    pretty_print_results(results)


def with_nas_config(rows: list[dict]) -> list[dict]:
    """Spell the tuple out in each row, beside the model id.

    The NAS timing tables of rf-detr-internal read these five fields by name.
    The model id holds the same tuple, but nothing parses an id back into data.
    """
    configs = {family.model_id(config): config for family in NAS_FAMILIES for config in family.configs}
    named_fields = ("resolution", "patch_size", "num_windows", "dec_layers", "num_queries")
    return [{**row, **dict(zip(named_fields, configs[row["model_id"]]))} for row in rows]


if __name__ == "__main__":
    fire.Fire(main)
