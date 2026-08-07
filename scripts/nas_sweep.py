"""NAS candidate latency sweep, hardware-agnostic.

Port of rf-detr-internal scripts/ai1_latencies/get_ai1_latencies.py, which
produced the production AI1 NAS timing tables. Differences: works for any
hardware target (T4, AI1), writes latency_stats keyed by hardware (the shape
rfdetr_internal.engine.load_timing_results_file consumes), records the ONNX
sha256 per entry, and writes an environment sidecar (<output>.meta.json).

The results file shape is load-bearing: every entry must keep the fields
resolution/patch_size/num_windows/dec_layers/num_queries/score/latency_stats/
throttled. New fields may only be added, never renamed or nested differently.
"""

import json
import os
import shutil
import time
import urllib.request

from sab.environment import collect_environment, file_sha256

# Per-backbone ONNX source. Keys are the NAS parent names exposed by
# get_sab_latencies.py as "nas/<key>".
BACKBONE_PATHS = {
    "dinov2_windowed_small": {
        "onnx_url_base": "https://storage.googleapis.com/rfdetr/qwetgapofdgi6a5lgha5dsasd31fpoikwer-od-full",
        "output_subdir": "small_backbone",
    },
    "dinov2_windowed_base": {
        "onnx_url_base": "TBD",
        "output_subdir": "large_backbone",
    },
    "PE-Core-T": {
        "onnx_url_base": "https://storage.googleapis.com/rfdetr/npzmp1n0cmx8h0ku5q7ycyxz8wrq9gby8m0-od-pecoret-full",
        "output_subdir": "pe_core_t_backbone",
    },
}

RESOLUTION_CANDIDATES = [320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960]
PATCH_SIZE_CANDIDATES = [8, 10, 12, 16, 20, 24, 32]
NUM_WINDOWS_CANDIDATES = [1, 2, 4]
DEC_LAYERS_CANDIDATES = [0, 1, 2, 3, 4, 5, 6]
NUM_QUERIES_CANDIDATES = [50, 100, 200, 300]


def iter_candidates():
    import itertools

    for resolution, patch_size, num_windows, dec_layers, num_queries in itertools.product(
        RESOLUTION_CANDIDATES,
        sorted(PATCH_SIZE_CANDIDATES, reverse=True),
        sorted(NUM_WINDOWS_CANDIDATES, reverse=True),
        DEC_LAYERS_CANDIDATES,
        NUM_QUERIES_CANDIDATES,
    ):
        # Resolution must be a multiple of the attention block size.
        block_size = num_windows * patch_size
        resolution = (resolution // block_size) * block_size
        yield resolution, patch_size, num_windows, dec_layers, num_queries


def _config_id(resolution, patch_size, num_windows, dec_layers, num_queries) -> str:
    return f"{resolution}_{patch_size}_{num_windows}_{dec_layers}_{num_queries}"


def _download_onnx(onnx_url: str, onnx_local_path: str) -> bool:
    if os.path.exists(onnx_local_path):
        return True
    os.makedirs(os.path.dirname(onnx_local_path), exist_ok=True)
    if not onnx_url.startswith(("http://", "https://")):
        raise ValueError(f"Refusing to fetch non-HTTP(S) ONNX URL: {onnx_url}")
    print(f"Downloading ONNX from {onnx_url} ...")
    try:
        # Scheme validated above; onnx_url_base entries are trusted GCS buckets.
        urllib.request.urlretrieve(onnx_url, onnx_local_path)  # noqa: S310
    except Exception as e:
        print(f"Failed to download ONNX: {e}, skipping.")
        return False
    return True


def _write_sidecar(results_file: str, backbone: str, hardware: str, buffer_time: float, max_images: int | None):
    sidecar = {
        "environment": collect_environment(),
        "sweep_config": {
            "backbone": backbone,
            "hardware": hardware,
            "buffer_time": buffer_time,
            "max_images": max_images,
            "needs_fp16": True,
            "batch_size": 1,
        },
    }
    with open(f"{results_file}.meta.json", "w") as f:
        json.dump(sidecar, f, indent=2)


def build_candidate_entry(
    resolution: int,
    patch_size: int,
    num_windows: int,
    dec_layers: int,
    num_queries: int,
    score,
    latency_stats: dict,
    hardware: str,
    throttled: bool,
    onnx_sha256: str,
) -> dict:
    # Field names and nesting are the production NAS table contract consumed by
    # rfdetr_internal.engine.load_timing_results_file; only additive changes allowed.
    return {
        "num_queries": num_queries,
        "resolution": resolution,
        "patch_size": patch_size,
        "num_windows": num_windows,
        "dec_layers": dec_layers,
        "score": score,
        "latency_stats": {hardware: latency_stats},
        "throttled": throttled,
        "onnx_sha256": onnx_sha256,
    }


def run_sweep(
    backbone: str,
    hardware: str,
    output_dir: str,
    coco_path: str,
    buffer_time: float = 0.2,
    max_images: int | None = None,
) -> list[dict]:
    # Deferred imports: keep this module importable on machines without
    # torch/tensorrt (tests, --list).
    from sab.models.benchmark_rfdetr import RFDETRTRTInference
    from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifact

    paths = BACKBONE_PATHS[backbone]
    onnx_url_base = paths["onnx_url_base"]
    if onnx_url_base == "TBD":
        raise ValueError(f"No ONNX source configured for backbone {backbone!r}")

    backbone_dir = os.path.join(output_dir, paths["output_subdir"])
    os.makedirs(backbone_dir, exist_ok=True)
    results_file = os.path.join(backbone_dir, "nas_results.json")

    annotation_file = os.path.join(coco_path, "_annotations.coco.json")
    is_jetson = hardware == "AI1"

    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        done = {
            _config_id(r["resolution"], r["patch_size"], r["num_windows"], r["dec_layers"], r["num_queries"])
            for r in results
        }
        print(f"Loaded {len(results)} existing results from {results_file}")
    else:
        results = []
        done = set()

    _write_sidecar(results_file, backbone, hardware, buffer_time, max_images)

    candidates = list(iter_candidates())
    print(f"Total number of candidates: {len(candidates)}")
    sweep_start = time.time()

    for resolution, patch_size, num_windows, dec_layers, num_queries in candidates:
        config_id = _config_id(resolution, patch_size, num_windows, dec_layers, num_queries)
        if config_id in done:
            continue
        done.add(config_id)

        print(
            f"Resolution: {resolution}, Patch Size: {patch_size}, Num Windows: {num_windows}, "
            f"Dec Layers: {dec_layers}, Num Queries: {num_queries}"
        )
        candidate_start = time.time()

        work_dir = os.path.join(backbone_dir, config_id)
        onnx_local_path = os.path.join(work_dir, "inference_model.onnx")
        if not _download_onnx(f"{onnx_url_base}/{config_id}/inference_model.onnx", onnx_local_path):
            continue

        onnx_sha256 = file_sha256(onnx_local_path)
        request = ArtifactBenchmarkRequest(
            onnx_path=onnx_local_path,
            inference_class=RFDETRTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            max_images=max_images,
            is_jetson=is_jetson,
        )
        score, stats, throttled = run_benchmark_on_artifact(
            request, images_dir=coco_path, annotations_file_path=annotation_file
        )
        shutil.rmtree(work_dir, ignore_errors=True)

        candidate_result = build_candidate_entry(
            resolution=resolution,
            patch_size=patch_size,
            num_windows=num_windows,
            dec_layers=dec_layers,
            num_queries=num_queries,
            score=score,
            latency_stats=stats,
            hardware=hardware,
            throttled=throttled,
            onnx_sha256=onnx_sha256,
        )
        results.append(candidate_result)
        print(candidate_result)
        print(f"Time taken to analyze candidate {config_id}: {time.time() - candidate_start:.1f} seconds")

        with open(results_file, "w") as f:
            json.dump(results, f)

    print(f"Time taken to analyze {len(candidates)} candidates: {time.time() - sweep_start:.1f} seconds")
    return results
