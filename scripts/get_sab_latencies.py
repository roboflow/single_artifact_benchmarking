"""Standardized latency benchmarking entry point.

This script benchmarks NAS candidates and named models for one hardware
target in the pinned docker environment (see docker/README.md). By default,
it runs all NAS parent sweeps and all named models. Use --models to run a
subset. Each flag defaults from a SAB_* environment variable. As a result,
the same script works as a container or service entrypoint.

Examples:
    python scripts/get_sab_latencies.py --list
    python scripts/get_sab_latencies.py --hardware t4 --coco-path /data --output-dir /results
    python scripts/get_sab_latencies.py --hardware ai1 --models nas/PE-Core-T,rfdetr-nano
"""

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nas_sweep
from named_models import NAMED_MODELS

from sab.environment import collect_environment
from sab.models.utils import pretty_print_results, run_benchmark_on_artifact

HARDWARE_LABELS = {"t4": "T4", "ai1": "AI1"}
NAS_PREFIX = "nas/"
ARTIFACT_URL_BASE = "https://storage.googleapis.com/single_artifact_benchmarking"

# Pinned methodology constant. The production NAS tables and the published
# RF-DETR benchmarks use a 200 ms buffer between passes. All standard runs
# use the same value, so new numbers stay comparable with the tables.
# Entries that throttle in spite of the buffer keep "throttled": true.
BUFFER_TIME_S = 0.2


def artifact_in_bucket(artifact: str) -> bool:
    request = urllib.request.Request(f"{ARTIFACT_URL_BASE}/{artifact}", method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=10):  # noqa: S310 -- fixed https base
            return True
    except Exception:
        return False


def missing_artifacts() -> dict[str, str]:
    return {
        key: model.onnx_artifact
        for key, model in NAMED_MODELS.items()
        if not os.path.exists(model.onnx_artifact) and not artifact_in_bucket(model.onnx_artifact)
    }


def all_model_names() -> list[str]:
    nas_names = [
        f"{NAS_PREFIX}{backbone}"
        for backbone, paths in nas_sweep.BACKBONE_PATHS.items()
        if paths["onnx_url_base"] != "TBD"
    ]
    return nas_names + list(NAMED_MODELS)


def parse_args(argv=None):
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hardware",
        choices=sorted(HARDWARE_LABELS),
        default=env("SAB_HARDWARE", "").lower() or None,
        help="Hardware target. Defaults from SAB_HARDWARE, which the docker images set.",
    )
    parser.add_argument(
        "--models",
        default=env("SAB_MODELS", ""),
        help="Comma list of model names to run (see --list). Default: everything. "
        f"NAS parent sweeps are named {NAS_PREFIX}<backbone>.",
    )
    parser.add_argument("--coco-path", default=env("SAB_COCO_PATH", "/data"), help="COCO-format val dir with _annotations.coco.json")
    parser.add_argument("--output-dir", default=env("SAB_OUTPUT_DIR", "/results"))
    parser.add_argument("--max-images", type=int, default=int(env("SAB_MAX_IMAGES", "0")) or None)
    parser.add_argument("--list", action="store_true", help="Print all valid model names and exit.")
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="Print the model entries whose ONNX artifacts are not in the bucket, then exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(all_model_names()))
        sys.exit(0)
    if args.list_missing:
        for key, artifact in sorted(missing_artifacts().items()):
            print(f"{key}: {artifact}")
        sys.exit(0)
    if args.hardware is None:
        parser.error("--hardware is required (or set SAB_HARDWARE)")

    valid = set(all_model_names())
    args.selected = [m.strip() for m in args.models.split(",") if m.strip()] or all_model_names()
    unknown = [m for m in args.selected if m not in valid]
    if unknown:
        parser.error(f"unknown model name(s): {', '.join(unknown)} (see --list)")
    return args


def run_named_model(key: str, args, hardware: str) -> dict:
    model = NAMED_MODELS[key]
    request = model.to_request(
        buffer_time=BUFFER_TIME_S,
        max_images=args.max_images,
        is_jetson=hardware == "AI1",
    )
    accuracy_stats, latency_stats, throttled = run_benchmark_on_artifact(
        request,
        images_dir=args.coco_path,
        annotations_file_path=os.path.join(args.coco_path, "_annotations.coco.json"),
    )
    return {
        "model": key,
        "family": model.family,
        "artifact_request": request.dump(),
        "accuracy_stats": accuracy_stats,
        "latency_stats": {hardware: latency_stats},
        "throttled": throttled,
    }


def main(argv=None):
    args = parse_args(argv)
    hardware = HARDWARE_LABELS[args.hardware]
    os.makedirs(args.output_dir, exist_ok=True)

    nas_selected = [m.removeprefix(NAS_PREFIX) for m in args.selected if m.startswith(NAS_PREFIX)]
    named_selected = [m for m in args.selected if not m.startswith(NAS_PREFIX)]
    print(f"Hardware: {hardware}; NAS sweeps: {nas_selected or 'none'}; named models: {named_selected or 'none'}")

    for backbone in nas_selected:
        nas_sweep.run_sweep(
            backbone=backbone,
            hardware=hardware,
            output_dir=os.path.join(args.output_dir, "nas"),
            coco_path=args.coco_path,
            buffer_time=BUFFER_TIME_S,
            max_images=args.max_images,
        )

    if named_selected:
        named_dir = os.path.join(args.output_dir, "named")
        os.makedirs(named_dir, exist_ok=True)
        environment = collect_environment()
        results = []
        skipped = []
        for key in named_selected:
            model = NAMED_MODELS[key]
            if not os.path.exists(model.onnx_artifact) and not artifact_in_bucket(model.onnx_artifact):
                print(f"Skipping {key}: {model.onnx_artifact} is not in the bucket")
                skipped.append(key)
                continue
            output_file = os.path.join(named_dir, f"{key}.json")
            if os.path.exists(output_file):
                print(f"Skipping {key}: {output_file} exists")
                with open(output_file) as f:
                    results.append(json.load(f)["results"][0])
                continue
            result = run_named_model(key, args, hardware)
            results.append(result)
            with open(output_file, "w") as f:
                json.dump({"environment": environment, "results": [result]}, f, indent=2)

        combined_file = os.path.join(named_dir, "combined.json")
        with open(combined_file, "w") as f:
            json.dump({"environment": environment, "results": results, "skipped_missing_artifact": skipped}, f, indent=2)
        print(f"Wrote {combined_file}")
        if skipped:
            print(f"Skipped {len(skipped)} entries with missing artifacts: {', '.join(skipped)}")
        pretty_print_results(results)


if __name__ == "__main__":
    main()
