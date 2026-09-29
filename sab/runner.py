import dataclasses
from typing import Iterable

from sab.artifacts import ensure_artifact
from sab.evaluation import evaluate
from sab.host import host_info
from sab.models.utils import get_coco_class_index_mapping
from sab.monitors import select_monitor
from sab.processors import Pipeline
from sab.request import ArtifactBenchmarkRequest
from sab.results import load_results, result_key, save_results
from sab.runtimes.base import runtime_class


def parse_filter(value: str | Iterable[str] | None) -> set[str] | None:
    """Turn a --runtimes or --devices value into a set. Fire gives a tuple for `a,b`."""
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else value
    names = {str(item).strip() for item in items if str(item).strip()}
    return names or None


def run_benchmark_on_artifact(
    request: ArtifactBenchmarkRequest, images_dir: str, annotations_file_path: str
) -> dict:
    artifact_path = ensure_artifact(request.artifact_path)
    if request.graph_surgery_func:
        artifact_path = request.graph_surgery_func(artifact_path)

    if request.needs_class_remapping:
        class_mapping = get_coco_class_index_mapping(annotations_file_path)
        inv_class_mapping = {v: k for k, v in class_mapping.items()}
    else:
        inv_class_mapping = None

    runtime = request.runtime(artifact_path, request.device, request.precision)
    processor = request.processor(runtime.input_spec, normalize=not request.normalized_in_graph)
    pipeline = Pipeline(runtime, processor, request.output_names)

    monitor = select_monitor(runtime_class(request.runtime), request.device)
    with monitor:
        accuracy_stats = evaluate(
            pipeline,
            images_dir,
            annotations_file_path,
            inv_class_mapping,
            buffer_time=request.buffer_time,
            max_images=request.max_images,
            max_dets=request.max_dets,
        )
    # After the with block the monitor has stopped, so the verdict is final.
    throttled = monitor.did_throttle()

    if throttled:
        print(f"🔴  Throttled during evaluation. Latency results are unreliable. Try increasing the buffer time. Current buffer time: {request.buffer_time}s")
    elif throttled is None:
        print("Throttle state unknown: this host gives no throttle signal for this device.")
    else:
        print("No throttling during evaluation.")

    return _row(request, accuracy_stats, pipeline.profiler.get_stats(), throttled)


def _row(request: ArtifactBenchmarkRequest, accuracy_stats, latency_stats, throttled) -> dict:
    return {
        "artifact_request": request.dump(),
        "host": host_info(runtime_class(request.runtime)),
        "accuracy_stats": accuracy_stats,
        "latency_stats": latency_stats,
        "throttled": throttled,
    }


def run_benchmark_on_artifacts(
    requests: list[ArtifactBenchmarkRequest],
    images_dir: str,
    annotations_file_path: str,
    *,
    output_file: str,
    runtimes: str | Iterable[str] | None = None,
    devices: str | Iterable[str] | None = None,
    max_images: int | None = None,
    rerun: bool = False,
) -> list[dict]:
    """Run each request and rewrite `output_file` after every row.

    Rows already in the file stay, and a row with the same key is not run again unless `rerun` is set.
    Returns the rows for these requests, in order.
    """
    runtime_filter = parse_filter(runtimes)
    device_filter = parse_filter(devices)

    stored_rows = load_results(output_file)
    stored_index = {result_key(row): i for i, row in enumerate(stored_rows)}
    rows = []

    def store(row: dict):
        key = result_key(row)
        if key in stored_index:
            stored_rows[stored_index[key]] = row
        else:
            stored_index[key] = len(stored_rows)
            stored_rows.append(row)
        save_results(output_file, stored_rows)

    for request in requests:
        if runtime_filter is not None and request.runtime_name not in runtime_filter:
            continue
        if device_filter is not None and request.device not in device_filter:
            continue
        if max_images is not None:
            request = dataclasses.replace(request, max_images=max_images)

        if request.unsupported:
            row = _row(request, None, None, None)
            store(row)
            rows.append(row)
            continue

        if not runtime_class(request.runtime).is_available(request.device):
            print(f"Skipping {request.artifact_path}: {request.runtime_name} is not available on {request.device} on this host.")
            continue

        key = request.key()
        if key in stored_index and not rerun:
            print(f"Keeping the saved row for {request.artifact_path} ({request.runtime_name}, {request.device}, {request.precision}). Use rerun to run it again.")
            rows.append(stored_rows[stored_index[key]])
            continue

        row = run_benchmark_on_artifact(request, images_dir, annotations_file_path)
        print(row)
        store(row)
        rows.append(row)

    return rows
