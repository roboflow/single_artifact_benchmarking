import json
import os
import tempfile


def load_results(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _json_default(value):
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_results(path: str, rows: list[dict]):
    """Write the rows to `path` atomically: a crash keeps the previous file whole."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rows, f, default=_json_default)
        os.replace(temp_path, path)
    except BaseException:
        os.unlink(temp_path)
        raise


def result_key(row: dict) -> tuple:
    """The same tuple as ArtifactBenchmarkRequest.key() for the request that made the row."""
    request = row["artifact_request"]
    return (
        request["artifact_path"],
        request["runtime"],
        request["device"],
        request["precision"],
        request["max_images"],
    )


def _pct(stats, idx):
    try:
        v = stats[idx]
        return None if v is None else v * 100.0
    except Exception:
        return None


def _fmt(x, width=6, prec=1):
    return f"{x:{width}.{prec}f}" if isinstance(x, (int, float)) else f"{'—':>{width}}"


def _throttled_label(throttled) -> str:
    return "?" if throttled is None else ("yes" if throttled else "no")


def pretty_print_results(results: list[dict]):
    """
    Prints summary runtime info plus COCO AP/AR breakdown.

    Assumes result['accuracy_stats'] is pycocotools COCOeval.stats with this order:
      0: AP@[.50:.95] (area=all,   maxDets=max_dets)
      1: AP@.50       (area=all,   maxDets=max_dets)
      2: AP@.75       (area=all,   maxDets=max_dets)
      3: AP@[.50:.95] (area=small, maxDets=max_dets)
      4: AP@[.50:.95] (area=medium,maxDets=max_dets)
      5: AP@[.50:.95] (area=large, maxDets=max_dets)
      6: AR@[.50:.95] (area=all,   maxDets=1)
      7: AR@[.50:.95] (area=all,   maxDets=10)
      8: AR@[.50:.95] (area=all,   maxDets=max_dets)
      9: AR@[.50:.95] (area=small, maxDets=max_dets)
     10: AR@[.50:.95] (area=medium,maxDets=max_dets)
     11: AR@[.50:.95] (area=large, maxDets=max_dets)
    """
    supported = [r for r in results if not r["artifact_request"]["unsupported"]]
    partial_image_counts = set()

    header = (
        f"{'Artifact':30} {'Runtime':12} {'Device':6} {'Precision':9} "
        f"{'mAP50':>6} {'mAP50-95':>9} {'AP75':>6} {'Latency':>9} {'Throttled':>9}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        request = result["artifact_request"]
        identity = f"{request['artifact_path']:30} {request['runtime']:12} {request['device']:6} {request['precision']:9}"
        if request["unsupported"]:
            print(f"{identity} unsupported: {request['unsupported']}")
            continue

        stats = result["accuracy_stats"]
        latency = (result.get("latency_stats") or {}).get("median")
        map50_95 = _fmt(_pct(stats, 0), 9)
        if request["max_images"] is not None:
            partial_image_counts.add(request["max_images"])
            map50_95 = f"{map50_95.strip() + '*':>9}"
        print(
            f"{identity} {_fmt(_pct(stats, 1))} {map50_95} {_fmt(_pct(stats, 2))} "
            f"{_fmt(latency, 9, 2)} {_throttled_label(result.get('throttled')):>9}"
        )

    for count in sorted(partial_image_counts):
        print(f"* evaluated on the first {count} images")

    print("\nAP breakdown (COCO):")
    ap_hdr = f"{'Artifact':30} {'AP_s':>6} {'AP_m':>6} {'AP_l':>6}"
    print(ap_hdr)
    print("-" * len(ap_hdr))
    for result in supported:
        name = result["artifact_request"]["artifact_path"]
        stats = result["accuracy_stats"]
        print(f"{name:30} {_fmt(_pct(stats, 3))} {_fmt(_pct(stats, 4))} {_fmt(_pct(stats, 5))}")

    print("\nAR breakdown (COCO):")
    ar_hdr = f"{'Artifact':30} {'AR@1':>6} {'AR@10':>6} {'AR@max_dets':>13} {'AR_s':>6} {'AR_m':>6} {'AR_l':>6}"
    print(ar_hdr)
    print("-" * len(ar_hdr))
    for result in supported:
        name = result["artifact_request"]["artifact_path"]
        stats = result["accuracy_stats"]
        print(
            f"{name:30} {_fmt(_pct(stats, 6))} {_fmt(_pct(stats, 7))} {_fmt(_pct(stats, 8), 13)} "
            f"{_fmt(_pct(stats, 9))} {_fmt(_pct(stats, 10))} {_fmt(_pct(stats, 11))}"
        )
