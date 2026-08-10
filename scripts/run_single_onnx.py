"""Debug helper that benchmarks one local ONNX file with the standard protocol.

This is a port of rf-detr-internal scripts/ai1_latencies/run_single_onnx_ai1.py.
It works for each hardware target.
"""

import argparse
import os
import time

from sab.environment import collect_environment
from sab.models.benchmark_rfdetr import RFDETRTRTInference
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", help="Path to the ONNX file to benchmark")
    parser.add_argument("--hardware", choices=["t4", "ai1"], default=os.environ.get("SAB_HARDWARE", "").lower() or "t4")
    parser.add_argument("--coco-path", default=os.environ.get("SAB_COCO_PATH", "/data"))
    parser.add_argument("--buffer-time", type=float, default=0.2)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    request = ArtifactBenchmarkRequest(
        onnx_path=args.onnx_path,
        inference_class=RFDETRTRTInference,
        needs_fp16=True,
        buffer_time=args.buffer_time,
        max_images=args.max_images,
        is_jetson=args.hardware == "ai1",
    )

    t0 = time.time()
    score, latency_stats, throttled = run_benchmark_on_artifact(
        request,
        images_dir=args.coco_path,
        annotations_file_path=os.path.join(args.coco_path, "_annotations.coco.json"),
    )
    print(f"Benchmark run time: {time.time() - t0:.1f} sec")
    print(f"\nScore: {score}")
    print(f"Latency stats: {latency_stats}")
    print(f"Throttled: {throttled}")
    print(f"Environment: {collect_environment()}")


if __name__ == "__main__":
    main()
