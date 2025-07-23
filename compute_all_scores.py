import os
import json
import requests
from tqdm import tqdm

from models.benchmark_yolov11 import YOLOv11TRTInference, YOLOv11ONNXInference
from models.benchmark_rfdetr import RFDETRTRTInference, RFDETRONNXInference
from models.benchmark_dfine import DFINETRTInference, DFINEONNXInference
from models.benchmark_lwdetr import LWDETRONNXInference, LWDETRTRTInference
from models.utils import get_coco_class_index_mapping

from clock_watch import ThrottleMonitor
from onnx_inference import ONNXInference
from trt_inference import TRTInference, build_engine
from evaluation import evaluate


class ArtifactBenchmarkRequest:
    def __init__(self,
            onnx_path: str,
            inference_class: type[ONNXInference|TRTInference],
            needs_class_remapping: bool = False,
            needs_fp16: bool = False,
            buffer_time: float = 0.0,
        ):
        self.onnx_path = onnx_path
        self.inference_class = inference_class
        self.needs_class_remapping = needs_class_remapping
        self.needs_fp16 = needs_fp16
        self.buffer_time = buffer_time

    def dump(self):
        return {
            "onnx_path": self.onnx_path,
            "inference_class": self.inference_class.__name__,
            "needs_fp16": self.needs_fp16,
            "buffer_time": self.buffer_time,
        }


def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers['content-length'])
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


artifact_requests = [
    ArtifactBenchmarkRequest(
        onnx_path="yolo11n_nms_conf_0.001.onnx",
        inference_class=YOLOv11ONNXInference,
        needs_class_remapping=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="yolo11n_nms_conf_0.001.onnx",
        inference_class=YOLOv11TRTInference,
        needs_class_remapping=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="yolo11n_nms_conf_0.001.onnx",
        inference_class=YOLOv11TRTInference,
        needs_class_remapping=True,
        needs_fp16=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="yolo11n_nms_conf_0.01.onnx",
        inference_class=YOLOv11ONNXInference,
        needs_class_remapping=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="yolo11n_nms_conf_0.01.onnx",
        inference_class=YOLOv11TRTInference,
        needs_class_remapping=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="yolo11n_nms_conf_0.01.onnx",
        inference_class=YOLOv11TRTInference,
        needs_class_remapping=True,
        needs_fp16=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="rf-detr-nano.onnx",
        inference_class=RFDETRONNXInference,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="rf-detr-nano.onnx",
        inference_class=RFDETRTRTInference,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="rf-detr-nano.onnx",
        inference_class=RFDETRTRTInference,
        needs_fp16=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="dfine_n_coco.onnx",
        inference_class=DFINEONNXInference,
        needs_class_remapping=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="dfine_n_coco.onnx",
        inference_class=DFINETRTInference,
        needs_class_remapping=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="dfine_n_coco.onnx",
        inference_class=DFINETRTInference,
        needs_class_remapping=True,
        needs_fp16=True,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="lw-detr-tiny.onnx",
        inference_class=LWDETRONNXInference,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="lw-detr-tiny.onnx",
        inference_class=LWDETRTRTInference,
    ),
    ArtifactBenchmarkRequest(
        onnx_path="lw-detr-tiny.onnx",
        inference_class=LWDETRTRTInference,
        needs_fp16=True,
    ),
]


def run_benchmark_on_artifact(artifact_request: ArtifactBenchmarkRequest, images_dir: str, annotations_file_path: str) -> tuple[dict, dict, bool]:
    if not os.path.exists(artifact_request.onnx_path):
        print(f"Downloading {artifact_request.onnx_path}...")
        download_file(f"https://storage.googleapis.com/single_artifact_benchmarking/{artifact_request.onnx_path}", artifact_request.onnx_path)

    if artifact_request.needs_class_remapping:
        class_mapping = get_coco_class_index_mapping(annotations_file_path)
        inv_class_mapping = {v: k for k, v in class_mapping.items()}
    else:
        inv_class_mapping = None

    if issubclass(artifact_request.inference_class, TRTInference):
        if not artifact_request.needs_fp16:
            engine_path = artifact_request.onnx_path.replace(".onnx", ".engine")
        else:
            engine_path = artifact_request.onnx_path.replace(".onnx", ".fp16.engine")
    
        if not os.path.exists(engine_path):
            print(f"Building engine for {artifact_request.onnx_path} and saving to {engine_path}...")
            with ThrottleMonitor() as throttle_monitor:
                build_engine(artifact_request.onnx_path, engine_path, use_fp16=artifact_request.needs_fp16)
                if throttle_monitor.did_throttle():
                    print("GPU throttled during engine build. This is expected and is a limitation of TensorRT.")
        else:
            print(f"Engine for {artifact_request.onnx_path} already exists at {engine_path}")

        inference = artifact_request.inference_class(engine_path)
    else:
        if artifact_request.needs_fp16:
            raise ValueError("FP16 is not supported for ONNX inference")
        
        inference = artifact_request.inference_class(artifact_request.onnx_path)
    
    throttled = False
    with ThrottleMonitor() as throttle_monitor:
        accuracy_stats = evaluate(inference, images_dir, annotations_file_path, inv_class_mapping, buffer_time=artifact_request.buffer_time)
        if throttle_monitor.did_throttle():
            throttled = True
            print(f"🔴  GPU throttled, latency results are unreliable. Try increasing the buffer time. Current buffer time: {artifact_request.buffer_time}s")
        else:
            print("GPU did not throttle during evaluation. Latency numbers should be reliable.")
    
    latency_stats = inference.profiler.get_stats()
    
    return accuracy_stats, latency_stats, throttled


if __name__ == "__main__":
    images_dir = "/home/isaac/cocodir/val2017"
    annotations_file_path = "/home/isaac/cocodir/annotations/instances_val2017.json"

    results = []

    for artifact_request in artifact_requests:
        accuracy_stats, latency_stats, throttled = run_benchmark_on_artifact(artifact_request, images_dir, annotations_file_path)
        result = {
            "artifact_request": artifact_request.dump(),
            "accuracy_stats": accuracy_stats,
            "latency_stats": latency_stats,
            "throttled": throttled,
        }
        print(result)
        results.append(result)

    with open("results.json", "w") as f:
        json.dump(results, f)
    
    print("\n=== Benchmark Results ===")
    print(f"{'Model':30} {'Runtime':8} {'FP16':5} {'mAP50':>6} {'mAP50-95':>9} {'Latency':>7} {'Throttled':>9}")
    print("-" * 80)
    for request, result in zip(artifact_requests, results):
        model = request.onnx_path
        runtime = "TRT" if issubclass(request.inference_class, TRTInference) else "ONNX"
        fp16 = request.needs_fp16
        map50 = result['accuracy_stats'][1] * 100
        map50_95 = result['accuracy_stats'][0] * 100
        latency = result['latency_stats']['median']
        throttled = result['throttled']
        
        print(f"{model:30} {runtime:8} {'yes' if fp16 else 'no':5} "
                f"{map50:6.1f} {map50_95:9.1f} {latency:7.2f} {'yes' if throttled else 'no':>9}")