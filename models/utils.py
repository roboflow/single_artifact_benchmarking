import os
import requests
from tqdm import tqdm

from supervision.utils.file import read_json_file
from supervision.dataset.formats.coco import coco_categories_to_classes, build_coco_class_index_mapping

from clock_watch import ThrottleMonitor
from onnx_inference import ONNXInference
from trt_inference import TRTInference, build_engine
from evaluation import evaluate


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


def get_coco_class_index_mapping(annotations_path: str):
    coco_data = read_json_file(annotations_path)
    classes = coco_categories_to_classes(coco_categories=coco_data["categories"])
    class_mapping = build_coco_class_index_mapping(
        coco_categories=coco_data["categories"], target_classes=classes
    )
    return class_mapping


def cxcywh_to_xyxy(boxes):
    boxes = boxes.clone()
    boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes


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
            "is_trt": issubclass(self.inference_class, TRTInference),
            "needs_fp16": self.needs_fp16,
            "buffer_time": self.buffer_time,
        }


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


def run_benchmark_on_artifacts(artifact_requests: list[ArtifactBenchmarkRequest], images_dir: str, annotations_file_path: str) -> list[tuple[dict, dict, bool]]:
    results = []
    for artifact_request in artifact_requests:
        accuracy_stats, latency_stats, throttled = run_benchmark_on_artifact(artifact_request, images_dir, annotations_file_path)
        result = {
            "artifact_request": artifact_request.dump(),
            "accuracy_stats": accuracy_stats,
            "latency_stats": latency_stats,
            "throttled": throttled,
        }
        results.append(result)
    return results


def pretty_print_results(results: list[dict]):
    print(f"{'Model':30} {'Runtime':8} {'FP16':5} {'mAP50':>6} {'mAP50-95':>9} {'Latency':>7} {'Throttled':>9}")
    print("-" * 80)
    for result in results:
        model = result['artifact_request']['onnx_path']
        runtime = "TRT" if result['artifact_request']['is_trt'] else "ONNX"
        fp16 = result['artifact_request']['needs_fp16']
        map50 = result['accuracy_stats'][1] * 100
        map50_95 = result['accuracy_stats'][0] * 100
        latency = result['latency_stats']['median']
        throttled = result['throttled']
        
        print(f"{model:30} {runtime:8} {'yes' if fp16 else 'no':5} "
                f"{map50:6.1f} {map50_95:9.1f} {latency:7.2f} {'yes' if throttled else 'no':>9}")