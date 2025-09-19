from rfdetr import RFDETRMedium, RFDETRSmall, RFDETRNano
import argparse
from models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifact
from models.benchmark_rfdetr import RFDETRTRTInference
import os
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

gcp_save_dir = os.path.join("gs://rf-detr-rf100-vl/medium-fixed-fp16", os.path.basename(args.dataset_dir))
model = RFDETRMedium(pretrain_weights=os.path.join(args.model_dir, "checkpoint_best_total.pth"))

model.export(output_dir = args.output_dir)

del model

onnx_file = os.path.join(args.output_dir, "weights.onnx")
buffer_time = 0.2 #?
rfdetr_artifacts = ArtifactBenchmarkRequest(
    onnx_path=onnx_file,
    inference_class=RFDETRTRTInference,
    needs_fp16=True,
    buffer_time=buffer_time,
)
subprocess.run(["gsutil", "cp", onnx_file, gcp_save_dir])


image_dir = os.path.join(args.dataset_dir, "test")
annotations_file_path = os.path.join(args.dataset_dir, "/test/_annotations.coco.json")
results = run_benchmark_on_artifact(rfdetr_artifacts, image_dir, annotations_file_path)
output_file_path = os.path.join(args.output_dir, "latencies_and_accs.json")
print(f"Saving results to {output_file_path}")
with open(output_file_path, "w") as f:
    json.dump(results, f)

subprocess.run(["gsutil", "cp", output_file_path, gcp_save_dir])