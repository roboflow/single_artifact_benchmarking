import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import io
import requests
import os

from onnx_inference import ONNXInference
from trt_inference import TRTInference, build_engine
from evaluation import evaluate
from clock_watch import ThrottleMonitor
from models.utils import cxcywh_to_xyxy


def preprocess_image(image: torch.Tensor, image_input_shape: tuple[int, int]) -> tuple[torch.Tensor, dict]:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    means = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)

    image = TF.normalize(image, means, stds)
    image = TF.resize(image, image_input_shape[2:])
    return image, {}


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["dets"]
    out_logits = outputs["labels"]
    scores = out_logits.sigmoid()

    topk_values, topk_indexes = torch.topk(scores.view(scores.shape[0], -1), 300, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

    bboxes = cxcywh_to_xyxy(bboxes)

    return bboxes.contiguous(), labels.contiguous(), scores.contiguous()


class RFDETRONNXInference(ONNXInference):
    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


class RFDETRTRTInference(TRTInference):
    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


if __name__ == "__main__":
    # model_path = "rf-detr-nano.onnx"
    model_path = "/home/isaac/LW-DETR/output/lwdetr_dinov2_small_flex_coco/inference_model.onnx"
    # engine_path = "rf-detr-nano.engine"
    # engine_path = "rf-detr-nano_fp16.engine"
    # engine_path = "inference_model_clamped_tgt.engine"
    engine_path = "inference_model_clamped_tgt_fp16.engine"
    coco_dir = "/home/isaac/cocodir/val2017"
    coco_annotations_file_path = "/home/isaac/cocodir/annotations/instances_val2017.json"
    buffer_time = 0.0

    # inference = RFDETRONNXInference(model_path)
    if not os.path.exists(engine_path):
        with ThrottleMonitor() as throttle_monitor:
            build_engine(model_path, engine_path, use_fp16=True)
            if throttle_monitor.did_throttle():
                print("GPU throttled during engine build. This is expected and is a limitation of TensorRT.")

    inference = RFDETRTRTInference(engine_path)

    with ThrottleMonitor() as throttle_monitor:
        evaluate(inference, coco_dir, coco_annotations_file_path, buffer_time=buffer_time)
        if throttle_monitor.did_throttle():
            print(f"🔴  GPU throttled, latency results are unreliable. Try increasing the buffer time. Current buffer time: {buffer_time}s")

    inference.print_latency_stats()
