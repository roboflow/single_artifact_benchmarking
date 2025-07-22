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


def cxcywh_to_xyxy(boxes):
    boxes = boxes.clone()
    boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes


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
    model_path = "rf-detr-nano.onnx"
    engine_path = "rf-detr-nano.engine"
    coco_dir = "/home/isaac/cocodir/val2017"
    coco_annotations_file_path = "/home/isaac/cocodir/annotations/instances_val2017.json"

    # onnx_inference = RFDETRONNXInference(model_path)
    if not os.path.exists(engine_path):
        build_engine(model_path, engine_path)
    trt_inference = RFDETRTRTInference(engine_path)

    evaluate(trt_inference, coco_dir, coco_annotations_file_path)

    trt_inference.print_latency_stats()
