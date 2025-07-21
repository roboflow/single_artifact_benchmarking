import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import io
import requests

from onnx_inference import ONNXInference
from evaluation import evaluate


def cxcywh_to_xyxy(boxes):
    boxes = boxes.clone()
    boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes


class RFDETRONNXInference(ONNXInference):
    def __init__(self, model_path: str, image_input_name: str|None=None):
        super().__init__(model_path, image_input_name)

        self.means = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.stds = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def preprocess(self, input_image: torch.Tensor) -> torch.Tensor:
        if len(input_image.shape) == 3:
            input_image = input_image.unsqueeze(0)

        input_image = TF.normalize(input_image, self.means, self.stds)
        input_image = TF.resize(input_image, self.image_input_shape[2:])

        return input_image
    
    def postprocess(self, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


if __name__ == "__main__":
    model_path = "rf-detr-nano.onnx"
    coco_dir = "/home/isaac/cocodir/val2017"
    coco_annotations_file_path = "/home/isaac/cocodir/annotations/instances_val2017.json"

    inference = RFDETRONNXInference(model_path)

    evaluate(inference, coco_dir, coco_annotations_file_path)

    inference.print_latency_stats()