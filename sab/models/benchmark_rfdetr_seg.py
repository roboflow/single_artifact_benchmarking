import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import requests
import os
import json
import fire


from sab.onnx_inference import ONNXInference
from sab.trt_inference import TRTInference
from sab.models.utils import cxcywh_to_xyxy, ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results


def preprocess_image(image: torch.Tensor, image_input_shape: tuple[int, int]) -> tuple[torch.Tensor, dict]:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    orig_target_sizes = torch.tensor([image.shape[2], image.shape[3]], device=image.device)
    
    means = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)

    image = TF.normalize(image, means, stds)
    image = TF.resize(image, image_input_shape[2:])

    return image, {
        "orig_target_sizes": orig_target_sizes,
    }


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["dets"]
    out_logits = outputs["labels"]
    masks = outputs["masks"]
    scores = out_logits.sigmoid()

    flat_scores = scores.view(scores.shape[0], -1)
    num_select = min(300, flat_scores.shape[1])

    topk_values, topk_indexes = torch.topk(flat_scores, num_select, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    masks = torch.gather(masks, 1, topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1,1,masks.shape[2],masks.shape[3]))

    bboxes = cxcywh_to_xyxy(bboxes)

    masks = F.interpolate(masks, size=metadata["orig_target_sizes"].tolist(), mode="bilinear", align_corners=False)
    masks = masks > 0

    return bboxes.contiguous(), labels.contiguous(), scores.contiguous(), masks.contiguous()


class RFDETRSegONNXInference(ONNXInference):
    def __init__(self, model_path: str, image_input_name: str|None=None):
        super().__init__(model_path, image_input_name, prediction_type="segm")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


class RFDETRSegTRTInference(TRTInference):
    def __init__(self, model_path: str, image_input_name: str|None=None):
        super().__init__(model_path, image_input_name, use_cuda_graph=True, prediction_type="segm")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)

class NoCudaGraphRFDETRSegTRTInference(RFDETRSegTRTInference):
    def __init__(self, model_path: str, image_input_name: str|None=None):
        super().__init__(model_path, image_input_name, use_cuda_graph=False, prediction_type="segm")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


def main(image_dir: str, annotations_file_path: str, buffer_time: float = 0.0, output_file_name: str = "rfdetr_seg_results.json"):
    requests = [
    ]

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)
    
    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
