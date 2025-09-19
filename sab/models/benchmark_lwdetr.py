import torch
import torchvision.transforms.functional as TF
import json
import fire

from sab.onnx_inference import ONNXInference
from sab.trt_inference import TRTInference
from sab.models.utils import cxcywh_to_xyxy, ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results


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

    flat_scores = scores.view(scores.shape[0], -1)
    num_select = min(300, flat_scores.shape[1])

    topk_values, topk_indexes = torch.topk(flat_scores, num_select, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

    bboxes = cxcywh_to_xyxy(bboxes)

    return bboxes.contiguous(), labels.contiguous(), scores.contiguous()


class LWDETRONNXInference(ONNXInference):
    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


class LWDETRTRTInference(TRTInference):
    def __init__(self, model_path: str, image_input_name: str|None=None):
        super().__init__(model_path, image_input_name, use_cuda_graph=True)

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


def main(image_dir: str, annotations_file_path: str, buffer_time: float = 0.0, output_file_name: str = "lwdetr_results.json"):
    requests = [
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-tiny.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-tiny.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-small.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-small.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-medium.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-medium.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-large.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-large.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-xlarge.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="lw-detr-xlarge.onnx",
            inference_class=LWDETRTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
        ),
    ]

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)
    
    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
