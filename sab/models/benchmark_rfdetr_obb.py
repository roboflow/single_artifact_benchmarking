"""SAB benchmark adapter for RF-DETR oriented-bounding-box (OBB) ONNX models, evaluated on YOLO-OBB tiles (e.g. DOTA).

Export from rf-detr (``RFDETRNanoOBB``): ``RFDETR.from_checkpoint(ckpt).export(format="onnx")``

  - Input:  input  (1, 3, 1024, 1024), ImageNet-normalized RGB
  - Output: dets   (1, 300, 5)  [cx, cy, w, h, theta]: first four normalized to the input, theta in radians
            labels (1, 300, C)  class logits; C = num_classes + 1 (RF-DETR's extra no-object column)

Preprocessing and top-k selection follow RF-DETR (``sab.models.benchmark_rfdetr``), keeping the angle. RF-DETR
resizes to a square without letterboxing, so images must be square (DOTA tiles are): a stretch would distort rotated
boxes.

    python -m sab.models.benchmark_rfdetr_obb <tiles>/images/val <tiles>/labels/val --onnx_path rf-detr-nano-obb.onnx
"""

import json

import fire
import torch

from sab.models.benchmark_rfdetr import preprocess_image
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results
from sab.onnx_inference import ONNXInferenceCPU
from sab.trt_inference import TRTInference

DOTA_NUM_CLASSES = 15


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict, num_classes: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-300 (query, class) pairs by sigmoid score -> oriented boxes in original pixels, class ids, scores.

    Only the first ``num_classes`` logit columns are scored when given, so the never-positive no-object column cannot
    take top-k places.

    Raises:
        ValueError: If the original image is not square.
    """
    height, width = metadata["original_shape"]
    if height != width:
        raise ValueError(f"RF-DETR OBB needs square images (a stretch distorts rotated boxes); got {width}x{height}")
    logits = outputs["labels"]
    if num_classes is not None:
        logits = logits[..., :num_classes]
    scores = logits.sigmoid().flatten(1)
    values, indexes = torch.topk(scores, min(300, scores.shape[1]), dim=1)
    queries = indexes // logits.shape[2]
    boxes = torch.gather(outputs["dets"], 1, queries.unsqueeze(-1).expand(-1, -1, 5))[0].float()
    scale = boxes.new_tensor([width, height, width, height, 1.0])
    return boxes * scale, (indexes % logits.shape[2])[0], values[0]


def _preprocess(input_image: torch.Tensor, image_input_shape) -> tuple[torch.Tensor, dict]:
    image, metadata = preprocess_image(input_image, image_input_shape)
    metadata["original_shape"] = tuple(input_image.shape[-2:])
    return image, metadata


class RFDETROBBONNXCPUInference(ONNXInferenceCPU):
    def __init__(self, model_path: str, image_input_name: str | None = None, num_classes: int | None = None):
        self.num_classes = num_classes
        super().__init__(model_path, image_input_name, prediction_type="obb")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return _preprocess(input_image, self.image_input_shape)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata, self.num_classes)


class RFDETROBBTRTInference(TRTInference):
    def __init__(self, model_path: str, image_input_name: str | None = None, num_classes: int | None = None):
        self.num_classes = num_classes
        super().__init__(model_path, image_input_name, use_cuda_graph=True, prediction_type="obb")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return _preprocess(input_image, self.image_input_shape)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata, self.num_classes)


def main(image_dir: str, labels_dir: str, onnx_path: str = "rf-detr-nano-obb.onnx", buffer_time: float = 0.0,
         output_file_name: str = "rfdetr_obb_results.json", num_classes: int = DOTA_NUM_CLASSES,
         include_cpu: bool = False, max_images: int | None = None, max_dets: int = 300):
    """Benchmark an RF-DETR OBB export: TensorRT FP32 and FP16, plus ONNX CPU with ``include_cpu``.

    ``max_dets`` defaults to the model's 300 outputs: DOTA tiles can hold more than COCO's usual 100 objects.
    """
    kwargs = {"num_classes": num_classes}
    requests = [
        ArtifactBenchmarkRequest(onnx_path=onnx_path, inference_class=RFDETROBBTRTInference, needs_fp16=fp16, buffer_time=buffer_time,
                                 max_images=max_images, max_dets=max_dets, inference_kwargs=kwargs)
        for fp16 in (False, True)
    ]
    if include_cpu:
        requests.append(ArtifactBenchmarkRequest(onnx_path=onnx_path, inference_class=RFDETROBBONNXCPUInference, buffer_time=buffer_time,
                                                 max_images=max_images, max_dets=max_dets, inference_kwargs=kwargs))

    results = run_benchmark_on_artifacts(requests, image_dir, labels_dir)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)

    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
