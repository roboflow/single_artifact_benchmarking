import torch
import torchvision.transforms.functional as TF
from supervision.utils.file import read_json_file
from supervision.dataset.formats.coco import coco_categories_to_classes, build_coco_class_index_mapping
import os

from onnx_inference import ONNXInference
from trt_inference import TRTInference, build_engine
from evaluation import evaluate


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


def preprocess_image(image: torch.Tensor, image_input_shape: tuple[int, int]) -> torch.Tensor:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = TF.resize(image, image_input_shape[2:])
    return image


def postprocess_output(outputs: dict[str, torch.Tensor], image_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["output0"][0, :, :4]
    scores = outputs["output0"][0, :, 4]
    labels = outputs["output0"][0, :, 5]

    bboxes /= torch.tensor([image_shape[2], image_shape[3], image_shape[2], image_shape[3]], device=bboxes.device)

    return bboxes, labels, scores


class YOLOv11ONNXInference(ONNXInference):
    def preprocess(self, input_image: torch.Tensor) -> torch.Tensor:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, self.image_input_shape)
    

class YOLOv11TRTInference(TRTInference):
    def preprocess(self, input_image: torch.Tensor) -> torch.Tensor:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, self.image_input_shape)
    

if __name__ == "__main__":
    model_path = "yolo11n.onnx"
    engine_path = "yolo11n.engine"
    coco_dir = "/home/isaac/cocodir/val2017"
    coco_annotations_file_path = "/home/isaac/cocodir/annotations/instances_val2017.json"

    class_mapping = get_coco_class_index_mapping(coco_annotations_file_path)
    inv_class_mapping = {v: k for k, v in class_mapping.items()}

    # onnx_inference = YOLOv11ONNXInference(model_path)
    if not os.path.exists(engine_path):
        build_engine(model_path, engine_path)

    inference = YOLOv11TRTInference(engine_path)

    evaluate(inference, coco_dir, coco_annotations_file_path, inv_class_mapping)

    inference.print_latency_stats()