from supervision.utils.file import read_json_file
from supervision.dataset.formats.coco import coco_categories_to_classes, build_coco_class_index_mapping


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