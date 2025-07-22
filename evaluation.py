import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm
import time


def evaluate(inference, image_dir: str, annotations_file_path: str, class_mapping: dict[int, str]|None=None, buffer_time: float=0.0, output_file_name: str="predictions.json"):
    predictions = []

    coco_annotations = COCO(annotations_file_path)

    image_ids = coco_annotations.getImgIds()

    # image_ids = image_ids[:50]

    for image_id in tqdm(image_ids):
        image_info = coco_annotations.loadImgs(image_id)[0]
        image_path = os.path.join(image_dir, image_info["file_name"])

        image = Image.open(image_path).convert("RGB")
        initial_shape = image.size
        image = TF.to_tensor(image).cuda()

        xyxy, class_id, score = inference.infer(image)

        xyxy = xyxy.squeeze(0)
        class_id = class_id.squeeze(0)
        score = score.squeeze(0)

        xywh = xyxy.clone()
        xywh[:, 2:4] -= xywh[:, 0:2]
        xywh[:, 0::2] *= initial_shape[0]
        xywh[:, 1::2] *= initial_shape[1]

        xywh = xywh.cpu().numpy()
        class_id = class_id.cpu().numpy()
        score = score.cpu().numpy()

        for this_xywh, this_class_id, this_score in zip(xywh, class_id, score):
            predictions.append({
                "image_id": image_id,
                "bbox": this_xywh.tolist(),
                "category_id": class_mapping[int(this_class_id)] if class_mapping is not None else int(this_class_id),
                "score": float(this_score)
            })
        
        time.sleep(buffer_time)

    with open(output_file_name, "w") as f:
        json.dump(predictions, f)

    coco_det = coco_annotations.loadRes(output_file_name)
    coco_eval = COCOeval(coco_annotations, coco_det, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]