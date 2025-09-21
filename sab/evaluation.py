import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
import faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
import json
from tqdm import tqdm
import time
import random


def evaluate(inference, image_dir: str, annotations_file_path: str, class_mapping: dict[int, str]|None=None, buffer_time: float=0.0, output_file_name: str|None=None, max_images: int|None=None):
    predictions = []

    coco_annotations = COCO(annotations_file_path)

    image_ids = coco_annotations.getImgIds()
    image_ids = sorted(image_ids)
    random.seed(0)
    random.shuffle(image_ids)

    if max_images is not None:
        image_ids = image_ids[:max_images]

    for image_id in tqdm(image_ids):
        image_info = coco_annotations.loadImgs(image_id)[0]
        image_path = os.path.join(image_dir, image_info["file_name"])

        image = Image.open(image_path).convert("RGB")
        initial_shape = image.size
        image = TF.to_tensor(image).cuda()

        if inference.prediction_type == "bbox":
            xyxy, class_id, score = inference.infer(image)
            masks = None
        elif inference.prediction_type == "segm":
            xyxy, class_id, score, masks = inference.infer(image)
        else:
            raise ValueError(f"Invalid prediction type: {inference.prediction_type}")

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

        if masks is not None:
            masks = masks.squeeze(0)
            masks = masks.cpu().numpy()

        # for this_xywh, this_class_id, this_score in zip(xywh, class_id, score):
        for i in range(xywh.shape[0]):
            this_xywh = xywh[i]
            this_class_id = class_id[i]
            this_score = score[i]

            prediction = {
                "image_id": image_id,
                "bbox": this_xywh.tolist(),
                "category_id": class_mapping[int(this_class_id)] if class_mapping is not None else int(this_class_id),
                "score": float(this_score)
            }
            
            if masks is not None:
                formatted_array = np.asfortranarray(masks[i, :, :, np.newaxis].astype(np.uint8))
                prediction["segmentation"] = mask_utils.encode(formatted_array)[0]
                prediction["segmentation"]["counts"] = prediction["segmentation"]["counts"].decode("utf-8")
            
            predictions.append(prediction)
        
        time.sleep(buffer_time)

    if output_file_name is not None:
        print(f"Saving predictions to {output_file_name}")
        with open(output_file_name, "w") as f:
            json.dump(predictions, f)

    print("Loading predictions into COCO format (in-memory)")
    coco_det = coco_annotations.loadRes(predictions)
    
    print("Evaluating predictions")
    coco_eval = COCOeval(coco_annotations, coco_det, inference.prediction_type)
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats.tolist()