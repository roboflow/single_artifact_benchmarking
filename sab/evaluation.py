import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm
import time
from contextlib import nullcontext
from typing import Callable

from sab.onnx_inference import ONNXInferenceCPU


def _load_coco_tools():
    """Import the COCO stack on the mAP path only.

    faster_coco_eval and pycocotools stay out of module scope so that
    run_timed_pass() imports and runs on a box that has neither.
    """
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as mask_utils
    return COCO, COCOeval, mask_utils


def run_timed_pass(
    inference,
    image_paths: list[str],
    buffer_time: float = 0.0,
    max_images: int | None = None,
    monitor=None,
    on_result: Callable[[int, tuple[int, int], tuple], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
):
    """Run the measured inference loop over `image_paths` and return latency stats.

    This is the timed path and nothing else: no annotations, no COCO, no mAP. Every
    image goes through open, to_tensor, transfer, infer, then the buffer sleep.

    Args:
        inference: an inference object exposing .prediction_type, .infer() and .profiler
        image_paths: images to run, in order
        buffer_time: seconds to sleep after each image, to let the GPU cool
        max_images: run only the first N images
        monitor: optional context manager to hold open for the pass, such as a
            ThrottleMonitor or a CPUFrequencyMonitor. Read its verdict after the call.
        on_result: called with (index, initial_shape, (xyxy, class_id, score, masks))
            for each image, before the buffer sleep. Lets a caller layer accumulation
            on top of the loop without changing what is timed.
        sleep_fn: seam for tests; the buffer sleep itself.

    Returns:
        The profiler statistics for the pass.
    """
    if max_images is not None:
        image_paths = image_paths[:max_images]

    monitor_context = monitor if monitor is not None else nullcontext()

    with monitor_context:
        for index, image_path in enumerate(tqdm(image_paths)):
            image = Image.open(image_path).convert("RGB")
            initial_shape = image.size
            image = TF.to_tensor(image)
            if not isinstance(inference, ONNXInferenceCPU):
                image = image.cuda()

            if inference.prediction_type == "bbox":
                xyxy, class_id, score = inference.infer(image)
                masks = None
            elif inference.prediction_type == "segm":
                xyxy, class_id, score, masks = inference.infer(image)
            else:
                raise ValueError(f"Invalid prediction type: {inference.prediction_type}")

            if on_result is not None:
                on_result(index, initial_shape, (xyxy, class_id, score, masks))

            sleep_fn(buffer_time)

    return inference.profiler.get_stats()


def evaluate(inference, image_dir: str, annotations_file_path: str, class_mapping: dict[int, str]|None=None, buffer_time: float=0.0, output_file_name: str|None=None, max_images: int|None=None, max_dets: int=100):
    COCO, COCOeval, mask_utils = _load_coco_tools()

    predictions = []

    coco_annotations = COCO(annotations_file_path)

    image_ids = coco_annotations.getImgIds()

    if max_images is not None:
        image_ids = image_ids[:max_images]

    image_paths = [
        os.path.join(image_dir, coco_annotations.loadImgs(image_id)[0]["file_name"])
        for image_id in image_ids
    ]

    def accumulate_predictions(index, initial_shape, outputs):
        image_id = image_ids[index]
        xyxy, class_id, score, masks = outputs

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

    run_timed_pass(
        inference,
        image_paths,
        buffer_time=buffer_time,
        on_result=accumulate_predictions,
    )

    if output_file_name is not None:
        print(f"Saving predictions to {output_file_name}")
        with open(output_file_name, "w") as f:
            json.dump(predictions, f)

    print("Loading predictions into COCO format (in-memory)")
    coco_det = coco_annotations.loadRes(predictions)

    print("Evaluating predictions")
    coco_eval = COCOeval(coco_annotations, coco_det, inference.prediction_type)
    coco_eval.params.maxDets = [1, 10, max_dets,]
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats.tolist()
