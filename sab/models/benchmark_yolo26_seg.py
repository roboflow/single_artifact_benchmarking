import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import os
import json
import fire


from sab.onnx_inference import ONNXInference
from sab.trt_inference import TRTInference
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results
from sab.models.graph_surgery import fuse_yolo_mask_postprocessing_into_onnx


def preprocess_image(image: torch.Tensor, image_input_shape: tuple[int, int]) -> tuple[torch.Tensor, dict]:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    original_shape = image.shape

    metadata = {
        "original_shape": original_shape,
        "image_input_shape": image_input_shape,
    }

    # Calculate letterbox dimensions
    input_h, input_w = image_input_shape[2:]
    orig_h, orig_w = image.shape[2:]
    
    # Calculate scaling factor and new unpadded dimensions
    scale = min(input_h / orig_h, input_w / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    
    # Calculate padding
    pad_h = input_h - new_h
    pad_w = input_w - new_w
    top = pad_h // 2
    left = pad_w // 2
    
    # Resize image
    image = TF.resize(image, (new_h, new_w))
    
    # Pad to target size
    padding = (left, top, pad_w - left, pad_h - top)
    image = TF.pad(image, padding, fill=0)
    
    # Save letterbox metadata for postprocessing
    metadata.update({
        "scale": scale,
        "padding": padding
    })
    
    return image, metadata


import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def postprocess_output(
    outputs: Dict[str, torch.Tensor],
    metadata: Dict,
    score_threshold: float = 0.0,
    mask_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Post-process YOLO11 segmentation ONNX outputs exported with NMS=True.

    Expected outputs:
      - outputs["output0"]: (1, 38, 300) or (1, 300, 38)
            Each detection row is:
              [x1, y1, x2, y2, score, class_id, (32 mask coeffs...)]
            (Some exports/models may output xywh instead of xyxy; this function auto-detects.)
      - outputs["output1"]: (1, 32, 160, 160)  (prototypes)

    Returns:
      bboxes: (N, 4) float32, xyxy in original image pixel coordinates
      labels: (N,) int64
      scores: (N,) float32
      masks:  (N, orig_h, orig_w) bool (thresholded at mask_threshold), at original image resolution
    """
    det = outputs["output0"]
    proto = outputs["output1"]
    device = det.device

    # Original + model input sizes
    orig_shape = metadata["original_shape"]  # (B,C,H,W)
    orig_h, orig_w = int(orig_shape[-2]), int(orig_shape[-1])

    img_input_shape = metadata["image_input_shape"]  # usually (B,C,H,W) (we use last 2 dims)
    input_h, input_w = int(img_input_shape[-2]), int(img_input_shape[-1])

    # padding = (left, top, right, bottom) in pixels (as used by torchvision TF.pad)
    left, top, right, bottom = [int(x) for x in metadata["padding"]]

    # -----------------------
    # Parse detections (N, 38)
    # -----------------------
    det = det.squeeze(0)  # -> (38,300) or (300,38)

    if det.ndim != 2:
        # Try to coerce to 2D if some runtime adds singleton dims
        det = det.reshape(det.shape[0], -1) if det.shape[0] == 38 else det.reshape(-1, det.shape[-1])

    # Normalize layout to (N, 38)
    if det.shape[0] == 38 and det.shape[1] != 38:
        det = det.transpose(0, 1)  # (300,38)
    elif det.shape[1] == 38:
        pass
    else:
        raise ValueError(f"Unexpected output0 shape after squeeze: {tuple(det.shape)}")

    scores = det[:, 4]
    keep = scores > score_threshold
    det = det[keep]
    scores = scores[keep]

    if det.numel() == 0:
        empty_boxes = torch.empty((0, 4), device=device, dtype=torch.float32)
        empty_labels = torch.empty((0,), device=device, dtype=torch.int64)
        empty_scores = torch.empty((0,), device=device, dtype=torch.float32)
        empty_masks = torch.empty((0, orig_h, orig_w), device=device, dtype=torch.bool)
        return empty_boxes, empty_labels, empty_scores, empty_masks

    boxes = det[:, 0:4]
    labels = det[:, 5].to(torch.int64)
    mask_coeff = det[:, 6:]  # (N, 32)

    # If boxes appear normalized (0..1), scale to model-input pixels
    if float(boxes.max()) <= 1.5:
        scale_xyxy = torch.tensor([input_w, input_h, input_w, input_h], device=device, dtype=boxes.dtype)
        boxes = boxes * scale_xyxy

    # Heuristic: if many detections violate x2>=x1 or y2>=y1, assume xywh and convert to xyxy
    if (boxes[:, 2] < boxes[:, 0]).float().mean() > 0.5 or (boxes[:, 3] < boxes[:, 1]).float().mean() > 0.5:
        cxcy = boxes[:, 0:2]
        wh = boxes[:, 2:4]
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], dim=1)  # xyxy

    # -----------------------
    # Undo letterbox for boxes
    # -----------------------
    # Compute the *actual* resized (unpadded) size used in preprocess
    new_w = input_w - left - right
    new_h = input_h - top - bottom

    # Gains (use separate gains to match the integer rounding in preprocess)
    gain_w = new_w / orig_w
    gain_h = new_h / orig_h

    boxes = boxes.clone()
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, [0, 2]] /= gain_w
    boxes[:, [1, 3]] /= gain_h
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

    # -----------------------
    # Masks: coeffs @ protos
    # -----------------------
    proto = proto.squeeze(0)  # -> (32, mh, mw) typically
    if proto.ndim != 3:
        raise ValueError(f"Unexpected output1 shape after squeeze: {tuple(proto.shape)}")

    # Handle possible HWC prototype layout
    if proto.shape[0] != mask_coeff.shape[1]:
        if proto.shape[-1] == mask_coeff.shape[1]:
            proto = proto.permute(2, 0, 1)  # (32, mh, mw)
        else:
            raise ValueError(f"Prototype channels {tuple(proto.shape)} don't match mask coeffs {tuple(mask_coeff.shape)}")

    c, mh, mw = proto.shape
    if mask_coeff.shape[1] != c:
        raise ValueError(f"mask_coeff dim {mask_coeff.shape[1]} != proto channels {c}")

    # (N, mh*mw) -> (N, mh, mw), then sigmoid
    masks = mask_coeff @ proto.reshape(c, -1)
    masks = masks.reshape(-1, mh, mw)

    # Upsample masks to model input size, remove padding, upsample to original size
    masks = F.interpolate(masks.unsqueeze(1), size=(input_h, input_w), mode="bilinear", align_corners=False).squeeze(1)
    masks = masks[:, top : input_h - bottom, left : input_w - right]
    masks = F.interpolate(masks.unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)

    # Crop masks to their boxes (vectorized)
    n, h, w = masks.shape
    x1, y1, x2, y2 = boxes.unbind(1)
    x1 = x1.clamp(0, w)
    x2 = x2.clamp(0, w)
    y1 = y1.clamp(0, h)
    y2 = y2.clamp(0, h)

    cols = torch.arange(w, device=device)[None, None, :]
    rows = torch.arange(h, device=device)[None, :, None]
    crop = (
        (cols >= x1[:, None, None])
        & (cols < x2[:, None, None])
        & (rows >= y1[:, None, None])
        & (rows < y2[:, None, None])
    )
    masks = masks * crop.to(masks.dtype)

    masks = masks > mask_threshold  # bool masks at original resolution

    return boxes.to(torch.float32), labels, scores.to(torch.float32), masks






class YOLOv11SegONNXInference(ONNXInference):
    def __init__(self, model_path: str, image_input_name: str|None=None):
        super().__init__(model_path, image_input_name, prediction_type="segm")

    # reference: https://github.com/ultralytics/ultralytics/blob/3c88bebc9514a4d7f70b771811ddfe3a625ef14d/examples/YOLOv8-OpenCV-ONNX-Python/main.py#L23C57-L31
    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)
    

class YOLOv11SegTRTInference(TRTInference):
    def __init__(self, model_path: str, image_input_name: str|None=None):
        super().__init__(model_path, image_input_name, use_cuda_graph=False, prediction_type="segm")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)
    

def main(image_dir: str, annotations_file_path: str, buffer_time: float = 0.0, output_file_name: str = "yolov11_results.json"):
    requests = [
        ArtifactBenchmarkRequest(
            onnx_path="yolo26n-seg.onnx",
            # graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
            inference_class=YOLOv11SegTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
            # max_images=20,
        ),
        # ArtifactBenchmarkRequest(
        #     onnx_path="yolo26s-seg.onnx",
        #     # graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
        #     inference_class=YOLOv11SegTRTInference,
        #     needs_fp16=True,
        #     buffer_time=buffer_time,
        #     needs_class_remapping=True,
        #     # max_images=20,
        # ),
        # ArtifactBenchmarkRequest(
        #     onnx_path="yolo26m-seg.onnx",
        #     # graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
        #     inference_class=YOLOv11SegTRTInference,
        #     needs_fp16=True,
        #     buffer_time=buffer_time,
        #     needs_class_remapping=True,
        #     # max_images=20,
        # ),
        # ArtifactBenchmarkRequest(
        #     onnx_path="yolo26l-seg.onnx",
        #     # graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
        #     inference_class=YOLOv11SegTRTInference,
        #     needs_fp16=True,
        #     buffer_time=buffer_time,
        #     needs_class_remapping=True,
        #     # max_images=20,
        # ),
        # ArtifactBenchmarkRequest(
        #     onnx_path="yolo26x-seg.onnx",
        #     # graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
        #     inference_class=YOLOv11SegTRTInference,
        #     needs_fp16=True,
        #     buffer_time=buffer_time,
        #     needs_class_remapping=True,
        #     # max_images=20,
        # ),
    ]

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)
    
    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)