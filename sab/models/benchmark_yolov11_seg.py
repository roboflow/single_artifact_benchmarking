from functools import partial

import fire
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from sab.models.graph_surgery import fuse_yolo_mask_postprocessing_into_onnx
from sab.processors import Processor
from sab.request import ArtifactBenchmarkRequest
from sab.results import pretty_print_results
from sab.runner import run_benchmark_on_artifacts
from sab.runtimes.tensorrt import TRTRuntime


def preprocess_image(image: torch.Tensor, image_input_shape: tuple[int, int], normalize: bool = True) -> tuple[torch.Tensor, dict]:
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

    if not normalize:
        image = image * 255.0
    
    # Save letterbox metadata for postprocessing
    metadata.update({
        "scale": scale,
        "padding": padding
    })
    
    return image, metadata


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["_det_meta"][0, :, :4]
    scores = outputs["_det_meta"][0, :, 4]
    labels = outputs["_det_meta"][0, :, 5]

    image_input_shape = metadata["image_input_shape"]

    # Denormalize from input shape
    bboxes /= torch.tensor([image_input_shape[2], image_input_shape[3], image_input_shape[2], image_input_shape[3]], device=bboxes.device)

    # Remove padding and scale to original image dimensions
    padding = metadata["padding"] # (left, top, right_pad, bottom_pad)
    
    # First remove padding in absolute coordinates
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * image_input_shape[3] - padding[0]  
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * image_input_shape[2] - padding[1]

    # Then scale back to original dimensions
    bboxes[:, [0, 2]] /= (image_input_shape[3] - padding[0] - padding[2]) 
    bboxes[:, [1, 3]] /= (image_input_shape[2] - padding[1] - padding[3])

    # Clip to [0, 1] 
    bboxes = torch.clamp(bboxes, 0, 1)

    # Get masks, upsample to padded input size, remove padding, then resize to original image size
    masks = outputs["_masks_cropped"]

    # Select batch dimension if present
    if masks.dim() == 4:
        masks = masks[0]  # shape: (num_masks, h, w)

    # Ensure shape is (N=1, C=num_masks, H, W) for interpolation
    if masks.dim() == 3:
        masks = masks.unsqueeze(0)

    # Upsample to model input size (with padding)
    input_h, input_w = image_input_shape[2], image_input_shape[3]
    masks = F.interpolate(masks.float(), size=(input_h, input_w), mode="bilinear", align_corners=False)

    # Remove letterbox padding
    left, top, right_pad, bottom_pad = metadata["padding"]
    masks = masks[:, :, top: input_h - bottom_pad, left: input_w - right_pad]

    # Resize to original image spatial size
    orig_h, orig_w = metadata["original_shape"][2], metadata["original_shape"][3]
    masks = F.interpolate(masks, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    # Binarize
    masks = (masks.squeeze(0) > 0.5)

    return bboxes, labels, scores, masks


class YOLOv11SegProcessor(Processor):
    prediction_type = "segm"

    # reference: https://github.com/ultralytics/ultralytics/blob/3c88bebc9514a4d7f70b771811ddfe3a625ef14d/examples/YOLOv8-OpenCV-ONNX-Python/main.py#L23C57-L31
    def preprocess(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(image, self.input_spec.shape, self.normalize)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, ...]:
        return postprocess_output(outputs, metadata)


def build_requests(buffer_time: float = 0.0) -> list[ArtifactBenchmarkRequest]:
    artifact_paths = [
        "yolo11n_seg_nms_conf_0.01.onnx",
        "yolo11s_seg_nms_conf_0.01.onnx",
        "yolo11m_seg_nms_conf_0.01.onnx",
        "yolo11l_seg_nms_conf_0.01.onnx",
        "yolo11x_seg_nms_conf_0.01.onnx",
    ]
    return [
        ArtifactBenchmarkRequest(
            artifact_path=artifact_path,
            graph_surgery_func=fuse_yolo_mask_postprocessing_into_onnx,
            runtime=partial(TRTRuntime, use_cuda_graph=False),
            processor=YOLOv11SegProcessor,
            device="gpu",
            precision="fp16",
            buffer_time=buffer_time,
            needs_class_remapping=True,
        )
        for artifact_path in artifact_paths
    ]

def main(
    image_dir: str,
    annotations_file_path: str,
    buffer_time: float = 0.0,
    output_file_name: str = "yolov11_results.json",
    runtimes: str | None = None,
    devices: str | None = None,
    max_images: int | None = None,
    rerun: bool = False,
):
    results = run_benchmark_on_artifacts(
        build_requests(buffer_time),
        image_dir,
        annotations_file_path,
        output_file=output_file_name,
        runtimes=runtimes,
        devices=devices,
        max_images=max_images,
        rerun=rerun,
    )
    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
