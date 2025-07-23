import torch
import torchvision.transforms.functional as TF
import os
import json
import fire


from onnx_inference import ONNXInference
from trt_inference import TRTInference
from models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results


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


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["output0"][0, :, :4]
    scores = outputs["output0"][0, :, 4]
    labels = outputs["output0"][0, :, 5]

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

    return bboxes, labels, scores


class YOLOv11ONNXInference(ONNXInference):
    # reference: https://github.com/ultralytics/ultralytics/blob/3c88bebc9514a4d7f70b771811ddfe3a625ef14d/examples/YOLOv8-OpenCV-ONNX-Python/main.py#L23C57-L31
    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)
    

class YOLOv11TRTInference(TRTInference):
    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)
    

def main(image_dir: str, annotations_file_path: str, buffer_time: float = 0.0, output_file_name: str = "yolov11_results.json"):
    requests = [
        ArtifactBenchmarkRequest(
            onnx_path="yolo11n_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11n_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11n_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11n_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11s_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11s_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11s_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11s_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11m_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11m_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11m_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11m_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11l_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11l_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11l_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11l_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11x_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11x_nms_conf_0.001.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11x_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="yolo11x_nms_conf_0.01.onnx",
            inference_class=YOLOv11TRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
    ]

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)
    
    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)