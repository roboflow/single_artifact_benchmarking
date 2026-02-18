import torch
import torchvision.transforms.functional as TF
import json
import fire


from sab.onnx_inference import ONNXInference
from sab.trt_inference import TRTInference
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results


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


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["output0"][0, :, :4]
    scores = outputs["output0"][0, :, 4]
    labels = outputs["output0"][0, :, 5]
    keypoints = outputs["output0"][0, :, 6:]  # [N, 51]

    image_input_shape = metadata["image_input_shape"]
    input_h, input_w = image_input_shape[2], image_input_shape[3]
    padding = metadata["padding"]  # (left, top, right_pad, bottom_pad)

    # --- Undo letterbox for bboxes ---
    # Normalize from pixel coords to 0-1 in padded space
    bboxes /= torch.tensor([input_w, input_h, input_w, input_h], device=bboxes.device)

    # Remove padding and scale to original image dimensions
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * input_w - padding[0]
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * input_h - padding[1]

    bboxes[:, [0, 2]] /= (input_w - padding[0] - padding[2])
    bboxes[:, [1, 3]] /= (input_h - padding[1] - padding[3])

    bboxes = torch.clamp(bboxes, 0, 1)

    # --- Undo letterbox for keypoints ---
    kps = keypoints.reshape(-1, 17, 3)

    # Remove padding offset and normalize to 0-1 relative to original image
    kps[:, :, 0] = (kps[:, :, 0] - padding[0]) / (input_w - padding[0] - padding[2])
    kps[:, :, 1] = (kps[:, :, 1] - padding[1]) / (input_h - padding[1] - padding[3])

    keypoints = kps.reshape(-1, 51)

    return bboxes, labels, scores, keypoints


class YOLO26PoseONNXInference(ONNXInference):
    def __init__(self, model_path: str, image_input_name: str | None = None):
        super().__init__(model_path, image_input_name, prediction_type="keypoints")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict):
        return postprocess_output(outputs, metadata)


class YOLO26PoseTRTInference(TRTInference):
    def __init__(self, model_path: str, image_input_name: str | None = None):
        super().__init__(model_path, image_input_name, use_cuda_graph=False, prediction_type="keypoints")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict):
        return postprocess_output(outputs, metadata)


def main(image_dir: str, annotations_file_path: str, buffer_time: float = 0.0, output_file_name: str = "yolo26_pose_results.json"):
    requests = [
        ArtifactBenchmarkRequest(
            onnx_path="yolo26n-pose.onnx",
            inference_class=YOLO26PoseTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
            max_dets=20,  # appropriate for keypoints
        ),
    ]

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)

    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
