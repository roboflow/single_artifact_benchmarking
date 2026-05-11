import torch
import torchvision.transforms.functional as TF
import json
import fire


from sab.onnx_inference import ONNXInference
from sab.trt_inference import TRTInference
from sab.models.utils import cxcywh_to_xyxy, ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results


# RF-DETR keypoint heads emit one slot per (class, max_keypoint), padded with zeros for
# classes that have fewer keypoints than the max. The preview model is trained with
# num_keypoints_per_class = [0, 17] (background, person), so K_padded = 17 * 2 = 34.
NUM_KEYPOINTS = 17


def preprocess_image(image: torch.Tensor, image_input_shape: tuple[int, int]) -> tuple[torch.Tensor, dict]:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    means = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)

    image = TF.normalize(image, means, stds)
    image = TF.resize(image, image_input_shape[2:])
    return image, {}


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["dets"]            # [B, N_q, 4] cxcywh, normalized [0,1]
    out_logits = outputs["labels"]      # [B, N_q, C]
    out_keypoints = outputs["keypoints"]  # [B, N_q, K_padded, D]

    B, N_q, C = out_logits.shape
    K_padded = out_keypoints.shape[2]
    D = out_keypoints.shape[3]
    assert K_padded % C == 0, f"K_padded={K_padded} not divisible by num_classes={C}"
    K_per_class = K_padded // C
    assert K_per_class == NUM_KEYPOINTS, f"expected {NUM_KEYPOINTS} keypoints per class, got {K_per_class}"

    scores = out_logits.sigmoid()
    flat_scores = scores.view(B, -1)
    num_select = min(300, flat_scores.shape[1])

    topk_values, topk_indexes = torch.topk(flat_scores, num_select, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // C  # [B, num_select] query indices
    labels = topk_indexes % C       # [B, num_select] class indices

    bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    bboxes = cxcywh_to_xyxy(bboxes)

    # Gather keypoints per top-k query, then select the per-class slot for each predicted label.
    kp_gather_idx = topk_boxes.unsqueeze(-1).unsqueeze(-1).expand(B, num_select, K_padded, D)
    keypoints_g = torch.gather(out_keypoints, 1, kp_gather_idx)  # [B, num_select, K_padded, D]
    keypoints_g = keypoints_g.view(B, num_select, C, K_per_class, D)[..., :4]

    batch_idx = torch.arange(B, device=labels.device).unsqueeze(-1).expand_as(labels)
    query_idx = torch.arange(num_select, device=labels.device).unsqueeze(0).expand_as(labels)
    keypoints_sel = keypoints_g[batch_idx, query_idx, labels]  # [B, num_select, K_per_class, 4]

    # Per the r-flow PostProcess: dim 0,1 are normalized x,y; dim 2 is the per-keypoint
    # confidence logit (sigmoid -> [0,1]); dim 3 is unused for COCO output.
    keypoints_xy = keypoints_sel[..., :2]
    keypoints_conf = keypoints_sel[..., 2:3].sigmoid()
    keypoints_final = torch.cat([keypoints_xy, keypoints_conf], dim=-1)  # [B, num_select, K_per_class, 3]
    keypoints_final = keypoints_final.reshape(B, num_select, K_per_class * 3)  # [B, num_select, 51]

    return bboxes.contiguous(), labels.contiguous(), scores.contiguous(), keypoints_final.contiguous()


class RFDETRKeypointONNXInference(ONNXInference):
    def __init__(self, model_path: str, image_input_name: str | None = None):
        super().__init__(model_path, image_input_name, prediction_type="keypoints")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict):
        return postprocess_output(outputs, metadata)


class RFDETRKeypointTRTInference(TRTInference):
    def __init__(self, model_path: str, image_input_name: str | None = None):
        super().__init__(model_path, image_input_name, use_cuda_graph=True, prediction_type="keypoints")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict):
        return postprocess_output(outputs, metadata)


def main(image_dir: str, annotations_file_path: str, buffer_time: float = 0.0, output_file_name: str = "rfdetr_keypoint_results.json"):
    requests = [
        ArtifactBenchmarkRequest(
            onnx_path="rf-detr-nano-keypoint-preview-small-arch.onnx",
            inference_class=RFDETRKeypointTRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            max_dets=20,
        ),
    ]

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)

    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
