import fire
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from sab.models.utils import cxcywh_to_xyxy
from sab.processors import Processor
from sab.request import ArtifactBenchmarkRequest
from sab.results import pretty_print_results
from sab.runner import run_benchmark_on_artifacts


def preprocess_image(image: torch.Tensor, image_input_shape: tuple[int, int], normalize: bool = True) -> tuple[torch.Tensor, dict]:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    orig_target_sizes = torch.tensor([image.shape[2], image.shape[3]], device=image.device)
    
    if normalize:
        means = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        stds = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)

        image = TF.normalize(image, means, stds)
    else:
        image = image * 255.0
    image = TF.resize(image, image_input_shape[2:])

    return image, {
        "orig_target_sizes": orig_target_sizes,
    }


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["dets"]
    out_logits = outputs["labels"]
    masks = outputs["masks"]
    scores = out_logits.sigmoid()

    flat_scores = scores.view(scores.shape[0], -1)
    num_select = min(300, flat_scores.shape[1])

    topk_values, topk_indexes = torch.topk(flat_scores, num_select, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    masks = torch.gather(masks, 1, topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1,1,masks.shape[2],masks.shape[3]))

    bboxes = cxcywh_to_xyxy(bboxes)

    masks = F.interpolate(masks, size=metadata["orig_target_sizes"].tolist(), mode="bilinear", align_corners=False)
    masks = masks > 0

    return bboxes.contiguous(), labels.contiguous(), scores.contiguous(), masks.contiguous()


class RFDETRSegProcessor(Processor):
    prediction_type = "segm"

    def preprocess(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(image, self.input_spec.shape, self.normalize)

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


def build_requests(buffer_time: float = 0.0) -> list[ArtifactBenchmarkRequest]:
    return []

def main(
    image_dir: str,
    annotations_file_path: str,
    buffer_time: float = 0.0,
    output_file_name: str = "rfdetr_seg_results.json",
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
