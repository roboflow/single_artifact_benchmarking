from functools import partial

import fire
import torch
import torchvision.transforms.functional as TF

from sab.processors import Processor
from sab.request import ArtifactBenchmarkRequest
from sab.results import pretty_print_results
from sab.runner import run_benchmark_on_artifacts
from sab.runtimes.tensorrt import TRTRuntime


def preprocess_image(image: torch.Tensor, image_input_shape: tuple[int, int], normalize: bool = True) -> tuple[torch.Tensor, dict]:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = TF.resize(image, image_input_shape[2:])

    if not normalize:
        image = image * 255.0

    return image, {}


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["boxes"].squeeze(0)
    labels = outputs["labels"].squeeze(0)
    scores = outputs["scores"].squeeze(0)

    return bboxes, labels, scores


class DFINEProcessor(Processor):
    def preprocess(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(image, self.input_spec.shape, self.normalize)

    def extra_inputs(self, image: torch.Tensor, metadata: dict) -> dict[str, torch.Tensor]:
        # spoof with ones because we want unnormalized bboxes
        return {"orig_target_sizes": torch.ones((1, 2), dtype=torch.int64, device=image.device)}

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


def build_requests(buffer_time: float = 0.0) -> list[ArtifactBenchmarkRequest]:
    artifact_paths = [
        "dfine_n_coco.opset17.onnx",
        "dfine_s_obj2coco.opset17.onnx",
        "dfine_m_obj2coco.opset17.onnx",
        "dfine_l_obj2coco_e25.opset17.onnx",
        "dfine_x_obj2coco.opset17.onnx",
    ]
    return [
        ArtifactBenchmarkRequest(
            artifact_path=artifact_path,
            runtime=partial(TRTRuntime, use_cuda_graph=False),
            processor=DFINEProcessor,
            device="gpu",
            precision=precision,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        )
        for artifact_path in artifact_paths
        for precision in ("fp32", "fp16")
    ]

def main(
    image_dir: str,
    annotations_file_path: str,
    buffer_time: float = 0.0,
    output_file_name: str = "dfine_results.json",
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
