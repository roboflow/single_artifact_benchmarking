"""SAB benchmark adapter for MMRotate oriented-box detectors exported with MMDeploy 0.x (TensorRT 8.x plugins).

Two-stage MMRotate detectors (Oriented R-CNN with a ResNet-50, LSKNet or StripNet backbone) need rotated RoI align and
rotated NMS, which TensorRT lacks; MMDeploy exports them as its own TensorRT plugins (``MMCVMultiLevelRotatedRoiAlign``,
``TRTBatchedRotatedNMS``), compiled into ``libmmdeploy_tensorrt_ops.so``. That library is loaded before the engine is
built and deserialized; its path comes from ``--plugin_path`` or ``MMDEPLOY_TRT_OPS``. The rotated NMS runs inside the
engine, so the timed engine execution includes it.

Export (MMDeploy ``torch2onnx``, static 1024x1024 rotated-detection config):

  - Input:  input  (1, 3, 1024, 1024), ImageNet-normalized RGB (mmdet ``Normalize`` with ``to_rgb=True``)
  - Output: dets   (1, K, 6)  [cx, cy, w, h, angle, score] in input pixels, angle in radians (MMRotate le90)
            labels (1, K)     class ids in MMRotate's DOTA order (remapped to the YOLO labels' order), padding rows
                              score 0

    python -m sab.models.benchmark_mmrotate_obb <tiles>/images/val <tiles>/labels/val --onnx_path strip_rcnn_s.onnx
"""

import ctypes
import json
import os

import fire
import torch

from sab.models.benchmark_rfdetr import preprocess_image
from sab.models.utils import ArtifactBenchmarkRequest, pretty_print_results, run_benchmark_on_artifacts
from sab.trt_inference import TRTInference

MAX_DETECTIONS = 300
#: MMRotate's DOTA v1.0 class order (``DOTADataset.CLASSES``) mapped to the YOLO-OBB labels' (Ultralytics DOTAv1.yaml):
#: plane, baseball-diamond, bridge, ground-track-field, small-vehicle, large-vehicle, ship, tennis-court,
#: basketball-court, storage-tank, soccer-ball-field, roundabout, harbor, swimming-pool, helicopter.
MMROTATE_TO_YOLO_DOTA = (0, 3, 8, 6, 10, 9, 1, 4, 5, 2, 13, 12, 7, 14, 11)


def load_plugins(plugin_path: str | None) -> None:
    """Load MMDeploy's TensorRT plugin library into the process, so engines using its operators build and load.

    Raises:
        FileNotFoundError: If no plugin library is given or found.
    """
    path = plugin_path or os.environ.get("MMDEPLOY_TRT_OPS")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"MMDeploy TensorRT plugin library not found: {path!r} (set --plugin_path or MMDEPLOY_TRT_OPS)")
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The highest-scoring ``MAX_DETECTIONS`` rotated boxes in original pixels, their class ids and scores."""
    dets, labels = outputs["dets"][0].float(), outputs["labels"][0].long()
    scores = dets[:, 5]
    keep = scores > 0
    dets, labels, scores = dets[keep], labels[keep], scores[keep]
    order = scores.argsort(descending=True)[:MAX_DETECTIONS]
    height, width = metadata["original_shape"]
    input_height, input_width = metadata["input_shape"]
    if width / input_width != height / input_height:
        raise ValueError("rotated boxes need an isotropic resize; use square images and a square input")
    boxes = dets[order, :5].clone()
    boxes[:, :4] *= width / input_width
    classes = labels[order]
    if metadata.get("remap_dota_classes", True):
        classes = torch.as_tensor(MMROTATE_TO_YOLO_DOTA, device=classes.device)[classes]
    return boxes, classes, scores[order]


class MMRotateOBBTRTInference(TRTInference):
    def __init__(self, model_path: str, image_input_name: str | None = None, use_cuda_graph: bool = True):
        super().__init__(model_path, image_input_name, use_cuda_graph=use_cuda_graph, prediction_type="obb")

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        image, metadata = preprocess_image(input_image, self.image_input_shape)
        metadata["original_shape"] = tuple(input_image.shape[-2:])
        metadata["input_shape"] = tuple(self.image_input_shape[2:])
        return image, metadata

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


def main(image_dir: str, labels_dir: str, onnx_path: str, plugin_path: str | None = None, buffer_time: float = 0.0,
         output_file_name: str = "mmrotate_obb_results.json", max_images: int | None = None,
         max_dets: int = MAX_DETECTIONS, fp32: bool = False):
    """Benchmark an MMDeploy-exported MMRotate detector: TensorRT FP16 (and FP32 with ``fp32``)."""
    load_plugins(plugin_path)
    requests = [
        ArtifactBenchmarkRequest(onnx_path=onnx_path, inference_class=MMRotateOBBTRTInference, needs_fp16=fp16,
                                 buffer_time=buffer_time, max_images=max_images, max_dets=max_dets)
        for fp16 in ((False, True) if fp32 else (True,))
    ]
    results = run_benchmark_on_artifacts(requests, image_dir, labels_dir)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)

    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
