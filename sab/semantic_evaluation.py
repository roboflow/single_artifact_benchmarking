"""Evaluate dense class maps against single-channel semantic label images."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticEvaluationConfig:
    num_classes: int
    ignore_index: int = 255
    label_offset: int = 0
    resize_mode: str = "native"
    iou_average: str = "union"

    def __post_init__(self):
        if self.num_classes < 2:
            raise ValueError("num_classes must include at least two pixel classes")
        if self.resize_mode not in {"native", "letterbox"}:
            raise ValueError("resize_mode must be native or letterbox")
        if self.iou_average not in {"all", "union", "gt_present"}:
            raise ValueError("iou_average must be all, union, or gt_present")

    def dump(self):
        return asdict(self)


# ADEChallengeData2016 annotations use 1..150; 0 is unlabeled.
ADE20K_CONFIG = SemanticEvaluationConfig(num_classes=150, ignore_index=0, label_offset=1, iou_average="all")
YOLO26_ADE20K_CONFIG = SemanticEvaluationConfig(
    num_classes=150, ignore_index=0, label_offset=1, resize_mode="letterbox", iou_average="gt_present")


def letterbox_geometry(original_shape, input_shape):
    """Ultralytics validation scales the long side, rounding resized sizes up."""
    height, width = original_shape
    input_h, input_w = input_shape
    scale = min(input_h / height, input_w / width)
    resized_h = min(math.ceil(height * scale), input_h)
    resized_w = min(math.ceil(width * scale), input_w)
    return resized_h, resized_w, (input_h - resized_h) // 2, (input_w - resized_w) // 2


def letterbox_target(target, input_shape):
    height, width, top, left = letterbox_geometry(target.shape, input_shape)
    # Nearest interpolation preserves class IDs, including the ignore sentinel.
    resized = F.interpolate(torch.from_numpy(target)[None, None].float(),
                            size=(height, width), mode="nearest")[0, 0].numpy().astype(np.int64)
    result = np.full(input_shape, -1, dtype=np.int64)
    result[top:top + height, left:left + width] = resized
    return result


class SemanticMetrics:
    """Accumulate pixel counts globally, without averaging per-image IoUs.

    Rows are ground-truth classes and columns are predictions. The averaging
    policy is explicit: all classes (ADE20K), nonempty unions (TorchMetrics),
    or ground-truth-present classes (Ultralytics).
    """

    def __init__(self, num_classes: int, iou_average: str = "union"):
        if iou_average not in {"all", "union", "gt_present"}:
            raise ValueError("Unknown IoU averaging policy")
        self.num_classes = num_classes
        self.iou_average = iou_average
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.num_images = 0

    def update(self, prediction: np.ndarray, target: np.ndarray, ignore_index: int):
        if prediction.shape != target.shape or target.ndim != 2:
            raise ValueError(f"Expected matching HxW class maps, got {prediction.shape} and {target.shape}")
        if not np.issubdtype(prediction.dtype, np.integer):
            raise ValueError("Semantic predictions must contain integer class IDs")
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError("Semantic targets must contain integer class IDs")

        valid = target != ignore_index
        truth = target[valid].astype(np.int64)
        pred = prediction[valid].astype(np.int64)
        for name, values in (("target", truth), ("prediction", pred)):
            if np.any((values < 0) | (values >= self.num_classes)):
                raise ValueError(f"Semantic {name} IDs must be in [0, {self.num_classes - 1}]")
        counts = np.bincount(self.num_classes * truth + pred, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)
        self.num_images += 1

    def get_stats(self):
        matrix = self.confusion_matrix
        support = matrix.sum(axis=1)
        intersection = matrix.diagonal()
        union = support + matrix.sum(axis=0) - intersection
        valid_pixels = int(support.sum())
        if valid_pixels == 0:
            raise ValueError("No labeled pixels were evaluated; check mask encoding and ignore_index")
        iou = np.divide(intersection, union, out=np.zeros(self.num_classes), where=union > 0)
        accuracy = np.divide(intersection, support, out=np.zeros(self.num_classes), where=support > 0)
        present = support > 0
        averaged = {"all": np.ones(self.num_classes, dtype=bool),
                    "union": union > 0, "gt_present": present}[self.iou_average]
        return {
            "mean_iou": float(iou[averaged].mean()),
            "iou_average": self.iou_average,
            "num_classes_averaged": int(averaged.sum()),
            "pixel_accuracy": float(intersection.sum() / valid_pixels),
            "mean_accuracy": float(accuracy[present].mean()),
            "per_class_iou": [float(v) if n else None for v, n in zip(iou, union)],
            "per_class_accuracy": [float(v) if n else None for v, n in zip(accuracy, support)],
            "class_pixel_counts": support.tolist(),
            "num_classes": self.num_classes,
            "num_images": self.num_images,
            "valid_pixels": valid_pixels,
        }


def semantic_image_pairs(image_dir: str, mask_dir: str, max_images: int | None = None):
    """Pair images with PNG masks by relative path and stem, in stable order."""
    image_root, mask_root = Path(image_dir), Path(mask_dir)
    for path in (image_root, mask_root):
        if not path.is_dir():
            raise ValueError(f"Semantic evaluation requires a directory: {path}")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise ValueError(f"No images found in {image_root}")
    pairs = [(p, mask_root / p.relative_to(image_root).with_suffix(".png")) for p in images]
    for _, mask_path in pairs:
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing semantic mask: {mask_path}")
    return pairs


def load_semantic_target(path: Path, config: SemanticEvaluationConfig):
    with Image.open(path) as mask:
        # Preserve palette indices. Converting a P-mode mask to L changes IDs.
        target = np.array(mask)
    if target.ndim != 2 or not np.issubdtype(target.dtype, np.integer):
        raise ValueError(f"Expected a single-channel integer class-ID mask: {path}")
    ignored = target == config.ignore_index
    target = target.astype(np.int64) - config.label_offset
    target[ignored] = -1
    # Validate raw non-ignored pixels too: subtracting the offset must not
    # accidentally turn an invalid label into the internal ignore sentinel.
    if np.any((~ignored) & ((target < 0) | (target >= config.num_classes))):
        raise ValueError(f"Invalid label IDs in {path}; check label_offset and ignore_index")
    return target


def evaluate_semantic(inference, image_dir: str, mask_dir: str, config: SemanticEvaluationConfig,
                      buffer_time: float = 0.0, max_images: int | None = None):
    """Score dense predictions on the configured evaluation grid.

    Inference adapters return an HxW integer tensor in 0..num_classes-1. Native
    evaluation keeps mask resolution; letterbox evaluation uses nearest-resized
    targets at the model input resolution and ignores padding. Only inference
    inside the runtime's profiler is timed.
    """
    from sab.evaluation import run_timed_pass

    pairs = semantic_image_pairs(image_dir, mask_dir, max_images)
    metrics = SemanticMetrics(config.num_classes, config.iou_average)

    def accumulate(index, initial_shape, prediction):
        target = load_semantic_target(pairs[index][1], config)
        if target.shape != (initial_shape[1], initial_shape[0]):
            raise ValueError(f"Mask dimensions do not match the image: {pairs[index][1]}")
        if config.resize_mode == "letterbox":
            target = letterbox_target(target, tuple(inference.image_input_shape[-2:]))
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Semantic adapters must return an HxW integer tensor")
        metrics.update(prediction.detach().cpu().numpy(), target, ignore_index=-1)

    run_timed_pass(inference, [str(p) for p, _ in pairs], buffer_time=buffer_time, on_result=accumulate)
    return metrics.get_stats()
