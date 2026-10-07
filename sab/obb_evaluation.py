"""Evaluate oriented bounding boxes (OBB) on YOLO-OBB datasets, such as DOTA cut into tiles.

Ground truth is read from YOLO-OBB label files (``class x1 y1 x2 y2 x3 y3 x4 y4``, corners normalized to the image),
one ``<stem>.txt`` per image; a missing file means no objects. Each labelled polygon is replaced by its minimum-area
rectangle, as Ultralytics validation does, and predictions are matched by exact rotated IoU.

The metric is COCO's: faster-coco-eval's matching and accumulation, with only ``computeIoU`` replaced. The returned
12 statistics have the order of ``COCOeval.stats``, so results print like the bbox benchmarks. With every angle at 0
the numbers equal the standard COCO bbox metric.

OBB adapters return ``(obboxes, class_id, score)``: ``obboxes`` is ``(N, 5)`` ``(cx, cy, w, h, theta)`` in pixels of
the original image, ``theta`` in radians, clockwise on screen (Ultralytics ``xywhr``).
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def obboxes_to_polygons(obboxes: torch.Tensor) -> torch.Tensor:
    """Corners ``(N, 4, 2)`` of oriented boxes ``(N, 5)`` ``(cx, cy, w, h, theta)``."""
    cx, cy, w, h, theta = obboxes.unbind(-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    along = torch.stack([w / 2 * cos, w / 2 * sin], dim=-1)
    across = torch.stack([-h / 2 * sin, h / 2 * cos], dim=-1)
    center = torch.stack([cx, cy], dim=-1)
    return torch.stack([center + along + across, center + along - across, center - along - across, center - along + across], dim=-2)


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inside(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Which points ``(..., P, 2)`` lie inside or on the convex polygon ``(..., V, 2)``, for either winding."""
    edges = torch.roll(polygon, -1, dims=-2) - polygon
    orientation = torch.sign(_cross(edges, torch.roll(edges, -1, dims=-2)).sum(-1, keepdim=True))[..., None]
    offsets = points[..., :, None, :] - polygon[..., None, :, :]
    cross = _cross(edges[..., None, :, :], offsets) * orientation
    return (cross >= -1e-9 * edges.norm(dim=-1)[..., None, :] * offsets.norm(dim=-1)).all(-1)


def _intersection_area(polygon1: torch.Tensor, polygon2: torch.Tensor) -> torch.Tensor:
    """Area of the intersection of convex quadrilaterals, batched: the hull of the corners of each inside the other
    and the edge crossings, sorted by angle around their centroid; unused slots repeat the first point."""
    starts1, starts2 = polygon1[..., :, None, :], polygon2[..., None, :, :]
    edges1 = (torch.roll(polygon1, -1, dims=-2) - polygon1)[..., :, None, :]
    edges2 = (torch.roll(polygon2, -1, dims=-2) - polygon2)[..., None, :, :]
    denominator = _cross(edges1, edges2)
    parallel = denominator.abs() <= 1e-12 * edges1.norm(dim=-1) * edges2.norm(dim=-1)
    safe = torch.where(parallel, torch.ones_like(denominator), denominator)
    t, u = _cross(starts2 - starts1, edges2) / safe, _cross(starts2 - starts1, edges1) / safe
    crosses = ~parallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    points = torch.cat([polygon1, polygon2, (starts1 + t[..., None] * edges1).flatten(-3, -2)], dim=-2)
    valid = torch.cat([_inside(polygon1, polygon2), _inside(polygon2, polygon1), crosses.flatten(-2)], dim=-1)
    count = valid.sum(-1)
    centroid = (points * valid[..., None]).sum(-2) / count.clamp_min(1)[..., None]
    offsets = points - centroid[..., None, :]
    angles = torch.atan2(offsets[..., 1], offsets[..., 0]).masked_fill(~valid, math.inf)
    ordered = torch.gather(points, -2, angles.argsort(dim=-1)[..., None].expand_as(points))
    slots = torch.arange(points.shape[-2], device=points.device)
    ordered = torch.where((slots < count[..., None])[..., None], ordered, ordered[..., :1, :])
    area = 0.5 * _cross(ordered, torch.roll(ordered, -1, dims=-2)).sum(-1).abs()
    return torch.where(count >= 3, area, torch.zeros_like(area))


def polygon_iou(polygons1: torch.Tensor, polygons2: torch.Tensor) -> torch.Tensor:
    """Exact pairwise IoU ``(N, M)`` of convex quadrilaterals ``(N, 4, 2)`` and ``(M, 4, 2)``, in float64.

    Only pairs whose circumscribed circles overlap can intersect; the polygon is built for those alone.
    """
    if len(polygons1) == 0 or len(polygons2) == 0:
        return torch.zeros((len(polygons1), len(polygons2)), dtype=torch.float64)
    first, second = polygons1.double(), polygons2.double()
    center1, center2 = first.mean(1), second.mean(1)
    radius1 = (first - center1[:, None]).norm(dim=-1).amax(1)
    radius2 = (second - center2[:, None]).norm(dim=-1).amax(1)
    rows, columns = (torch.cdist(center1, center2) <= radius1[:, None] + radius2[None]).nonzero(as_tuple=True)
    intersection = first.new_zeros((len(first), len(second)))
    intersection[rows, columns] = _intersection_area(first[rows], second[columns])
    area1 = 0.5 * _cross(first, torch.roll(first, -1, dims=1)).sum(-1).abs()
    area2 = 0.5 * _cross(second, torch.roll(second, -1, dims=1)).sum(-1).abs()
    return intersection / (area1[:, None] + area2[None] - intersection).clamp_min(torch.finfo(torch.float64).tiny)


def load_yolo_obb_labels(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO-OBB label file into pixel corners ``(M, 4, 2)`` and class ids ``(M,)``; missing means empty."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    for number, row in enumerate(rows, start=1):
        if len(row) != 9:
            raise ValueError(f"{path}:{number}: expected 9 fields (class + 4 corners), got {len(row)}")
    if not rows:
        return np.zeros((0, 4, 2)), np.zeros(0, dtype=np.int64)
    values = np.array(rows, dtype=np.float64)
    return values[:, 1:].reshape(-1, 4, 2) * np.array([width, height]), values[:, 0].astype(np.int64)


def min_area_rectangles(polygons: np.ndarray) -> torch.Tensor:
    """Corners ``(M, 4, 2)`` of the minimum-area rectangle of each labelled polygon (DOTA labels are quadrilaterals)."""
    if len(polygons) == 0:
        return torch.zeros((0, 4, 2), dtype=torch.float64)
    return torch.from_numpy(np.stack([cv2.boxPoints(cv2.minAreaRect(p.astype(np.float32))) for p in polygons])).double()


def obb_image_label_pairs(image_dir: str, label_dir: str, max_images: int | None = None) -> list[tuple[Path, Path]]:
    """Images in stable order, each paired with ``<label_dir>/<stem>.txt`` (which may be missing: no objects)."""
    image_root, label_root = Path(image_dir), Path(label_dir)
    for path in (image_root, label_root):
        if not path.is_dir():
            raise ValueError(f"OBB evaluation requires a directory: {path}")
    images = sorted(p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise ValueError(f"No images found in {image_root}")
    return [(p, label_root / f"{p.stem}.txt") for p in images]


def _faster_coco():
    """COCO / COCOeval from faster-coco-eval, imported on the evaluation path only (as sab.evaluation does)."""
    from faster_coco_eval import COCO
    from faster_coco_eval.core.faster_eval_api import COCOeval

    class ObbCOCOeval(COCOeval):
        """COCOeval scoring detections by exact rotated IoU of their ``poly`` corners; all else is COCO's."""

        def computeIoU(self, imgId, catId):  # noqa: N802, N803 - faster-coco-eval's override signature
            params = self.params
            gt = self.gt_dataset.get_instances([imgId], [catId], True)[0][0]
            dt = self.dt_dataset.get_instances([imgId], [catId], True)[0][0]
            if len(gt) == 0 or len(dt) == 0:
                return []
            order = np.argsort([-d["score"] for d in dt], kind="mergesort")[: params.maxDets[-1]]
            detections = torch.tensor([dt[i]["poly"] for i in order], dtype=torch.float64).reshape(-1, 4, 2)
            truth = torch.tensor([g["poly"] for g in gt], dtype=torch.float64).reshape(-1, 4, 2)
            return polygon_iou(detections, truth).numpy()

    return COCO, ObbCOCOeval


def _annotation(polygon: torch.Tensor) -> dict:
    """COCO fields of one rectangle: ``poly`` for IoU, ``area`` = its area (COCO area ranges), and an un-rotated
    ``bbox`` of the same area, which ``loadRes`` requires and derives detection areas from."""
    corners = polygon.double()
    side1 = float((corners[1] - corners[0]).norm())
    side2 = float((corners[2] - corners[1]).norm())
    cx, cy = corners.mean(0).tolist()
    return {"poly": corners.flatten().tolist(), "bbox": [cx - side1 / 2, cy - side2 / 2, side1, side2], "area": side1 * side2}


def obb_coco_stats(detections: list[dict], ground_truth: list[tuple[torch.Tensor, torch.Tensor]], max_dets: int = 100) -> list[float]:
    """COCO's 12 statistics for oriented boxes.

    Args:
        detections: Per image, ``obboxes`` ``(N, 5)`` in pixels, ``scores`` ``(N,)`` and ``labels`` ``(N,)``.
        ground_truth: Per image, rectangle corners ``(M, 4, 2)`` and class ids ``(M,)``.
        max_dets: Largest detection limit (``maxDets = [1, 10, max_dets]``).
    """
    COCO, ObbCOCOeval = _faster_coco()
    images, annotations, results, classes = [], [], [], set()
    for image_id, (dets, (polygons, labels)) in enumerate(zip(detections, ground_truth, strict=True)):
        images.append({"id": image_id, "width": 1, "height": 1})
        for polygon, label in zip(polygons, labels.tolist()):
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": int(label), "iscrowd": 0, **_annotation(polygon)})
            classes.add(int(label))
        for polygon, score, label in zip(obboxes_to_polygons(dets["obboxes"].double()), dets["scores"].tolist(), dets["labels"].tolist()):
            results.append({"image_id": image_id, "category_id": int(label), "score": float(score), **_annotation(polygon)})
            classes.add(int(label))
    gt = COCO({"images": images, "annotations": annotations, "categories": [{"id": c, "name": str(c)} for c in sorted(classes)]})
    coco_eval = ObbCOCOeval(gt, gt.loadRes(results), iouType="bbox")
    coco_eval.params.maxDets = [1, 10, max_dets]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return [float(v) for v in coco_eval.stats]


def evaluate_obb(inference, image_dir: str, label_dir: str, buffer_time: float = 0.0, max_images: int | None = None, max_dets: int = 100) -> list[float]:
    """Run the timed pass over ``image_dir`` and score the oriented boxes against ``label_dir``.

    Only inference inside the runtime's profiler is timed; label reading and IoU happen in the result callback and
    after the pass.
    """
    from sab.evaluation import run_timed_pass

    pairs = obb_image_label_pairs(image_dir, label_dir, max_images)
    detections, ground_truth = [], []

    def host_copy(tensor):
        # Only NumPy is kept per image; tensors are built after the pass. Keeping torch CPU tensors created during the
        # pass grew host memory by several MB per image (OOM on DOTA val with TensorRT CUDA-graph adapters), NumPy did
        # not.
        return tensor.detach().cpu().numpy().copy()

    def accumulate(index, initial_shape, outputs):
        obboxes, class_id, score = outputs
        detections.append({"obboxes": host_copy(obboxes).astype(np.float32).reshape(-1, 5),
                           "scores": host_copy(score).astype(np.float32).reshape(-1),
                           "labels": host_copy(class_id).astype(np.int64).reshape(-1)})
        ground_truth.append(load_yolo_obb_labels(pairs[index][1], width=initial_shape[0], height=initial_shape[1]))

    run_timed_pass(inference, [str(p) for p, _ in pairs], buffer_time=buffer_time, on_result=accumulate)
    detections = [{key: torch.from_numpy(value) for key, value in image.items()} for image in detections]
    ground_truth = [(min_area_rectangles(polygons), torch.from_numpy(labels)) for polygons, labels in ground_truth]
    return obb_coco_stats(detections, ground_truth, max_dets=max_dets)
