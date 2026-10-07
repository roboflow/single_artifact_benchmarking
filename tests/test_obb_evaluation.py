"""Tests for oriented-bounding-box (OBB) evaluation on YOLO-OBB tiles."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from sab.obb_evaluation import (
    obb_coco_stats,
    load_yolo_obb_labels,
    obb_image_label_pairs,
    obboxes_to_polygons,
    polygon_iou,
)
from sab.models.benchmark_yolo26_obb import postprocess_output


def rectangle(cx, cy, w, h, theta):
    """Corners (4, 2) of a rotated rectangle, Ultralytics xywhr convention."""
    return obboxes_to_polygons(torch.tensor([[cx, cy, w, h, theta]], dtype=torch.float64))[0].numpy()


def opencv_iou(a, b):
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return inter / union


class TestPolygonIou:
    def test_matches_opencv_on_random_rectangles(self):
        rng = np.random.default_rng(0)
        boxes = np.column_stack([rng.uniform(0, 20, (300, 2)), rng.uniform(0.5, 15, (300, 2)), rng.uniform(0, 6.3, 300)])
        ours = [float(polygon_iou(torch.tensor(rectangle(*a))[None], torch.tensor(rectangle(*b))[None])) for a, b in zip(boxes[::2], boxes[1::2])]
        reference = [opencv_iou(rectangle(*a), rectangle(*b)) for a, b in zip(boxes[::2], boxes[1::2])]
        assert ours == pytest.approx(reference, abs=1e-5)

    def test_identical(self):
        quad = torch.tensor(rectangle(10, 10, 8, 2, 0.4))[None]
        assert float(polygon_iou(quad, quad)) == pytest.approx(1.0)

    def test_disjoint(self):
        assert float(polygon_iou(torch.tensor(rectangle(0, 0, 2, 2, 0))[None], torch.tensor(rectangle(50, 50, 2, 2, 0))[None])) == 0.0

    def test_pairwise_shape(self):
        quads = torch.tensor(np.stack([rectangle(i, i, 4, 2, 0.1 * i) for i in range(5)]))
        assert polygon_iou(quads[:2], quads).shape == (2, 5)


class TestLabels:
    def test_corners_scale_to_pixels(self, tmp_path):
        (tmp_path / "a.txt").write_text("3 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")
        polygons, classes = load_yolo_obb_labels(tmp_path / "a.txt", width=200, height=100)
        assert (classes.tolist(), polygons[0].tolist()) == ([3], [[20, 20], [60, 20], [60, 40], [20, 40]])

    def test_missing_label_file_is_background(self, tmp_path):
        polygons, classes = load_yolo_obb_labels(tmp_path / "none.txt", width=10, height=10)
        assert (polygons.shape, classes.shape) == ((0, 4, 2), (0,))

    def test_five_field_line_is_refused(self, tmp_path):
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        with pytest.raises(ValueError, match="9 fields"):
            load_yolo_obb_labels(tmp_path / "a.txt", width=10, height=10)

    def test_pairs_images_with_labels_by_stem(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for stem in ("b", "a"):
            Image.new("RGB", (8, 8)).save(tmp_path / "images" / f"{stem}.jpg")
        pairs = obb_image_label_pairs(str(tmp_path / "images"), str(tmp_path / "labels"))
        assert [(p.name, l.name) for p, l in pairs] == [("a.jpg", "a.txt"), ("b.jpg", "b.txt")]


def detections_from(truth, score=0.9):
    """Detections equal to ground truth rectangles (cx, cy, w, h, theta, class)."""
    return {"obboxes": truth[:, :5], "scores": torch.full((len(truth),), score), "labels": truth[:, 5].long()}


class TestCocoStats:
    def test_perfect_detections_score_one(self):
        truth = torch.tensor([[100.0, 100, 80, 20, 0.4, 1], [300, 200, 40, 30, 1.2, 2]])
        polygons = obboxes_to_polygons(truth[:, :5])
        stats = obb_coco_stats([detections_from(truth)], [(polygons, truth[:, 5].long())], max_dets=100)
        assert (stats[0], stats[1]) == (pytest.approx(1.0), pytest.approx(1.0))

    def test_axis_aligned_boxes_match_standard_coco_bbox_metric(self):
        """With every angle 0, rotated IoU is box IoU: the stats equal pycocotools' bbox evaluation."""
        import faster_coco_eval
        faster_coco_eval.init_as_pycocotools()
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        rng = np.random.default_rng(1)
        images, detections, truths, coco_dets, annotations = [], [], [], [], []
        for image_id in range(6):
            n = int(rng.integers(1, 8))
            truth = torch.tensor(np.column_stack([rng.uniform(50, 400, (n, 2)), rng.uniform(8, 120, (n, 2)), np.zeros(n), rng.integers(0, 3, n)]), dtype=torch.float32)
            noisy = truth.repeat(2, 1)
            noisy[:, :4] += torch.tensor(rng.normal(0, 5, (2 * n, 4)), dtype=torch.float32)
            noisy[:, 2:4] = noisy[:, 2:4].abs() + 1
            scores = torch.tensor(rng.uniform(0, 1, 2 * n), dtype=torch.float32)
            detections.append({"obboxes": noisy[:, :5], "scores": scores, "labels": noisy[:, 5].long()})
            truths.append((obboxes_to_polygons(truth[:, :5]), truth[:, 5].long()))
            images.append({"id": image_id, "width": 512, "height": 512})
            for row in truth.tolist():
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": int(row[5]), "iscrowd": 0,
                                    "bbox": [row[0] - row[2] / 2, row[1] - row[3] / 2, row[2], row[3]], "area": row[2] * row[3]})
            for row, score in zip(noisy.tolist(), scores.tolist()):
                coco_dets.append({"image_id": image_id, "category_id": int(row[5]), "score": score,
                                  "bbox": [row[0] - row[2] / 2, row[1] - row[3] / 2, row[2], row[3]]})
        gt = COCO()
        gt.dataset = {"images": images, "annotations": annotations, "categories": [{"id": c} for c in range(3)]}
        gt.createIndex()
        reference = COCOeval(gt, gt.loadRes(coco_dets), "bbox")
        reference.params.maxDets = [1, 10, 100]
        reference.evaluate()
        reference.accumulate()
        reference.summarize()

        stats = obb_coco_stats(detections, truths, max_dets=100)

        assert stats == pytest.approx(list(reference.stats), abs=1e-6)

    def test_wrong_angle_is_a_miss(self):
        truth = torch.tensor([[200.0, 200, 160, 20, 0.3, 1]])
        rotated = truth.clone()
        rotated[0, 4] += math.pi / 2
        stats = obb_coco_stats([detections_from(rotated)], [(obboxes_to_polygons(truth[:, :5]), truth[:, 5].long())], max_dets=100)
        assert stats[1] == pytest.approx(0.0)


class TestYolo26ObbPostprocess:
    def test_letterbox_is_undone(self):
        """A 512 x 256 image letterboxed to 1024: scale 2, 256 px grey bands top and bottom."""
        metadata = {"original_shape": (256, 512), "input_shape": (1024, 1024), "crop": (256, 0, 512, 1024)}
        output = torch.tensor([[[512.0, 512, 100, 40, 0.9, 3, 0.25]]])

        obboxes, labels, scores = postprocess_output({"output0": output}, metadata)

        assert (obboxes[0].tolist(), labels.tolist(), scores.tolist()) == (
            pytest.approx([256.0, 128.0, 50.0, 20.0, 0.25]), [3], pytest.approx([0.9]))


class TestRfdetrObbPostprocess:
    """RF-DETR OBB exports: dets [1, Q, 5] (cx, cy, w, h normalized, theta) and labels [1, Q, C] logits."""

    def outputs(self):
        dets = torch.tensor([[[0.5, 0.25, 0.1, 0.05, 0.3], [0.2, 0.2, 0.1, 0.1, 0.0]]])
        logits = torch.full((1, 2, 4), -10.0)
        logits[0, 0, 2] = 3.0   # query 0: class 2
        logits[0, 1, 3] = 8.0   # query 1: the no-object slot (column 3) of a 3-class model
        return {"dets": dets, "labels": logits}

    def test_boxes_scale_to_pixels_and_keep_angle(self):
        from sab.models.benchmark_rfdetr_obb import postprocess_output
        obboxes, labels, scores = postprocess_output(self.outputs(), {"original_shape": (200, 200)}, num_classes=3)
        assert (obboxes[0].tolist(), int(labels[0])) == (pytest.approx([100.0, 50.0, 20.0, 10.0, 0.3]), 2)

    def test_columns_past_num_classes_are_not_scored(self):
        from sab.models.benchmark_rfdetr_obb import postprocess_output
        _, labels, _ = postprocess_output(self.outputs(), {"original_shape": (200, 200)}, num_classes=3)
        assert int(labels.max()) < 3

    def test_non_square_image_is_refused(self):
        from sab.models.benchmark_rfdetr_obb import postprocess_output
        with pytest.raises(ValueError, match="square"):
            postprocess_output(self.outputs(), {"original_shape": (200, 300)}, num_classes=3)
