"""The YOLO26 seg adapter: what it reads off a plain export, and what it returns.

YOLO26 is NMS-free, so nothing is folded into the graph. The mask matmul, the
letterbox undo, and the box crop all happen here, on the outputs of the plain
ultralytics export.
"""

import pytest
import torch

from sab.models.benchmark_yolo26_seg import (
    YOLO26SegTRTInference,
    postprocess_output,
)
from sab.trt_inference import TRTInference

NUM_COEFFS = 32
PROTO_SIZE = 160

# A 640x480 source image, letterboxed into a 640x640 input: 80 rows of padding
# top and bottom, none left or right.
ORIG_H, ORIG_W = 480, 640
METADATA = {
    "original_shape": (1, 3, ORIG_H, ORIG_W),
    "image_input_shape": (1, 3, 640, 640),
    "scale": 1.0,
    "padding": (0, 80, 0, 80),
}


def detection(x1, y1, x2, y2, score, class_id):
    """One (38,) row: xyxy in input-image pixels, score, class, then mask coeffs."""
    row = torch.zeros(6 + NUM_COEFFS)
    row[:6] = torch.tensor([x1, y1, x2, y2, score, class_id])
    # Uniform coefficients against all-ones prototypes give a mask of 1.0
    # everywhere, so the only thing left shaping it is the box crop.
    row[6:] = 1.0 / NUM_COEFFS
    return row


def outputs_for(*rows):
    return {
        "output0": torch.stack(rows).unsqueeze(0),
        "output1": torch.ones(1, NUM_COEFFS, PROTO_SIZE, PROTO_SIZE),
    }


def test_boxes_come_back_normalized_to_the_original_image():
    # In the 640x640 letterboxed frame the box sits at (100,180)-(300,380).
    # Dropping the 80-row pad puts it at (100,100)-(300,300) on the 640x480
    # source, and sab.evaluation reads that as a fraction of the source size.
    boxes, _, _, _ = postprocess_output(outputs_for(detection(100, 180, 300, 380, 0.9, 3)), METADATA)

    assert boxes.shape == (1, 4)
    assert torch.allclose(boxes[0], torch.tensor([100 / 640, 100 / 480, 300 / 640, 300 / 480]))


def test_labels_and_scores_survive_the_row_split():
    _, labels, scores, _ = postprocess_output(outputs_for(detection(10, 90, 50, 130, 0.75, 17)), METADATA)

    assert labels.dtype == torch.int64
    assert labels.tolist() == [17]
    assert scores.tolist() == [0.75]


def test_masks_are_original_resolution_and_cropped_to_their_box():
    _, _, _, masks = postprocess_output(outputs_for(detection(100, 180, 300, 380, 0.9, 3)), METADATA)

    assert masks.shape == (1, ORIG_H, ORIG_W)
    assert masks.dtype == torch.bool
    # Positive everywhere before the crop, so what is left is exactly the box.
    assert masks[0, 100:300, 100:300].all()
    assert int(masks.sum()) == 200 * 200


def test_zero_score_rows_are_dropped():
    # A NMS-free export pads its 300 slots with zero-score rows.
    outputs = outputs_for(
        detection(100, 180, 300, 380, 0.9, 3),
        detection(0, 0, 0, 0, 0.0, 0),
    )
    boxes, labels, scores, masks = postprocess_output(outputs, METADATA)

    assert len(boxes) == len(labels) == len(scores) == len(masks) == 1
    assert scores.tolist() == pytest.approx([0.9])


def test_an_empty_detection_set_returns_empty_tensors():
    boxes, labels, scores, masks = postprocess_output(outputs_for(detection(0, 0, 0, 0, 0.0, 0)), METADATA)

    assert boxes.shape == (0, 4)
    assert labels.shape == (0,)
    assert scores.shape == (0,)
    assert masks.shape == (0, ORIG_H, ORIG_W)


def test_a_channels_first_output0_is_transposed_back():
    # Some exports emit (1, 38, 300) instead of (1, 300, 38).
    outputs = outputs_for(detection(100, 180, 300, 380, 0.9, 3))
    outputs["output0"] = outputs["output0"].transpose(1, 2).contiguous()

    boxes, labels, _, _ = postprocess_output(outputs, METADATA)

    assert labels.tolist() == [3]
    assert torch.allclose(boxes[0], torch.tensor([100 / 640, 100 / 480, 300 / 640, 300 / 480]))


def test_the_trt_class_replays_a_cuda_graph_like_every_other_yolo26_row(monkeypatch):
    # b388156 settled this for yolo26: TRT rows are timed as graph replay. The
    # yolo26_seg branch inherited use_cuda_graph=False from the yolov11 seg
    # copy it started as.
    captured = {}

    def fake_init(self, model_path, image_input_name=None, use_cuda_graph=True, prediction_type="bbox"):
        captured.update(use_cuda_graph=use_cuda_graph, prediction_type=prediction_type)

    monkeypatch.setattr(TRTInference, "__init__", fake_init)
    YOLO26SegTRTInference("yolo26n-seg.engine")

    assert captured["use_cuda_graph"] is True
    assert captured["prediction_type"] == "segm"
