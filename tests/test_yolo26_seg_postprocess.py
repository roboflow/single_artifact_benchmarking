"""The YOLO26 seg adapter: the upstream fused-mask variant and its postprocess.

`fuse_yolo26_mask_postprocessing_into_onnx` (ported from the upstream yolo26_seg
branch, the pipeline behind the published yolo26-seg rows) folds the mask
matmul, the resize to the input grid, and the box crop into the graph, with NO
sigmoid — masks leave the engine as logits at input resolution. The postprocess
undoes the letterbox for boxes (yolov11-seg math), drops the padding from the
masks, resizes to the original image, and binarizes at logit 0.
"""

import numpy as np
import pytest
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from sab.models import benchmark_yolo26_seg as y26seg
from sab.models.graph_surgery import fuse_yolo26_mask_postprocessing_into_onnx
from sab.trt_inference import TRTInference

# A 640x480 source image letterboxed into 640x640: 80 rows of padding top and
# bottom, none left or right.
ORIG_H, ORIG_W = 480, 640
METADATA = {
    "original_shape": (1, 3, ORIG_H, ORIG_W),
    "image_input_shape": (1, 3, 640, 640),
    "scale": 1.0,
    "padding": (0, 80, 0, 80),
}
K = 300


def outputs_for(x1, y1, x2, y2, score, class_id, logit=4.0):
    """Fused-graph outputs for one detection: meta rows plus an input-resolution
    logit mask that is `logit` inside the box and a strong negative outside."""
    det_meta = torch.zeros(1, K, 6)
    det_meta[0, 0] = torch.tensor([x1, y1, x2, y2, score, class_id])
    masks = torch.full((1, K, 640, 640), -9.0)
    masks[0, 0, int(y1) : int(y2), int(x1) : int(x2)] = logit
    return {"_det_meta": det_meta, "_masks_cropped": masks}


def test_boxes_come_back_normalized_to_the_original_image():
    outputs = outputs_for(100, 180, 300, 380, 0.9, 3)
    boxes, labels, scores, _ = y26seg.postprocess_output(outputs, METADATA)
    assert torch.allclose(boxes[0], torch.tensor([100 / 640, 100 / 480, 300 / 640, 300 / 480]))
    assert labels[0] == 3
    assert float(scores[0]) == pytest.approx(0.9)


def test_masks_are_original_resolution_and_cut_at_logit_zero():
    outputs = outputs_for(100, 180, 300, 380, 0.9, 3)
    _, _, _, masks = y26seg.postprocess_output(outputs, METADATA)
    assert masks.shape == (K, ORIG_H, ORIG_W)
    m = masks[0]
    # The positive-logit region maps to (100,100)-(300,300) after the pad drop.
    assert m[200, 200]
    assert not m[50, 50]
    assert not m[400, 500]


def test_negative_logits_everywhere_give_an_empty_mask():
    outputs = outputs_for(100, 180, 300, 380, 0.9, 3, logit=-1.0)
    _, _, _, masks = y26seg.postprocess_output(outputs, METADATA)
    assert not masks[0].any()


def test_the_trt_class_replays_a_cuda_graph_like_every_other_yolo26_row(monkeypatch):
    seen = {}

    def fake_init(self, model_path, image_input_name=None, use_cuda_graph=False, prediction_type="bbox"):
        seen.update(use_cuda_graph=use_cuda_graph, prediction_type=prediction_type)

    monkeypatch.setattr(TRTInference, "__init__", fake_init)
    y26seg.YOLO26SegTRTInference("model.onnx")

    assert seen == {"use_cuda_graph": True, "prediction_type": "segm"}


def test_the_module_rows_use_the_upstream_fused_variant():
    import unittest.mock as mock

    captured = []
    with mock.patch.object(y26seg, "run_benchmark_on_artifacts", side_effect=lambda reqs, *a: captured.extend(reqs) or []):
        with mock.patch.object(y26seg, "pretty_print_results"):
            y26seg.main("images", "annotations.json", output_file_name="/dev/null")

    assert len(captured) == 5
    assert all(r.graph_surgery_func is fuse_yolo26_mask_postprocessing_into_onnx for r in captured)


def _plain_seg_model(path, k=8, coeffs=32, proto=20):
    det = numpy_helper.from_array(np.zeros((1, k, 6 + coeffs), np.float32), name="_det_const")
    pro = numpy_helper.from_array(np.ones((1, coeffs, proto, proto), np.float32), name="_pro_const")
    image = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    out0 = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, k, 6 + coeffs])
    out1 = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, coeffs, proto, proto])
    nodes = [
        helper.make_node("Shape", ["images"], ["_shape_unused"]),
        helper.make_node("Identity", ["_det_const"], ["output0"]),
        helper.make_node("Identity", ["_pro_const"], ["output1"]),
    ]
    graph = helper.make_graph(nodes, "plain_seg", [image], [out0, out1], initializer=[det, pro])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    onnx.save(model, path)
    return path


def test_the_upstream_surgery_emits_input_resolution_logit_masks(tmp_path):
    plain = _plain_seg_model(str(tmp_path / "tiny-seg.onnx"))

    fused_path = fuse_yolo26_mask_postprocessing_into_onnx(plain)

    fused = onnx.load(fused_path)
    assert fused_path.endswith("-fused26.onnx")
    assert [o.name for o in fused.graph.output] == ["_det_meta", "_masks_cropped"]
    mask_out = fused.graph.output[1]
    dims = [d.dim_value for d in mask_out.type.tensor_type.shape.dim]
    assert dims[2:] == [640, 640], f"masks not at input resolution: {dims}"
    op_types = [node.op_type for node in fused.graph.node]
    assert "Resize" in op_types
    assert "Sigmoid" not in op_types, "the upstream variant keeps masks as logits"
