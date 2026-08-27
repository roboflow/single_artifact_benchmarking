"""The YOLO26 seg adapter: fused mask surgery plus the shared YOLOv11 seg path.

The adapter reuses `benchmark_yolov11_seg`'s preprocess and postprocess, exactly
as the OD path reuses `benchmark_yolov11`'s. The mask matmul, sigmoid, and box
crop run inside the graph via `fuse_yolo_mask_postprocessing_into_onnx` — the
same surgery, and the same timed region, as the yolov8/yolov11 seg rows behind
the published numbers.
"""

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from sab.models import benchmark_yolo26_seg as y26seg
from sab.models import benchmark_yolov11_seg as y11seg
from sab.models.graph_surgery import fuse_yolo_mask_postprocessing_into_onnx
from sab.trt_inference import TRTInference

NUM_COEFFS = 32
PROTO_SIZE = 160


def test_the_adapter_reuses_the_yolov11_seg_path():
    assert y26seg.postprocess_output is y11seg.postprocess_output
    assert y26seg.preprocess_image is y11seg.preprocess_image


def test_the_module_rows_fold_the_mask_surgery_into_the_graph():
    import unittest.mock as mock

    captured = []
    with mock.patch.object(y26seg, "run_benchmark_on_artifacts", side_effect=lambda reqs, *a: captured.extend(reqs) or []):
        with mock.patch.object(y26seg, "pretty_print_results"):
            y26seg.main("images", "annotations.json", output_file_name="/dev/null")

    assert len(captured) == 5
    assert all(request.graph_surgery_func is fuse_yolo_mask_postprocessing_into_onnx for request in captured)
    assert all(request.needs_class_remapping for request in captured)


def test_the_trt_class_replays_a_cuda_graph_like_every_other_yolo26_row(monkeypatch):
    seen = {}

    def fake_init(self, model_path, image_input_name=None, use_cuda_graph=False, prediction_type="bbox"):
        seen.update(use_cuda_graph=use_cuda_graph, prediction_type=prediction_type)

    monkeypatch.setattr(TRTInference, "__init__", fake_init)
    y26seg.YOLO26SegTRTInference("model.onnx")

    assert seen == {"use_cuda_graph": True, "prediction_type": "segm"}


def _plain_seg_model(path, k=8, coeffs=NUM_COEFFS, proto=20):
    """A minimal graph with the plain seg export's output contract: constant
    detections (1,k,6+coeffs) and prototypes (1,coeffs,proto,proto)."""
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


def test_the_surgery_rewrites_a_plain_export_to_the_fused_contract(tmp_path):
    plain = _plain_seg_model(str(tmp_path / "tiny-seg.onnx"))

    fused_path = fuse_yolo_mask_postprocessing_into_onnx(plain)

    fused = onnx.load(fused_path)
    output_names = [o.name for o in fused.graph.output]
    assert output_names == ["_det_meta", "_masks_cropped"]
    op_types = {node.op_type for node in fused.graph.node}
    # The mask assembly the published numbers time inside the engine.
    assert {"Split", "MatMul", "Sigmoid"} <= op_types
