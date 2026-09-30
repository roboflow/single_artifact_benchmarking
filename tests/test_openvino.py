import os

import pytest
import torch

ov = pytest.importorskip("openvino")
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from sab.runtimes.base import UnavailableOnHost  # noqa: E402
from sab.runtimes.openvino import OpenVINORuntime  # noqa: E402

pytestmark = pytest.mark.openvino


def cpu_has_native_fp16() -> bool:
    return "FP16" in ov.Core().get_property("CPU", "OPTIMIZATION_CAPABILITIES")


needs_native_fp16 = pytest.mark.skipif(not cpu_has_native_fp16(), reason="this CPU compiles f16 as f32")


def convert(graph, directory, name="model"):
    """Convert an ONNX graph to an OpenVINO `.xml` + `.bin` pair and return the `.xml` path."""
    directory.mkdir(exist_ok=True)
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(onnx_model)
    onnx_path = directory / f"{name}.onnx"
    onnx.save(onnx_model, str(onnx_path))
    xml_path = directory / f"{name}.xml"
    ov.save_model(ov.convert_model(str(onnx_path)), str(xml_path))
    return xml_path


def double_graph(shape=(1, 3, 4, 4)):
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, list(shape))
    outputs = helper.make_tensor_value_info("outputs", TensorProto.FLOAT, list(shape))
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    node = helper.make_node("Mul", ["images", "two"], ["outputs"])
    return helper.make_graph([node], "double", [images], [outputs], [two])


def two_input_graph():
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])
    sizes = helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, [1, 2])
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3, 4, 4])
    labels = helper.make_tensor_value_info("labels", TensorProto.INT64, [1, 2])
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    one = helper.make_tensor("one", TensorProto.INT64, [], [1])
    nodes = [
        helper.make_node("Mul", ["images", "two"], ["boxes"]),
        helper.make_node("Add", ["orig_target_sizes", "one"], ["labels"]),
    ]
    return helper.make_graph(nodes, "two_inputs", [images, sizes], [boxes, labels], [two, one])


def dynamic_double_graph():
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", "channels", "height", "width"])
    outputs = helper.make_tensor_value_info("outputs", TensorProto.FLOAT, ["batch", "channels", "height", "width"])
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    node = helper.make_node("Mul", ["images", "two"], ["outputs"])
    return helper.make_graph([node], "dynamic_double", [images], [outputs], [two])


def test_input_shape_makes_a_dynamic_image_input_static(tmp_path):
    runtime = OpenVINORuntime(str(convert(dynamic_double_graph(), tmp_path)), "cpu", "fp32", input_shape=(1, 3, 4, 4))
    images = torch.rand(1, 3, 4, 4)

    outputs = runtime.run({"images": images})["outputs"]

    assert runtime.input_spec.shape == (1, 3, 4, 4)
    torch.testing.assert_close(outputs, images * 2)


def test_a_dynamic_image_input_without_input_shape_fails_at_load(tmp_path):
    with pytest.raises(ValueError, match="input_shape"):
        OpenVINORuntime(str(convert(dynamic_double_graph(), tmp_path)), "cpu", "fp32")


def test_is_available_only_with_the_cpu_plugin(monkeypatch):
    assert OpenVINORuntime.is_available("cpu")

    monkeypatch.setattr(ov.Core, "available_devices", property(lambda self: ["GPU"]))
    assert not OpenVINORuntime.is_available("cpu")


def test_run_gives_numerically_correct_outputs_that_own_their_memory(tmp_path):
    runtime = OpenVINORuntime(str(convert(double_graph(), tmp_path)), "cpu", "fp32")
    images = torch.rand(1, 3, 4, 4)

    first = runtime.run({"images": images})["outputs"]
    runtime.run({"images": torch.zeros(1, 3, 4, 4)})

    torch.testing.assert_close(first, images * 2)


def test_a_directory_must_hold_exactly_one_xml_file(tmp_path):
    convert(double_graph(), tmp_path / "one", name="yolo")
    images = torch.rand(1, 3, 4, 4)
    runtime = OpenVINORuntime(str(tmp_path / "one"), "cpu", "fp32")
    torch.testing.assert_close(runtime.run({"images": images})["outputs"], images * 2)

    (tmp_path / "none").mkdir()
    with pytest.raises(FileNotFoundError):
        OpenVINORuntime(str(tmp_path / "none"), "cpu", "fp32")

    convert(double_graph(), tmp_path / "two", name="a")
    convert(double_graph(), tmp_path / "two", name="b")
    with pytest.raises(ValueError, match="several"):
        OpenVINORuntime(str(tmp_path / "two"), "cpu", "fp32")


@pytest.mark.parametrize(
    "precision, expected",
    [("fp32", ov.Type.f32), ("int8", ov.Type.f32), pytest.param("fp16", ov.Type.f16, marks=needs_native_fp16)],
)
def test_compiles_with_the_inference_precision_hint(tmp_path, precision, expected):
    runtime = OpenVINORuntime(str(convert(double_graph(), tmp_path)), "cpu", precision)

    hint = runtime.compiled_model.get_property(ov.properties.hint.inference_precision())

    assert hint == expected


def test_uses_every_logical_cpu(tmp_path):
    runtime = OpenVINORuntime(str(convert(double_graph(), tmp_path)), "cpu", "fp32")

    threads = runtime.compiled_model.get_property(ov.properties.inference_num_threads())

    assert threads == os.cpu_count()


def test_fp16_on_a_cpu_without_native_fp16_is_unavailable_on_host(tmp_path):
    if cpu_has_native_fp16():
        pytest.skip("this CPU runs f16 natively")
    xml_path = convert(double_graph(), tmp_path)

    with pytest.raises(UnavailableOnHost):
        OpenVINORuntime(str(xml_path), "cpu", "fp16")


def test_extra_int64_input_reaches_the_model_as_int64(tmp_path):
    runtime = OpenVINORuntime(str(convert(two_input_graph(), tmp_path)), "cpu", "fp32")
    images = torch.rand(1, 3, 4, 4)
    sizes = torch.tensor([[640, 480]], dtype=torch.int64)

    outputs = runtime.run({"images": images, "orig_target_sizes": sizes})

    assert runtime.input_spec.name == "images"
    torch.testing.assert_close(outputs["boxes"], images * 2)
    assert outputs["labels"].tolist() == [[641, 481]]


def test_warmup_leaves_no_timings_and_each_run_adds_one(tmp_path):
    runtime = OpenVINORuntime(str(convert(two_input_graph(), tmp_path)), "cpu", "fp32")
    assert runtime.profiler.timings == []

    inputs = {"images": torch.rand(1, 3, 4, 4), "orig_target_sizes": torch.ones(1, 2, dtype=torch.int64)}
    for expected in (1, 2, 3):
        runtime.run(inputs)
        assert len(runtime.profiler.timings) == expected
