import onnx
import pytest
import torch
from onnx import TensorProto, helper

pytest.importorskip("onnxruntime")

from sab.runtimes.onnxruntime import ONNXRuntime  # noqa: E402

pytestmark = pytest.mark.onnxruntime


def save_model(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def make_single_input_model(path, batch="batch"):
    """outputs = images * 2, with a symbolic batch dimension by default."""
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [batch, 3, 4, 4])
    outputs = helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [batch, 3, 4, 4])
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    node = helper.make_node("Mul", ["images", "two"], ["outputs"])
    return save_model(helper.make_graph([node], "single", [images], [outputs], [two]), path)


def make_two_input_model(path, image_name="images"):
    """boxes = images * 2 and labels = orig_target_sizes + 1, with an int64 second input."""
    images = helper.make_tensor_value_info(image_name, TensorProto.FLOAT, [1, 3, 4, 4])
    sizes = helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, [1, 2])
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3, 4, 4])
    labels = helper.make_tensor_value_info("labels", TensorProto.INT64, [1, 2])
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    one = helper.make_tensor("one", TensorProto.INT64, [], [1])
    nodes = [
        helper.make_node("Mul", [image_name, "two"], ["boxes"]),
        helper.make_node("Add", ["orig_target_sizes", "one"], ["labels"]),
    ]
    graph = helper.make_graph(nodes, "double", [images, sizes], [boxes, labels], [two, one])
    return save_model(graph, path)


def make_dynamic_output_model(path):
    """outputs = NonZero(images): the output shape depends on the values."""
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])
    outputs = helper.make_tensor_value_info("indices", TensorProto.INT64, [4, None])
    node = helper.make_node("NonZero", ["images"], ["indices"])
    return save_model(helper.make_graph([node], "nonzero", [images], [outputs]), path)


def cpu_runtime(path, **kwargs):
    return ONNXRuntime(path, "cpu", "fp32", **kwargs)


def test_is_available_on_cpu_and_on_gpu_only_with_the_cuda_provider_and_a_cuda_device(monkeypatch):
    import onnxruntime as ort

    assert ONNXRuntime.is_available("cpu")
    assert not ONNXRuntime.is_available("npu")

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert not ONNXRuntime.is_available("gpu")

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not ONNXRuntime.is_available("gpu")


def test_input_spec_replaces_a_symbolic_batch_with_one_and_keeps_a_static_one(tmp_path):
    symbolic = cpu_runtime(make_single_input_model(tmp_path / "symbolic.onnx"))
    static = cpu_runtime(make_single_input_model(tmp_path / "static.onnx", batch=2))

    assert symbolic.input_spec.name == "images"
    assert symbolic.input_spec.shape == (1, 3, 4, 4)
    assert static.input_spec.shape == (2, 3, 4, 4)


def test_run_records_one_timing_per_call_after_warmup(tmp_path):
    runtime = cpu_runtime(make_single_input_model(tmp_path / "m.onnx"))
    assert runtime.profiler.timings == []

    for expected in (1, 2, 3):
        runtime.run({"images": torch.rand(1, 3, 4, 4)})
        assert len(runtime.profiler.timings) == expected


def test_image_input_defaults_to_images_when_the_model_has_several_inputs(tmp_path):
    runtime = cpu_runtime(make_two_input_model(tmp_path / "m.onnx"))
    assert runtime.input_spec.name == "images"


def test_image_input_name_overrides_the_default(tmp_path):
    path = make_two_input_model(tmp_path / "m.onnx", image_name="pixels")
    runtime = cpu_runtime(path, image_input_name="pixels")
    assert runtime.input_spec.name == "pixels"
    assert runtime.input_spec.shape == (1, 3, 4, 4)


def test_image_input_name_is_required_when_ambiguous_and_must_exist(tmp_path):
    with pytest.raises(ValueError, match="image_input_name"):
        cpu_runtime(make_two_input_model(tmp_path / "two.onnx", image_name="pixels"))
    with pytest.raises(ValueError, match="nope"):
        cpu_runtime(make_single_input_model(tmp_path / "one.onnx"), image_input_name="nope")


def test_run_binds_every_input_with_its_own_dtype(tmp_path):
    runtime = cpu_runtime(make_two_input_model(tmp_path / "m.onnx"))
    images = torch.rand(1, 3, 4, 4)
    sizes = torch.tensor([[10, 20]], dtype=torch.int64)

    outputs = runtime.run({"images": images, "orig_target_sizes": sizes})

    torch.testing.assert_close(outputs["boxes"], images * 2)
    assert outputs["labels"].dtype == torch.int64
    assert outputs["labels"].tolist() == [[11, 21]]


def test_dynamic_output_shapes_follow_the_values_of_each_call(tmp_path):
    runtime = cpu_runtime(make_dynamic_output_model(tmp_path / "m.onnx"), dynamic_output_shapes=True)
    images = torch.zeros(1, 3, 4, 4)
    images[0, 1, 2, 3] = 1.0
    images[0, 2, 0, 0] = 1.0

    two = runtime.run({"images": images})["indices"]
    none = runtime.run({"images": torch.zeros(1, 3, 4, 4)})["indices"]

    assert torch.equal(two, torch.nonzero(images).T)
    assert none.shape == (4, 0)
    assert len(runtime.profiler.timings) == 2


def test_outputs_do_not_share_memory_between_calls(tmp_path):
    runtime = cpu_runtime(make_single_input_model(tmp_path / "m.onnx"))

    first = runtime.run({"images": torch.ones(1, 3, 4, 4)})["outputs"]
    runtime.run({"images": torch.zeros(1, 3, 4, 4)})

    assert first.eq(2).all()
