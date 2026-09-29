import sys

import pytest
import torch

from sab.runtimes.tensorrt import TRTRuntime, builder_flag_names, engine_path_for


def make_two_input_model(path):
    from tests.test_onnxruntime import make_two_input_model as make_model

    return make_model(path)


def requires_tensorrt(test):
    test = pytest.mark.tensorrt(test)
    return pytest.mark.skipif(not TRTRuntime.is_available("gpu"), reason="needs TensorRT and an NVIDIA GPU")(test)


def test_declares_name_and_devices():
    assert TRTRuntime.name == "tensorrt"
    assert TRTRuntime.devices == frozenset({"gpu"})


@pytest.mark.parametrize(
    "precision, engine_name",
    [
        ("fp32", "model.engine"),
        ("fp16", "model.fp16.engine"),
        ("int8", "model.int8.engine"),
    ],
)
def test_engine_path_names_the_precision(precision, engine_name):
    assert engine_path_for("cache/model.onnx", precision) == f"cache/{engine_name}"


def test_engine_path_keeps_dots_in_the_model_name():
    assert engine_path_for("dfine_n_coco.opset17.onnx", "fp32") == "dfine_n_coco.opset17.engine"
    assert engine_path_for("dfine_n_coco.opset17.onnx", "fp16") == "dfine_n_coco.opset17.fp16.engine"


def test_engine_path_replaces_only_the_suffix():
    assert engine_path_for("a.onnx.dir/model.onnx", "fp32") == "a.onnx.dir/model.engine"


def test_engine_path_rejects_a_file_that_is_not_onnx():
    with pytest.raises(ValueError, match="onnx"):
        engine_path_for("model.tflite", "fp32")


def test_engine_path_rejects_an_unknown_precision():
    with pytest.raises(ValueError, match="bf16"):
        engine_path_for("model.onnx", "bf16")


def test_fp32_sets_no_builder_flag():
    assert builder_flag_names("fp32") == ()


def test_fp16_sets_the_fp16_flag():
    assert builder_flag_names("fp16") == ("FP16",)


def test_int8_sets_the_int8_and_fp16_flags():
    assert set(builder_flag_names("int8")) == {"INT8", "FP16"}


def test_is_unavailable_when_tensorrt_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert not TRTRuntime.is_available("gpu")


def test_is_unavailable_without_a_cuda_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not TRTRuntime.is_available("gpu")


def test_is_available_with_tensorrt_and_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert TRTRuntime.is_available("gpu")


def test_is_unavailable_on_other_devices(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert not TRTRuntime.is_available("cpu")


def test_rejects_a_device_other_than_gpu():
    with pytest.raises(ValueError, match="cpu"):
        TRTRuntime("model.onnx", "cpu", "fp32")


def test_version_names_the_library(monkeypatch):
    class FakeTensorRT:
        __version__ = "10.4.0"

    monkeypatch.setitem(sys.modules, "tensorrt", FakeTensorRT)
    assert TRTRuntime.version() == "tensorrt 10.4.0"


@requires_tensorrt
@pytest.mark.parametrize("use_cuda_graph", [True, False])
def test_gpu_run_matches_the_model(tmp_path, use_cuda_graph):
    path = make_two_input_model(tmp_path / "m.onnx")
    runtime = TRTRuntime(path, "gpu", "fp32", use_cuda_graph=use_cuda_graph)
    images = torch.rand(1, 3, 4, 4, device="cuda")
    sizes = torch.tensor([[10, 20]], dtype=torch.int64, device="cuda")

    for _ in range(3):
        outputs = runtime.run({"images": images, "orig_target_sizes": sizes})

    torch.testing.assert_close(outputs["boxes"], images * 2)
    assert outputs["labels"].tolist() == [[11, 21]]
    assert (tmp_path / "m.engine").exists()


@requires_tensorrt
def test_gpu_graph_capture_sample_is_not_recorded(tmp_path):
    runtime = TRTRuntime(make_two_input_model(tmp_path / "m.onnx"), "gpu", "fp32")
    inputs = {
        "images": torch.rand(1, 3, 4, 4, device="cuda"),
        "orig_target_sizes": torch.ones(1, 2, dtype=torch.int64, device="cuda"),
    }

    for _ in range(3):
        runtime.run(inputs)

    assert len(runtime.profiler.timings) == 2


@requires_tensorrt
def test_gpu_rejects_an_input_the_engine_does_not_have(tmp_path):
    runtime = TRTRuntime(make_two_input_model(tmp_path / "m.onnx"), "gpu", "fp32")
    with pytest.raises(ValueError, match="nope"):
        runtime.run({"images": torch.rand(1, 3, 4, 4, device="cuda"), "nope": torch.ones(1, device="cuda")})
