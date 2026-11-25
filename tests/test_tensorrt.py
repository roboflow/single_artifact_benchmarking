import sys

import pytest
import torch

from sab import hosts
from sab.runtimes import tensorrt
from sab.runtimes.tensorrt import TRTRuntime, builder_flag_names, engine_path_for
from tests.fakes import make_two_input_model


def requires_tensorrt(test):
    test = pytest.mark.tensorrt(test)
    return pytest.mark.skipif(not TRTRuntime.is_available("gpu"), reason="needs TensorRT and an NVIDIA GPU")(test)


@pytest.mark.parametrize(
    "artifact_path, precision, engine_path",
    [
        ("cache/model.onnx", "fp32", "cache/model.trt10.16.1.11.engine"),
        ("cache/model.onnx", "fp16", "cache/model.trt10.16.1.11.fp16.engine"),
        ("cache/model.onnx", "int8", "cache/model.trt10.16.1.11.int8.engine"),
        ("dfine_n_coco.opset17.onnx", "fp16", "dfine_n_coco.opset17.trt10.16.1.11.fp16.engine"),
    ],
)
def test_engine_path_names_the_tensorrt_version_and_the_precision(artifact_path, precision, engine_path):
    assert engine_path_for(artifact_path, precision, "10.16.1.11") == engine_path


def test_engine_path_rejects_a_file_that_is_not_onnx_and_an_unknown_precision():
    with pytest.raises(ValueError, match="onnx"):
        engine_path_for("model.tflite", "fp32", "10.16.1.11")
    with pytest.raises(ValueError, match="bf16"):
        engine_path_for("model.onnx", "bf16", "10.16.1.11")


def test_builder_flags_per_precision():
    assert builder_flag_names("fp32") == ()
    assert builder_flag_names("fp16") == ("FP16",)
    assert set(builder_flag_names("int8")) == {"INT8", "FP16"}


@pytest.mark.parametrize(
    "tensorrt_module, cuda, expected",
    [(None, True, False), (object(), False, False), (object(), True, True)],
)
def test_is_available_needs_tensorrt_and_a_cuda_device(monkeypatch, tensorrt_module, cuda, expected):
    monkeypatch.setitem(sys.modules, "tensorrt", tensorrt_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert TRTRuntime.is_available("gpu") is expected


def test_rejects_a_device_other_than_gpu():
    with pytest.raises(ValueError, match="cpu"):
        TRTRuntime("model.onnx", "cpu", "fp32")


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
    import tensorrt

    assert (tmp_path / f"m.trt{tensorrt.__version__}.engine").exists()


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


class RecordingMonitor:
    def __init__(self):
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        return None

    def did_throttle(self):
        return None


def test_an_engine_build_uses_the_throttle_monitor_of_the_host(monkeypatch, tmp_path):
    """A Jetson without NVML must build an engine, so the build must not go to NVML directly."""
    monitor = RecordingMonitor()
    selected = []

    def select_monitor(runtime, device):
        selected.append((runtime, device))
        return monitor

    monkeypatch.setattr(hosts, "select_monitor", select_monitor)
    monkeypatch.setattr(tensorrt, "build_engine", lambda artifact_path, engine_path, precision: object())

    artifact = str(tmp_path / "model.onnx")
    engine_path = tensorrt._ensure_engine(artifact, "fp16", "10.16.1")

    assert selected == [(TRTRuntime, "gpu")]
    assert monitor.entered
    assert engine_path == engine_path_for(artifact, "fp16", "10.16.1")
