import sys

import numpy as np
import pytest
import torch

ct = pytest.importorskip("coremltools")
if sys.platform != "darwin":
    pytest.skip("Core ML runs only on macOS", allow_module_level=True)

from sab.runtimes.coreml import CoreMLRuntime  # noqa: E402

pytestmark = pytest.mark.coreml

SIDE = 8


class Double(torch.nn.Module):
    def forward(self, x):
        return x * 2


class ImageAndSizes(torch.nn.Module):
    def forward(self, image, sizes):
        return image * 1.0, sizes + 1


def convert(module, inputs, outputs, path, **options):
    example_inputs = [torch.zeros(tuple(input.shape.to_list())) for input in inputs]
    traced = torch.jit.trace(module.eval(), example_inputs)
    model = ct.convert(traced, inputs=inputs, outputs=outputs, convert_to="mlprogram", **options)
    model.save(str(path))
    return str(path)


def tensor_type(name, shape=(1, 3, SIDE, SIDE), dtype=np.float32):
    return ct.TensorType(name=name, shape=shape, dtype=dtype)


def image_type(name="image"):
    return ct.ImageType(name=name, shape=(1, 3, SIDE, SIDE), scale=1 / 255)


@pytest.fixture(scope="session")
def double_model(tmp_path_factory):
    return convert(
        Double(),
        [tensor_type("images")],
        [ct.TensorType(name="outputs")],
        tmp_path_factory.mktemp("double") / "x.mlpackage",
        compute_precision=ct.precision.FLOAT32,
    )


@pytest.fixture(scope="session")
def image_model(tmp_path_factory):
    return convert(
        Double(),
        [image_type()],
        [ct.TensorType(name="outputs")],
        tmp_path_factory.mktemp("image") / "x.mlpackage",
        compute_precision=ct.precision.FLOAT32,
    )


@pytest.fixture(scope="session")
def image_and_sizes_model(tmp_path_factory):
    return convert(
        ImageAndSizes(),
        [image_type(), tensor_type("orig_target_sizes", shape=(1, 2), dtype=np.int32)],
        [ct.TensorType(name="boxes"), ct.TensorType(name="labels")],
        tmp_path_factory.mktemp("two") / "x.mlpackage",
        compute_precision=ct.precision.FLOAT32,
    )


@pytest.fixture(scope="session")
def float16_model(tmp_path_factory):
    return convert(
        Double(),
        [tensor_type("images", dtype=np.float16)],
        [ct.TensorType(name="outputs")],
        tmp_path_factory.mktemp("half") / "x.mlpackage",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )


def test_run_gives_numerically_correct_outputs_that_own_their_memory(double_model):
    runtime = CoreMLRuntime(double_model, "cpu", "fp32")
    images = torch.rand(1, 3, SIDE, SIDE)

    first = runtime.run({"images": images})["outputs"]
    runtime.run({"images": torch.zeros(1, 3, SIDE, SIDE)})

    torch.testing.assert_close(first, images * 2)


@pytest.mark.parametrize(
    "device, expected",
    [("cpu", "CPU_ONLY"), ("gpu", "CPU_AND_GPU"), ("npu", "CPU_AND_NE")],
)
def test_each_device_maps_to_its_compute_unit(double_model, device, expected):
    runtime = CoreMLRuntime(double_model, device, "fp32")

    assert runtime.model.compute_unit == ct.ComputeUnit[expected]


def test_image_input_takes_a_0_255_nchw_tensor(image_model):
    runtime = CoreMLRuntime(image_model, "cpu", "fp32")
    pixels = torch.rand(1, 3, SIDE, SIDE) * 255

    outputs = runtime.run({"image": pixels})["outputs"]

    assert runtime.input_spec.shape == (1, 3, SIDE, SIDE)
    torch.testing.assert_close(outputs, pixels * 2 / 255, atol=1 / 255 + 1e-5, rtol=0)


def test_extra_int32_input_reaches_the_model_with_its_values(image_and_sizes_model):
    runtime = CoreMLRuntime(image_and_sizes_model, "cpu", "fp32", image_input_name="image")
    pixels = torch.rand(1, 3, SIDE, SIDE) * 255
    sizes = torch.tensor([[640, 480]], dtype=torch.int64)

    outputs = runtime.run({"image": pixels, "orig_target_sizes": sizes})

    assert runtime.input_spec.name == "image"
    torch.testing.assert_close(outputs["boxes"], pixels / 255, atol=1 / 255 + 1e-5, rtol=0)
    assert outputs["labels"].tolist() == [[641, 481]]


def test_float16_input_takes_a_float32_tensor(float16_model):
    runtime = CoreMLRuntime(float16_model, "cpu", "fp16")
    images = torch.rand(1, 3, SIDE, SIDE)

    outputs = runtime.run({"images": images})["outputs"]

    torch.testing.assert_close(outputs.float(), images * 2, atol=1e-2, rtol=1e-2)


def test_warmup_leaves_no_timings_and_each_run_adds_one(double_model):
    runtime = CoreMLRuntime(double_model, "cpu", "fp32")
    assert runtime.profiler.timings == []

    for expected in (1, 2, 3):
        runtime.run({"images": torch.rand(1, 3, SIDE, SIDE)})
        assert len(runtime.profiler.timings) == expected


def test_is_available_is_false_for_an_unknown_device_and_off_macos(monkeypatch):
    assert CoreMLRuntime.is_available("npu")
    assert not CoreMLRuntime.is_available("cuda")

    monkeypatch.setattr(sys, "platform", "linux")
    assert not CoreMLRuntime.is_available("cpu")
