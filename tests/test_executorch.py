import pytest
import torch

pytest.importorskip("executorch.runtime")

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner  # noqa: E402
from executorch.exir import to_edge_transform_and_lower  # noqa: E402

from sab.runtimes import executorch as executorch_runtime  # noqa: E402
from sab.runtimes.executorch import ExecuTorchRuntime  # noqa: E402

pytestmark = pytest.mark.executorch


class ConvModel(torch.nn.Module):
    """One NCHW image input."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, images):
        return self.conv(images)


class ImageAndSizesModel(ConvModel):
    """Image first, then an int64 input, like the `orig_target_sizes` input of some detectors."""

    def forward(self, images, sizes):
        return self.conv(images), sizes + 1


class SizesAndImageModel(ConvModel):
    def forward(self, sizes, images):
        return sizes + 1, self.conv(images)


CASES = {
    "conv": (ConvModel, (torch.rand(1, 3, 8, 8),)),
    "image_and_sizes": (ImageAndSizesModel, (torch.rand(1, 3, 8, 8), torch.tensor([[10, 20]]))),
    "sizes_and_image": (SizesAndImageModel, (torch.tensor([[10, 20]]), torch.rand(1, 3, 8, 8))),
}


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("pte")


@pytest.fixture(scope="session")
def build_pte(model_dir):
    """Export each tiny model once per session. Returns (module, example inputs, path)."""
    built = {}

    def build(case: str):
        if case not in built:
            torch.manual_seed(0)
            model_class, example_inputs = CASES[case]
            module = model_class().eval()
            exported = torch.export.export(module, example_inputs)
            program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
            path = model_dir / f"{case}.pte"
            path.write_bytes(program.buffer)
            built[case] = (module, example_inputs, str(path))
        return built[case]

    return build


@pytest.mark.parametrize(
    "registered, cpu, npu",
    [({"XnnpackBackend"}, True, False), ({"XnnpackBackend", "CoreMLBackend"}, True, True), (set(), False, False)],
)
def test_each_device_needs_its_backend_in_the_installed_executorch(monkeypatch, registered, cpu, npu):
    monkeypatch.setattr(executorch_runtime, "_registered_backends", lambda: registered)

    assert ExecuTorchRuntime.is_available("cpu") is cpu
    assert ExecuTorchRuntime.is_available("npu") is npu
    assert not ExecuTorchRuntime.is_available("gpu")


@pytest.mark.skipif(not ExecuTorchRuntime.is_available("npu"), reason="needs the ExecuTorch Core ML backend")
def test_npu_runs_a_core_ml_program(model_dir):
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner

    torch.manual_seed(0)
    module = ConvModel().eval()
    compile_specs = CoreMLBackend.generate_compile_specs(compute_unit=ct.ComputeUnit.CPU_AND_NE)
    exported = torch.export.export(module, (torch.rand(1, 3, 8, 8),))
    program = to_edge_transform_and_lower(exported, partitioner=[CoreMLPartitioner(compile_specs=compile_specs)])
    path = model_dir / "conv_coreml.pte"
    path.write_bytes(program.to_executorch().buffer)
    runtime = ExecuTorchRuntime(str(path), "npu", "fp16")
    images = torch.rand(1, 3, 8, 8)

    outputs = runtime.run({"input0": images})

    torch.testing.assert_close(outputs["output0"], module(images), atol=1e-2, rtol=1e-2)


def test_run_matches_the_torch_model_and_outputs_own_their_memory(build_pte):
    module, _, path = build_pte("conv")
    runtime = ExecuTorchRuntime(path, "cpu", "fp32")
    images = torch.rand(1, 3, 8, 8)

    first = runtime.run({"input0": images})
    runtime.run({"input0": torch.rand(1, 3, 8, 8)})

    assert runtime.input_spec.name == "input0"
    assert set(first) == {"output0"}
    torch.testing.assert_close(first["output0"], module(images), atol=1e-4, rtol=1e-4)


def test_run_passes_inputs_in_positional_order(build_pte):
    module, _, path = build_pte("image_and_sizes")
    runtime = ExecuTorchRuntime(path, "cpu", "fp32")
    images = torch.rand(1, 3, 8, 8)
    sizes = torch.tensor([[10, 20]])

    # The dict order is the reverse of the positional order.
    outputs = runtime.run({"input1": sizes, "input0": images})

    assert runtime.input_spec.name == "input0"
    torch.testing.assert_close(outputs["output0"], module(images, sizes)[0], atol=1e-4, rtol=1e-4)
    assert outputs["output1"].tolist() == [[11.0, 21.0]]


def test_run_finds_the_image_at_a_later_position(build_pte):
    module, _, path = build_pte("sizes_and_image")
    runtime = ExecuTorchRuntime(path, "cpu", "fp32")
    images = torch.rand(1, 3, 8, 8)
    sizes = torch.tensor([[10, 20]])

    outputs = runtime.run({"input0": sizes, "input1": images})

    assert runtime.input_spec.name == "input1"
    assert runtime.input_spec.shape == (1, 3, 8, 8)
    assert outputs["output0"].tolist() == [[11.0, 21.0]]
    torch.testing.assert_close(outputs["output1"], module(sizes, images)[1], atol=1e-4, rtol=1e-4)


def test_run_rejects_input_names_that_are_not_positional(build_pte):
    *_, path = build_pte("conv")
    runtime = ExecuTorchRuntime(path, "cpu", "fp32")

    with pytest.raises(ValueError, match=r"\['images'\].*\['input0'\]"):
        runtime.run({"images": torch.rand(1, 3, 8, 8)})
    assert runtime.profiler.timings == []


def test_run_records_one_timing_per_call_after_warmup(build_pte):
    *_, path = build_pte("conv")
    runtime = ExecuTorchRuntime(path, "cpu", "fp32")
    assert runtime.profiler.timings == []

    for expected in (1, 2, 3):
        runtime.run({"input0": torch.rand(1, 3, 8, 8)})
        assert len(runtime.profiler.timings) == expected
