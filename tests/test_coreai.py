"""The fixtures in tests/data/coreai come from coreai-torch 0.4.2: `images * 2` in fp32 and fp16, and a
two-input graph that gives `images * 2` and `orig_target_sizes + 1` (int32)."""

import gc
import sys
from pathlib import Path

import pytest
import torch

rt = pytest.importorskip("coreai.runtime")

from sab.runtimes.coreai import CoreAIRuntime  # noqa: E402

pytestmark = [
    pytest.mark.coreai,
    pytest.mark.skipif(not CoreAIRuntime.is_available("cpu"), reason="needs the macOS 27 Core AI runtime"),
]

FIXTURES = Path(__file__).parent / "data" / "coreai"


def fixture(name: str) -> str:
    return str(FIXTURES / f"{name}.aimodel")


def test_run_gives_numerically_correct_outputs_that_own_their_memory():
    runtime = CoreAIRuntime(fixture("double_fp32"), "cpu", "fp32")
    images = torch.rand(1, 3, 4, 4)

    first = runtime.run({"images": images})["outputs"]
    runtime.run({"images": torch.zeros(1, 3, 4, 4)})

    torch.testing.assert_close(first, images * 2)


def test_fp16_artifact_takes_a_float32_tensor_and_gives_float16():
    runtime = CoreAIRuntime(fixture("double_fp16"), "cpu", "fp16")
    images = torch.rand(1, 3, 4, 4)

    outputs = runtime.run({"images": images})["outputs"]

    assert outputs.dtype == torch.float16
    torch.testing.assert_close(outputs.float(), images * 2, atol=1e-2, rtol=1e-2)


def test_extra_int64_input_reaches_the_model_as_int32():
    runtime = CoreAIRuntime(fixture("two_inputs"), "cpu", "fp32")
    images = torch.rand(1, 3, 4, 4)

    outputs = runtime.run({"images": images, "orig_target_sizes": torch.tensor([[640, 480]])})

    assert runtime.input_spec.name == "images"
    assert runtime.input_spec.shape == (1, 3, 4, 4)
    torch.testing.assert_close(outputs["boxes"], images * 2)
    assert outputs["labels"].tolist() == [[641, 481]]


def test_cpu_allows_only_the_cpu():
    runtime = CoreAIRuntime(fixture("double_fp32"), "cpu", "fp32")

    assert [str(kind) for kind in runtime.options.allowed_compute_unit_kinds] == ["CPU"]


@pytest.mark.parametrize("device, preferred", [("gpu", "GPU"), ("npu", "Neural Engine")])
def test_gpu_and_npu_set_the_preferred_compute_unit(device, preferred):
    if not CoreAIRuntime.is_available(device):
        pytest.skip(f"this Mac has no {device}")

    runtime = CoreAIRuntime(fixture("double_fp32"), device, "fp32")

    assert str(runtime.options.preferred_compute_unit_kind) == preferred


def test_warmup_leaves_no_timings_and_each_run_adds_one():
    runtime = CoreAIRuntime(fixture("two_inputs"), "cpu", "fp32")
    assert runtime.profiler.timings == []

    inputs = {"images": torch.rand(1, 3, 4, 4), "orig_target_sizes": torch.ones(1, 2, dtype=torch.int64)}
    for expected in (1, 2, 3):
        runtime.run(inputs)
        assert len(runtime.profiler.timings) == expected


def test_is_available_needs_macos_a_known_device_and_that_compute_unit(monkeypatch):
    assert not CoreAIRuntime.is_available("tpu")

    monkeypatch.setattr(rt.ComputeUnitKind, "available_kinds", staticmethod(lambda: [rt.ComputeUnitKind.cpu()]))
    assert CoreAIRuntime.is_available("cpu")
    assert not CoreAIRuntime.is_available("npu")

    monkeypatch.setattr(sys, "platform", "linux")
    assert not CoreAIRuntime.is_available("cpu")


def test_the_event_loop_closes_when_the_runtime_is_collected():
    runtime = CoreAIRuntime(fixture("double_fp32"), "cpu", "fp32")
    loop = runtime._loop

    del runtime
    gc.collect()

    assert loop.is_closed()
