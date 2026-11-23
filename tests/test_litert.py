from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("ai_edge_litert")

from sab.runtimes.litert import LiteRTRuntime, pick_image_input_name  # noqa: E402

pytestmark = pytest.mark.litert

DATA_DIR = Path(__file__).parent / "data" / "litert"
FLOAT_MODEL = str(DATA_DIR / "float32_nhwc.tflite")
INT8_MODEL = str(DATA_DIR / "int8_nhwc.tflite")
INPUT_NAME = "serving_default_images:0"
OUTPUT_NAME = "StatefulPartitionedCall_1:0"

# The 1x1 convolution of tests/data/litert/generate.py: 3 input channels mixed into 6.
WEIGHTS = torch.tensor(
    [[0.5, -0.25, 1.0, 0.0, 0.125, -1.0], [0.25, 0.5, -0.5, 1.0, 0.0, 0.75], [-0.5, 0.125, 0.25, 0.5, 1.0, 0.0]]
)


def expected_output(canonical: torch.Tensor) -> torch.Tensor:
    """The model function on a canonical (1, 3, 4, 5) image: a per-pixel channel mix, flattened to (1, 20, 6)."""
    pixels = canonical.permute(0, 2, 3, 1).reshape(1, 20, 3)
    return pixels @ WEIGHTS


def test_nhwc_model_takes_a_canonical_input_and_gives_outputs_that_own_their_memory():
    runtime = LiteRTRuntime(FLOAT_MODEL, "cpu", "fp32")
    image = torch.rand(1, 3, 4, 5)

    first = runtime.run({INPUT_NAME: image})
    runtime.run({INPUT_NAME: torch.zeros(1, 3, 4, 5)})

    assert runtime.input_spec.name == INPUT_NAME
    assert runtime.input_spec.shape == (1, 3, 4, 5)
    assert set(first) == {OUTPUT_NAME}
    torch.testing.assert_close(first[OUTPUT_NAME], expected_output(image), rtol=1e-4, atol=1e-4)


def test_int8_model_quantizes_the_input_and_dequantizes_the_output():
    float_runtime = LiteRTRuntime(FLOAT_MODEL, "cpu", "fp32")
    int8_runtime = LiteRTRuntime(INT8_MODEL, "cpu", "int8")
    image = torch.rand(1, 3, 4, 5)

    reference = float_runtime.run({INPUT_NAME: image})[OUTPUT_NAME]
    quantized = int8_runtime.run({INPUT_NAME: image})[OUTPUT_NAME]

    assert quantized.dtype == torch.float32
    assert (quantized - reference).abs().max() < 0.05
    assert not torch.equal(quantized, reference)  # the int8 grid is coarser than float32


def test_warmup_leaves_no_timings_and_each_run_adds_one():
    runtime = LiteRTRuntime(FLOAT_MODEL, "cpu", "fp32")
    assert runtime.profiler.timings == []

    for expected in (1, 2, 3):
        runtime.run({INPUT_NAME: torch.rand(1, 3, 4, 5)})
        assert len(runtime.profiler.timings) == expected


IMAGE = ("serving_default_images:0", (1, 4, 5, 3), np.float32)
SCALE = ("serving_default_scale:0", (1,), np.float32)
MASK = ("serving_default_mask:0", (1, 4, 5, 1), np.uint8)


def test_the_only_four_dimensional_input_is_the_image_input_of_a_multi_input_model():
    assert pick_image_input_name([SCALE, IMAGE], None) == IMAGE[0]


def test_an_explicit_image_input_name_wins_over_the_shape():
    assert pick_image_input_name([SCALE, IMAGE, MASK], MASK[0]) == MASK[0]


def test_a_single_input_model_needs_no_shape_match():
    assert pick_image_input_name([SCALE], None) == SCALE[0]


def test_several_four_dimensional_inputs_without_a_name_raise():
    with pytest.raises(ValueError, match="several inputs"):
        pick_image_input_name([IMAGE, MASK], None)


def test_several_inputs_without_a_four_dimensional_one_raise():
    with pytest.raises(ValueError, match="several inputs"):
        pick_image_input_name([SCALE, ("serving_default_size:0", (2,), np.int32)], None)
