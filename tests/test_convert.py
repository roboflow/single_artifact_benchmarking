import numpy as np
import pytest
import torch

from sab.runtimes.convert import from_artifact_output, guess_layout, nchw_shape, to_artifact_input


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((1, 3, 640, 640), "NCHW"),
        ((1, 640, 640, 3), "NHWC"),
        ((1, 3, 4, 3), "NCHW"),  # the second dim is 3, so the channels lead
        ((1, 3), "NCHW"),
    ],
)
def test_guess_layout(shape, expected):
    assert guess_layout(shape) == expected


def test_nhwc_input_round_trips_to_the_canonical_nchw_tensor():
    canonical = torch.arange(1 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 3, 4, 5).requires_grad_()

    array = to_artifact_input(canonical, "NHWC", np.float32)

    assert array.shape == (1, 4, 5, 3)
    assert array.flags["C_CONTIGUOUS"]
    assert nchw_shape(array.shape, "NHWC") == tuple(canonical.shape)
    np.testing.assert_array_equal(array.transpose(0, 3, 1, 2), canonical.detach().numpy())


def test_integer_input_without_quantization_rounds_and_clips():
    tensor = torch.tensor([[[[-20.0, 0.4, 0.6, 127.5, 254.6, 300.0]]]])

    array = to_artifact_input(tensor, "NCHW", np.uint8)

    assert array.flatten().tolist() == [0, 0, 1, 128, 255, 255]  # 127.5 rounds to the even value 128


@pytest.mark.parametrize(
    "dtype, quantization, values, expected",
    [
        (np.int8, (0.5, -10), [0.0, 0.3, 1.0, 100.0, -100.0], [-10, -9, -8, 127, -128]),
        (np.uint8, (0.1, 100), [0.0, 1.0, 20.0, -20.0], [100, 110, 255, 0]),
    ],
)
def test_quantized_input_uses_scale_and_zero_point_and_clips(dtype, quantization, values, expected):
    array = to_artifact_input(torch.tensor([[[values]]]), "NCHW", dtype, quantization=quantization)

    assert array.dtype == dtype
    assert array.flatten().tolist() == expected


@pytest.mark.parametrize(
    "array, quantization, expected",
    [
        (np.array([-128, 0, 127], dtype=np.int8), (0.5, -128), [0.0, 64.0, 127.5]),
        (np.array([0, 100, 255], dtype=np.uint8), (0.1, 100), [-10.0, 0.0, 15.5]),
    ],
)
def test_quantized_output_is_dequantized_to_float32(array, quantization, expected):
    tensor = from_artifact_output(array, quantization=quantization)

    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, torch.tensor(expected))
