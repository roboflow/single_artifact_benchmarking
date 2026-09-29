"""Conversion between the canonical tensors of SAB and the tensors of an artifact.

Processors give NCHW float32. A runtime calls these helpers outside the timed call.
"""

from typing import Literal

import numpy as np
import torch

Layout = Literal["NCHW", "NHWC"]
Quantization = tuple[float, int]  # (scale, zero_point)

_CHANNEL_COUNTS = (1, 3)


def guess_layout(shape: tuple[int, ...]) -> Layout:
    """NHWC when the channels are last and the second dimension is not a channel count."""
    is_channels_last = len(shape) == 4 and shape[3] in _CHANNEL_COUNTS and shape[1] not in _CHANNEL_COUNTS
    return "NHWC" if is_channels_last else "NCHW"


def nchw_shape(shape: tuple[int, ...], layout: Layout) -> tuple[int, ...]:
    if layout == "NHWC":
        batch, height, width, channels = shape
        return (batch, channels, height, width)
    return tuple(shape)


def to_artifact_input(
    tensor: torch.Tensor, layout: Layout, dtype: np.dtype | type, quantization: Quantization | None = None
) -> np.ndarray:
    """A contiguous array in the layout and dtype of the artifact.

    An integer dtype takes a float tensor. With `quantization`, the values map through the
    scale and zero-point of the artifact. Without it, the values round to the nearest integer.
    Either way they clip to the range of the dtype.
    """
    dtype = np.dtype(dtype)
    if layout == "NHWC":
        tensor = tensor.permute(0, 2, 3, 1)
    tensor = tensor.detach().cpu()

    if np.issubdtype(dtype, np.integer) and tensor.is_floating_point():
        if quantization is not None:
            scale, zero_point = quantization
            tensor = tensor / scale + zero_point
        limits = np.iinfo(dtype)
        tensor = tensor.round().clamp(limits.min, limits.max)

    return np.ascontiguousarray(tensor.numpy().astype(dtype))


def from_artifact_output(array: np.ndarray, quantization: Quantization | None = None) -> torch.Tensor:
    """A float32 tensor that owns its memory. Integer arrays dequantize when `quantization` is given."""
    values = np.array(array, dtype=np.float32, order="C")
    if quantization is not None and np.issubdtype(array.dtype, np.integer):
        scale, zero_point = quantization
        values = (values - zero_point) * scale
    return torch.from_numpy(values)
