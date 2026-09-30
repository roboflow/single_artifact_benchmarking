import importlib.metadata
import sys
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from sab.profiler import CPUProfiler
from sab.runtimes.base import InputSpec, Runtime, pick_image_input

WARMUP_ITERATIONS = 10


@dataclass(frozen=True)
class _Feature:
    shape: tuple[int, ...]
    dtype: np.dtype
    is_image: bool


def _read_feature(input_description) -> _Feature:
    from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType, ImageFeatureType

    name = input_description.name
    kind = input_description.type.WhichOneof("Type")
    if kind == "multiArrayType":
        array_type = input_description.type.multiArrayType
        dtypes = {
            ArrayFeatureType.FLOAT32: np.float32,
            ArrayFeatureType.FLOAT16: np.float16,
            ArrayFeatureType.DOUBLE: np.float64,
            ArrayFeatureType.INT32: np.int32,
        }
        if array_type.dataType not in dtypes:
            raise ValueError(f"Input {name!r} has an unsupported array data type {array_type.dataType}")
        return _Feature(tuple(array_type.shape), np.dtype(dtypes[array_type.dataType]), is_image=False)
    if kind == "imageType":
        image_type = input_description.type.imageType
        if image_type.colorSpace not in (ImageFeatureType.RGB, ImageFeatureType.BGR):
            raise ValueError(f"Input {name!r} is not an RGB or BGR image")
        return _Feature((1, 3, image_type.height, image_type.width), np.dtype(np.float32), is_image=True)
    raise ValueError(f"Input {name!r} has an unsupported kind {kind!r}")


def _to_pil_image(pixels: np.ndarray):
    from PIL import Image

    hwc = np.transpose(pixels[0], (1, 2, 0))
    return Image.fromarray(np.ascontiguousarray(np.clip(np.rint(hwc), 0, 255).astype(np.uint8)), "RGB")


def _to_artifact_form(feature: _Feature, tensor: torch.Tensor):
    array = tensor.detach().cpu().numpy()
    if feature.is_image:
        return _to_pil_image(array)
    return np.ascontiguousarray(array.astype(feature.dtype))


class CoreMLRuntime(Runtime):
    """Runs a Core ML artifact. The timed call is `MLModel.predict()`.

    The timed call includes the conversion of the inputs to `MLMultiArray` and of the outputs
    back to numpy, because Core ML gives no smaller call. On gpu and npu, Core ML runs the
    operations that the device does not support on the CPU. An image-type input scales its pixels
    inside the model, so it takes 0-255 values: its rows set `normalized_in_graph`.
    """

    name: ClassVar[str] = "coreml"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu", "gpu", "npu"})

    def __init__(self, artifact_path: str, device: str, precision: str, *, image_input_name: str | None = None):
        super().__init__(artifact_path, device, precision)
        import coremltools as ct

        compute_units = {
            "cpu": ct.ComputeUnit.CPU_ONLY,
            "gpu": ct.ComputeUnit.CPU_AND_GPU,
            "npu": ct.ComputeUnit.CPU_AND_NE,
        }[device]
        self.model = ct.models.MLModel(artifact_path, compute_units=compute_units)
        self.profiler = CPUProfiler()
        self.input_device = "cpu"

        self._features = {
            description.name: _read_feature(description) for description in self.model.get_spec().description.input
        }
        image_name = pick_image_input(list(self._features), image_input_name)
        self.input_spec = InputSpec(name=image_name, shape=self._features[image_name].shape)

        self.warmup()

    @classmethod
    def is_available(cls, device: str) -> bool:
        if device not in cls.devices or sys.platform != "darwin":
            return False
        try:
            import coremltools  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        return f"coremltools {importlib.metadata.version('coremltools')}"

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feed = self._to_feed(inputs)

        with self.profiler.profile():
            outputs = self.model.predict(feed)

        return {name: torch.from_numpy(np.array(value)) for name, value in outputs.items()}

    def warmup(self, num_iterations: int = WARMUP_ITERATIONS):
        """Run zero inputs through the model to warm caches before real measurements begin."""
        feed = self._to_feed({name: torch.zeros(feature.shape) for name, feature in self._features.items()})
        for _ in range(num_iterations):
            self.model.predict(feed)
        self.profiler.reset()

    def _to_feed(self, inputs: dict[str, torch.Tensor]) -> dict:
        return {name: _to_artifact_form(self._features[name], tensor) for name, tensor in inputs.items()}
