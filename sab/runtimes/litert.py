import importlib.metadata
import os
from typing import ClassVar

import numpy as np
import torch

from sab.profiler import CPUProfiler
from sab.runtimes.base import InputSpec, Runtime, pick_image_input
from sab.runtimes.convert import Layout, from_artifact_output, guess_layout, nchw_shape, to_artifact_input

WARMUP_ITERATIONS = 10


def _quantization(details: dict) -> tuple[float, int] | None:
    scale, zero_point = details["quantization"]
    return (scale, zero_point) if scale != 0 else None


class LiteRTRuntime(Runtime):
    """Runs a `.tflite` artifact with the LiteRT interpreter. The timed call is `invoke()`.

    The exporters `litert` and `tflite` both give `.tflite` files. The runtime sets the inputs
    with `set_tensor` before the timed call and reads the outputs with `get_tensor` after it.
    An artifact with int8 or uint8 tensors gets quantized inputs and gives dequantized outputs.
    """

    name: ClassVar[str] = "litert"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu"})

    def __init__(self, artifact_path: str, device: str, precision: str, *, image_input_name: str | None = None):
        super().__init__(artifact_path, device, precision)
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=artifact_path, num_threads=os.cpu_count())
        self.interpreter.allocate_tensors()
        self.profiler = CPUProfiler()
        self.input_device = "cpu"

        self._inputs = {details["name"]: details for details in self.interpreter.get_input_details()}
        self._outputs = self.interpreter.get_output_details()

        image_name = pick_image_input(list(self._inputs), image_input_name)
        image_shape = tuple(int(dim) for dim in self._inputs[image_name]["shape"])
        self._image_layout: Layout = guess_layout(image_shape)
        self.input_spec = InputSpec(name=image_name, shape=nchw_shape(image_shape, self._image_layout))

        self.warmup()

    @classmethod
    def is_available(cls, device: str) -> bool:
        if device not in cls.devices:
            return False
        try:
            import ai_edge_litert.interpreter  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        return f"ai-edge-litert {importlib.metadata.version('ai-edge-litert')}"

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name, tensor in inputs.items():
            details = self._inputs[name]
            layout: Layout = self._image_layout if name == self.input_spec.name else "NCHW"
            array = to_artifact_input(tensor, layout, details["dtype"], _quantization(details))
            self.interpreter.set_tensor(details["index"], array)

        with self.profiler.profile():
            self.interpreter.invoke()

        return {
            details["name"]: from_artifact_output(
                self.interpreter.get_tensor(details["index"]), _quantization(details)
            )
            for details in self._outputs
        }

    def warmup(self, num_iterations: int = WARMUP_ITERATIONS):
        """Run zero inputs through the model to warm caches before real measurements begin."""
        for details in self._inputs.values():
            self.interpreter.set_tensor(details["index"], np.zeros(details["shape"], dtype=details["dtype"]))
        for _ in range(num_iterations):
            self.interpreter.invoke()
        self.profiler.reset()
