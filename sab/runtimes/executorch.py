import importlib.metadata
from typing import ClassVar

import torch

from sab.profiler import CPUProfiler
from sab.runtimes.base import InputSpec, Runtime

# ExecuTorch ScalarType value -> torch dtype
_DTYPES = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    11: torch.bool,
}
_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64)

WARMUP_ITERATIONS = 10


def _input_name(index: int) -> str:
    return f"input{index}"


class ExecuTorchRuntime(Runtime):
    """Runs a `.pte` artifact. The timed call is `Method.execute`.

    A `.pte` has no input names, so inputs are named by position: `input0`, `input1`, and so on.
    Outputs are named `output0`, `output1`, and so on. The artifact fixes the backend when it is
    exported. The CPU artifacts use XNNPACK.
    """

    name: ClassVar[str] = "executorch"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu"})

    def __init__(self, artifact_path: str, device: str, precision: str, *, method_name: str = "forward"):
        super().__init__(artifact_path, device, precision)
        from executorch.runtime import Runtime as ExecuTorch

        self.input_device = "cpu"
        self.profiler = CPUProfiler()
        program = ExecuTorch.get().load_program(artifact_path)
        self.method = program.load_method(method_name)

        metadata = self.method.metadata
        input_infos = [metadata.input_tensor_meta(i) for i in range(metadata.num_inputs())]
        self._input_shapes = [tuple(info.sizes()) for info in input_infos]
        self._input_dtypes = [_DTYPES[info.dtype()] for info in input_infos]

        self._image_index = self._find_image_input()
        self.input_spec = InputSpec(name=_input_name(self._image_index), shape=self._input_shapes[self._image_index])

        self.warmup()

    @classmethod
    def is_available(cls, device: str) -> bool:
        if device not in cls.devices:
            return False
        try:
            import executorch.runtime  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        return f"executorch {importlib.metadata.version('executorch')}"

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        arguments = self._to_artifact_inputs(inputs)

        with self.profiler.profile():
            raw_outputs = self.method.execute(arguments)

        # ExecuTorch can reuse its output buffers on the next call, so each caller gets its own copy.
        return {f"output{i}": output.detach().clone() for i, output in enumerate(raw_outputs)}

    def warmup(self, num_iterations: int = WARMUP_ITERATIONS):
        """Run zeros through the method to warm the caches before real measurements begin."""
        zeros = [
            torch.zeros(shape, dtype=dtype)
            for shape, dtype in zip(self._input_shapes, self._input_dtypes)
        ]
        for _ in range(num_iterations):
            self.method.execute(zeros)
        self.profiler.reset()

    def _find_image_input(self) -> int:
        """Input 0, unless input 0 is not a 4-D float and exactly one other input is."""
        image_like = [
            index
            for index, (shape, dtype) in enumerate(zip(self._input_shapes, self._input_dtypes))
            if len(shape) == 4 and dtype in _FLOAT_DTYPES
        ]
        if 0 in image_like or len(image_like) != 1:
            return 0
        return image_like[0]

    def _to_artifact_inputs(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return [
            inputs[_input_name(index)].detach().to("cpu", dtype).contiguous()
            for index, dtype in enumerate(self._input_dtypes)
        ]
