import asyncio
import importlib.metadata
import sys
from typing import ClassVar

import numpy as np
import torch

from sab.profiler import CPUProfiler
from sab.runtimes.base import InputSpec, Runtime, pick_image_input

WARMUP_ITERATIONS = 10

# SAB device -> the name of the Core AI compute unit kind
_COMPUTE_UNIT_KINDS = {"cpu": "CPU", "gpu": "GPU", "npu": "Neural Engine"}


def _specialization_options(device: str):
    import coreai.runtime as rt

    if device == "cpu":
        return rt.SpecializationOptions.cpu_only()
    kind = rt.ComputeUnitKind.gpu() if device == "gpu" else rt.ComputeUnitKind.neural_engine()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(kind)


class CoreAIRuntime(Runtime):
    """Runs an Apple Core AI `.aimodel` on macOS 27. The timed call is one call of the inference function.

    The call includes the copy of the inputs to the compute unit, because Core AI gives no smaller call.
    The cpu device allows only the CPU. The gpu and npu devices set the preferred compute unit, and
    Core AI can run the ops that the preferred unit does not support on another unit.
    """

    name: ClassVar[str] = "coreai"
    devices: ClassVar[frozenset[str]] = frozenset(_COMPUTE_UNIT_KINDS)

    def __init__(
        self,
        artifact_path: str,
        device: str,
        precision: str,
        *,
        function_name: str = "main",
        image_input_name: str | None = None,
    ):
        super().__init__(artifact_path, device, precision)
        import coreai.runtime as rt

        # The Core AI Python API is async. One loop serves every call, so the timed call does not start a loop.
        self._loop = asyncio.new_event_loop()
        self.options = _specialization_options(device)
        model = self._loop.run_until_complete(rt.AIModel.load(artifact_path, self.options))
        self.function = model.load_function(function_name)
        self.profiler = CPUProfiler()
        self.input_device = "cpu"

        descriptor = self.function.desc
        input_descriptors = {name: descriptor.input_descriptor(name) for name in descriptor.input_names}
        self._input_dtypes = {name: np.dtype(input.dtype) for name, input in input_descriptors.items()}
        self._input_shapes = {name: tuple(input.shape) for name, input in input_descriptors.items()}

        image_name = pick_image_input(list(descriptor.input_names), image_input_name)
        self.input_spec = InputSpec(name=image_name, shape=self._input_shapes[image_name])

        self.warmup()

    def __del__(self):
        loop = getattr(self, "_loop", None)
        if loop is not None and not loop.is_closed():
            loop.close()

    @classmethod
    def is_available(cls, device: str) -> bool:
        if device not in cls.devices or sys.platform != "darwin":
            return False
        try:
            import coreai.runtime as rt
        except ImportError:
            return False
        return _COMPUTE_UNIT_KINDS[device] in {str(kind) for kind in rt.ComputeUnitKind.available_kinds()}

    @classmethod
    def version(cls) -> str:
        return f"coreai-core {importlib.metadata.version('coreai-core')}"

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        arrays = self._to_artifact_inputs(inputs)

        with self.profiler.profile():
            outputs = self._loop.run_until_complete(self.function(arrays))

        return {name: torch.from_numpy(np.array(output.numpy())) for name, output in outputs.items()}

    def warmup(self, num_iterations: int = WARMUP_ITERATIONS):
        """Run zero inputs through the function to warm the caches before real measurements begin."""
        import coreai.runtime as rt

        zeros = {
            name: rt.NDArray(np.zeros(self._input_shapes[name], dtype=dtype))
            for name, dtype in self._input_dtypes.items()
        }
        for _ in range(num_iterations):
            self._loop.run_until_complete(self.function(zeros))
        self.profiler.reset()

    def _to_artifact_inputs(self, inputs: dict[str, torch.Tensor]) -> dict:
        import coreai.runtime as rt

        return {
            name: rt.NDArray(np.ascontiguousarray(tensor.detach().cpu().numpy().astype(self._input_dtypes[name])))
            for name, tensor in inputs.items()
        }
