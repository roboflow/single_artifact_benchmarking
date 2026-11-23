import os
from typing import ClassVar

import numpy as np
import torch

from sab.profiler import CPUProfiler, CUDAProfiler
from sab.runtimes.base import InputSpec, Runtime, pick_image_input

_PROVIDERS = {"cpu": "CPUExecutionProvider", "gpu": "CUDAExecutionProvider"}
_INPUT_DEVICES = {"cpu": "cpu", "gpu": "cuda"}

# ONNX Runtime type string -> (torch dtype, numpy dtype)
_DTYPES = {
    "tensor(float)": (torch.float32, np.float32),
    "tensor(float16)": (torch.float16, np.float16),
    "tensor(double)": (torch.float64, np.float64),
    "tensor(int64)": (torch.int64, np.int64),
    "tensor(int32)": (torch.int32, np.int32),
    "tensor(int8)": (torch.int8, np.int8),
    "tensor(uint8)": (torch.uint8, np.uint8),
    "tensor(bool)": (torch.bool, np.bool_),
}
_NUMPY_DTYPE_OF = {torch_dtype: numpy_dtype for torch_dtype, numpy_dtype in _DTYPES.values()}

WARMUP_ITERATIONS = 10


def _static_shape(shape) -> list[int]:
    """Replace every symbolic or unknown dimension with 1."""
    return [dim if isinstance(dim, int) and dim > 0 else 1 for dim in shape]


def _static_image_shape(name: str, shape) -> list[int]:
    """Replace a symbolic batch dimension with 1. Any other symbolic or unknown dimension is an error."""
    for index, dim in enumerate(shape):
        if index > 0 and not (isinstance(dim, int) and dim > 0):
            raise ValueError(f"Image input {name!r} has a symbolic or unknown dimension {dim!r} in {list(shape)}. Export the model with a fixed image size.")
    return _static_shape(shape)


class ONNXRuntime(Runtime):
    """Runs an ONNX artifact with IOBinding. The timed call is `run_with_iobinding`.

    Every input in the dict of `run()` is bound by name, with the dtype of its tensor.
    Outputs go into preallocated buffers, unless `dynamic_output_shapes` is set. Then
    ONNX Runtime allocates them, and the copy to torch happens after the timed call.
    """

    name: ClassVar[str] = "onnxruntime"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu", "gpu"})

    def __init__(
        self,
        artifact_path: str,
        device: str,
        precision: str,
        *,
        dynamic_output_shapes: bool = False,
        image_input_name: str | None = None,
    ):
        super().__init__(artifact_path, device, precision)
        import onnxruntime as ort

        self.dynamic_output_shapes = dynamic_output_shapes
        self.input_device = _INPUT_DEVICES[device]

        session_options = None
        if device == "cpu":
            # Fixed thread counts keep latency stable across runs. ONNX Runtime otherwise picks
            # its own counts, and the scheduling of several inter-op threads adds variance.
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = os.cpu_count()
            session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            artifact_path, providers=[_PROVIDERS[device]], sess_options=session_options
        )
        self.profiler = CPUProfiler() if device == "cpu" else CUDAProfiler()

        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        image_name = pick_image_input([node.name for node in self.inputs], image_input_name)
        image_node = next(node for node in self.inputs if node.name == image_name)
        self.input_spec = InputSpec(name=image_name, shape=tuple(_static_image_shape(image_name, image_node.shape)))

        self.warmup()

    @classmethod
    def is_available(cls, device: str) -> bool:
        if device not in cls.devices:
            return False
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        if device == "gpu":
            return _PROVIDERS["gpu"] in ort.get_available_providers() and torch.cuda.is_available()
        return True

    @classmethod
    def version(cls) -> str:
        import onnxruntime as ort

        return f"onnxruntime {ort.__version__}"

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        binding, buffers = self._bind(inputs)
        binding.synchronize_inputs()

        with self.profiler.profile():
            self.session.run_with_iobinding(binding)

        binding.synchronize_outputs()

        if self.dynamic_output_shapes:
            return self._collect_allocated_outputs(binding)
        return buffers

    def warmup(self, num_iterations: int = WARMUP_ITERATIONS):
        """Run dummy data through the model to trigger JIT optimizations
        and warm CPU/GPU caches before real measurements begin."""
        dummy_inputs = self._dummy_inputs()
        for _ in range(num_iterations):
            binding, _ = self._bind(dummy_inputs)
            binding.synchronize_inputs()
            self.session.run_with_iobinding(binding)
            binding.synchronize_outputs()
        self.profiler.reset()

    def _dummy_inputs(self) -> dict[str, torch.Tensor]:
        dummy_inputs = {}
        for node in self.inputs:
            dtype, _ = _DTYPES[node.type]
            shape = _static_shape(node.shape)
            if node.name == self.input_spec.name:
                tensor = torch.randn(shape, dtype=torch.float32, device=self.input_device).to(dtype)
            else:
                tensor = torch.ones(shape, dtype=dtype, device=self.input_device)
            dummy_inputs[node.name] = tensor
        return dummy_inputs

    def _bind(self, inputs: dict[str, torch.Tensor]):
        binding = self.session.io_binding()
        device_type = self.input_device
        device_id = 0

        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            binding.bind_input(
                name=name,
                device_type=device_type,
                device_id=device_id,
                element_type=_NUMPY_DTYPE_OF[tensor.dtype],
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        buffers = {}
        for node in self.outputs:
            if self.dynamic_output_shapes:
                binding.bind_output(node.name, device_type, device_id)
                continue
            dtype, numpy_dtype = _DTYPES[node.type]
            shape = _static_shape(node.shape)
            buffer = torch.empty(shape, dtype=dtype, device=self.input_device)
            binding.bind_output(
                name=node.name,
                device_type=device_type,
                device_id=device_id,
                element_type=numpy_dtype,
                shape=shape,
                buffer_ptr=buffer.data_ptr(),
            )
            buffers[node.name] = buffer
        return binding, buffers

    def _collect_allocated_outputs(self, binding) -> dict[str, torch.Tensor]:
        arrays = binding.copy_outputs_to_cpu()
        return {
            node.name: torch.from_numpy(array).to(self.input_device)
            for node, array in zip(self.outputs, arrays)
        }
