import importlib.metadata
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from sab.profiler import CPUProfiler
from sab.runtimes.base import InputSpec, Runtime, UnavailableOnHost, pick_image_input

WARMUP_ITERATIONS = 10


def _find_xml(artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.is_file():
        return path
    xml_files = sorted(path.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml file found in {path}")
    if len(xml_files) > 1:
        raise ValueError(f"Found several .xml files in {path}: {[file.name for file in xml_files]}")
    return xml_files[0]


def _static_shape(partial_shape) -> tuple[int, ...]:
    """Replace every dynamic dimension with 1."""
    return tuple(dim.get_length() if dim.is_static else 1 for dim in partial_shape)


def _output_name(port, index: int) -> str:
    return port.any_name if port.get_names() else f"output{index}"


class OpenVINORuntime(Runtime):
    """Runs an OpenVINO artifact on the CPU plugin. The timed call is `InferRequest.infer()`.

    `artifact_path` is an `.xml` or `.onnx` file, or the directory that holds one `.xml` file.
    `input_shape` gives the static shape of an image input whose height, width or channels are dynamic.
    One `InferRequest` serves every call. The input tensors are set before the timed call,
    and the outputs are copied after it, because the request reuses its output buffers.
    """

    name: ClassVar[str] = "openvino"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu"})

    def __init__(
        self,
        artifact_path: str,
        device: str,
        precision: str,
        *,
        image_input_name: str | None = None,
        input_shape: tuple[int, ...] | None = None,
    ):
        super().__init__(artifact_path, device, precision)
        import openvino as ov
        import openvino.properties as properties
        import openvino.properties.hint as hint

        # The CPU plugin defaults to f16 on ARM and to bf16 on x86 with AMX. Without this hint,
        # an fp32 row does not run in fp32. An int8 artifact keeps its own quantization.
        inference_precision = ov.Type.f16 if precision == "fp16" else ov.Type.f32

        core = ov.Core()
        model = core.read_model(str(_find_xml(artifact_path)))
        # Compilation can give an input port another `any_name`, so the names come from the model.
        self._input_names = [port.any_name for port in model.inputs]
        image_name = pick_image_input(self._input_names, image_input_name)
        if input_shape is not None:
            model.reshape({image_name: ov.PartialShape(list(input_shape))})
        elif any(dim.is_dynamic for dim in list(model.input(image_name).get_partial_shape())[1:]):
            raise ValueError(f"The image input {image_name!r} has a dynamic shape. Give its static shape as input_shape.")
        self.compiled_model = core.compile_model(
            model,
            "CPU",
            {
                hint.performance_mode(): hint.PerformanceMode.LATENCY,
                # The LATENCY hint uses only the physical cores, even when more threads are asked for.
                # ONNX Runtime uses every logical CPU, so both runtimes get the same threads.
                hint.enable_hyper_threading(): True,
                properties.inference_num_threads(): os.cpu_count(),
                hint.inference_precision(): inference_precision,
            },
        )
        # A CPU with no native f16 compiles an f16 hint as f32. Such a row would measure fp32 under an fp16 label.
        compiled_precision = self.compiled_model.get_property(hint.inference_precision())
        if compiled_precision != inference_precision:
            raise UnavailableOnHost(
                f"the OpenVINO CPU plugin compiles {precision} as {compiled_precision.get_type_name()} on this CPU"
            )
        self.infer_request = self.compiled_model.create_infer_request()
        self.profiler = CPUProfiler()
        self.input_device = "cpu"

        input_ports = dict(zip(self._input_names, self.compiled_model.inputs))
        self._input_dtypes = {name: port.get_element_type().to_dtype() for name, port in input_ports.items()}
        self._input_shapes = {name: _static_shape(port.get_partial_shape()) for name, port in input_ports.items()}
        self._outputs = [(_output_name(port, index), port) for index, port in enumerate(self.compiled_model.outputs)]

        self.input_spec = InputSpec(name=image_name, shape=self._input_shapes[image_name])

        self.warmup()

    @classmethod
    def is_available(cls, device: str) -> bool:
        if device not in cls.devices:
            return False
        try:
            import openvino as ov
        except ImportError:
            return False
        return "CPU" in ov.Core().available_devices

    @classmethod
    def version(cls) -> str:
        return f"openvino {importlib.metadata.version('openvino')}"

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._set_inputs(inputs)

        with self.profiler.profile():
            self.infer_request.infer()

        return {name: torch.from_numpy(np.array(self.infer_request.get_tensor(port).data)) for name, port in self._outputs}

    def warmup(self, num_iterations: int = WARMUP_ITERATIONS):
        """Run zero inputs through the model to warm caches before real measurements begin."""
        for name in self._input_names:
            self._set_array(name, np.zeros(self._input_shapes[name], dtype=self._input_dtypes[name]))
        for _ in range(num_iterations):
            self.infer_request.infer()
        self.profiler.reset()

    def _set_inputs(self, inputs: dict[str, torch.Tensor]):
        for name, tensor in inputs.items():
            array = tensor.detach().cpu().numpy().astype(self._input_dtypes[name])
            self._set_array(name, np.ascontiguousarray(array))

    def _set_array(self, name: str, array: np.ndarray):
        import openvino as ov

        self.infer_request.set_tensor(name, ov.Tensor(array))
