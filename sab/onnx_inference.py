import os
import onnxruntime as ort
import torch
import numpy as np

from sab.profiler import CUDAProfiler, CPUProfiler


class ONNXInferenceBase:
    def __init__(self, model_path: str, providers: list[str], profiler, device: str, image_input_name: str|None=None, prediction_type: str="bbox", session_options: ort.SessionOptions|None=None):
        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)

        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shapes = [input.shape for input in self.session.get_inputs()]
        self.output_shapes = [output.shape for output in self.session.get_outputs()]

        if len(self.input_names) != 1 and image_input_name is None:
            raise ValueError("Model has multiple inputs, but no image input name was provided")
        elif len(self.input_names) == 1 and image_input_name is not None:
            assert image_input_name in self.input_names, f"Image input name {image_input_name} not found in model inputs"

        self.image_input_name = image_input_name if image_input_name is not None else self.input_names[0]
        self.image_input_shape = self.session.get_inputs()[self.input_names.index(self.image_input_name)].shape

        self.profiler = profiler
        self.device = device

        self.prediction_type = prediction_type

        self.warmup()

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError("Subclasses must implement this method")

    def construct_bindings(self, input_image: torch.Tensor) -> tuple[ort.IOBinding, dict[str, torch.Tensor]]:
        # Construct IOBinding for the input and output tensors
        if len(self.input_names) != 1:
            raise RuntimeError("Default implementation only supports models with a single input, please subclass and implement this method")

        binding = self.session.io_binding()

        input_image = input_image.contiguous()

        if len(input_image.shape) == 3:
            input_image = input_image.unsqueeze(0)

        device_type = input_image.device.type
        device_id = input_image.device.index if input_image.device.index is not None else 0

        binding.bind_input(
            name=self.image_input_name,
            device_type=device_type,
            device_id=device_id,
            element_type=np.float16 if input_image.dtype == torch.float16 else np.float32,
            shape=input_image.shape,
            buffer_ptr=input_image.data_ptr(),
        )

        outputs = {}

        for i, output_name in enumerate(self.output_names):
            output_shape = self.output_shapes[i]
            buffer = torch.empty(output_shape, dtype=torch.float32, device=input_image.device)

            binding.bind_output(
                name=output_name,
                device_type=device_type,
                device_id=device_id,
                element_type=np.float32,
                shape=output_shape,
                buffer_ptr=buffer.data_ptr(),
            )

            outputs[output_name] = buffer

        return binding, outputs

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Postprocess the outputs into bbox, class, and score
        # bbox must be in normalized coordinates (0-1) and in xyxy format
        raise NotImplementedError("Subclasses must implement this method")

    def infer(self, input_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_image, metadata = self.preprocess(input_image)

        binding, outputs = self.construct_bindings(input_image)

        binding.synchronize_inputs()

        with self.profiler.profile():
            self.session.run_with_iobinding(binding)

        binding.synchronize_outputs()

        return self.postprocess(outputs, metadata)

    def warmup(self, num_iterations: int = 10):
        """Run dummy data through the model to trigger JIT optimizations
        and warm CPU/GPU caches before real measurements begin."""
        device = torch.device(self.device)
        dummy_input = torch.randn(self.image_input_shape, dtype=torch.float32, device=device)
        for _ in range(num_iterations):
            binding, _ = self.construct_bindings(dummy_input)
            binding.synchronize_inputs()
            self.session.run_with_iobinding(binding)
            binding.synchronize_outputs()
        self.profiler.reset()

    def print_latency_stats(self):
        self.profiler.print_stats()


class ONNXInferenceCUDA(ONNXInferenceBase):
    def __init__(self, model_path: str, image_input_name: str|None=None, prediction_type: str="bbox"):
        super().__init__(model_path, ['CUDAExecutionProvider'], CUDAProfiler(),
                         'cuda', image_input_name, prediction_type)


class ONNXInferenceCPU(ONNXInferenceBase):
    def __init__(self, model_path: str, image_input_name: str|None=None, prediction_type: str="bbox"):
        # Fix thread counts for stable latency across runs:
        # - Fixed intra_op threads avoids ORT picking different counts per run
        # - Single inter_op thread eliminates scheduling variance between ops
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count()
        sess_options.inter_op_num_threads = 1
        super().__init__(model_path, ['CPUExecutionProvider'], CPUProfiler(),
                         'cpu', image_input_name, prediction_type,
                         session_options=sess_options)
