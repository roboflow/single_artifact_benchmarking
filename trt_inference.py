import tensorrt as trt
import torch
import numpy as np

from profiler import CUDAProfiler


def build_engine(model_path, engine_path, use_fp16=False):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    config = builder.create_builder_config()
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, logger)

    with open(model_path, "rb") as f:
        model_data = f.read()
    parser.parse(model_data)

    print(f"Building engine from {model_path} to {engine_path}")
    engine = builder.build_serialized_network(network, config)
    print(f"Engine built successfully")

    with open(engine_path, "wb") as f:
        f.write(engine)

    return engine


class TRTInference:
    def __init__(self, engine_path: str, image_input_name: str|None=None):
        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, "")

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

        self.input_names = [name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        self.output_names = [name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        self.output_shapes = [list(self.engine.get_tensor_shape(name)) for name in self.output_names]

        self.initialize_bindings()

        # Cache binding indices
        self.input_binding_idxs = {
            name: i
            for i, name in enumerate(names)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        }
        self.output_binding_idxs = {
            name: i
            for i, name in enumerate(names)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        }

        if len(self.input_names) != 1 and image_input_name is None:
            raise ValueError("Model has multiple inputs, but no image input name was provided")
        elif len(self.input_names) == 1 and image_input_name is not None:
            assert image_input_name in self.input_names, f"Image input name {image_input_name} not found in model inputs"
        
        self.image_input_name = image_input_name if image_input_name is not None else self.input_names[0]
        self.image_input_shape = self.engine.get_tensor_shape(self.image_input_name)

        self.profiler = CUDAProfiler()
    
    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError("Subclasses must implement this method")

    def construct_bindings(self, input_image: torch.Tensor) -> list[int]:
        # Construct bindings for the input and output tensors
        if len(self.input_names) != 1:
            raise RuntimeError("Default implementation only supports models with a single input, please subclass and implement this method")
        
        input_image = input_image.contiguous()

        bindings = [None] * self.engine.num_io_tensors

        for name, binding_idx in self.output_binding_idxs.items():
            bindings[binding_idx] = self.binding_ptrs[name]
        
        bindings[self.input_binding_idxs[self.image_input_name]] = input_image.data_ptr()

        return bindings
    
    def initialize_bindings(self):
        self.bindings = {}
        self.binding_ptrs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if not self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = self.engine.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                self.bindings[name] = torch.from_numpy(np.empty(shape, dtype=dtype)).cuda()
                self.binding_ptrs[name] = self.bindings[name].data_ptr()
    
    def postprocess(self, output_tensors: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Postprocess the output tensors into bbox, class, and score
        # bbox must be in normalized coordinates (0-1) and in xyxy format
        raise NotImplementedError("Subclasses must implement this method")

    def infer(self, input_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_image, metadata = self.preprocess(input_image)

        bindings = self.construct_bindings(input_image)

        with self.profiler.profile():
            success = self.context.execute_v2(bindings)
            assert success, "Execution failed"
                
        outputs = {name: self.bindings[name].clone() for name in self.output_names}

        return self.postprocess(outputs, metadata)

    def print_latency_stats(self):
        self.profiler.print_stats()