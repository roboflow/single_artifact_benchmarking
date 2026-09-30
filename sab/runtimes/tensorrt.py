import os
from typing import ClassVar

import numpy as np
import torch

from sab.profiler import CUDAProfiler
from sab.runtimes.base import InputSpec, Runtime, pick_image_input

# The fp32 name stays `<stem>.engine`, so engines that earlier runs cached still match.
_ENGINE_SUFFIXES = {"fp32": ".engine", "fp16": ".fp16.engine", "int8": ".int8.engine"}

# int8 keeps FP16 on: layers without a Q/DQ pair fall back to fp16 instead of fp32.
_BUILDER_FLAG_NAMES = {"fp32": (), "fp16": ("FP16",), "int8": ("INT8", "FP16")}


def engine_path_for(artifact_path: str, precision: str) -> str:
    if not artifact_path.endswith(".onnx"):
        raise ValueError(f"TensorRT builds engines from .onnx artifacts, got {artifact_path!r}")
    if precision not in _ENGINE_SUFFIXES:
        raise ValueError(f"Unknown precision {precision!r}. Supported: {sorted(_ENGINE_SUFFIXES)}")
    return artifact_path[: -len(".onnx")] + _ENGINE_SUFFIXES[precision]


def builder_flag_names(precision: str) -> tuple[str, ...]:
    """Names of the `tensorrt.BuilderFlag` members that the engine build sets for a precision."""
    if precision not in _BUILDER_FLAG_NAMES:
        raise ValueError(f"Unknown precision {precision!r}. Supported: {sorted(_BUILDER_FLAG_NAMES)}")
    return _BUILDER_FLAG_NAMES[precision]


def build_engine(model_path, engine_path, precision="fp32"):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    config = builder.create_builder_config()
    for flag_name in builder_flag_names(precision):
        config.set_flag(getattr(trt.BuilderFlag, flag_name))

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, logger)

    with open(model_path, "rb") as f:
        model_data = f.read()

    if not parser.parse(model_data):
        print("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    # Create optimization profile to fix dynamic batch dimensions
    profile = builder.create_optimization_profile()

    # Handle dynamic input shapes - fix batch size to 1
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_shape = input_tensor.shape
        print(f"Input {i} ({input_tensor.name}): {input_shape}")

        # Check if batch dimension is dynamic (typically -1)
        if input_shape[0] == -1:
            # Fix batch size to 1
            fixed_shape = (1,) + tuple(input_shape[1:])
            print(f"  Setting fixed batch shape: {fixed_shape}")

            # Set min, optimal, and max shapes all to batch size 1
            profile.set_shape(input_tensor.name, fixed_shape, fixed_shape, fixed_shape)

    # Add the optimization profile to the configuration
    config.add_optimization_profile(profile)

    print(f"Building engine from {model_path} to {engine_path}")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("Failed to build engine")
        return None

    print("Engine built successfully")

    with open(engine_path, "wb") as f:
        f.write(engine)

    return engine


def _ensure_engine(artifact_path: str, precision: str) -> str:
    """Give the path of the engine for an ONNX artifact, and build the engine when it is missing."""
    from sab.monitors.nvidia import ThrottleMonitor

    engine_path = engine_path_for(artifact_path, precision)

    if os.path.exists(engine_path):
        print(f"Engine for {artifact_path} already exists at {engine_path}")
        return engine_path

    print(f"Building engine for {artifact_path} and saving to {engine_path}...")
    with ThrottleMonitor() as throttle_monitor:
        engine = build_engine(artifact_path, engine_path, precision)
    # After the with block the worker has joined, so the verdict is final.
    if throttle_monitor.did_throttle():
        print("GPU throttled during engine build. This is expected and is a limitation of TensorRT.")
    if engine is None:
        raise RuntimeError(f"TensorRT could not build an engine from {artifact_path}")
    return engine_path


class TRTRuntime(Runtime):
    """Runs an ONNX artifact as a TensorRT engine that the runtime builds on first use.

    The timed call is the replay of a CUDA graph, or `execute_async_v3` without one. Every
    input in the dict of `run()` is copied into a persistent tensor. Those tensors never
    move, because a captured CUDA graph holds their addresses.
    """

    name: ClassVar[str] = "tensorrt"
    devices: ClassVar[frozenset[str]] = frozenset({"gpu"})

    def __init__(
        self,
        artifact_path: str,
        device: str,
        precision: str,
        *,
        use_cuda_graph: bool = True,
        image_input_name: str | None = None,
    ):
        super().__init__(artifact_path, device, precision)
        import tensorrt as trt

        self.input_device = "cuda"
        engine_path = _ensure_engine(artifact_path, precision)

        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, "")

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        print(f"Using {self.engine.num_aux_streams} auxiliary streams for TensorRT inference")

        self.context = self.engine.create_execution_context()

        # Create dedicated CUDA stream for inference
        self.torch_stream = torch.cuda.Stream()
        self.cuda_stream_ptr = self.torch_stream.cuda_stream

        # Create separate stream for warm-up to avoid capture state conflicts
        self.warmup_stream = torch.cuda.Stream()
        self.warmup_stream_ptr = self.warmup_stream.cuda_stream
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph_compatible = True  # Track if model is compatible with CUDA graphs

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

        self.input_names = [name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        self.output_names = [name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        self.output_shapes = [tuple(self.engine.get_tensor_shape(name)) for name in self.output_names]

        # Initialize persistent tensors for CUDA graphs
        self.initialize_persistent_tensors()

        image_name = pick_image_input(self.input_names, image_input_name)
        self.input_spec = InputSpec(name=image_name, shape=tuple(self.engine.get_tensor_shape(image_name)))

        # CUDA graph management
        self.graph_cache = {}  # Cache graphs for different input shapes
        self.current_input_shape = None

        self.profiler = CUDAProfiler(stream=self.torch_stream)  # Stream-aware profiling

        print(f"TensorRT inference initialized with CUDA graphs: {self.use_cuda_graph}")
        if self.use_cuda_graph:
            print("Note: CUDA graphs will be tested on first inference. If incompatible, will fall back to standard execution.")

    @classmethod
    def is_available(cls, device: str) -> bool:
        if device not in cls.devices:
            return False
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            return False
        return torch.cuda.is_available()

    @classmethod
    def version(cls) -> str:
        import tensorrt as trt

        return f"tensorrt {trt.__version__}"

    def initialize_persistent_tensors(self):
        """Initialize persistent PyTorch tensors and bind them to TensorRT"""
        import tensorrt as trt

        self.persistent_tensors = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))  # Convert Dims to tuple
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # Convert numpy dtype to torch dtype
            if dtype == np.float32:
                torch_dtype = torch.float32
            elif dtype == np.float16:
                torch_dtype = torch.float16
            elif dtype == np.int32:
                torch_dtype = torch.int32
            elif dtype == np.int64:
                torch_dtype = torch.int64
            else:
                torch_dtype = torch.float32  # Default fallback

            # Create persistent tensor
            tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
            self.persistent_tensors[name] = tensor

            # Bind tensor address to TensorRT context
            self.context.set_tensor_address(name, tensor.data_ptr())

            print(f"Allocated persistent tensor '{name}': {shape} ({torch_dtype})")

    def _reallocate_tensor_for_shape(self, tensor_name: str, new_shape: tuple):
        """Reallocate a tensor for a new shape and update TensorRT binding"""
        current_tensor = self.persistent_tensors[tensor_name]

        # Create new tensor with the same dtype
        new_tensor = torch.empty(new_shape, dtype=current_tensor.dtype, device='cuda')

        # Update our reference and TensorRT binding
        self.persistent_tensors[tensor_name] = new_tensor
        self.context.set_tensor_address(tensor_name, new_tensor.data_ptr())

        print(f"Reallocated tensor '{tensor_name}' from {tuple(current_tensor.shape)} to {new_shape}")

    def _capture_cuda_graph(self, input_shape: tuple):
        """Capture a CUDA graph for the given input shape"""
        print(f"Capturing CUDA graph for input shape: {input_shape}")

        # Update input tensor shape if needed
        current_input_shape = tuple(self.persistent_tensors[self.input_spec.name].shape)
        if input_shape != current_input_shape:
            self._reallocate_tensor_for_shape(self.input_spec.name, input_shape)

        try:
            # IMPORTANT: Warm-up execution on SEPARATE stream (not capture stream or default)
            print("Performing warm-up execution before graph capture...")

            # Do warm-up runs on separate warm-up stream to avoid capture state confusion
            for _ in range(3):
                success = self.context.execute_async_v3(self.warmup_stream_ptr)
                if not success:
                    raise RuntimeError("TensorRT warm-up execution failed")
                self.warmup_stream.synchronize()  # Sync the warm-up stream

            # Now capture the CUDA graph on our dedicated stream
            print("Starting CUDA graph capture...")
            graph = torch.cuda.CUDAGraph()

            # Capture on the dedicated stream
            with torch.cuda.graph(graph, stream=self.torch_stream):
                success = self.context.execute_async_v3(self.cuda_stream_ptr)
                if not success:
                    raise RuntimeError("TensorRT execution failed during graph capture")

            # Cache the graph
            shape_key = input_shape
            self.graph_cache[shape_key] = graph

            print(f"Successfully captured CUDA graph for shape {input_shape}")
            return graph

        except Exception as e:
            print(f"CUDA graph capture failed: {e}")

            # Mark this shape as incompatible with CUDA graphs
            self.graph_cache[input_shape] = None
            return None

    def _recover_cuda_context(self):
        """Recover from CUDA context corruption"""
        try:
            print("Recovering CUDA context...")

            # Force synchronization
            torch.cuda.synchronize()

            # Create new streams
            self.torch_stream = torch.cuda.Stream()
            self.cuda_stream_ptr = self.torch_stream.cuda_stream
            self.warmup_stream = torch.cuda.Stream()
            self.warmup_stream_ptr = self.warmup_stream.cuda_stream

            # Update profiler to use new stream
            self.profiler.set_stream(self.torch_stream)

            # Clear cache
            torch.cuda.empty_cache()

            print("CUDA context recovery completed")

        except Exception as e:
            print(f"CUDA context recovery failed: {e}")

    def _execute_with_graph(self, input_shape: tuple):
        """Execute inference using CUDA graph"""
        shape_key = input_shape

        # Get or create graph for this shape
        if shape_key not in self.graph_cache:
            self._capture_cuda_graph(input_shape)

        # Check if graph capture failed for this shape
        graph = self.graph_cache[shape_key]
        if graph is None:
            # Fall back to standard execution
            self._execute_standard()
        else:
            # Execute the cached graph
            graph.replay()

    def _execute_standard(self):
        """Execute inference using standard TensorRT execution"""
        success = self.context.execute_async_v3(self.cuda_stream_ptr)
        if not success:
            raise RuntimeError("TensorRT execution failed")

    def copy_input_data(self, inputs: dict[str, torch.Tensor]) -> tuple:
        """Copy every input to its persistent tensor. Give the shape of the image input."""
        if set(inputs) != set(self.input_names):
            raise ValueError(f"Inputs {sorted(inputs)} do not match the engine inputs {sorted(self.input_names)}")

        image_name = self.input_spec.name
        image = inputs[image_name].contiguous()
        input_shape = tuple(image.shape)

        # Ensure input tensor has the right shape
        current_shape = tuple(self.persistent_tensors[image_name].shape)
        if input_shape != current_shape:
            # If using CUDA graphs, we need to manage shape changes carefully
            if self.use_cuda_graph:
                # This will trigger graph re-capture if needed
                self.current_input_shape = input_shape
            else:
                # For standard execution, just reallocate
                self._reallocate_tensor_for_shape(image_name, input_shape)

        # Copy data to persistent tensors. The other inputs never change shape.
        self.persistent_tensors[image_name].copy_(image)
        for name, tensor in inputs.items():
            if name != image_name:
                self.persistent_tensors[name].copy_(tensor)

        return input_shape

    def get_outputs(self) -> dict[str, torch.Tensor]:
        """Get output tensors (cloned to avoid data races)"""
        return {name: self.persistent_tensors[name].clone() for name in self.output_names}

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        torch.cuda.synchronize()

        with torch.cuda.stream(self.torch_stream):
            input_shape = self.copy_input_data(inputs)
            self._execute_timed(input_shape)

        torch.cuda.synchronize()

        return self.get_outputs()

    def _execute_timed(self, input_shape: tuple):
        # The first call for a shape captures the CUDA graph inside the profiled
        # region: warm-up executions plus the capture itself. That sample measures
        # capture, not replay, so it is dropped from the series after execution.
        captures_graph = (
            self.use_cuda_graph
            and self.cuda_graph_compatible
            and input_shape not in self.graph_cache
        )
        samples_before_execution = len(self.profiler.timings)

        if self.use_cuda_graph and self.cuda_graph_compatible:
            # Use async profiling for CUDA graphs to avoid interference
            with self.profiler.profile_async(stream=self.torch_stream):
                try:
                    self._execute_with_graph(input_shape)
                except Exception as e:
                    print(f"CUDA graph execution failed: {e}")
                    print("Disabling CUDA graphs for this model")
                    self.cuda_graph_compatible = False
                    # Perform recovery and retry with standard execution
                    self._recover_cuda_context()
                    self._execute_standard()

            # Get async timing result after synchronization
            self.torch_stream.synchronize()
            async_timing = self.profiler.get_last_timing_async()
            if async_timing is None:
                print("Warning: Could not retrieve async timing measurement")

        else:
            # Use regular profiling for standard execution
            with self.profiler.profile(stream=self.torch_stream):
                self._execute_standard()

        if captures_graph:
            self.profiler.discard_timings_since(samples_before_execution)

    def cleanup(self):
        """Clean up CUDA graph resources"""
        # PyTorch CUDA graphs are automatically cleaned up by Python's garbage collector
        # Just clear our cache
        self.graph_cache.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA graph resources cleaned up")

        # Note: PyTorch streams (self.torch_stream, self.warmup_stream) are automatically
        # cleaned up when they go out of scope

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup
