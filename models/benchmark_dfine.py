import torch
import torchvision.transforms.functional as TF
import numpy as np
import onnxruntime as ort
import os

from onnx_inference import ONNXInference
from trt_inference import TRTInference, build_engine
from evaluation import evaluate
from clock_watch import ThrottleMonitor
from models.utils import get_coco_class_index_mapping


def preprocess_image(image: torch.Tensor, image_input_shape: tuple[int, int]) -> tuple[torch.Tensor, dict]:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = TF.resize(image, image_input_shape[2:])
    
    return image, {}


def postprocess_output(outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = outputs["boxes"].squeeze(0)
    labels = outputs["labels"].squeeze(0)
    scores = outputs["scores"].squeeze(0)

    return bboxes, labels, scores


class DFINEONNXInference(ONNXInference):
    def __init__(self, model_path: str, image_input_name: str|None="images"):
        super().__init__(model_path, image_input_name)

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def construct_bindings(self, input_image: torch.Tensor) -> tuple[ort.IOBinding, dict[str, torch.Tensor]]:
        # Construct IOBinding for the input and output tensors
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

        # spoof with ones because we want unnormalized bboxes
        target_shapes_buffer = torch.ones((1, 2), dtype=torch.int64, device=input_image.device)
        binding.bind_input(
            name="orig_target_sizes",
            device_type=device_type,
            device_id=device_id,
            element_type=np.int64,
            shape=[1, 2],
            buffer_ptr=target_shapes_buffer.data_ptr(),
        )

        outputs = {}

        for i, output_name in enumerate(self.output_names):
            output_shape = self.output_shapes[i]
            if not isinstance(output_shape[0], int):
                output_shape[0] = 1
            dtype = torch.float32 if output_name != "labels" else torch.int64
            np_dtype = np.float32 if output_name != "labels" else np.int64
            buffer = torch.empty(output_shape, dtype=dtype, device=input_image.device)

            binding.bind_output(
                name=output_name,
                device_type=device_type,
                device_id=device_id,
                element_type=np_dtype,
                shape=output_shape,
                buffer_ptr=buffer.data_ptr(),
            )

            outputs[output_name] = buffer

        return binding, outputs

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


class DFINETRTInference(TRTInference):
    def __init__(self, model_path: str, image_input_name: str|None="images"):
        super().__init__(model_path, image_input_name)

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def construct_bindings(self, input_image: torch.Tensor) -> list[int]:
        # Construct bindings for the input and output tensors
        input_image = input_image.contiguous()

        bindings = [None] * self.engine.num_io_tensors

        for name, binding_idx in self.output_binding_idxs.items():
            bindings[binding_idx] = self.binding_ptrs[name]
        
        bindings[self.input_binding_idxs[self.image_input_name]] = input_image.data_ptr()

        target_shapes_buffer = torch.ones((1, 2), dtype=torch.int64, device=input_image.device)
        bindings[self.input_binding_idxs["orig_target_sizes"]] = target_shapes_buffer.data_ptr()

        return bindings
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


if __name__ == "__main__":
    model_path = "dfine_n_coco.onnx"
    # engine_path = "dfine_n_coco.engine"
    engine_path = "dfine_n_coco_fp16.engine"
    coco_dir = "/home/isaac/cocodir/val2017"
    coco_annotations_file_path = "/home/isaac/cocodir/annotations/instances_val2017.json"
    buffer_time = 0.0

    class_mapping = get_coco_class_index_mapping(coco_annotations_file_path)
    inv_class_mapping = {v: k for k, v in class_mapping.items()}

    # inference = DFINEONNXInference(model_path)
    if not os.path.exists(engine_path):
        with ThrottleMonitor() as throttle_monitor:
            build_engine(model_path, engine_path, use_fp16=True)
            if throttle_monitor.did_throttle():
                print("GPU throttled during engine build. This is expected and is a limitation of TensorRT.")

    inference = DFINETRTInference(engine_path)

    # evaluate(onnx_inference, coco_dir, coco_annotations_file_path, inv_class_mapping)

    # onnx_inference.print_latency_stats()

    with ThrottleMonitor() as throttle_monitor:
        evaluate(inference, coco_dir, coco_annotations_file_path, inv_class_mapping, buffer_time=buffer_time)
        if throttle_monitor.did_throttle():
            print(f"🔴  GPU throttled, latency results are unreliable. Try increasing the buffer time. Current buffer time: {buffer_time}s")
        else:
            print("GPU did not throttle during evaluation. Latency numbers should be reliable.")

    inference.print_latency_stats()