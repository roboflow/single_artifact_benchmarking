import torch
import torchvision.transforms.functional as TF
import numpy as np
import onnxruntime as ort
import json
import fire


from sab.onnx_inference import ONNXInference
from sab.trt_inference import TRTInference
from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifacts, pretty_print_results


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
        super().__init__(model_path, image_input_name, use_cuda_graph=False)

    def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return preprocess_image(input_image, self.image_input_shape)
    
    def copy_input_data(self, input_image: torch.Tensor):
        """Copy input data to persistent tensors, handling multiple inputs"""
        input_image = input_image.contiguous()
        input_shape = tuple(input_image.shape)
        
        # Handle main image input (same as base class)
        current_shape = tuple(self.persistent_tensors[self.image_input_name].shape)
        if input_shape != current_shape:
            if self.use_cuda_graph:
                self.current_input_shape = input_shape
            else:
                self._reallocate_tensor_for_shape(self.image_input_name, input_shape)
        
        # Copy main image data
        self.persistent_tensors[self.image_input_name].copy_(input_image)
        
        # Handle the additional "orig_target_sizes" input
        if "orig_target_sizes" in self.persistent_tensors:
            # Fill with ones as in the original construct_bindings method
            # The persistent tensor should already have shape (1, 2) and dtype int64
            self.persistent_tensors["orig_target_sizes"].fill_(1)
        
        return input_shape
    
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return postprocess_output(outputs, metadata)


def main(image_dir: str, annotations_file_path: str, buffer_time: float = 0.0, output_file_name: str = "dfine_results.json"):
    requests = [
        ArtifactBenchmarkRequest(
            onnx_path="dfine_n_coco.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="dfine_n_coco.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="dfine_s_obj2coco.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="dfine_s_obj2coco.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="dfine_m_obj2coco.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="dfine_m_obj2coco.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="dfine_l_obj2coco_e25.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="dfine_l_obj2coco_e25.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="dfine_x_obj2coco.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=False,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
        ArtifactBenchmarkRequest(
            onnx_path="dfine_x_obj2coco.opset17.onnx",
            inference_class=DFINETRTInference,
            needs_fp16=True,
            buffer_time=buffer_time,
            needs_class_remapping=True,
        ),
    ]

    results = run_benchmark_on_artifacts(requests, image_dir, annotations_file_path)

    print(f"Saving results to {output_file_name}")
    with open(output_file_name, "w") as f:
        json.dump(results, f)
    
    pretty_print_results(results)


if __name__ == "__main__":
    fire.Fire(main)