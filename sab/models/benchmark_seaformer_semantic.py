"""SeaFormer fixed-shape semantic segmentation on ADE20K."""

import fire

from sab.models.semantic_dense import DenseSemanticAdapter, run_dense_benchmark
from sab.onnx_inference import ONNXInferenceCPU, ONNXInferenceCUDA
from sab.trt_inference import TRTInference


class _SeaFormerAdapter(DenseSemanticAdapter):
    family = "seaformer"


class SeaFormerSemanticTRTInference(_SeaFormerAdapter, TRTInference):
    def __init__(self, model_path):
        super().__init__(model_path, use_cuda_graph=True, prediction_type="semantic")


class SeaFormerSemanticONNXInference(_SeaFormerAdapter, ONNXInferenceCUDA):
    def __init__(self, model_path):
        super().__init__(model_path, prediction_type="semantic")


class SeaFormerSemanticONNXCPUInference(_SeaFormerAdapter, ONNXInferenceCPU):
    def __init__(self, model_path):
        super().__init__(model_path, prediction_type="semantic")


def main(image_dir: str, mask_dir: str, buffer_time: float = 0.0,
         output_file_name: str = "seaformer_semantic_results.json",
         onnx_path: str = "seaformer-t-ade20k.onnx", runtime: str = "trt",
         fp16: bool | None = None, max_images: int | None = None):
    """Benchmark a fixed [1,3,H,W] ONNX with raw [1,150,h,w] logits.

    Images are resized to the artifact's input dimensions. Predictions are
    scored against original-resolution labels. FP16 defaults on for TensorRT.
    """
    runtimes = {"trt": SeaFormerSemanticTRTInference, "onnx-cuda": SeaFormerSemanticONNXInference,
                "onnx-cpu": SeaFormerSemanticONNXCPUInference}
    run_dense_benchmark(image_dir, mask_dir, onnx_path, runtimes, runtime, fp16,
                        buffer_time, output_file_name, max_images)


if __name__ == "__main__":
    fire.Fire(main)
