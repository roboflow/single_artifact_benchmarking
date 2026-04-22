"""SAB benchmark adapter for YOLO26 ONNX models.

YOLO26 uses the same ultralytics export format as YOLOv11:
  - Input:  images  (1, 3, 640, 640)
  - Output: output0 (1, 300, 6)  [x1, y1, x2, y2, conf, cls]

Preprocessing and postprocessing are identical to YOLOv11.
"""

from sab.models.benchmark_yolov11 import preprocess_image, postprocess_output
from sab.onnx_inference import ONNXInferenceCPU, ONNXInferenceCUDA
from sab.trt_inference import TRTInference


class YOLO26ONNXCPUInference(ONNXInferenceCPU):
    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata)


class YOLO26ONNXInference(ONNXInferenceCUDA):
    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata)


class YOLO26TRTInference(TRTInference):
    def __init__(self, model_path, image_input_name=None):
        super().__init__(model_path, image_input_name, use_cuda_graph=False)

    def preprocess(self, input_image):
        return preprocess_image(input_image, self.image_input_shape)

    def postprocess(self, outputs, metadata):
        return postprocess_output(outputs, metadata)
