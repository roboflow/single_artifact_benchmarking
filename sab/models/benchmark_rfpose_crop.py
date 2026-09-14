"""SAB handler for formatted RF-Pose crop images, never patch tensors.

Native-aspect crops share a batch. Normalization, patch extraction, network,
GMM decode and crop score are inside; crop warping and all NMS are outside.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

from sab.models.benchmark_decoded_pose import DecodedPoseTRTInference
from sab.models.benchmark_rfpose import digest
from sab.models.benchmark_rfpose_split import require_fused_attention
from sab.trt_inference import TRTInference


def formatted_crop(rgb, box, buckets, padding):
    box = np.asarray(box, np.float64)
    if rgb.dtype != np.uint8 or box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
        raise ValueError('RGB uint8 image and valid detector xywh box required')
    aspect = box[2] / box[3]
    ratios = np.asarray([w / h for h, w in buckets])
    bucket = int(np.abs(np.log(aspect) - np.log(ratios)).argmin())
    h, w = buckets[bucket]
    width = max(box[2], box[3] * w / h) * padding
    span = np.asarray([width, width * h / w])
    origin = box[:2] + box[2:] / 2 - span / 2
    scale = np.asarray([w - 1, h - 1]) / span
    matrix = np.array([[scale[0], 0, -origin[0]*scale[0]], [0, scale[1], -origin[1]*scale[1]]])
    crop = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    canvas = np.zeros((3, max(x[0] for x in buckets), max(x[1] for x in buckets)), np.uint8)
    canvas[:, :h, :w] = crop.transpose(2, 0, 1)
    return canvas, np.float32(aspect), np.int32(bucket)


def crop_profiles(contract, maximum=45):
    tail = tuple(contract['input_shape'][1:])
    return dict(image=[(n, *tail) for n in (1, 1, maximum)],
                box_aspect=[(n,) for n in (1, 1, maximum)], bucket_id=[(n,) for n in (1, 1, maximum)])


class RFPoseCropTRTInference(TRTInference):
    _capture_cuda_graph = DecodedPoseTRTInference._capture_cuda_graph
    fresh_context = DecodedPoseTRTInference.fresh_context
    get_outputs = DecodedPoseTRTInference.get_outputs

    def __init__(self, engine_path, onnx_path, batch=1):
        self.contract = json.loads(Path(onnx_path).with_suffix('.contract.json').read_text())
        receipt = json.loads(Path(engine_path).with_name('build.json').read_text())
        if (receipt['identity']['onnx_sha256'] != digest(onnx_path)
                or receipt['engine_sha256'] != digest(engine_path)
                or self.contract['onnx_sha256'] != digest(onnx_path)):
            raise ValueError('artifact identity mismatch')
        if receipt['identity']['tensorrt'] != trt.__version__ or self.contract.get('flip') is not False:
            raise ValueError('requires matching TRT environment and single-view contract')
        if self.contract['family'] != 'rfpose' or self.contract['nms_inside_engine']:
            raise ValueError('requires NMS-free RF-Pose crop contract')
        self.batch = int(batch)
        super().__init__(str(engine_path), 'image', prediction_type='keypoints')
        self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self.graph_status = dict(attempted=False, active=False)
        required = self.contract['precision_policy'].get('required_fused_mha_blocks')
        if required:
            inspector = self.engine.create_engine_inspector()
            info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            self.fused_attention_blocks = require_fused_attention(info, required)

    def initialize_persistent_tensors(self):
        if set(self.input_names) != {'image', 'box_aspect', 'bucket_id'} or set(self.output_names) != {'keypoints', 'scores'}:
            raise ValueError('unexpected image-crop contract')
        self.persistent_tensors, self.allocators = {}, {}
        for name in self.input_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            if not self.context.set_input_shape(name, (self.batch, *shape[1:])):
                raise ValueError('invalid batch shape')
        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(s <= 0 for s in shape):
                raise ValueError('unresolved shape')
            dtype = torch.from_numpy(np.empty(0, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            buffer = torch.empty(shape, dtype=dtype, device='cuda')
            self.persistent_tensors[name] = buffer
            if not self.context.set_tensor_address(name, buffer.data_ptr()):
                raise RuntimeError('failed binding')

    def copy_input_data(self, values):
        if set(values) != set(self.input_names):
            raise ValueError('requires formatted image, box aspect and bucket ID')
        for name, value in values.items():
            destination = self.persistent_tensors[name]
            if destination.shape != value.shape or destination.dtype != value.dtype:
                raise ValueError('wrong input shape/dtype: ' + name)
            destination.copy_(value)
        return tuple(values['image'].shape)

    def preprocess(self, images_and_boxes):
        rows = [formatted_crop(rgb, box, self.contract['aspect_buckets'], self.contract['padding'])
                for rgb, box in images_and_boxes]
        if len(rows) != self.batch:
            raise ValueError('batch count mismatch')
        return {name: torch.from_numpy(np.stack([r[i] for r in rows]))
                for i, name in enumerate(('image', 'box_aspect', 'bucket_id'))}, {}
