"""Small SAB handler for decoded third-party pose ONNX artifacts.

The engine owns normalization, the network, native decode, scoring and any
native NMS. Only image formatting/coordinate restoration live in this handler.
No source-model dependency, graph surgery or custom TRT plugin is used.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
import torch

from sab.models.benchmark_rfpose import BoundedOutput, digest
from sab.trt_inference import TRTInference


def format_image(rgb, height, width, kind, box=None):
    """Reference pixel geometry; return uint8 NCHW and crop-to-image affine."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB uint8 HWC required')
    h, w = rgb.shape[:2]
    if kind == 'yolo_letterbox':
        gain = min(height / h, width / w)
        nh, nw = round(h * gain), round(w * gain)
        top, left = round((height - nh) / 2 - .1), round((width - nw) / 2 - .1)
        bottom, right = round((height - nh) / 2 + .1), round((width - nw) / 2 + .1)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else rgb
        out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        inverse = np.array([[1 / gain, 0, -left / gain], [0, 1 / gain, -top / gain]])
    elif kind == 'pil_resize':
        out = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))
        inverse = np.array([[w / width, 0, 0], [0, h / height, 0]])
    elif kind == 'mmpose_bottomup_fit':
        actual_w = min(width, height * w / h)
        source_width = np.float32(w * width / actual_w)
        center = np.array([w / 2, h / 2], np.float32)
        source = np.zeros((3, 2), np.float32)
        source[0] = center
        source[1] = center + [-.5 * source_width, 0]
        delta = source[0] - source[1]
        source[2] = source[1] + [-delta[1], delta[0]]
        target = np.array([[width / 2, height / 2], [0, height / 2],
                           [0, height / 2 + width / 2]], np.float32)
        matrix = cv2.getAffineTransform(source, target)
        out = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        inverse = cv2.invertAffineTransform(matrix)
    elif kind == 'topdown_affine':
        box = np.asarray(box, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
            raise ValueError('valid detector xywh box required')
        center = box[:2] + .5 * box[2:]
        crop_width = max(box[2], box[3] * width / height) * 1.25
        span = np.array([crop_width, crop_width * height / width])
        sx, sy = np.array([width, height]) / span
        matrix = np.array([[sx, 0, width / 2 - center[0] * sx],
                           [0, sy, height / 2 - center[1] * sy]], np.float64)
        out = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inverse = cv2.invertAffineTransform(matrix)
    else:
        raise ValueError('unknown formatting contract: ' + str(kind))
    return np.ascontiguousarray(out.transpose(2, 0, 1)[None]), inverse


def restore_points(xy, inverse):
    return np.asarray(xy, dtype=np.float64) @ inverse[:, :2].T + inverse[:, 2]


class DecodedPoseTRTInference(TRTInference):
    def __init__(self, engine_path, onnx_path, use_cuda_graph=True):
        self.contract = json.loads(Path(onnx_path).with_suffix('.contract.json').read_text())
        receipt = json.loads(Path(engine_path).with_name('build.json').read_text())
        identity = receipt.get('identity', receipt)
        if identity['onnx_sha256'] != digest(onnx_path) or receipt['engine_sha256'] != digest(engine_path):
            raise ValueError('engine/ONNX/build identity mismatch')
        if identity['tensorrt'] != trt.__version__:
            raise ValueError('use the pinned TRT environment matching this engine')
        if self.contract['onnx_sha256'] != identity['onnx_sha256']:
            raise ValueError('contract ONNX identity mismatch')
        self.stage = self.contract['stage']
        family = self.contract['family']
        if self.stage == 'person_crop':
            self.format_kind = 'topdown_affine'
        elif family == 'yolo26':
            self.format_kind = 'yolo_letterbox'
        elif family in ('ecpose', 'detrpose'):
            self.format_kind = 'pil_resize'
        elif family.startswith('rtmo'):
            self.format_kind = 'mmpose_bottomup_fit'
        else:
            raise ValueError('unreviewed decoded pose contract')
        super().__init__(str(engine_path), 'image', use_cuda_graph=use_cuda_graph, prediction_type='keypoints')
        self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self.graph_status = dict(attempted=False, active=False)
        if not {'keypoints', 'scores'} <= set(self.output_names):
            raise ValueError('requires in-engine keypoint decode AND crop/person scores')

    def initialize_persistent_tensors(self):
        self.persistent_tensors, self.allocators = {}, {}
        if not set(self.input_names) <= {'image', 'confidence_threshold'}:
            raise ValueError('unreviewed auxiliary engine input')
        for name in self.input_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            if shape and shape[0] == -1 and all(s > 0 for s in shape[1:]):
                self.context.set_input_shape(name, (1, *shape[1:]))
        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = torch.from_numpy(np.empty(0, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            if any(s < 0 for s in shape):
                capacity = self.contract.get('person_capacity')
                if name in self.input_names or capacity is None or len(shape) < 2 or shape[1] != -1:
                    raise ValueError(f'unknown dynamic output capacity: {name} {shape}')
                maximum = (shape[0], capacity, *shape[2:])
                allocator = BoundedOutput(maximum, dtype)
                self.allocators[name] = allocator
                self.context.set_output_allocator(name, allocator)
                self.context.set_tensor_address(name, allocator.buffer.data_ptr())
            else:
                buffer = torch.zeros(shape, dtype=dtype, device='cuda')
                self.persistent_tensors[name] = buffer
                if not self.context.set_tensor_address(name, buffer.data_ptr()):
                    raise RuntimeError('failed to bind ' + name)
        if self.persistent_tensors['image'].dtype != torch.uint8:
            raise ValueError('expected formatted uint8 image; normalization must remain inside engine')

    def preprocess(self, rgb, box=None):
        height, width = self.persistent_tensors['image'].shape[-2:]
        array, inverse = format_image(rgb, height, width, self.format_kind, box)
        return torch.from_numpy(array), dict(inverse=inverse)

    def copy_input_data(self, image):
        destination = self.persistent_tensors['image']
        if image.dtype != destination.dtype or image.shape != destination.shape:
            raise ValueError('prepared input must match exact engine shape and dtype')
        destination.copy_(image)
        # Optional RTMO input is fixed at zero; all confidence cuts are offline.
        return tuple(image.shape)

    def get_outputs(self):
        self.torch_stream.synchronize()
        return {name: (self.allocators[name].value() if name in self.allocators else
                       self.persistent_tensors[name]).clone() for name in self.output_names}

    def _capture_cuda_graph(self, shape):
        self.graph_status = dict(attempted=True, active=False)
        try:
            for _ in range(3):
                self._execute_standard()
            self.torch_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.torch_stream):
                self._execute_standard()
            self.graph_cache[shape] = graph
            self.graph_status['active'] = True
            return graph
        except Exception as error:
            self.graph_cache[shape] = None
            self.graph_status['error'] = str(error)
            return None

    def fresh_context(self):
        """Failed capture can disturb allocator state; never time that context."""
        self.graph_cache.clear()
        torch.cuda.synchronize()
        self.context = self.engine.create_execution_context()
        self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self.initialize_persistent_tensors()


def checked_outputs(runner):
    values = {name: v.cpu().numpy().copy() for name, v in runner.get_outputs().items()}
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError('nonfinite decoded output')
    if 'valid' in values and values['valid'].dtype != np.bool_:
        raise ValueError('valid mask must be boolean')
    return values
