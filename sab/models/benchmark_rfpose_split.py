"""SAB two-stage RF-Pose runner: one detector engine, one dynamic pose engine.

No RF-Pose imports or graph surgery. Only a count is read on the host.
Geometry stays in persistent GPU buffers, directly bound to the pose engine.
All count-specific execution contexts share one engine and activation arena,
and are strictly sequential. CUDA graph use is independent for each stage.
"""

import json
import math
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
import torch

from sab.models.benchmark_rfpose import RFPoseJointTRTInference, build_stock_typed, digest
from sab.profiler import CUDAProfiler


OUTPUT_NAMES = ('keypoints', 'scores', 'boxes', 'valid', 'detector_scores', 'bucket_ids')


def read_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest['contract'].get('contract_version') != 8:
        raise ValueError('requires a two-stage RF-Pose contract')
    for key, stage in [('detector', 'detector_compact'), ('pose', 'selected_pose')]:
        artifact = path.parent / manifest[key]['path']
        if digest(artifact) != manifest[key]['sha256']:
            raise ValueError('stage ONNX hash mismatch: ' + key)
        model = onnx.load(artifact, load_external_data=False)
        contract = json.loads({prop.key: prop.value for prop in model.metadata_props}['sab.rfpose'])
        if contract.get('stage') != stage or contract.get('contract_version') != 8:
            raise ValueError('incorrect stage contract')
        for identity in ('pose_checkpoint_sha256', 'detector_checkpoint_sha256', 'calibration_sha256',
                         'tokens', 'padding', 'aspect_buckets', 'decode', 'score', 'split_source_sha256'):
            if contract.get(identity) != manifest['contract'].get(identity):
                raise ValueError('incompatible stage provenance: ' + identity)
        if contract.get('custom_plugins') or any(node.domain not in ('', 'ai.onnx')
                or node.op_type in ('NonZero', 'If', 'Loop') for node in model.graph.node):
            raise ValueError('requires standard, externally shaped engine stages')
        if key == 'pose' and contract.get('pose_precision_experiment') != manifest['contract'].get('pose_precision_experiment'):
            raise ValueError('pose precision policy does not match the artifact')
    return manifest


def pose_profiles(capacity=300, optimal=1):
    if not 1 <= optimal <= capacity <= 300:
        raise ValueError('invalid pose profile bounds')
    return {name: [(n,) + tail for n in (1, optimal, capacity)] for name, tail in
            [('center', (2,)), ('size', (2,)), ('selected_scores', ()), ('selected_valid', ())]}


def tensor_dtype(engine, name):
    return torch.from_numpy(np.empty(0, dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype


def aligned_arena(size):
    storage = torch.empty(int(size) + 4096, dtype=torch.uint8, device='cuda')
    offset = -storage.data_ptr() % 4096
    return storage[offset:offset + int(size)]


def require_fused_attention(information, required):
    """Fail closed when a precision policy relies on fused wide accumulation."""
    fused = sum('_gemm_mha_v2' in str(layer)
                for layer in json.loads(information)['Layers'])
    if fused != required:
        raise ValueError(f'precision policy requires {required} fused MHA blocks, found {fused}')
    return fused


class RFPoseSplitTRTInference:
    preprocess = RFPoseJointTRTInference.preprocess
    postprocess = RFPoseJointTRTInference.postprocess

    def __init__(self, manifest_path, cache, *, threshold=.48, optimal_batch=1,
                 optimization_level=3, max_aux_streams=None):
        self.manifest_path = Path(manifest_path)
        self.manifest = read_manifest(self.manifest_path)
        self.contract = self.manifest['contract']
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('threshold must be finite and in [0,1]')
        self.capacity = self.contract['person_capacity']
        self.torch_stream = torch.cuda.Stream()
        self.cuda_stream_ptr = self.torch_stream.cuda_stream
        self.profiler = CUDAProfiler(self.torch_stream)
        options = dict(optimization_level=optimization_level, max_aux_streams=max_aux_streams)
        self.detector_path, self.detector_receipt = build_stock_typed(
            self.manifest_path.parent / self.manifest['detector']['path'], cache, **options)
        self.pose_path, self.pose_receipt = build_stock_typed(
            self.manifest_path.parent / self.manifest['pose']['path'], cache,
            input_profiles=pose_profiles(self.capacity, optimal_batch), **options)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.detector = self.runtime.deserialize_cuda_engine(self.detector_path.read_bytes())
        self.pose = self.runtime.deserialize_cuda_engine(self.pose_path.read_bytes())
        if self.detector is None or self.pose is None:
            raise RuntimeError('engine deserialization failed')
        precision = self.contract.get('pose_precision_experiment', {})
        required_fused = precision.get('required_fused_mha_blocks')
        if required_fused is not None:
            inspector = self.pose.create_engine_inspector()
            information = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            # This target-specific policy is deliberately fail-closed. New TRT
            # naming/tactics require a fresh audit rather than a silent fallback.
            require_fused_attention(information, required_fused)
        self.front_context = self.detector.create_execution_context()
        self.front_context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self.front_buffers = {}
        for i in range(self.detector.num_io_tensors):
            name = self.detector.get_tensor_name(i)
            shape = tuple(self.detector.get_tensor_shape(name))
            if any(d < 1 for d in shape):
                raise ValueError('detector stage must have fixed storage shapes')
            value = torch.empty(shape, dtype=tensor_dtype(self.detector, name), device='cuda')
            self.front_buffers[name] = value
            if not self.front_context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError('failed detector binding')
        self.persistent_tensors = {name: self.front_buffers[name] for name in
                                  ('image', 'source_hw', 'confidence_threshold')}
        self.persistent_tensors['source_image'] = torch.empty(self.contract['source_image_shape'],
                                                             dtype=torch.uint8, device='cuda')
        self.image_input_shape = tuple(self.front_buffers['image'].shape)
        self.pose_buffers = {name: self.front_buffers[name] for name in
                             ('center', 'size', 'selected_scores', 'selected_valid')}
        self.pose_buffers['source_image'] = self.persistent_tensors['source_image']
        for name in OUTPUT_NAMES:
            maximum = tuple(self.capacity if d == -1 else d for d in self.pose.get_tensor_shape(name))
            self.pose_buffers[name] = torch.empty(maximum, dtype=tensor_dtype(self.pose, name), device='cuda')
        self.activation_arena = aligned_arena(self.pose.device_memory_size)
        self.host_count = torch.empty(1, dtype=torch.int32, pin_memory=True)
        self.host_count_numpy = self.host_count.numpy()
        self.contexts, self.pose_graphs = {}, {}
        self.front_graph = None
        self.fallback_context = self._new_context(1)
        self.fallback_count = 1
        self.last_count = 0
        self.uncaptured_pose_counts = set()
        self.cache_outputs, self.output_cache = False, []
        self.prediction_type = 'keypoints'

    def _set_count(self, context, count):
        if not 1 <= count <= self.capacity:
            raise ValueError('invalid person count; never truncate')
        for name, shape in pose_profiles(self.capacity).items():
            actual = (count,) + tuple(shape[0][1:])
            if not context.set_input_shape(name, actual):
                raise RuntimeError('failed dynamic pose shape: ' + name)

    def _new_context(self, count):
        context = self.pose.create_execution_context_without_device_memory()
        context.device_memory = self.activation_arena.data_ptr()
        context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self._set_count(context, count)
        for name, value in self.pose_buffers.items():
            if not context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError('failed pose binding: ' + name)
        if context.infer_shapes():
            raise ValueError('unresolved pose input shapes')
        for name in OUTPUT_NAMES:
            shape = tuple(context.get_tensor_shape(name))
            if any(d < 0 for d in shape) or shape[1] != count:
                raise ValueError('pose stage still has data-dependent output shapes')
        return context

    def copy_input_data(self, image):
        for name, value in self.pending_inputs.items():
            destination = self.persistent_tensors[name]
            if destination.shape != value.shape or destination.dtype != value.dtype:
                raise ValueError('wrong input shape/type: ' + name)
            destination.copy_(value)

    def _front_work(self):
        if not self.front_context.execute_async_v3(self.cuda_stream_ptr):
            raise RuntimeError('detector execution failed')
        # Capture this asynchronous, pinned four-byte copy WITH the detector.
        # The synchronization/host dispatch below is still inside total timing.
        self.host_count.copy_(self.front_buffers['person_count'], non_blocking=True)

    def prepare(self, image, counts=(1, 2, 3, 4, 5), warmup=20, *,
                capture_detector=True, capture_pose=False):
        """Capture known counts outside measurement; one engine serves all."""
        prepared, metadata = self.preprocess(image)
        torch.cuda.synchronize()
        with torch.cuda.stream(self.torch_stream):
            self.copy_input_data(prepared)
            for _ in range(warmup):
                self._front_work()
        self.torch_stream.synchronize()
        if capture_detector and self.front_graph is None:
            self.front_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.front_graph, stream=self.torch_stream):
                self._front_work()
        for count in sorted(set(counts)):
            if count == 0:
                continue
            context = self.contexts.get(count)
            if context is None:
                context = self._new_context(count)
                self.contexts[count] = context
                with torch.cuda.stream(self.torch_stream):
                    for _ in range(warmup):
                        if not context.execute_async_v3(self.cuda_stream_ptr):
                            raise RuntimeError('pose warmup failed')
                self.torch_stream.synchronize()
            if capture_pose and count not in self.pose_graphs:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.torch_stream):
                    if not context.execute_async_v3(self.cuda_stream_ptr):
                        raise RuntimeError('pose capture failed')
                self.pose_graphs[count] = graph
            print('SAB_SPLIT_PREPARED', count, 'pose_graph', capture_pose,
                  'shared_engine', str(self.pose_path), flush=True)
        self.profiler.reset()
        return metadata

    def execute(self, detector_graph=True, pose_graph=False):
        """Caller brackets this WHOLE method in CUDA events; never sum stages."""
        if detector_graph:
            if self.front_graph is None:
                raise RuntimeError('detector graph was not prepared')
            self.front_graph.replay()
        else:
            self._front_work()
        self.torch_stream.synchronize()
        count = int(self.host_count_numpy[0])
        if not 0 <= count <= self.capacity:
            raise ValueError('invalid device count')
        self.last_count = count
        if not count:
            return
        if pose_graph and count in self.pose_graphs:
            self.pose_graphs[count].replay()
        else:
            if pose_graph:
                self.uncaptured_pose_counts.add(count)
            context = self.contexts.get(count)
            if context is None:
                context = self.fallback_context
                if self.fallback_count != count:
                    self._set_count(context, count)
                    self.fallback_count = count
            if not context.execute_async_v3(self.cuda_stream_ptr):
                raise RuntimeError('dynamic pose execution failed')

    def get_outputs(self):
        # All output tensors have a leading singleton image dimension, so a
        # prefix is contiguous and shares the engine's exact N-row layout.
        count = self.last_count
        return {name: value[:, :count].clone() for name, value in self.pose_buffers.items()
                if name in OUTPUT_NAMES}

    def cleanup(self):
        self.torch_stream.synchronize()
        self.front_graph = None
        self.pose_graphs.clear()
        self.contexts.clear()
        self.fallback_context = None
