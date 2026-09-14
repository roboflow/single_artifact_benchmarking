"""SAB handler for an exported, single-engine RF-DETR + RF-Pose model.

Only initial image formatting and trivial result formatting happen here.
Detector filtering, all retained crops, mixed-aspect pose inference, GMM
mode search and scores are in the ONNX/TRT artifact. No RF-Pose source-code
dependency, model construction, ONNX rewriting or custom plugin is needed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import onnx
import torch
import torchvision.transforms.functional as TF
import tensorrt as trt

from sab.trt_inference import TRTInference


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(source)


def _digest(source):
    value = hashlib.sha256()
    for chunk in iter(lambda: source.read(8 << 20), b''):
        value.update(chunk)
    return value.hexdigest()


def read_contract(path):
    model = onnx.load(path, load_external_data=False)
    props = {entry.key: entry.value for entry in model.metadata_props}
    contract = json.loads(props['sab.rfpose'])
    if contract.get('contract_version') != 7 or not contract.get('sab_input'):
        raise ValueError('expected a SAB joint RF-Pose artifact, not a raw-forward/control graph')
    def standard_ops(graph):
        for node in graph.node:
            if node.domain not in ('', 'ai.onnx'):
                return False
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH and not standard_ops(attribute.g):
                    return False
                if attribute.type == onnx.AttributeProto.GRAPHS and not all(standard_ops(g) for g in attribute.graphs):
                    return False
        return True

    if contract.get('custom_plugins') or not standard_ops(model.graph):
        raise ValueError('this handler accepts stock ONNX/TRT operators only')
    return contract


def processed_pose_counts(counts, contract):
    """Account for padding/fast paths without confusing them with predictions."""
    counts = np.asarray(counts, dtype=np.int64)
    batches = contract.get('static_person_batches')
    if not batches:
        return counts if contract.get('skip_empty') else np.maximum(counts, 1)
    batches = np.asarray(batches, dtype=np.int64)
    indices = np.searchsorted(batches, counts)
    outside = indices == len(batches)
    if outside.any() and not contract.get('dynamic_person_fallback'):
        raise ValueError('returned count exceeds the compiled pose batch capacity')
    processed = batches[np.minimum(indices, len(batches) - 1)]
    processed = np.where(outside, counts, processed)
    return np.where(counts == 0, 0, processed)


class BoundedOutput(trt.IOutputAllocator):
    """Reserve storage once; actual person count still comes from the engine."""

    def __init__(self, shape, dtype):
        super().__init__()
        itemsize = torch.empty((), dtype=dtype).element_size()
        size = math.ceil(math.prod(shape) * itemsize / 4096) * 4096
        self.storage = torch.empty(size + 4096, dtype=torch.uint8, device='cuda')
        offset = -self.storage.data_ptr() % 4096
        self.buffer = self.storage[offset:offset + size].view(dtype)
        self.maximum_shape, self.shape, self.error = tuple(shape), None, None

    def reallocate_output(self, name, memory, size, alignment):
        if size > self.buffer.numel() * self.buffer.element_size() or self.buffer.data_ptr() % alignment:
            self.error = f'{name}: output allocation exceeds declared capacity/alignment'
            return 0
        return self.buffer.data_ptr()

    def reallocate_output_async(self, name, memory, size, alignment, stream):
        return self.reallocate_output(name, memory, size, alignment)

    def notify_shape(self, name, shape):
        self.shape = tuple(shape)
        if len(shape) != len(self.maximum_shape) or any(not 0 <= a <= b for a, b in zip(shape, self.maximum_shape)):
            self.error = f'{name}: invalid actual shape {shape}'

    def value(self):
        if self.error or self.shape is None:
            raise RuntimeError(self.error or 'engine did not report an output shape')
        return self.buffer[:math.prod(self.shape)].reshape(self.shape)


class RFPoseJointTRTInference(TRTInference):
    def __init__(self, engine_path, onnx_path, *, threshold=0.48, use_cuda_graph=True):
        self.contract = read_contract(onnx_path)
        receipt = json.loads(Path(engine_path).with_name('build.json').read_text())
        if receipt['identity']['onnx_sha256'] != digest(onnx_path) or receipt['engine_sha256'] != digest(engine_path):
            raise ValueError('engine, ONNX and build receipt must identify the same artifact')
        if receipt['identity']['tensorrt'] != trt.__version__:
            raise ValueError('engine was built with a different TensorRT version; use its pinned uv extra or rebuild')
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('threshold must be finite and in [0,1]')
        super().__init__(str(engine_path), 'image', use_cuda_graph=use_cuda_graph,
                         prediction_type='keypoints')
        self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self.graph_status = dict(requested=bool(use_cuda_graph), active=False, attempted=False)
        self.benchmark_prepared = False
        self.cache_outputs = False
        self.output_cache = []

    def prepare_for_benchmark(self, image, warmup=100):
        """Capture/preflight outside timing; rebuild context after failed capture."""
        prepared, _ = self.preprocess(image)
        torch.cuda.synchronize()

        def warm():
            with torch.cuda.stream(self.torch_stream):
                shape = self.copy_input_data(prepared)
                for _ in range(warmup):
                    self._execute_standard()
            self.torch_stream.synchronize()
            return shape

        shape = warm()
        if self.use_cuda_graph:
            graph = self._capture_cuda_graph(shape)
            if graph is None:
                # TRT can change allocator state during an unsupported capture.
                # A fresh context must be used for the real uncaptured result.
                torch.cuda.synchronize()
                self.context = self.engine.create_execution_context()
                self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
                self.initialize_persistent_tensors()
                warm()
                self.graph_cache[shape] = None
                self.graph_status['fresh_context_after_failure'] = True
        self.profiler.reset()
        self.benchmark_prepared = True

    def infer(self, image):
        if not self.benchmark_prepared:
            self.prepare_for_benchmark(image)
        return super().infer(image)

    def initialize_persistent_tensors(self):
        self.persistent_tensors, self.allocators = {}, {}
        expected = {'image', 'source_image', 'source_hw', 'confidence_threshold'}
        if set(self.input_names) != expected:
            raise ValueError('unexpected joint input contract')
        for name in self.input_names + self.output_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = torch.from_numpy(np.empty(0, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            if name in self.input_names:
                if any(d <= 0 for d in shape):
                    raise ValueError('joint image/metadata input shapes must be fixed')
                value = torch.empty(shape, dtype=dtype, device='cuda')
            else:
                if len(shape) < 2 or any(d < 0 for d in shape[:1] + shape[2:]):
                    raise ValueError('only the output person axis may be data-dependent')
                maximum = tuple(self.contract['person_capacity'] if i == 1 and d < 0 else d for i, d in enumerate(shape))
                allocator = BoundedOutput(maximum, dtype)
                if all(d >= 0 for d in shape):
                    allocator.shape = shape
                self.allocators[name] = allocator
                self.context.set_output_allocator(name, allocator)
                value = allocator.buffer
            self.persistent_tensors[name] = value
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError('failed binding: ' + name)

    def preprocess(self, image):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        _, channels, h, w = image.shape
        shape = tuple(self.persistent_tensors['source_image'].shape)
        if image.shape[0] != 1 or channels != 3 or not (0 < h <= shape[2] and 0 < w <= shape[3]):
            raise ValueError('expected one original RGB image fitting the declared source canvas')
        pixels = image if image.dtype == torch.uint8 else (image * 255).round().clamp(0, 255).to(torch.uint8)
        canvas = pixels.new_zeros(shape)
        canvas[:, :, :h, :w] = pixels
        # Same normalize-then-antialiased-resize order as SAB's RF-DETR handler.
        prepared = TF.normalize(pixels.float() / 255, self.contract['mean'], self.contract['std'])
        prepared = TF.resize(prepared, list(self.image_input_shape[2:]), antialias=True)
        self.pending_inputs = dict(image=prepared, source_image=canvas,
            source_hw=torch.tensor([[h, w]], dtype=torch.float32, device=image.device),
            confidence_threshold=torch.tensor([self.threshold], dtype=torch.float32, device=image.device))
        return prepared, dict(height=h, width=w)

    def copy_input_data(self, image):
        for name, value in self.pending_inputs.items():
            buffer = self.persistent_tensors[name]
            if value.shape != buffer.shape or value.dtype != buffer.dtype:
                raise ValueError('wrong input shape or type: ' + name)
            buffer.copy_(value)
        return tuple(image.shape)

    def _capture_cuda_graph(self, input_shape):
        self.graph_status['attempted'] = True
        # Use SAB's actual capture attempt; never mislabel fallback as captured.
        self.torch_stream.synchronize()
        graph = super()._capture_cuda_graph(input_shape)
        self.graph_status.update(active=graph is not None,
            reason=None if graph is not None else
            'SAB capture failed; runtime conditional/data-dependent execution remains uncaptured')
        return graph

    def get_outputs(self):
        return {name: allocator.value().clone() for name, allocator in self.allocators.items()}

    def postprocess(self, outputs, metadata):
        keep = outputs['valid'][0]
        if getattr(self, 'cache_outputs', False):
            self.output_cache.append({name: value[0, keep].cpu().numpy().copy()
                for name, value in outputs.items() if name != 'valid'})
        xy = outputs['keypoints'][:, keep]
        scale = xy.new_tensor([metadata['width'], metadata['height']])
        xy = xy / scale
        keypoints = torch.cat([xy, torch.ones_like(xy[..., :1])], -1).flatten(-2)
        boxes = outputs['boxes'][:, keep] / scale.repeat(2)
        scores = outputs['scores'][:, keep]
        labels = torch.ones_like(scores, dtype=torch.int64)
        # Formatting only: learned presence weighting, GMM decode and score
        # have already executed in the single TensorRT artifact.
        return boxes.contiguous(), labels, scores.contiguous(), keypoints.contiguous()


class BuildProgress(trt.IProgressMonitor):
    """Low-frequency, thread-safe progress for large standard-ONNX builds."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.phases = {}
        self.last_report = time.monotonic()

    def phase_start(self, phase_name, parent_phase, num_steps):
        with self.lock:
            self.phases[phase_name] = (parent_phase, num_steps, time.monotonic())
            if not parent_phase:
                print('SAB_BUILD_PHASE', phase_name, 'steps', num_steps, flush=True)

    def step_complete(self, phase_name, step):
        with self.lock:
            now = time.monotonic()
            if now - self.last_report >= 30:
                _, total, started = self.phases.get(phase_name, (None, None, now))
                print('SAB_BUILD_PROGRESS', phase_name, step, '/', total,
                      'seconds', round(now - started, 1), flush=True)
                self.last_report = now
        return True

    def phase_finish(self, phase_name):
        with self.lock:
            now = time.monotonic()
            parent, _, started = self.phases.pop(phase_name, (None, None, now))
            if not parent or now - started >= 10:
                print('SAB_BUILD_FINISHED', phase_name, 'seconds', round(now - started, 1), flush=True)


def build_joint(onnx_path, cache, workspace_gib=4, **build_options):
    """Stock TRT build preserving the explicit PyTorch FP16/FP32 islands."""
    read_contract(onnx_path)
    return build_stock_typed(onnx_path, cache, workspace_gib, **build_options)


def build_stock_typed(onnx_path, cache, workspace_gib=4, *, optimization_level=3, max_aux_streams=None,
                      input_profiles=None):
    """Same typed builder for joint artifacts and separately labeled controls.

    A raw detector control is not accepted by the full-pose handler/evaluator.
    This helper changes no graph or precision choices.
    """
    if not 0 <= optimization_level <= 5 or (max_aux_streams is not None and max_aux_streams < 0):
        raise ValueError('optimization_level must be 0..5 and max_aux_streams nonnegative or None')
    identity = dict(onnx_sha256=digest(onnx_path), tensorrt=trt.__version__,
        strongly_typed=True, workspace_gib=workspace_gib, optimization_level=optimization_level,
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        custom_plugins=False)
    if max_aux_streams is not None:
        identity['max_aux_streams'] = max_aux_streams
    if input_profiles is not None:
        # One dynamic-batch engine, not an engine rebuilt for each capture.
        # Make the complete min/opt/max specification part of cache identity.
        normalized = {}
        for name, bounds in input_profiles.items():
            if len(bounds) != 3 or not bounds[0] or any(len(s) != len(bounds[0]) for s in bounds):
                raise ValueError('profiles require three equal-rank min/opt/max shapes')
            if any(not 1 <= lo <= opt <= hi for lo, opt, hi in zip(*bounds)):
                raise ValueError('profile dimensions must satisfy 1 <= min <= opt <= max')
            normalized[name] = [list(s) for s in bounds]
        identity['input_profiles'] = normalized
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(cache) / key
    engine_path, receipt_path = folder / 'model.engine', folder / 'build.json'
    if engine_path.exists() or receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['identity'] != json.loads(json.dumps(identity)) or digest(engine_path) != receipt['engine_sha256']:
            raise ValueError('engine cache identity mismatch')
        return engine_path, receipt
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    started = time.monotonic()
    print('SAB_BUILD_STAGE', 'onnx_import', str(onnx_path), flush=True)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    imported = time.monotonic()
    print('SAB_BUILD_STAGE', 'tactic_build', 'import_seconds', imported - started, flush=True)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    config.builder_optimization_level = optimization_level
    if max_aux_streams is not None:
        config.max_aux_streams = max_aux_streams
    if input_profiles is not None:
        profile = builder.create_optimization_profile()
        dynamic = {network.get_input(i).name for i in range(network.num_inputs)
                   if -1 in tuple(network.get_input(i).shape)}
        if dynamic != set(input_profiles):
            raise ValueError('input profiles must identify every dynamic input, and only dynamic inputs')
        for name, bounds in input_profiles.items():
            # Python returns None on success and raises ValueError on failure.
            profile.set_shape(name, *bounds)
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError('could not add optimization profile')
    progress = BuildProgress()
    config.progress_monitor = progress
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('TensorRT build failed')
    folder.mkdir(parents=True)
    engine_path.write_bytes(bytes(engine))
    receipt = dict(identity=identity, engine_sha256=digest(engine_path),
                   onnx_import_seconds=imported - started, engine_build_seconds=time.monotonic() - imported)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return engine_path, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('onnx', type=Path)
    p.add_argument('--engine-cache', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.48)
    p.add_argument('--cuda-graph', choices=['on', 'off'], default='on')
    p.add_argument('--max-images', type=int)
    p.add_argument('--buffer-seconds', type=float, default=0.2)
    p.add_argument('--workspace-gib', type=int, default=4)
    p.add_argument('--optimization-level', type=int, choices=range(6), default=3)
    p.add_argument('--max-aux-streams', type=int,
                   help='optional stock TRT stream limit; omitted leaves the compiler default')
    p.add_argument('--cuda-pool-mib', type=int, default=1024)
    p.add_argument('--save-predictions', action='store_true',
                   help='opt-in compact pose/detector-output cache for offline F1 and parity')
    a = p.parse_args()
    source_identity = digest(Path(__file__))
    if a.output.exists():
        raise FileExistsError(a.output)
    from sab.evaluation import evaluate
    from sab.clock_watch import ThrottleMonitor
    from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool, telemetry

    if a.max_images is not None and a.max_images < 1:
        p.error('--max-images must be positive')
    if (a.buffer_seconds < 0 or a.cuda_pool_mib < 0 or a.workspace_gib < 1
            or (a.max_aux_streams is not None and a.max_aux_streams < 0)):
        p.error('invalid timing/build policy')
    cache_path = a.output.with_suffix('.npz')
    if a.save_predictions and cache_path.exists():
        raise FileExistsError(cache_path)
    native = json.loads(a.annotations.read_text())
    # Match the inherited evaluator's COCO.getImgIds() insertion order exactly.
    image_ids = [image['id'] for image in native['images']][:a.max_images]
    with exclusive_gpu(), retain_cuda_pool(a.cuda_pool_mib):
        engine, receipt = build_joint(a.onnx, a.engine_cache, a.workspace_gib,
                                     optimization_level=a.optimization_level, max_aux_streams=a.max_aux_streams)
        runner = RFPoseJointTRTInference(engine, a.onnx, threshold=a.threshold,
                                        use_cuda_graph=a.cuda_graph == 'on')
        if runner.contract['keypoints'] != 17:
            raise ValueError('SAB COCO evaluator currently supports body-17 only')
        runner.cache_outputs = a.save_predictions
        # Warmup and the failed-capture preflight can hit the power cap.
        # Finish them before starting the measured-window throttle monitor.
        from PIL import Image

        first = TF.to_tensor(Image.open(a.images / native['images'][0]['file_name']).convert('RGB')).cuda()
        runner.prepare_for_benchmark(first)
        before = telemetry()
        with ThrottleMonitor() as monitor:
            accuracy = evaluate(runner, str(a.images), str(a.annotations),
                buffer_time=a.buffer_seconds, max_images=a.max_images, max_dets=20)
            throttled = monitor.did_throttle()
            during = telemetry()
        after = telemetry()
    if len(runner.profiler.timings) != len(image_ids):
        raise ValueError('one warmed engine timing is required for every evaluation image')
    saved = None
    if a.save_predictions:
        if len(runner.output_cache) != len(image_ids):
            raise ValueError('output cache/evaluator image count mismatch')
        counts = [len(row['scores']) for row in runner.output_cache]
        joined = {key: np.concatenate([row[key] for row in runner.output_cache])
                  for key in runner.output_cache[0]}
        if any(not np.isfinite(value).all() for value in joined.values()):
            raise ValueError('nonfinite output cache')
        if not (joined['detector_scores'] >= np.float32(a.threshold)).all():
            raise ValueError('engine returned a detection below its runtime cutoff')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open('xb') as out:
            processed = processed_pose_counts(counts, runner.contract)
            np.savez_compressed(out, image_ids=np.asarray(image_ids, dtype=np.int64),
                prediction_image_ids=np.repeat(image_ids, counts),
                gate_scores=joined['detector_scores'], keypoints=joined['keypoints'],
                boxes=joined['boxes'], crop_scores=joined['scores'],
                pose_quality=joined['scores'] / np.maximum(joined['detector_scores'], 1e-30),
                bucket_ids=joined['bucket_ids'], returned_counts=counts,
                processed_counts=processed,
                engine_ms=np.asarray(runner.profiler.timings).reshape(-1, 1))
        saved = dict(path=str(cache_path), sha256=digest(cache_path), predictions=sum(counts))
    result = dict(engine=str(engine), build=receipt, onnx=str(a.onnx),
        contract=runner.contract, accuracy_stats=accuracy, latency_stats=runner.profiler.get_stats(),
        cuda_graph=runner.graph_status, throttled=throttled, threshold=a.threshold,
        max_images=a.max_images, buffer_seconds=a.buffer_seconds,
        warmup_excluded=100, capture_excluded=True, cuda_pool_mib=a.cuda_pool_mib,
        throttle_monitor_scope='evaluation only; warmup and capture preflight completed before monitor starts',
        telemetry=dict(before=before, during=during, after=after), prediction_cache=saved,
        images=len(image_ids), full_val=len(image_ids) == len(native['images']),
        annotation_sha256=digest(a.annotations), source_sha256=source_identity,
        clock_watch_sha256=digest(Path(__file__).resolve().parents[1] / 'clock_watch.py'),
        boundary='formatted detector image + original source pixels -> final pose points and scores; '
                 'filtering, crops, GMM decode and scoring all in one engine')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as out:
        json.dump(result, out, indent=2)
    print(json.dumps({k: result[k] for k in ('latency_stats', 'cuda_graph', 'throttled')}), flush=True)
    runner.cleanup()


if __name__ == '__main__':
    main()
