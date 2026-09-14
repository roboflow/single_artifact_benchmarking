"""Paired graph-on/off timing through SAB handlers, with explicit policies.

One serialized artifact and identical inputs in both modes; no graph edits.
Capture/warmup, image formatting and output copies are outside CUDA events.
Unavailable capture is reported, never relabeled as graph-enabled timing.
"""

import argparse
from contextlib import contextmanager, nullcontext
import ctypes
import fcntl
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import tensorrt as trt

from sab.models.benchmark_rfpose import digest


def telemetry():
    fields = ['uuid', 'name', 'driver_version', 'temperature.gpu', 'clocks.current.sm',
              'clocks.current.memory', 'power.draw', 'power.limit', 'clocks_event_reasons.active']
    result = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=' + ','.join(fields),
                                     '--format=csv,noheader,nounits'], text=True)
    return dict(zip(fields, result.strip().split(', ')))


@contextmanager
def exclusive_gpu():
    with open('/tmp/rfpose-t4-engine-benchmark.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'], text=True)
        if any(int(pid) != os.getpid() for pid in active.split()):
            raise RuntimeError('other GPU process present; refusing benchmark')
        yield


@contextmanager
def retain_cuda_pool(mib):
    torch.cuda.init()
    api = ctypes.CDLL('libcudart.so.12')
    pool, old = ctypes.c_void_p(), ctypes.c_uint64()
    limit = ctypes.c_uint64(mib << 20)

    def check(code):
        if code != 0:
            raise RuntimeError(f'CUDA pool API failed: {code}')

    check(api.cudaDeviceGetDefaultMemPool(ctypes.byref(pool), 0))
    check(api.cudaMemPoolGetAttribute(pool, 4, ctypes.byref(old)))
    check(api.cudaMemPoolSetAttribute(pool, 4, ctypes.byref(limit)))
    try:
        yield
    finally:
        check(api.cudaMemPoolSetAttribute(pool, 4, ctypes.byref(old)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handler', required=True, help='trusted SAB module:class')
    p.add_argument('--handler-kwargs', default='{}', help='JSON constructor keywords')
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--image-ids', type=int, nargs='+', default=[785])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-block', type=int, default=25)
    p.add_argument('--buffer-seconds', type=float, default=0.2)
    p.add_argument('--reference-clocks', action='store_true')
    p.add_argument('--cuda-pool-mib', type=int, default=1024)
    p.add_argument('--save-outputs', action='store_true')
    a = p.parse_args()
    if a.output.exists() or a.per_block < 1 or a.buffer_seconds < 0 or a.cuda_pool_mib < 0:
        raise ValueError('fresh output and valid sample/policy values required')
    a.output.mkdir(parents=True)
    module, name = a.handler.split(':')
    if not module.startswith('sab.models.'):
        raise ValueError('use a SAB model handler')
    handler = getattr(importlib.import_module(module), name)
    source_identity = dict(source_sha256=digest(Path(__file__)),
                           handler_sha256=digest(inspect.getfile(handler)))
    torch.set_num_threads(2)
    from sab.clock_watch import ThrottleMonitor

    cases = []
    graph_status = None
    with exclusive_gpu(), retain_cuda_pool(a.cuda_pool_mib):
        runner = handler(str(a.engine), **json.loads(a.handler_kwargs))
        source_identity['runner_sha256'] = digest(inspect.getfile(type(runner).__mro__[1]))
        runner.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        with ThrottleMonitor() if a.reference_clocks else nullcontext():
            for image_id in a.image_ids:
                path = a.image_root / f'{image_id:012d}.jpg'
                image = TF.to_tensor(Image.open(path).convert('RGB')).cuda()
                prepared, metadata = runner.preprocess(image)
                torch.cuda.synchronize()
                with torch.cuda.stream(runner.torch_stream):
                    shape = runner.copy_input_data(prepared)
                    for _ in range(100):
                        runner._execute_standard()
                runner.torch_stream.synchronize()
                if shape not in runner.graph_cache:
                    graph = runner._capture_cuda_graph(shape)
                    graph_status = getattr(runner, 'graph_status', dict(active=graph is not None))
                    if graph is None:
                        # A failed capture can alter TRT allocator state.
                        # Time a clean context, retaining the explicit failed
                        # preflight rather than contaminating the off result.
                        runner.cleanup()
                        del runner
                        runner = handler(str(a.engine), **json.loads(a.handler_kwargs))
                        runner.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
                        prepared, metadata = runner.preprocess(image)
                        torch.cuda.synchronize()
                        with torch.cuda.stream(runner.torch_stream):
                            runner.copy_input_data(prepared)
                            for _ in range(100):
                                runner._execute_standard()
                        runner.torch_stream.synchronize()
                        runner.graph_cache[shape] = None
                else:
                    graph = runner.graph_cache[shape]
                blocks, outputs = [], {}
                # Keep ABBA order for supported capture. Unsupported capture
                # has two explicit off blocks, not invented on-mode numbers.
                modes = [False, True, True, False] if graph is not None else [False, False]
                for block, captured in enumerate(modes):
                    execute = graph.replay if captured else runner._execute_standard
                    before, samples = telemetry(), []
                    for _ in range(a.per_block):
                        time.sleep(a.buffer_seconds)
                        with torch.cuda.stream(runner.torch_stream):
                            with runner.profiler.profile_async(stream=runner.torch_stream):
                                execute()
                        runner.torch_stream.synchronize()
                        elapsed = runner.profiler.get_last_timing_async()
                        if elapsed is None or not np.isfinite(elapsed) or elapsed <= 0:
                            raise RuntimeError('invalid CUDA-event timing')
                        samples.append(float(elapsed))
                    values = {key: value.cpu().numpy().copy() for key, value in runner.get_outputs().items()}
                    if any(not np.isfinite(value).all() for value in values.values()):
                        raise ValueError('nonfinite engine output')
                    outputs[captured] = values
                    blocks.append(dict(block=block, cuda_graph=captured, samples_ms=samples,
                                       before=before, after=telemetry()))
                if graph is not None:
                    for key in outputs[False]:
                        np.testing.assert_array_equal(outputs[False][key], outputs[True][key])
                summary = []
                for captured in sorted(outputs):
                    samples = [x for row in blocks if row['cuda_graph'] == captured for x in row['samples_ms']]
                    summary.append(dict(cuda_graph=captured, count=len(samples),
                        median_ms=float(np.median(samples)), mean_ms=float(np.mean(samples))))
                    if a.save_outputs:
                        with (a.output / f'{image_id}-graph-{int(captured)}.npz').open('xb') as out:
                            np.savez_compressed(out, **outputs[captured])
                row = dict(image_id=image_id, image_sha256=digest(path), blocks=blocks, summary=summary,
                           cuda_graph_supported=graph is not None,
                           cuda_graph_preflight=graph_status,
                           returned=int(outputs[False]['valid'].sum()) if 'valid' in outputs[False] else None,
                           graph_on_off_equal=True if graph is not None else None)
                cases.append(row)
                print('SAB_GRAPH_MODES', image_id, json.dumps(summary), 'capture_supported', graph is not None, flush=True)
        result = dict(engine=str(a.engine), engine_sha256=digest(a.engine),
            handler=a.handler, handler_kwargs=json.loads(a.handler_kwargs),
            **source_identity, cases=cases,
            reference_clocks=a.reference_clocks, buffer_seconds=a.buffer_seconds,
            cuda_pool_mib=a.cuda_pool_mib, nvtx='none', warmup=100,
            tensorrt=trt.__version__, torch=torch.__version__, cuda=torch.version.cuda,
            purpose='matched-image mode diagnostic, not full-COCO latency or AP')
        with (a.output / 'result.json').open('x') as out:
            json.dump(result, out, indent=2)
        runner.cleanup()


if __name__ == '__main__':
    main()
