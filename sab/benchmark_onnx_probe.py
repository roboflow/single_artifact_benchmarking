"""Stock-TRT numerical and graph-on/off diagnostics for fixed-shape ONNX.

Consumes only portable ONNX and exact-input NPZ references. No source model,
graph rewriting, plugins or claimed end-to-end latency. Nonfinite experiments
are recorded as failures, not silently filtered or accepted.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import tensorrt as trt
import torch

from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool, telemetry
from sab.models.benchmark_rfpose import build_stock_typed, digest
from sab.models.benchmark_rfpose_split import tensor_dtype


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('probes', type=Path)
    p.add_argument('--engine-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--names', nargs='+')
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--buffer-seconds', type=float, default=.1)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    paths = [a.probes / (name + '.onnx') for name in a.names] if a.names else sorted(a.probes.glob('*.onnx'))
    with exclusive_gpu(), retain_cuda_pool(1024):
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        for path in paths:
            engine_path, build = build_stock_typed(path, a.engine_cache)
            engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
            context = engine.create_execution_context()
            context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
            inspector = engine.create_engine_inspector()
            layers = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            (a.output / (path.stem + '.layers.json')).write_text(layers)
            with np.load(path.with_suffix('.reference.npz')) as source:
                reference = dict(source)
            buffers = {}
            stream = torch.cuda.Stream()
            output_names = []
            with torch.cuda.stream(stream):
                for i in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(i)
                    buffer = torch.empty(tuple(engine.get_tensor_shape(name)),
                                         dtype=tensor_dtype(engine, name), device='cuda')
                    buffers[name] = buffer
                    context.set_tensor_address(name, buffer.data_ptr())
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        buffer.copy_(torch.from_numpy(reference[name]))
                    else:
                        output_names.append(name)
                for _ in range(10):
                    if not context.execute_async_v3(stream.cuda_stream):
                        raise RuntimeError('probe execution failed')
            stream.synchronize()
            actual = {name: buffers[name].cpu().numpy().copy() for name in output_names}
            numerical = {}
            for name, value in actual.items():
                difference = np.abs(value.astype(np.float64)-reference[name].astype(np.float64))
                numerical[name] = dict(finite=bool(np.isfinite(value).all()),
                    nonfinite_count=int((~np.isfinite(value)).sum()),
                    max_abs=float(difference.max()) if np.isfinite(difference).all() else None,
                    mean_abs=float(difference.mean()) if np.isfinite(difference).all() else None)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError('probe capture failed')
            timings = []
            before = telemetry()
            for captured in (False, True, True, False):
                samples = []
                for _ in range(a.samples // 2):
                    time.sleep(a.buffer_seconds)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        start.record(stream)
                        if captured:
                            graph.replay()
                        elif not context.execute_async_v3(stream.cuda_stream):
                            raise RuntimeError('timed probe failed')
                        end.record(stream)
                    stream.synchronize()
                    samples.append(start.elapsed_time(end))
                timings.append(dict(cuda_graph=captured, samples_ms=samples))
            result = dict(onnx_sha256=digest(path), build=build, engine_path=str(engine_path),
                numerical=numerical, timings=timings, before=before, after=telemetry(),
                inspector_layers=engine.num_layers, source_sha256=digest(Path(__file__)),
                purpose='isolated fixed-input operator diagnostic, NOT whole-image latency')
            with (a.output / (path.stem + '.json')).open('x') as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
            np.savez_compressed(a.output / (path.stem + '.npz'), **actual)
            print('SAB_ONNX_PROBE', path.stem, json.dumps(numerical),
                  'layers', engine.num_layers, flush=True)
            graph = None
            context = inspector = engine = None
            buffers.clear()


if __name__ == '__main__':
    main()
