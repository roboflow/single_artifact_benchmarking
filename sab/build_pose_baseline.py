"""Fresh stock SAB FP16 build, with maximum clocks applied BEFORE tactic timing."""

import argparse
import inspect
import json
from pathlib import Path
import time

import onnx
import tensorrt as trt
import torch

from sab.baseline_clocks import MaximumClocks
from sab.benchmark_modes import exclusive_gpu, telemetry
from sab.models.benchmark_rfpose import digest
from sab.trt_inference import build_engine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('fresh engine directory required: ' + str(a.output))
    contract_path = a.onnx.with_suffix('.contract.json')
    contract = json.loads(contract_path.read_text())
    onnx_hash = digest(a.onnx)
    if contract['onnx_sha256'] != onnx_hash:
        raise ValueError('ONNX contract identity mismatch')
    model = onnx.load(a.onnx, load_external_data=False)
    external = {}
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            entries = {v.key: v.value for v in tensor.external_data}
            path = a.onnx.parent / entries['location']
            external[str(path)] = digest(path)
    a.output.mkdir(parents=True)
    engine = a.output / 'model.engine'
    started = time.monotonic()
    error = None
    clocks = MaximumClocks()
    try:
        with exclusive_gpu(), clocks:
            before = telemetry()
            serialized = build_engine(str(a.onnx), str(engine), use_fp16=True,
                                      profiling_verbosity=trt.ProfilingVerbosity.DETAILED)
            if serialized is None:
                raise RuntimeError('SAB stock engine build failed')
            after = telemetry()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        record = dict(onnx=str(a.onnx.resolve()), onnx_sha256=onnx_hash,
            contract_sha256=digest(contract_path), external_weights=external,
            engine=str(engine.resolve()), engine_sha256=digest(engine) if engine.exists() else None,
            tensorrt=trt.__version__, torch=torch.__version__, cuda=torch.version.cuda,
            build_seconds=time.monotonic()-started, error=error,
            builder='sab.trt_inference.build_engine', builder_sha256=digest(inspect.getfile(build_engine)),
            source_sha256=digest(Path(__file__)), clocks_source_sha256=digest(inspect.getfile(MaximumClocks)),
            settings=dict(fp16=True, tf32='SAB/TRT default (enabled)',
                          optimization_level='TRT default', workspace='TRT default',
                          profiling_verbosity='detailed (inspection only)',
                          custom_plugins=False, timing_cache_reused=False, engine_reused=False),
            clocks=clocks.report() if hasattr(clocks, 'sm_mhz') else None,
            before=locals().get('before'), after=locals().get('after'))
        with (a.output / 'build.json').open('x') as output:
            json.dump(record, output, indent=2, allow_nan=False)
    print('SAB_BASELINE_BUILT', a.onnx.stem, record['build_seconds'], record['engine_sha256'], flush=True)


if __name__ == '__main__':
    main()
