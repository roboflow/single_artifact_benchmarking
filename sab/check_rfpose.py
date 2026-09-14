"""Validate a joint engine on exact exported inputs and changing cutoffs.

No annotations are loaded. Numerical differences from the PyTorch reference
are reported, not mistaken for a full-COCO accuracy result. This preflight
uses SAB's real bindings and execution path, including dynamic outputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool
from sab.models.benchmark_rfpose import RFPoseJointTRTInference, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--onnx', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    reference_path = a.onnx.with_suffix('.reference.npz')
    source_hash = digest(Path(__file__))
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = dict(archive)
    example_cutoff = float(reference['confidence_threshold'][0])
    probes, comparison = [], {}
    with exclusive_gpu(), retain_cuda_pool(1024):
        runner = RFPoseJointTRTInference(a.engine, a.onnx, use_cuda_graph=False)
        inputs = {name: torch.from_numpy(reference[name]).cuda() for name in runner.input_names}
        for cutoff in sorted(set([0., example_cutoff, 1.])):
            inputs['confidence_threshold'] = torch.tensor([cutoff], dtype=torch.float32, device='cuda')
            runner.pending_inputs = inputs
            runner.torch_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(runner.torch_stream):
                runner.copy_input_data(inputs['image'])
                for _ in range(3):
                    runner._execute_standard()
            runner.torch_stream.synchronize()
            output = {name: value.cpu().numpy() for name, value in runner.get_outputs().items()}
            if any(not np.isfinite(value).all() for value in output.values()):
                raise ValueError('nonfinite engine output')
            keep = output['valid'][0]
            if not (output['detector_scores'][0, keep] >= np.float32(cutoff)).all():
                raise ValueError('runtime detector cutoff was not honored')
            count = int(keep.sum())
            probes.append(dict(threshold=cutoff, returned=count, output_rows=len(keep)))
            if cutoff == example_cutoff:
                other = reference['valid'][0]
                comparison['same_retained_count'] = count == int(other.sum())
                if comparison['same_retained_count']:
                    for name in ('keypoints', 'scores', 'boxes', 'detector_scores', 'bucket_ids'):
                        x, y = output[name][0, keep], reference[name][0, other]
                        delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                        comparison[name] = dict(max_abs=float(delta.max()) if delta.size else 0.,
                                                mean_abs=float(delta.mean()) if delta.size else 0.)
        if any(x['returned'] < y['returned'] for x, y in zip(probes, probes[1:])):
            raise ValueError('higher detector cutoff increased retained count')
        runner.cleanup()
    result = dict(engine_sha256=digest(a.engine), onnx_sha256=digest(a.onnx),
        reference_sha256=digest(reference_path), source_sha256=source_hash,
        threshold_probes=probes, pytorch_reference_comparison=comparison,
        finite_outputs_and_runtime_threshold_checks=True,
        accuracy_status='one-image preflight only; full-COCO validation still required')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
