"""Compare each split stage against its exact-input CPU PyTorch reference.

No annotations are loaded. Also exercise the full dynamic profile, including
batch 300, without truncating. These checks diagnose compiler differences;
they do not replace full-COCO accuracy validation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool
from sab.models.benchmark_rfpose import digest
from sab.models.benchmark_rfpose_split import RFPoseSplitTRTInference, OUTPUT_NAMES


def delta(x, y):
    difference = np.abs(x.astype(np.float64) - y.astype(np.float64))
    return dict(max_abs=float(difference.max()) if difference.size else 0.,
                mean_abs=float(difference.mean()) if difference.size else 0.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--engine-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    with exclusive_gpu(), retain_cuda_pool(1024):
        runner = RFPoseSplitTRTInference(a.manifest, a.engine_cache)
        root = a.manifest.parent
        with np.load(root / Path(runner.manifest['detector']['path']).with_suffix('.reference.npz')) as source:
            front = dict(source)
        with np.load(root / Path(runner.manifest['pose']['path']).with_suffix('.reference.npz')) as source:
            pose = dict(source)
        with torch.cuda.stream(runner.torch_stream):
            for name in ('image', 'source_hw', 'confidence_threshold'):
                runner.front_buffers[name].copy_(torch.from_numpy(front[name]))
            for _ in range(3):
                runner._front_work()
        runner.torch_stream.synchronize()
        actual_front = {name: runner.front_buffers[name].cpu().numpy().copy() for name in
                         ('center', 'size', 'selected_scores', 'selected_valid', 'person_count')}
        front_delta = {name: delta(value, front[name]) for name, value in actual_front.items()}
        np.savez_compressed(a.output / 'detector-output.npz', **actual_front)
        counts = []
        for count in (1, 2, 3, 5, 40, 300):
            with torch.cuda.stream(runner.torch_stream):
                for name in ('source_image', 'center', 'size', 'selected_scores', 'selected_valid'):
                    array = pose[name]
                    if name == 'source_image':
                        runner.pose_buffers[name].copy_(torch.from_numpy(array))
                    else:
                        array = np.concatenate([array] * ((count + len(array) - 1) // len(array)))[:count]
                        runner.pose_buffers[name][:count].copy_(torch.from_numpy(array))
                context = runner._new_context(count)
                for _ in range(3):
                    if not context.execute_async_v3(runner.cuda_stream_ptr):
                        raise RuntimeError('dynamic profile execution failed')
            runner.torch_stream.synchronize()
            runner.last_count = count
            actual = {name: value.cpu().numpy().copy() for name, value in runner.get_outputs().items()}
            if any(not np.isfinite(value).all() for value in actual.values()):
                raise ValueError('nonfinite dynamic pose output')
            expected = {name: np.concatenate([pose[name]] * ((count + 2) // 3), axis=1)[:, :count]
                        for name in OUTPUT_NAMES}
            comparison = {name: delta(actual[name], expected[name]) for name in OUTPUT_NAMES}
            counts.append(dict(count=count, reference_comparison=comparison, finite=True))
            np.savez_compressed(a.output / f'pose-{count}.npz', **actual)
        result = dict(manifest_sha256=digest(a.manifest), source_sha256=digest(Path(__file__)),
            detector_comparison=front_delta, pose_batch_checks=counts,
            torch_reference_count=int(front['person_count'][0]),
            trt_count=int(actual_front['person_count'][0]),
            purpose='exact stage inputs; full-COCO evaluation is a separate required check')
        with (a.output / 'result.json').open('x') as handle:
            json.dump(result, handle, indent=2)
        print('SAB_SPLIT_REFERENCE', json.dumps(result), flush=True)
        runner.cleanup()


if __name__ == '__main__':
    main()
