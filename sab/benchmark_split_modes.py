"""Measure all four graph policies including the actual two-stage handoff.

The primary CUDA-event interval spans detector, pinned count copy, host
synchronization/dispatch and pose. Wall time is also retained. It is NOT a
sum of separately measured stages. Captures/warmup and initial image
formatting are outside timing. Exact outputs must agree across graph modes.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import tensorrt as trt

from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool, telemetry
from sab.clock_watch import ThrottleMonitor
from sab.models.benchmark_rfpose import RFPoseJointTRTInference, digest
from sab.models.benchmark_rfpose_split import RFPoseSplitTRTInference


def outputs_numpy(runner):
    values = {name: value.cpu().numpy().copy() for name, value in runner.get_outputs().items()}
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError('nonfinite output')
    return values


def compare(reference, actual):
    before, after = reference['valid'][0], actual['valid'][0]
    counts_equal = int(before.sum()) == int(after.sum())
    result = dict(same_retained_count=counts_equal, reference_count=int(before.sum()),
                  actual_count=int(after.sum()))
    if not counts_equal:
        # Different TRT compilations need not be prediction-identical. Record
        # the failure of reference parity; NEVER silently align/drop rows or
        # call this accuracy-validated. Graph-mode equality remains mandatory.
        return result
    for name in reference:
        x, y = reference[name][0, before], actual[name][0, after]
        delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
        result[name] = dict(max_abs=float(delta.max()) if delta.size else 0.,
                            mean_abs=float(delta.mean()) if delta.size else 0.)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--engine-cache', type=Path, required=True)
    p.add_argument('--image-root', type=Path, default=Path('/home/isaac/r-flow/coco/val2017'))
    p.add_argument('--image-ids', nargs='+', type=int, default=[785, 872, 6954, 285])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--joint-engine', type=Path)
    p.add_argument('--joint-onnx', type=Path)
    p.add_argument('--threshold', type=float, default=.48)
    p.add_argument('--per-block', type=int, default=10)
    p.add_argument('--buffer-seconds', type=float, default=.2)
    p.add_argument('--reference-clocks', action='store_true')
    p.add_argument('--capture-counts', nargs='+', type=int, default=list(range(1, 17)))
    p.add_argument('--optimal-batch', type=int, default=1)
    p.add_argument('--optimization-level', type=int, default=3)
    p.add_argument('--max-aux-streams', type=int)
    p.add_argument('--build-only', action='store_true')
    a = p.parse_args()
    if a.output.exists() or a.per_block < 1 or a.buffer_seconds < 0:
        raise ValueError('fresh output and valid timing policy required')
    if bool(a.joint_engine) != bool(a.joint_onnx):
        p.error('supply both joint reference paths')
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    with exclusive_gpu(), retain_cuda_pool(1024):
        runner = RFPoseSplitTRTInference(a.manifest, a.engine_cache, threshold=a.threshold,
            optimal_batch=a.optimal_batch, optimization_level=a.optimization_level,
            max_aux_streams=a.max_aux_streams)
        identity = dict(detector=runner.detector_receipt, pose=runner.pose_receipt,
                        detector_path=str(runner.detector_path), pose_path=str(runner.pose_path),
                        manifest_sha256=digest(a.manifest), source_sha256=digest(Path(__file__)),
                        handler_sha256=digest(Path(__file__).parent / 'models/benchmark_rfpose_split.py'),
                        builder_sha256=digest(Path(__file__).parent / 'models/benchmark_rfpose.py'),
                        tensorrt=trt.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                        clock_watch_sha256=digest(Path(__file__).parent / 'clock_watch.py'))
        with (a.output / 'build.json').open('x') as out:
            json.dump(identity, out, indent=2)
        if a.build_only:
            runner.cleanup()
            return
        first = TF.pil_to_tensor(Image.open(a.image_root / f'{a.image_ids[0]:012d}.jpg').convert('RGB')).cuda()
        runner.prepare(first, counts=a.capture_counts, warmup=30, capture_pose=True)
        joint = None
        if a.joint_engine:
            joint = RFPoseJointTRTInference(a.joint_engine, a.joint_onnx,
                                            threshold=a.threshold, use_cuda_graph=False)
            joint.prepare_for_benchmark(first)

        cases = []
        # Palindromic mode order reduces drift; a joint baseline bookends it.
        modes = [(False, False), (True, False), (True, True), (False, True)]
        modes += list(reversed(modes))
        if joint:
            modes = [None] + modes + [None]
        with ThrottleMonitor() if a.reference_clocks else nullcontext():
            for image_id in a.image_ids:
                image_path = a.image_root / f'{image_id:012d}.jpg'
                pixels = TF.pil_to_tensor(Image.open(image_path).convert('RGB')).cuda()
                runner.threshold = a.threshold
                prepared, _ = runner.preprocess(pixels)
                torch.cuda.synchronize()
                with torch.cuda.stream(runner.torch_stream):
                    runner.copy_input_data(prepared)
                runner.torch_stream.synchronize()
                if joint:
                    joint.pending_inputs = runner.pending_inputs
                    with torch.cuda.stream(joint.torch_stream):
                        joint.copy_input_data(prepared)
                    joint.torch_stream.synchronize()
                blocks, mode_outputs = [], {}
                for block, mode in enumerate(modes):
                    target = joint if mode is None else runner
                    label = 'joint_uncaptured' if mode is None else f'detector_{int(mode[0])}_pose_{int(mode[1])}'
                    samples, wall = [], []
                    before = telemetry()
                    for _ in range(a.per_block):
                        time.sleep(a.buffer_seconds)
                        with torch.cuda.stream(target.torch_stream):
                            started = time.perf_counter_ns()
                            with target.profiler.profile_async(stream=target.torch_stream):
                                if mode is None:
                                    target._execute_standard()
                                else:
                                    target.execute(*mode)
                        target.torch_stream.synchronize()
                        wall.append((time.perf_counter_ns() - started) / 1e6)
                        elapsed = target.profiler.get_last_timing_async()
                        if elapsed is None or not np.isfinite(elapsed) or elapsed <= 0:
                            raise ValueError('invalid timing')
                        samples.append(float(elapsed))
                    actual = outputs_numpy(target)
                    if label in mode_outputs:
                        for name, value in actual.items():
                            np.testing.assert_array_equal(value, mode_outputs[label][name])
                    mode_outputs[label] = actual
                    blocks.append(dict(block=block, mode=label, samples_ms=samples, wall_ms=wall,
                                       before=before, after=telemetry()))
                reference = mode_outputs['detector_0_pose_0']
                for mode, actual in mode_outputs.items():
                    if mode != 'joint_uncaptured':
                        for name, value in reference.items():
                            np.testing.assert_array_equal(value, actual[name])
                    with (a.output / f'{image_id}-{mode}.npz').open('xb') as handle:
                        np.savez_compressed(handle, **actual)
                summary = {}
                for mode in mode_outputs:
                    samples = [x for b in blocks if b['mode'] == mode for x in b['samples_ms']]
                    wall = [x for b in blocks if b['mode'] == mode for x in b['wall_ms']]
                    summary[mode] = dict(median_ms=float(np.median(samples)), mean_ms=float(np.mean(samples)),
                                        wall_median_ms=float(np.median(wall)), samples=len(samples))
                count = int(reference['valid'].sum())
                comparison = compare(mode_outputs['joint_uncaptured'], reference) if joint else None
                row = dict(image_id=image_id, image_sha256=digest(image_path), count=count,
                           summary=summary, blocks=blocks, graph_mode_outputs_identical=True,
                           joint_comparison=comparison)
                cases.append(row)
                with (a.output / f'{image_id}-timings.json').open('x') as out:
                    json.dump(row, out, indent=2)
                print('SAB_SPLIT_RESULT', image_id, 'people', count, json.dumps(summary), flush=True)
        # Change the cutoff on the SAME input buffers/captured detector; then
        # return to the original cutoff. Captures must not freeze selection.
        threshold_probes = []
        for probe_index, cutoff in enumerate((0., a.threshold, 1., a.threshold)):
            runner.threshold = cutoff
            prepared, _ = runner.preprocess(first)
            torch.cuda.synchronize()
            with torch.cuda.stream(runner.torch_stream):
                runner.copy_input_data(prepared)
                runner.execute(True, True)
            runner.torch_stream.synchronize()
            actual = outputs_numpy(runner)
            with (a.output / f'cutoff-{probe_index}-split.npz').open('xb') as handle:
                np.savez_compressed(handle, **actual)
            count = int(actual['valid'].sum())
            if count != runner.last_count or not (actual['detector_scores'] >= np.float32(cutoff)).all():
                raise ValueError('count/cutoff contract failed')
            comparison = None
            if joint:
                joint.pending_inputs = runner.pending_inputs
                with torch.cuda.stream(joint.torch_stream):
                    joint.copy_input_data(prepared)
                    joint._execute_standard()
                joint.torch_stream.synchronize()
                reference = outputs_numpy(joint)
                with (a.output / f'cutoff-{probe_index}-joint.npz').open('xb') as handle:
                    np.savez_compressed(handle, **reference)
                comparison = compare(reference, actual)
            threshold_probes.append(dict(cutoff=cutoff, count=count, joint_comparison=comparison))
            print('SAB_SPLIT_CUTOFF', cutoff, count, json.dumps(comparison), flush=True)
        result = dict(**identity, cases=cases, threshold_probes=threshold_probes,
            threshold=a.threshold, reference_clocks=a.reference_clocks, buffer_seconds=a.buffer_seconds,
            cuda_pool_mib=1024, capture_counts=sorted(runner.pose_graphs),
            uncaptured_pose_counts=sorted(runner.uncaptured_pose_counts),
            warmup_and_capture_excluded=True, exact_batches=True, shared_pose_engine=True,
            activation_bytes=runner.activation_arena.numel(),
            timed_boundary='detector + pinned count D2H + host wait/dispatch + crops/pose/GMM/scores',
            purpose='matched-image diagnostic; not full-COCO accuracy/latency')
        with (a.output / 'result.json').open('x') as out:
            json.dump(result, out, indent=2)
        if joint:
            joint.cleanup()
        runner.cleanup()


if __name__ == '__main__':
    main()
