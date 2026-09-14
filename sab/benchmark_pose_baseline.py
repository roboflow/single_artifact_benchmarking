"""Matched real-input graph-on/off timing of freshly built decoded engines.

Timing is a deterministic representative sample, not full-COCO mean latency.
Optional full-COCO candidate caching is separate and is NOT used for timing.
"""

import argparse
import gc
import inspect
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
import torch

from sab.baseline_clocks import MaximumClocks
from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool, telemetry
from sab.models.benchmark_decoded_pose import DecodedPoseTRTInference, checked_outputs, restore_points
from sab.models.benchmark_rfpose import digest


def read_rgb(path, kind):
    # Set-pose references use Pillow decoding as well as Pillow resizing.
    if kind == 'pil_resize':
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB')).copy()
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare(runner, case):
    rgb = read_rgb(case['image'], runner.format_kind)
    tensor, metadata = runner.preprocess(rgb, case.get('box'))
    with torch.cuda.stream(runner.torch_stream):
        shape = runner.copy_input_data(tensor)
    runner.torch_stream.synchronize()
    return shape, metadata


def summarize(samples):
    values = np.asarray(samples, dtype=np.float64)
    return dict(count=len(values), mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), min_ms=float(values.min()))


def native_smoke(runner, onnx):
    """Existing set-pose native-reference gate; not full-dataset certification."""
    if runner.contract['family'] not in ('ecpose', 'detrpose'):
        return dict(status='not_applicable', reason='no set-pose native reference contract')
    from scipy.optimize import linear_sum_assignment

    path = onnx.with_suffix('.reference.npz')
    if digest(path) != runner.contract['reference_sha256']:
        raise ValueError('native reference identity mismatch')
    with np.load(path, allow_pickle=False) as data:
        expected = dict(data)
    with torch.cuda.stream(runner.torch_stream):
        runner.copy_input_data(torch.from_numpy(expected.pop('image')))
        runner._execute_standard()
    actual = checked_outputs(runner)
    if any(actual[k].shape != v.shape for k, v in expected.items()):
        raise ValueError('native smoke output shape mismatch')
    distances = np.linalg.norm(expected['keypoints'][0, :, None] - actual['keypoints'][0, None], axis=-1).mean(-1)
    distances += (expected['valid'][0, :, None] != actual['valid'][0, None]) * 1e6
    left, right = linear_sum_assignment(distances)
    high = np.maximum(expected['scores'][0, left], actual['scores'][0, right]) >= .1
    if not high.any():
        raise ValueError('no high-confidence row for native smoke')
    errors = {}
    for name in expected:
        first, second = expected[name][0, left][high], actual[name][0, right][high]
        if name == 'valid':
            np.testing.assert_array_equal(first, second)
        else:
            errors[name] = float(np.abs(first-second).max())
            np.testing.assert_allclose(first, second, atol=.02 if name == 'scores' else 2., rtol=1e-4,
                                       err_msg='native reduced-precision smoke: ' + name)
    return dict(status='passed', reference=str(path), reference_sha256=digest(path),
                checked_rows=int(high.sum()), minimum_score=.1, coordinate_atol=2., score_atol=.02,
                max_abs=errors, scope='single native reference; full-COCO validation still required')


def cache_full_val(runner, graph, annotation, image_root, output, identity):
    native = json.loads(annotation.read_text())
    images = sorted(native['images'], key=lambda row: row['id'])
    ids, points, boxes, scores = [], [], [], []
    for index, info in enumerate(images):
        case = dict(image=str(image_root / info['file_name']))
        _, meta = prepare(runner, case)
        with torch.cuda.stream(runner.torch_stream):
            graph.replay() if graph is not None else runner._execute_standard()
        result = checked_outputs(runner)
        valid = result.get('valid', np.ones(result['scores'].shape, dtype=bool))[0]
        xy = restore_points(result['keypoints'][0, valid], meta['inverse'])
        bb = restore_points(result['boxes'][0, valid].reshape(-1, 2, 2), meta['inverse']).reshape(-1, 4)
        if runner.format_kind == 'yolo_letterbox':
            xy[..., 0] = xy[..., 0].clip(0, info['width'])
            xy[..., 1] = xy[..., 1].clip(0, info['height'])
            bb[:, [0, 2]] = bb[:, [0, 2]].clip(0, info['width'])
            bb[:, [1, 3]] = bb[:, [1, 3]].clip(0, info['height'])
        ids.extend([info['id']] * int(valid.sum()))
        points.append(xy.astype(np.float32))
        boxes.append(bb.astype(np.float32))
        scores.append(result['scores'][0, valid].astype(np.float32))
        if (index + 1) % 500 == 0:
            print('SAB_BASELINE_CACHE', identity['id'], index + 1, len(images), flush=True)
    cache = output / 'predictions.npz'
    with cache.open('xb') as handle:
        np.savez_compressed(handle, image_ids=np.asarray([i['id'] for i in images], np.int64),
            prediction_image_ids=np.asarray(ids, np.int64), gate_scores=np.concatenate(scores),
            keypoints=np.concatenate(points), boxes=np.concatenate(boxes),
            pose_quality=np.full(len(ids), np.nan, np.float32))
    record = dict(**identity, schema='decoded_candidates_v1', threshold=0.,
        images=len(images), full_val=len(images) == 5000, predictions=len(ids),
        cache=str(cache), cache_sha256=digest(cache), annotation_sha256=digest(annotation),
        post_pose_threshold=False, all_engine_candidates=True, nms=runner.contract.get('nms', False),
        cuda_graph=graph is not None, status='requires offline F1/AP evaluation',
        protocol='all val images including no-person images; native mask/NMS; confidence cutoff in post')
    with cache.with_suffix('.json').open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--buffer-seconds', type=float, default=.2)
    p.add_argument('--cache-full-val', action='store_true')
    a = p.parse_args()
    if a.output.exists() or a.buffer_seconds < 0:
        raise ValueError('fresh output and nonnegative pacing required')
    spec = json.loads(a.spec.read_text())
    case_path = Path(spec['cases'])
    if digest(case_path) != spec['cases_sha256']:
        raise ValueError('input manifest identity mismatch')
    cases = json.loads(case_path.read_text())['cases']
    if not cases:
        raise ValueError('empty timing sample')
    onnx = Path(spec['onnx'])
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    identity = dict(id=spec['id'], engine=str(a.engine), engine_sha256=digest(a.engine),
        onnx=str(onnx), onnx_sha256=digest(onnx), spec_sha256=digest(a.spec),
        handler_sha256=digest(inspect.getfile(DecodedPoseTRTInference)), source_sha256=digest(Path(__file__)))
    timings, summaries, saved, error = [], [], [], None
    clocks = MaximumClocks()
    try:
        with exclusive_gpu(), retain_cuda_pool(1024), clocks:
            runner = DecodedPoseTRTInference(a.engine, onnx)
            # Hold failures remain visible as diagnostic timing, not accuracy admission.
            try:
                smoke = native_smoke(runner, onnx)
            except Exception as exc:
                smoke = dict(status='failed', error=str(exc))
            shape, _ = prepare(runner, cases[0])
            with torch.cuda.stream(runner.torch_stream):
                for _ in range(100):
                    runner._execute_standard()
            runner.torch_stream.synchronize()
            graph = runner._capture_cuda_graph(shape)
            capture = dict(runner.graph_status)
            if graph is None:
                runner.fresh_context()
                prepare(runner, cases[0])
                with torch.cuda.stream(runner.torch_stream):
                    for _ in range(100):
                        runner._execute_standard()
                runner.torch_stream.synchronize()
                capture['fresh_context_after_failure'] = True
            # Identical cooling/pacing policy for all architectures, no sample dropping.
            time.sleep(2.)
            before = telemetry()
            timing_start = len(clocks.samples)
            for index, case in enumerate(cases):
                if digest(Path(case['image'])) != case['image_sha256']:
                    raise ValueError('timing image changed')
                prepare(runner, case)
                modes = [False, True, True, False] if graph is not None else [False, False]
                if index % 2 and graph is not None:
                    modes = [True, False, False, True]
                outputs = {}
                for block, captured in enumerate(modes):
                    time.sleep(a.buffer_seconds)
                    with torch.cuda.stream(runner.torch_stream):
                        with runner.profiler.profile_async(stream=runner.torch_stream):
                            graph.replay() if captured else runner._execute_standard()
                    runner.torch_stream.synchronize()
                    elapsed = runner.profiler.get_last_timing_async()
                    if elapsed is None or not np.isfinite(elapsed) or elapsed <= 0:
                        raise ValueError('invalid timing')
                    outputs[captured] = checked_outputs(runner)
                    timings.append(dict(case=index, image_id=case['image_id'], block=block,
                                        cuda_graph=captured, ms=float(elapsed)))
                if graph is not None:
                    for name, value in outputs[False].items():
                        np.testing.assert_array_equal(value, outputs[True][name], err_msg='graph/off ' + name)
                if index < 4:
                    saved.append(outputs[False])
                if (index + 1) % 16 == 0:
                    print('SAB_BASELINE_TIMING', spec['id'], index + 1, len(cases), flush=True)
            timing_stop = len(clocks.samples)
            after = telemetry()
            for captured in (False, True):
                values = [r['ms'] for r in timings if r['cuda_graph'] == captured]
                if values:
                    summaries.append(dict(cuda_graph=captured, **summarize(values)))
            print('SAB_BASELINE_TIMED', spec['id'], json.dumps(summaries), flush=True)
            # Retain inspector data for an audit, never alter layers/precision here.
            inspector = runner.engine.create_engine_inspector()
            inspector.execution_context = runner.context
            (a.output / 'layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
            cache = None
            if a.cache_full_val and runner.stage == 'full_image' and smoke['status'] != 'failed':
                cache = cache_full_val(runner, graph, Path(spec['annotation']), Path(spec['image_root']), a.output, identity)
            graph = None
            inspector = None
            runner.cleanup()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
            # Same-source previous engine is a numerical diagnostic only. It does
            # not authorize assigning a prior AP/F1 to the fresh engine hash.
            reference_comparison = []
            previous = DecodedPoseTRTInference(spec['previous_engine'], onnx, use_cuda_graph=False)
            for case, expected in zip(cases[:4], saved):
                prepare(previous, case)
                with torch.cuda.stream(previous.torch_stream):
                    previous._execute_standard()
                actual = checked_outputs(previous)
                delta = {}
                for name, value in expected.items():
                    if actual[name].shape != value.shape:
                        delta[name] = dict(shape_changed=True)
                    elif value.dtype == np.bool_:
                        delta[name] = dict(equal=bool(np.array_equal(value, actual[name])))
                    else:
                        difference = np.abs(value.astype(np.float64) - actual[name].astype(np.float64))
                        delta[name] = dict(max_abs=float(difference.max()) if difference.size else 0.,
                                           mean_abs=float(difference.mean()) if difference.size else 0.)
                reference_comparison.append(dict(image_id=case['image_id'], differences=delta))
            previous.cleanup()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        record = dict(**identity, error=error, summary=summaries, samples=timings,
            capture=locals().get('capture'), native_smoke=locals().get('smoke'),
            previous_engine_comparison=locals().get('reference_comparison'),
            previous_engine_comparison_scope='four same-source inputs; raw row-wise differences, not matched accuracy',
            prediction_cache=locals().get('cache'),
            timing_clock_sample_range=[locals().get('timing_start'), locals().get('timing_stop')],
            clocks=clocks.report() if hasattr(clocks, 'sm_mhz') else None,
            before=locals().get('before'), after=locals().get('after'),
            cases=len(cases), buffer_seconds=a.buffer_seconds, warmup=100, cuda_pool_mib=1024,
            input_population='deterministic uniform sample of all val images' if spec['stage'] == 'full_image'
                             else 'deterministic uniform sample of det56 boxes; one crop per engine invocation',
            boundary='device-resident formatted uint8 image -> decoded keypoints and crop/person scores; '
                     'includes normalization/native decode/scoring; excludes initial formatting, transfers, inverse affine',
            nms_timing_boundary='two-stage pose module: NMS outside engine; excluded (detector-dependent)'
                                if spec['stage'] == 'person_crop' else 'one-stage: native NMS inside and timed when used',
            timing_scope='representative real-input paired ABBA/BAAB samples; NOT full-COCO mean latency',
            tensorrt=trt.__version__, torch=torch.__version__, cuda=torch.version.cuda,
            held_reason=spec.get('held_reason'), accuracy_policy='no automatic reuse of old-engine AP/F1')
        with (a.output / 'result.json').open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
