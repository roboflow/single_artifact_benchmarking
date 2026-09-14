"""Matched RF-Pose timing at the baseline campaign's 200 ms pacing.

Fresh maximum-clock builds; same 64 real inputs and paired ABBA/BAAB graph
modes. Whole pipeline is measured directly, never by summing stage timings.
Separate selected-pose and formatted-crop measurements have explicit boundaries.
"""

import argparse
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from sab.baseline_clocks import MaximumClocks
from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool
from sab.benchmark_pose_baseline import summarize
from sab.benchmark_split_modes import outputs_numpy
from sab.models.benchmark_decoded_pose import checked_outputs
from sab.models.benchmark_rfpose import build_stock_typed, digest
from sab.models.benchmark_rfpose_crop import RFPoseCropTRTInference, crop_profiles
from sab.models.benchmark_rfpose_split import RFPoseSplitTRTInference


def order(index, captured=True):
    return ([False, True, True, False] if index % 2 == 0 else [True, False, False, True]) if captured else [False, False]


def assert_equal(before, after):
    if set(before) != set(after):
        raise ValueError('output contract changed between executions')
    for name in before:
        np.testing.assert_array_equal(before[name], after[name], err_msg='graph/repetition ' + name)


def summaries(rows):
    result = []
    for mode in (False, True):
        selected = [r['ms'] for r in rows if r['cuda_graph'] == mode]
        if selected:
            result.append(dict(cuda_graph=mode, **summarize(selected)))
    return result


def elapsed(runner, call):
    # Cooling and all input copies/captures are excluded from the timer.
    time.sleep(.2)
    with torch.cuda.stream(runner.torch_stream):
        with runner.profiler.profile_async(stream=runner.torch_stream):
            call()
    runner.torch_stream.synchronize()
    value = runner.profiler.get_last_timing_async()
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError('invalid CUDA event latency')
    return float(value)


def pipeline_input(runner, path):
    with Image.open(path) as im:
        pixels = TF.pil_to_tensor(im.convert('RGB')).cuda()
    prepared, _ = runner.preprocess(pixels)
    torch.cuda.synchronize()
    with torch.cuda.stream(runner.torch_stream):
        runner.copy_input_data(prepared)
    runner.torch_stream.synchronize()
    return pixels


def fresh_pipeline(a, record):
    runner = RFPoseSplitTRTInference(a.manifest, a.output / 'engines', threshold=a.threshold)
    record['builds'] = dict(detector=runner.detector_receipt, pose=runner.pose_receipt)
    record['contract'] = runner.contract
    return runner


def cache_val(runner, a, record, detector_graph):
    if not a.annotation or not a.image_root:
        raise ValueError('full-val cache needs annotation image manifest and image root')
    payload = json.loads(a.annotation.read_text())
    images = sorted(payload['images'], key=lambda r: r['id'])
    if len(images) != 5000 or len({i['id'] for i in images}) != 5000:
        raise ValueError('requires all 5000 COCO val images, including negatives')
    saved, counts = [], []
    for index, info in enumerate(images):
        pipeline_input(runner, a.image_root / info['file_name'])
        with torch.cuda.stream(runner.torch_stream):
            runner.execute(detector_graph, False)
        runner.torch_stream.synchronize()
        result = outputs_numpy(runner)
        if not result['valid'].all() or not (result['detector_scores'] >= np.float32(a.threshold)).all():
            raise ValueError('invalid rows or below-threshold predictions')
        saved.append({k: v[0] for k, v in result.items()})
        counts.append(runner.last_count)
        if (index + 1) % 500 == 0:
            print('RFPOSE_PACED_ACCURACY', a.id, index + 1, 5000, flush=True)
    joined = {k: np.concatenate([r[k] for r in saved]) for k in saved[0]}
    cache = a.output / 'predictions.npz'
    with cache.open('xb') as handle:
        np.savez_compressed(handle, image_ids=np.asarray([i['id'] for i in images], np.int64),
            prediction_image_ids=np.repeat([i['id'] for i in images], counts),
            keypoints=joined['keypoints'], boxes=joined['boxes'], gate_scores=joined['detector_scores'],
            crop_scores=joined['scores'], pose_quality=joined['scores'] / np.maximum(joined['detector_scores'], 1e-30),
            bucket_ids=joined['bucket_ids'], returned_counts=counts)
    record['prediction_cache'] = dict(path=str(cache), sha256=digest(cache), images=5000,
        predictions=sum(counts), threshold=a.threshold, pose_nms=False, flip=False,
        annotation_sha256=digest(a.annotation), timed=False,
        caveat='gated cache; only confidence thresholds >= the engine cutoff can be evaluated')
    with cache.with_suffix('.json').open('x') as handle:
        json.dump(dict(schema='decoded_candidates_v1', threshold=a.threshold,
            cache_sha256=digest(cache), images=5000, full_val=True, predictions=sum(counts),
            annotation_sha256=digest(a.annotation), builds=record['builds'],
            manifest_sha256=record['manifest_sha256'], pose_nms=False, flip=False,
            post_pose_threshold=False), handle, indent=2, allow_nan=False)


def pipeline(a, cases, clocks, record):
    runner = fresh_pipeline(a, record)
    first = pipeline_input(runner, cases[0]['image'])
    try:
        runner.prepare(first, counts=(1, 2, 3, 5), warmup=100, capture_detector=True, capture_pose=False)
        capture = dict(active=True)
    except Exception as exc:
        capture = dict(active=False, error=str(exc), fresh_context_after_failure=True)
        runner.cleanup()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
        runner = fresh_pipeline(a, record)
        first = pipeline_input(runner, cases[0]['image'])
        runner.prepare(first, counts=(1, 2, 3, 5), warmup=100, capture_detector=False, capture_pose=False)
    record['detector_capture'] = capture
    rows, stage_rows, stage_capture = [], [], {}
    time.sleep(2.)
    record['timing_clock_start'] = len(clocks.samples)
    for index, case in enumerate(cases):
        if digest(case['image']) != case['image_sha256']:
            raise ValueError('timing image changed')
        pipeline_input(runner, case['image'])
        # Resolve the exact dynamic batch and warm its context before timing;
        # one-time context creation is not steady-state engine execution.
        with torch.cuda.stream(runner.torch_stream):
            runner.execute(False, False)
        runner.torch_stream.synchronize()
        reference = None
        for mode in order(index, capture['active']):
            ms = elapsed(runner, lambda: runner.execute(mode, False))
            current = outputs_numpy(runner)
            if reference is not None:
                assert_equal(reference, current)
            reference = current
            rows.append(dict(case=index, image_id=case['image_id'], people=runner.last_count,
                             cuda_graph=mode, pose_graph=False, ms=ms))
        count = runner.last_count
        # Independently time the actual deployed second stage. This includes
        # crop extraction and must NOT be compared to formatted-crop timings.
        if count:
            context = runner.contexts.get(count)
            if context is None:
                context = runner._new_context(count)
                runner.contexts[count] = context
            if count not in stage_capture:
                try:
                    with torch.cuda.stream(runner.torch_stream):
                        for _ in range(3):
                            if not context.execute_async_v3(runner.cuda_stream_ptr):
                                raise RuntimeError('pose warmup failed')
                    runner.torch_stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=runner.torch_stream):
                        if not context.execute_async_v3(runner.cuda_stream_ptr):
                            raise RuntimeError('pose capture failed')
                    runner.pose_graphs[count] = graph
                    stage_capture[count] = dict(active=True)
                except Exception as exc:
                    stage_capture[count] = dict(active=False, error=str(exc))
                    runner.contexts[count] = runner._new_context(count)
                    context = runner.contexts[count]
            graph = runner.pose_graphs.get(count)
            for mode in order(index, graph is not None):
                def call():
                    if mode:
                        graph.replay()
                    elif not context.execute_async_v3(runner.cuda_stream_ptr):
                        raise RuntimeError('pose execution failed')
                ms = elapsed(runner, call)
                assert_equal(reference, outputs_numpy(runner))
                stage_rows.append(dict(case=index, image_id=case['image_id'], people=count, cuda_graph=mode, ms=ms))
        else:
            stage_rows.extend(dict(case=index, image_id=case['image_id'], people=0, cuda_graph=mode, ms=0., skipped=True)
                              for mode in order(index))
        if (index + 1) % 16 == 0:
            print('RFPOSE_PACED_TIMING', a.id, index + 1, len(cases), flush=True)
    record.update(timing_clock_stop=len(clocks.samples), summary=summaries(rows), samples=rows,
        stage2_summary=summaries(stage_rows), stage2_samples=stage_rows, stage2_capture=stage_capture,
        stage2_boundary='source image + selected geometry -> crops + pose + GMM decode + scores; no detector or count handoff; zero on empty images',
        stage2_is_formatted_crop_benchmark=False,
        boundary='detector + GPU selection + 4-byte count handoff + host wait/dispatch + crops + pose + GMM + score; no NMS',
        graph_on_off_equal=True, pose_capture_in_primary=False)
    print('RFPOSE_PACED_RESULT', a.id, json.dumps(record['summary']), flush=True)
    # Publish timing before the independent, untimed full-val forward pass.
    with (a.output / 'timing.json').open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    if a.cache_full_val:
        cache_val(runner, a, record, capture['active'])
    runner.cleanup()


def crop(a, cases, clocks, record):
    contract = json.loads(a.onnx.with_suffix('.contract.json').read_text())
    engine, build = build_stock_typed(a.onnx, a.output / 'engines', input_profiles=crop_profiles(contract))
    record.update(builds=dict(crop=build), contract=contract)
    with np.load(a.onnx.with_suffix('.reference.npz'), allow_pickle=False) as archive:
        native = dict(archive)
    if digest(a.onnx.with_suffix('.reference.npz')) != contract['reference_sha256']:
        raise ValueError('reference cache changed')
    ref_runner = RFPoseCropTRTInference(engine, a.onnx, batch=len(native['image']))
    with torch.cuda.stream(ref_runner.torch_stream):
        ref_runner.copy_input_data({n: torch.from_numpy(native[n]) for n in ref_runner.input_names})
        ref_runner._execute_standard()
    actual = checked_outputs(ref_runner)
    errors = {n: float(np.abs(actual[n] - native[n]).max()) for n in actual}
    record['native_reference'] = dict(max_abs=errors, reference_sha256=contract['reference_sha256'],
                                     scope='four real det56 crops; not full-dataset AP certification')
    # Crop comparisons retain all joints; no score masking to hide differences.
    for name in actual:
        np.testing.assert_allclose(actual[name], native[name], atol=2. if name == 'keypoints' else .02, rtol=1e-4)
    ref_runner.cleanup()
    del ref_runner
    gc.collect()
    torch.cuda.empty_cache()
    runner = RFPoseCropTRTInference(engine, a.onnx, batch=1)
    prepared = []
    for case in cases:
        if digest(case['image']) != case['image_sha256']:
            raise ValueError('timing image changed')
        bgr = cv2.imread(case['image'])
        if bgr is None:
            raise ValueError('image decode failed')
        inputs, _ = runner.preprocess([(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), case['box'])])
        prepared.append(inputs)
    with torch.cuda.stream(runner.torch_stream):
        shape = runner.copy_input_data(prepared[0])
        for _ in range(100):
            runner._execute_standard()
    runner.torch_stream.synchronize()
    graph = runner._capture_cuda_graph(shape)
    record['capture'] = dict(runner.graph_status)
    if graph is None:
        runner.fresh_context()
        record['capture']['fresh_context_after_failure'] = True
    time.sleep(2.)
    record['timing_clock_start'] = len(clocks.samples)
    rows = []
    for index, (case, inputs) in enumerate(zip(cases, prepared)):
        with torch.cuda.stream(runner.torch_stream):
            runner.copy_input_data(inputs)
        runner.torch_stream.synchronize()
        reference = None
        for mode in order(index, graph is not None):
            ms = elapsed(runner, graph.replay if mode else runner._execute_standard)
            current = checked_outputs(runner)
            if reference is not None:
                assert_equal(reference, current)
            reference = current
            rows.append(dict(case=index, image_id=case['image_id'], bucket=int(inputs['bucket_id'][0]), cuda_graph=mode, ms=ms))
        if (index + 1) % 16 == 0:
            print('RFPOSE_PACED_CROP', a.id, index + 1, len(cases), flush=True)
    record.update(timing_clock_stop=len(clocks.samples), summary=summaries(rows), samples=rows,
        graph_on_off_equal=True, batch=1, fused_attention_blocks=getattr(runner, 'fused_attention_blocks', None),
        boundary='formatted uint8 crop + relative aspect + bucket -> normalization + patchification + pose + GMM decode + crop score; crop extraction and NMS outside')
    print('RFPOSE_PACED_RESULT', a.id, json.dumps(record['summary']), flush=True)
    runner.cleanup()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part', choices=['pipeline', 'crop'], required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--onnx', type=Path)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.48)
    p.add_argument('--cache-full-val', action='store_true')
    p.add_argument('--annotation', type=Path)
    p.add_argument('--image-root', type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('fresh build/output directory required')
    cases = json.loads(a.cases.read_text())['cases']
    if len(cases) != 64 or not 0 <= a.threshold <= 1:
        raise ValueError('same 64-case baseline manifest and legal threshold required')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    a.output.mkdir(parents=True)
    record = dict(id=a.id, part=a.part, cases_sha256=digest(a.cases), source_sha256=digest(__file__),
        handler_sha256=digest(Path(__file__).parent / 'models' / ('benchmark_rfpose_split.py' if a.part == 'pipeline' else 'benchmark_rfpose_crop.py')),
        threshold=a.threshold if a.part == 'pipeline' else None, buffer_seconds=.2, warmup=100, cases=64,
        timing_population='same baseline full-image cases' if a.part == 'pipeline' else 'same baseline det56 crop cases',
        maximum_clocks_before_build=True, old_engine_reused=False, nms_inside_engine=False,
        manifest_sha256=digest(a.manifest) if a.manifest else None, onnx_sha256=digest(a.onnx) if a.onnx else None,
        scope='representative-input timing; not a full-COCO latency average')
    clocks, error, started = MaximumClocks(), None, time.monotonic()
    try:
        with exclusive_gpu(), retain_cuda_pool(1024), clocks:
            (pipeline if a.part == 'pipeline' else crop)(a, cases, clocks, record)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        record.update(error=error, elapsed_seconds=time.monotonic()-started,
                      clocks=clocks.report() if hasattr(clocks, 'sm_mhz') else None)
        with (a.output / 'result.json').open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
