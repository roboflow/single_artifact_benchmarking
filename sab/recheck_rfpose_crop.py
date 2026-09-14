"""Resolve a strict crop smoke failure with complete exact-engine det56 evidence.

Keep the original failure, engine, and tolerance unchanged. Reuse only the
newly maximum-clock-built engine from this paced campaign, measure graph
on/off, and cache every det56 crop at batch one for independent AP evaluation.
No annotations, keypoint labels, or GT-derived geometry enter this process.
"""

import argparse
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from sab.baseline_clocks import MaximumClocks
from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool
from sab.benchmark_rfpose_paced import assert_equal, elapsed, order, summaries
from sab.cache_pose_crops import load_inputs
from sab.models.benchmark_decoded_pose import checked_outputs
from sab.models.benchmark_rfpose import digest
from sab.models.benchmark_rfpose_crop import RFPoseCropTRTInference


def restore_points(points, box, bucket, buckets, padding):
    box = np.asarray(box, np.float64)
    h, w = buckets[bucket]
    width = max(box[2], box[3] * w / h) * padding
    span = np.array([width, width * h / w])
    return points * (span / [w - 1, h - 1]) + box[:2] + box[2:] / 2 - span / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--image-manifest', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    previous = json.loads(a.previous.read_text())
    if (previous['part'] != 'crop' or previous['old_engine_reused']
            or not previous['maximum_clocks_before_build'] or not previous.get('native_reference')
            or previous['onnx_sha256'] != digest(a.onnx)):
        raise ValueError('requires this campaign\'s fresh maximum-clock crop artifact and smoke evidence')
    build = previous['builds']['crop']
    receipts = [p for p in a.previous.parent.glob('engines/**/build.json')
                if json.loads(p.read_text()) == build]
    if len(receipts) != 1:
        raise ValueError('missing/ambiguous exact-engine receipt')
    engine = receipts[0].with_name('model.engine')
    if digest(engine) != build['engine_sha256']:
        raise ValueError('engine hash mismatch')
    cases = json.loads(a.cases.read_text())['cases']
    if len(cases) != 64:
        raise ValueError('same 64 det56 timing crops required')
    images, detections, grouped = load_inputs(a.image_manifest, a.detections)
    n = len(detections)
    arrays = dict(image_ids=np.array([r['id'] for r in images], np.int64),
        prediction_image_ids=np.array([r['image_id'] for r in detections], np.int64),
        boxes_xywh=np.array([r['bbox'] for r in detections], np.float64),
        detector_scores=np.array([r['score'] for r in detections], np.float32),
        source_indices=np.array([r['source_index'] for r in detections], np.int64),
        keypoints=np.empty((n, 17, 2), np.float32), crop_scores=np.empty(n, np.float32),
        bucket_ids=np.empty(n, np.int32))
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    a.output.mkdir(parents=True)
    record = dict(id=previous['id'], part='crop', error=None, contract=previous['contract'],
        previous_measurement=str(a.previous), previous_measurement_sha256=digest(a.previous),
        previous_native_failure=previous['error'], native_reference=previous['native_reference'],
        native_smoke_status='retained unchanged; full-dataset accuracy evaluation required',
        builds=previous['builds'], engine=str(engine), engine_sha256=digest(engine),
        onnx_sha256=digest(a.onnx), handler_sha256=digest(Path(__file__).parent/'models/benchmark_rfpose_crop.py'),
        source_sha256=digest(__file__), cases_sha256=digest(a.cases), buffer_seconds=.2, cases=64, warmup=100,
        batch=1, maximum_clocks_before_build=True, old_engine_reused=False,
        reuses_this_campaign_fresh_build=True, image_manifest_sha256=digest(a.image_manifest),
        detections_sha256=digest(a.detections), nms_inside_engine=False,
        boundary='formatted uint8 crop + aspect + bucket -> normalization + patches + model + GMM + crop score; no crop extraction/NMS')
    clocks, started, completed = MaximumClocks(), time.monotonic(), 0
    try:
        with exclusive_gpu(), retain_cuda_pool(1024), clocks:
            # Save actual smoke outputs, without changing its 2-pixel tolerance.
            native = dict(np.load(a.onnx.with_suffix('.reference.npz'), allow_pickle=False))
            ref = RFPoseCropTRTInference(engine, a.onnx, batch=len(native['image']))
            with torch.cuda.stream(ref.torch_stream):
                ref.copy_input_data({k:torch.from_numpy(native[k]) for k in ref.input_names})
                ref._execute_standard()
            actual = checked_outputs(ref)
            with (a.output/'native-actual.npz').open('xb') as handle:
                np.savez_compressed(handle, **actual)
            ref.cleanup()
            del ref
            gc.collect()
            torch.cuda.empty_cache()
            runner = RFPoseCropTRTInference(engine, a.onnx, batch=1)
            inputs = []
            for case in cases:
                if digest(case['image']) != case['image_sha256']:
                    raise ValueError('timing image changed')
                rgb = cv2.cvtColor(cv2.imread(case['image']), cv2.COLOR_BGR2RGB)
                inputs.append(runner.preprocess([(rgb, case['box'])])[0])
            with torch.cuda.stream(runner.torch_stream):
                shape = runner.copy_input_data(inputs[0])
                for _ in range(100):
                    runner._execute_standard()
            runner.torch_stream.synchronize()
            graph = runner._capture_cuda_graph(shape)
            record['capture'] = dict(runner.graph_status)
            if graph is None:
                runner.fresh_context()
            time.sleep(2.)
            record['timing_clock_start'] = len(clocks.samples)
            samples = []
            for index, (case, tensors) in enumerate(zip(cases, inputs)):
                with torch.cuda.stream(runner.torch_stream):
                    runner.copy_input_data(tensors)
                runner.torch_stream.synchronize()
                before = None
                for mode in order(index, graph is not None):
                    ms = elapsed(runner, graph.replay if mode else runner._execute_standard)
                    output = checked_outputs(runner)
                    if before is not None:
                        assert_equal(before, output)
                    before = output
                    samples.append(dict(case=index, image_id=case['image_id'], cuda_graph=mode, ms=ms))
            record.update(timing_clock_stop=len(clocks.samples), samples=samples,
                          summary=summaries(samples), graph_on_off_equal=True)
            print('RFPOSE_CROP_RECHECK_TIMING', record['id'], json.dumps(record['summary']), flush=True)
            with (a.output/'timing.json').open('x') as handle:
                json.dump(record, handle, indent=2)
            for image_number, info in enumerate(images):
                indices = grouped.get(info['id'], [])
                if indices:
                    path = a.image_root / info['file_name']
                    if digest(path) != info['sha256']:
                        raise ValueError('source image changed')
                    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                    for index in indices:
                        box = detections[index]['bbox']
                        tensors, _ = runner.preprocess([(rgb, box)])
                        with torch.cuda.stream(runner.torch_stream):
                            runner.copy_input_data(tensors)
                            graph.replay() if graph is not None else runner._execute_standard()
                        output = checked_outputs(runner)
                        bucket = int(tensors['bucket_id'][0])
                        arrays['keypoints'][index] = restore_points(output['keypoints'][0], box, bucket,
                            runner.contract['aspect_buckets'], runner.contract['padding'])
                        arrays['crop_scores'][index] = output['scores'][0]
                        arrays['bucket_ids'][index] = bucket
                        completed += 1
                if (image_number+1) % 250 == 0:
                    print('RFPOSE_CROP_RECHECK_CACHE', record['id'], image_number+1, 5000,
                          completed, n, round(time.monotonic()-started,1), flush=True)
            runner.cleanup()
        if completed != n:
            raise ValueError('incomplete detector cache')
        cache = a.output/'predictions.npz'
        with cache.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        record.update(cache=str(cache), cache_sha256=digest(cache), images=5000, detections=n,
                      flip=False, views=1, detector_threshold=0., nms_applied=False)
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record.update(completed=completed, clocks=clocks.report(), elapsed_seconds=time.monotonic()-started)
        with (a.output/'result.json').open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
