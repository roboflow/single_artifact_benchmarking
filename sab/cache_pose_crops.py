"""Cache single-view detector crops from the exact measured SAB engine.

Only an image manifest and detector predictions reach this process: no keypoint
annotations, GT boxes, visibility labels or segmentation areas. Native dataset
rescoring/NMS and official AP run separately on CPU. This is not a timing run.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as handle:
        return json.load(handle)


def load_inputs(image_manifest, detections):
    payload = read_json(image_manifest)
    if set(payload) != {'images'}:
        raise ValueError('image-only manifest required; annotation fields are forbidden')
    images = sorted(payload['images'], key=lambda row: row['id'])
    ids = [row['id'] for row in images]
    if len(ids) != 5000 or len(set(ids)) != len(ids):
        raise ValueError('all 5000 unique COCO val images required')
    for row in images:
        if set(row) != {'id', 'file_name', 'width', 'height', 'sha256'}:
            raise ValueError('unexpected metadata in image-only manifest')
    ids = set(ids)
    rows, grouped = [], defaultdict(list)
    for source_index, row in enumerate(read_json(detections)):
        if row.get('category_id', 1) != 1:
            continue
        box = np.asarray(row['bbox'], np.float64)
        score = float(row['score'])
        if row['image_id'] not in ids or box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('invalid detector box/image')
        if np.any(box[2:] <= 0) or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid detector extent/confidence')
        clean = dict(image_id=int(row['image_id']), bbox=box, score=score, source_index=source_index)
        grouped[clean['image_id']].append(len(rows))
        rows.append(clean)
    if not rows:
        raise ValueError('empty detections')
    return images, rows, grouped


def decoded_row(result, inverse, height, width, num_keypoints):
    """Restore coordinates; area is the same detector-derived crop span as MMPose."""
    expected = dict(keypoints=(1, num_keypoints, 2), scores=(1,), joint_scores=(1, num_keypoints))
    if set(result) != set(expected):
        raise ValueError('unreviewed crop output contract')
    for name, shape in expected.items():
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise ValueError('invalid crop output: ' + name)
    inverse = np.asarray(inverse, np.float64)
    if inverse.shape != (2, 3) or not np.isfinite(inverse).all():
        raise ValueError('invalid inverse affine')
    xy = result['keypoints'][0].astype(np.float64) @ inverse[:, :2].T + inverse[:, 2]
    area = abs(float(np.linalg.det(inverse[:, :2]))) * height * width
    if not np.isfinite(area) or area <= 0:
        raise ValueError('invalid detector crop area')
    return xy.astype(np.float32), result['joint_scores'][0], float(result['scores'][0]), area


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement', type=Path, required=True)
    p.add_argument('--image-manifest', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('fresh cache directory required: ' + str(a.output))
    images, rows, grouped = load_inputs(a.image_manifest, a.detections)
    measurement = read_json(a.measurement)
    contract_path = Path(measurement['onnx']).with_suffix('.contract.json')
    contract = read_json(contract_path)
    if (measurement['error'] is not None or contract['stage'] != 'person_crop'
            or contract['family'] not in ('simcc', 'poseur') or contract.get('flip') is not False):
        raise ValueError('requires a measured, explicitly single-view third-party crop engine')
    for field in ('engine', 'onnx'):
        if digest(measurement[field]) != measurement[field + '_sha256']:
            raise ValueError('measured artifact changed: ' + field)
    import cv2
    import torch
    from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool
    from sab.models.benchmark_decoded_pose import DecodedPoseTRTInference, checked_outputs

    if digest(inspect.getfile(DecodedPoseTRTInference)) != measurement['handler_sha256']:
        raise ValueError('use the same handler source as the timing run')
    n, k = len(rows), contract['keypoints']
    arrays = dict(image_ids=np.asarray([r['id'] for r in images], np.int64),
        prediction_image_ids=np.asarray([r['image_id'] for r in rows], np.int64),
        source_indices=np.asarray([r['source_index'] for r in rows], np.int64),
        detector_scores=np.asarray([r['score'] for r in rows], np.float32),
        boxes_xywh=np.asarray([r['bbox'] for r in rows], np.float32),
        keypoints=np.empty((n, k, 2), np.float32), joint_scores=np.empty((n, k), np.float32),
        crop_scores=np.empty(n, np.float32), crop_areas=np.empty(n, np.float64))
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    a.output.mkdir(parents=True)
    started, error, completed, graph_status = time.monotonic(), None, 0, None
    record = dict(schema='single_view_detector_crops_v1', id=measurement['id'],
        engine_sha256=measurement['engine_sha256'], onnx_sha256=measurement['onnx_sha256'],
        measurement=str(a.measurement), measurement_sha256=digest(a.measurement),
        contract_sha256=digest(contract_path), handler_sha256=measurement['handler_sha256'],
        source_sha256=digest(__file__), image_manifest_sha256=digest(a.image_manifest),
        detections_sha256=digest(a.detections), family=contract['family'], keypoints=k,
        images=5000, detections=n, detector_threshold=0., flip=False, views=1,
        nms_applied=False, area_source='detector-derived padded aspect-corrected crop span product',
        timing=False, protocol='all det56 boxes; one unchanged measured engine invocation per crop')
    try:
        with exclusive_gpu(), retain_cuda_pool(1024):
            runner = DecodedPoseTRTInference(measurement['engine'], measurement['onnx'])
            shape = tuple(runner.persistent_tensors['image'].shape)
            if shape[:2] != (1, 3) or set(runner.input_names) != {'image'}:
                raise ValueError('batch-one image-only crop engine required')
            height, width = shape[-2:]
            record['input_hw'] = [height, width]
            graph, initialized = None, False
            for image_number, info in enumerate(images):
                indices = grouped.get(info['id'], [])
                if indices:
                    path = a.image_root / info['file_name']
                    if digest(path) != info['sha256']:
                        raise ValueError('source image changed: ' + str(path))
                    bgr = cv2.imread(str(path))
                    if bgr is None or bgr.shape[:2] != (info['height'], info['width']):
                        raise ValueError('image decode/size mismatch')
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    for index in indices:
                        tensor, meta = runner.preprocess(rgb, rows[index]['bbox'])
                        with torch.cuda.stream(runner.torch_stream):
                            runner.copy_input_data(tensor)
                        if not initialized:
                            with torch.cuda.stream(runner.torch_stream):
                                for _ in range(10):
                                    runner._execute_standard()
                            runner.torch_stream.synchronize()
                            graph = runner._capture_cuda_graph(shape)
                            graph_status = dict(runner.graph_status)
                            if graph is None:
                                runner.fresh_context()
                                with torch.cuda.stream(runner.torch_stream):
                                    runner.copy_input_data(tensor)
                            initialized = True
                        with torch.cuda.stream(runner.torch_stream):
                            graph.replay() if graph is not None else runner._execute_standard()
                        xy, js, score, area = decoded_row(checked_outputs(runner), meta['inverse'], height, width, k)
                        arrays['keypoints'][index] = xy
                        arrays['joint_scores'][index] = js
                        arrays['crop_scores'][index] = score
                        arrays['crop_areas'][index] = area
                        completed += 1
                if (image_number + 1) % 250 == 0:
                    print('SAB_CROP_CACHE', record['id'], image_number + 1, 5000, completed, n,
                          round(time.monotonic() - started, 1), flush=True)
            runner.cleanup()
        if completed != n:
            raise ValueError('incomplete crop cache')
        cache = a.output / 'predictions.npz'
        with cache.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        record.update(cache=str(cache), cache_sha256=digest(cache))
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        record.update(error=error, completed=completed, capture=graph_status,
                      elapsed_seconds=time.monotonic() - started)
        with (a.output / 'result.json').open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
