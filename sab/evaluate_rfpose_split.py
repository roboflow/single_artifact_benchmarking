"""Full-COCO validation of a two-stage, detector-gated pose pipeline.

All substantial work stays in the two engines. Total timing includes the
four-byte count transfer and host dispatch. Optional prediction caches use
the same offline F1 schema as the single-engine handler.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool, telemetry
from sab.clock_watch import ThrottleMonitor
from sab.evaluation import evaluate
from sab.models.benchmark_rfpose import digest
from sab.models.benchmark_rfpose_split import RFPoseSplitTRTInference


class EvaluationAdapter:
    def __init__(self, runner, detector_graph, pose_graph):
        self.runner = runner
        self.detector_graph, self.pose_graph = detector_graph, pose_graph
        self.profiler, self.prediction_type = runner.profiler, 'keypoints'

    def infer(self, image):
        runner = self.runner
        prepared, metadata = runner.preprocess(image)
        torch.cuda.synchronize()
        with torch.cuda.stream(runner.torch_stream):
            runner.copy_input_data(prepared)
            with runner.profiler.profile_async(stream=runner.torch_stream):
                runner.execute(self.detector_graph, self.pose_graph)
        runner.torch_stream.synchronize()
        if runner.profiler.get_last_timing_async() is None:
            raise RuntimeError('missing pipeline timing')
        return runner.postprocess(runner.get_outputs(), metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--engine-cache', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.48)
    p.add_argument('--graphs', choices=['none', 'detector', 'pose', 'both'], default='detector')
    p.add_argument('--capture-counts', nargs='+', type=int, default=list(range(1, 17)))
    p.add_argument('--buffer-seconds', type=float, default=.2)
    p.add_argument('--max-images', type=int)
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--optimal-batch', type=int, default=1)
    a = p.parse_args()
    source_identity = dict(source_sha256=digest(Path(__file__)),
        handler_sha256=digest(Path(__file__).parent / 'models/benchmark_rfpose_split.py'),
        builder_sha256=digest(Path(__file__).parent / 'models/benchmark_rfpose.py'),
        clock_watch_sha256=digest(Path(__file__).parent / 'clock_watch.py'))
    if a.output.exists() or a.output.with_suffix('.npz').exists():
        raise FileExistsError(a.output)
    if a.buffer_seconds < 0 or (a.max_images is not None and a.max_images < 1):
        raise ValueError('invalid evaluation policy')
    native = json.loads(a.annotations.read_text())
    image_ids = [image['id'] for image in native['images']][:a.max_images]
    torch.set_num_threads(2)
    with exclusive_gpu(), retain_cuda_pool(1024):
        runner = RFPoseSplitTRTInference(a.manifest, a.engine_cache, threshold=a.threshold,
                                        optimal_batch=a.optimal_batch)
        if runner.contract['keypoints'] != 17:
            raise ValueError('this COCO evaluator is body-17 only')
        first = TF.pil_to_tensor(Image.open(a.images / native['images'][0]['file_name']).convert('RGB')).cuda()
        runner.prepare(first, counts=a.capture_counts, warmup=30,
            capture_detector=a.graphs in ('detector', 'both'), capture_pose=a.graphs in ('pose', 'both'))
        runner.cache_outputs = a.save_predictions
        adapter = EvaluationAdapter(runner, a.graphs in ('detector', 'both'), a.graphs in ('pose', 'both'))
        before = telemetry()
        with ThrottleMonitor() as monitor:
            accuracy = evaluate(adapter, str(a.images), str(a.annotations), buffer_time=a.buffer_seconds,
                                max_images=a.max_images, max_dets=20)
            during, throttled = telemetry(), monitor.did_throttle()
        after = telemetry()
        if len(runner.profiler.timings) != len(image_ids):
            raise ValueError('every image requires a timed pipeline invocation')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        saved = None
        if a.save_predictions:
            if len(runner.output_cache) != len(image_ids):
                raise ValueError('image/output count mismatch')
            counts = [len(row['scores']) for row in runner.output_cache]
            joined = {key: np.concatenate([row[key] for row in runner.output_cache])
                      for key in runner.output_cache[0]}
            if any(not np.isfinite(value).all() for value in joined.values()):
                raise ValueError('nonfinite predictions')
            if not (joined['detector_scores'] >= np.float32(a.threshold)).all():
                raise ValueError('prediction below detector threshold')
            cache = a.output.with_suffix('.npz')
            with cache.open('xb') as out:
                np.savez_compressed(out, image_ids=np.asarray(image_ids, dtype=np.int64),
                    prediction_image_ids=np.repeat(image_ids, counts),
                    gate_scores=joined['detector_scores'], keypoints=joined['keypoints'],
                    boxes=joined['boxes'], crop_scores=joined['scores'],
                    pose_quality=joined['scores'] / np.maximum(joined['detector_scores'], 1e-30),
                    bucket_ids=joined['bucket_ids'], returned_counts=counts, processed_counts=counts,
                    engine_ms=np.asarray(runner.profiler.timings).reshape(-1, 1))
            saved = dict(path=str(cache), sha256=digest(cache), predictions=sum(counts))
        result = dict(manifest=str(a.manifest), manifest_sha256=digest(a.manifest),
            contract=runner.contract, detector_build=runner.detector_receipt, pose_build=runner.pose_receipt,
            graph_policy=a.graphs, captured_pose_counts=sorted(runner.pose_graphs),
            uncaptured_pose_counts=sorted(runner.uncaptured_pose_counts),
            accuracy_stats=accuracy, latency_stats=runner.profiler.get_stats(), threshold=a.threshold,
            images=len(image_ids), full_val=len(image_ids) == len(native['images']),
            buffer_seconds=a.buffer_seconds, prediction_cache=saved, throttled=throttled,
            warmup_and_capture_excluded=True, activation_bytes=runner.activation_arena.numel(),
            telemetry=dict(before=before, during=during, after=after),
            annotation_sha256=digest(a.annotations), **source_identity,
            timed_boundary='detector + pinned count D2H + host wait/dispatch + crops/pose/GMM/scores; not stage-time sum')
        with a.output.open('x') as out:
            json.dump(result, out, indent=2)
        print('SAB_SPLIT_FULL', json.dumps({key: result[key] for key in
            ('latency_stats', 'accuracy_stats', 'uncaptured_pose_counts', 'throttled')}), flush=True)
        runner.cleanup()


if __name__ == '__main__':
    main()
