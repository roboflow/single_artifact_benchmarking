"""Paired baseline/variant timing through the unchanged SAB split handler.

ABBA order, identical images/thresholds, whole-interval CUDA timing. Detector
graph on and off are measured; pose is always uncaptured. This never adds
separately measured stage times or drops difficult images/counts.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool, telemetry
from sab.benchmark_split_modes import outputs_numpy, compare
from sab.clock_watch import ThrottleMonitor
from sab.models.benchmark_rfpose import digest
from sab.models.benchmark_rfpose_split import RFPoseSplitTRTInference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('variant', type=Path)
    p.add_argument('--engine-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-root', type=Path, default=Path('/home/isaac/r-flow/coco/val2017'))
    p.add_argument('--image-ids', nargs='+', type=int, default=[785, 872, 6954, 285])
    p.add_argument('--threshold', type=float, default=.48)
    p.add_argument('--per-block', type=int, default=10)
    p.add_argument('--buffer-seconds', type=float, default=.2)
    a = p.parse_args()
    if a.output.exists() or a.per_block < 1 or a.buffer_seconds <= 0:
        raise ValueError('fresh output and positive timing policy required')
    a.output.mkdir(parents=True)
    source_identity = dict(source_sha256=digest(Path(__file__)),
        handler_sha256=digest(Path(__file__).parent/'models/benchmark_rfpose_split.py'))
    torch.set_num_threads(2)
    with exclusive_gpu(), retain_cuda_pool(1024):
        runners = [RFPoseSplitTRTInference(path, a.engine_cache, threshold=a.threshold)
                   for path in (a.baseline, a.variant)]
        if runners[0].detector_receipt != runners[1].detector_receipt:
            raise ValueError('precision comparison must reuse the exact detector engine')
        image = TF.pil_to_tensor(Image.open(a.image_root/f'{a.image_ids[0]:012d}.jpg').convert('RGB')).cuda()
        for runner in runners:
            runner.prepare(image, counts=(1, 2, 3, 5), warmup=20, capture_pose=False)
        cases = []
        with ThrottleMonitor() as monitor:
            for image_id in a.image_ids:
                image = TF.pil_to_tensor(Image.open(a.image_root/f'{image_id:012d}.jpg').convert('RGB')).cuda()
                for runner in runners:
                    prepared, _ = runner.preprocess(image)
                    torch.cuda.synchronize()
                    with torch.cuda.stream(runner.torch_stream):
                        runner.copy_input_data(prepared)
                    runner.torch_stream.synchronize()
                blocks, values = [], {}
                # ABBA across the Cartesian product of engine and front graph.
                order = [(0, False), (1, False), (0, True), (1, True)]
                order += list(reversed(order))
                before = telemetry()
                for which, captured in order:
                    runner = runners[which]
                    samples = []
                    for _ in range(a.per_block):
                        time.sleep(a.buffer_seconds)
                        with torch.cuda.stream(runner.torch_stream):
                            with runner.profiler.profile_async(stream=runner.torch_stream):
                                runner.execute(captured, False)
                        runner.torch_stream.synchronize()
                        elapsed = runner.profiler.get_last_timing_async()
                        if elapsed is None or not np.isfinite(elapsed):
                            raise ValueError('invalid whole-interval timing')
                        samples.append(float(elapsed))
                    actual = outputs_numpy(runner)
                    if which in values:
                        for key, value in actual.items():
                            if not np.array_equal(value, values[which][key]):
                                raise ValueError('graph policy/repetition changed engine outputs')
                    values[which] = actual
                    blocks.append(dict(variant=bool(which), detector_graph=captured, pose_graph=False,
                                       samples_ms=samples, people=runner.last_count))
                summaries = []
                for which in (False, True):
                    for captured in (False, True):
                        samples = [v for row in blocks if row['variant']==which and row['detector_graph']==captured
                                   for v in row['samples_ms']]
                        summaries.append(dict(variant=which, detector_graph=captured,
                            mean_ms=float(np.mean(samples)), median_ms=float(np.median(samples))))
                result = dict(image_id=image_id, blocks=blocks, summaries=summaries,
                    comparison=compare(values[0], values[1]), before=before, after=telemetry())
                cases.append(result)
                (a.output/f'{image_id}.json').write_text(json.dumps(result, indent=2))
                print('SAB_SPLIT_PAIR', image_id, json.dumps(summaries), flush=True)
            throttled = monitor.did_throttle()
        receipt = dict(baseline_sha256=digest(a.baseline), variant_sha256=digest(a.variant),
            detector_build=runners[0].detector_receipt,
            pose_builds=[runner.pose_receipt for runner in runners], cases=cases,
            buffer_seconds=a.buffer_seconds, throttled=throttled, **source_identity,
            timing_boundary='whole detector + count handoff + pose/crops/GMM; initial input formatting excluded')
        (a.output/'result.json').write_text(json.dumps(receipt, indent=2))
        for runner in runners:
            runner.cleanup()


if __name__ == '__main__':
    main()
