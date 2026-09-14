"""Exercise runtime threshold endpoints and both detector graph policies.

Uses the same engines and images, not annotations. No detections may be
silently dropped, and graph on/off must return identical outputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool
from sab.benchmark_split_modes import outputs_numpy
from sab.models.benchmark_rfpose import digest
from sab.models.benchmark_rfpose_split import RFPoseSplitTRTInference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--engine-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-root', type=Path, default=Path('/home/isaac/r-flow/coco/val2017'))
    p.add_argument('--image-ids', type=int, nargs='+', default=[785, 872, 6954, 285])
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    with exclusive_gpu(), retain_cuda_pool(1024):
        runner = RFPoseSplitTRTInference(a.manifest, a.engine_cache)
        first = TF.pil_to_tensor(Image.open(a.image_root/f'{a.image_ids[0]:012d}.jpg').convert('RGB')).cuda()
        runner.prepare(first, counts=(1,), warmup=3, capture_pose=False)
        cases = []
        for image_id in a.image_ids:
            image = TF.pil_to_tensor(Image.open(a.image_root/f'{image_id:012d}.jpg').convert('RGB')).cuda()
            for threshold in (0., .01, .48, .99, 1., .48, 0.):
                # Return to earlier values to catch stale count/threshold data.
                expected = None
                runner.threshold = threshold
                prepared, _ = runner.preprocess(image)
                torch.cuda.synchronize()
                with torch.cuda.stream(runner.torch_stream):
                    runner.copy_input_data(prepared)
                    for graph in (False, True):
                        runner.execute(graph, False)
                        runner.torch_stream.synchronize()
                        actual = outputs_numpy(runner)
                        valid = runner.front_buffers['selected_valid'].cpu().numpy()
                        count = runner.last_count
                        if int(valid.sum()) != count or not valid[:count].all() or valid[count:].any():
                            raise ValueError('count/compaction mismatch')
                        if not np.all(actual['detector_scores'] >= np.float32(threshold)):
                            raise ValueError('returned a detection below threshold')
                        if threshold == 1. and count != 0:
                            raise ValueError('finite logits must be rejected at threshold 1')
                        if expected is None:
                            expected = actual
                        else:
                            for key in actual:
                                np.testing.assert_array_equal(actual[key], expected[key])
                cases.append(dict(image_id=image_id, threshold=threshold, count=count,
                                  graph_outputs_identical=True))
                print('SAB_THRESHOLD_CHECK', json.dumps(cases[-1]), flush=True)
        receipt = dict(manifest_sha256=digest(a.manifest), source_sha256=digest(Path(__file__)),
                       handler_sha256=digest(Path(__file__).parent/'models/benchmark_rfpose_split.py'),
                       detector_build=runner.detector_receipt, pose_build=runner.pose_receipt,
                       cases=cases, annotation_access=False)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x') as handle:
            json.dump(receipt, handle, indent=2)
        runner.cleanup()


if __name__ == '__main__':
    main()
