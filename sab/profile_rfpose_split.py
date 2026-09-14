"""One warmed split-pipeline execution for Nsight kernel/precision inspection.

This is a profiling diagnostic, not an admissible latency measurement. Use
Nsight's cudaProfilerApi capture range to omit loading, warmup and captures.
"""

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.functional as TF
import tensorrt as trt

from sab.benchmark_modes import exclusive_gpu, retain_cuda_pool
from sab.models.benchmark_rfpose import digest
from sab.models.benchmark_rfpose_split import RFPoseSplitTRTInference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--engine-cache', type=Path, required=True)
    p.add_argument('--image', type=Path, default=Path('/home/isaac/r-flow/coco/val2017/000000000785.jpg'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    with exclusive_gpu(), retain_cuda_pool(1024):
        runner = RFPoseSplitTRTInference(a.manifest, a.engine_cache)
        pixels = TF.pil_to_tensor(Image.open(a.image).convert('RGB')).cuda()
        runner.prepare(pixels, counts=(1, 2, 3, 5), warmup=10)
        with torch.cuda.stream(runner.torch_stream):
            for _ in range(5):
                runner.execute(True, False)
        runner.torch_stream.synchronize()
        for name, engine in [('detector', runner.detector), ('pose', runner.pose)]:
            inspector = engine.create_engine_inspector()
            layers = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            (a.output / (name + '.layers.json')).write_text(layers)
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.stream(runner.torch_stream), torch.cuda.nvtx.range('rfpose_split_detector_graph_pose_uncaptured'):
            runner.execute(True, False)
        runner.torch_stream.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        result = dict(manifest_sha256=digest(a.manifest), detector_engine=str(runner.detector_path),
            pose_engine=str(runner.pose_path), people=runner.last_count,
            source_sha256=digest(Path(__file__)), purpose='kernel inspection only; not benchmark timing')
        (a.output / 'receipt.json').write_text(json.dumps(result, indent=2))
        print('SAB_SPLIT_PROFILED', json.dumps(result), flush=True)
        runner.cleanup()


if __name__ == '__main__':
    main()
