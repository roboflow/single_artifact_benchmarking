# RF-Pose joint artifacts in SAB

This handler accepts one ONNX artifact containing RF-DETR, global query/class
top-300 selection, person filtering and a runtime cutoff, native mixed-aspect
crop construction, RF-Pose, GMM/mean-shift decoding and crop-level scores.
It does not import RF-Pose or construct its model. Export changes live in
RF-Pose's PyTorch code; SAB uses stock TensorRT, without graph rewriting or
custom plugins.

## Install and run

Use the pinned Python 3.10/CUDA 12/TensorRT 10.4 environment:

```bash
uv sync --frozen --extra trt --extra dev
uv run --frozen --extra trt --extra dev python -m pytest -q tests
uv run --frozen --extra trt python -m sab.models.benchmark_rfpose joint.onnx \
  --engine-cache artifacts/engines --images /data/coco/val2017 \
  --annotations /data/coco/annotations/person_keypoints_val2017.json \
  --threshold 0.48 --output artifacts/joint-result.json --save-predictions
```

TensorRT 10.6 is also pinned as the mutually exclusive `trt106` extra. Keep
both runtimes isolated when comparing compilers:

```bash
UV_PROJECT_ENVIRONMENT=.venv-trt106 uv sync --frozen --extra trt106 --extra dev
UV_PROJECT_ENVIRONMENT=.venv-trt106 uv run --frozen --extra trt106 python -m sab.models.benchmark_rfpose --help
```

An engine built with one runtime must be rebuilt for the other. Both remain
in the content-addressed cache; the handler rejects runtime mismatches.
The RF-Pose fixed-batch conditional prototype requires testing with 10.6:
10.4 rejects its minimal ONNX test even with weak typing. Successful parsing
does not by itself validate full-engine accuracy, CUDA graphs or speed.

The isolated `trt1013` extra pins TensorRT 10.13.3.9 for conditional-shape
compiler investigations. Use `UV_PROJECT_ENVIRONMENT=.venv-trt1013` with
`--extra trt1013`; do not mix runtime extras or silently replace the 10.4/10.6
engines. Minimal conditional/DDS graphs parse in 10.6 but can fail during
engine construction. A newer compiler is a compatibility experiment, not
an accuracy or speed claim; its Nano control and complete joint output
must be revalidated before comparison.

The engine cache is keyed by the ONNX, TensorRT version, GPU capability and
build settings, with engine hashes verified on reuse. `sab.rfpose` metadata
inside the ONNX declares the interface and checkpoint/calibration provenance.
The handler rejects network-only controls, custom operators (including in
conditional subgraphs), and mismatched ONNX/engine/build receipts.

The default stock build policy is optimization level 3 with TensorRT choosing
auxiliary streams. `--optimization-level 0..5` and `--max-aux-streams N` expose
ordinary compiler settings for controlled performance experiments. They are
part of the cache identity; changing them requires a separate engine and
fresh output validation. These flags do not rewrite ONNX or weaken its
explicit precision types. In particular, no aggressive setting is silently
enabled for just one model family.

The current COCO evaluation entry point is body-17 only. It evaluates every
image in the supplied annotation file, including no-person images. A reduced
`--max-images` run is explicitly a smoke test, not a full-validation result.
`--save-predictions` is opt-in: it writes a compact NPZ with detector gate
scores, keypoints, boxes, final crop scores and per-image engine timings for
offline F1/accuracy checks. There is no post-pose cutoff. At a nonzero detector
cutoff, the reported AP is that gated operating point, not zero-cutoff AP.

## Timed boundary

Inputs are a normalized/resized detector image, original RGB uint8 pixels on
the declared source canvas, source height/width, and a confidence cutoff.
SAB performs the initial detector formatting exactly as its RF-DETR handler
(ImageNet normalize, then antialiased resize). Crops use original pixels,
not the downsampled detector input. Only initial formatting, I/O transfers
and trivial final formatting are outside CUDA events.

All nontrivial detection selection, inter-stage cropping, pose inference,
mode search and scoring are inside the single engine. No engine-time sum,
host-side crop loop, frozen detector selections or omitted decoding is an
admissible joint measurement. Different aspect crops share a pose batch.
Invalid output padding is discarded only as trivial result formatting.

The original RF-Pose measurements included the initial detector resize in
the engine. Report that boundary difference explicitly; do not present it
as a computational optimization.

## CUDA graphs and reference clocks

`--cuda-graph on` actually attempts SAB capture. Unsupported data-dependent
execution is recorded as an uncaptured fallback. The failed context is
discarded before timing; capture/warmup are excluded from results. The
default retained CUDA async pool is bounded to 1 GiB and restored afterward.

The main evaluator follows SAB's existing `ThrottleMonitor`: temporarily
request maximum supported clocks, retain 200 ms gaps, then reset clocks.
This needs NVIDIA clock-control permission. It does not increase the power
limit. Record warmup throttling separately from actual sample telemetry;
the monitor's historical memory-clock label read NVML domain 1 (SM), not
domain 2 (memory). The local port now fixes that and the application-clock
reason label, and releases its event monitor even when no new event arrives.
The receipt also includes independent `nvidia-smi` telemetry and identifies
the clock-monitor source. These diagnostic-label fixes do not rewrite old
timings. The inherited monitor restores
persistence disabled, matching this T4's initial state; do not assume it
preserves a different preexisting persistence configuration on other hosts.
The current entry point finishes warmup/capture before starting the measured
evaluation monitor; older development receipts explicitly include warmup in
their throttle-monitoring window.

For paired graph-on/off diagnostics through the actual SAB handler:

```bash
uv run --frozen --extra trt python -m sab.benchmark_modes \
  --handler sab.models.benchmark_rfpose:RFPoseJointTRTInference \
  --handler-kwargs '{"onnx_path":"joint.onnx","threshold":0.48}' \
  --engine artifacts/engines/HASH/model.engine --image-root /data/coco/val2017 \
  --image-ids 785 872 6954 285 --output artifacts/joint-modes \
  --reference-clocks --save-outputs
```

Available capture uses ABBA off/on/on/off blocks and requires bit-identical
outputs. Failed capture produces only real off-mode samples, never invented
on-mode timing. These short probes are not full-COCO latency/accuracy results.
The exclusive GPU lock also used by the RF-Pose queue prevents concurrent
benchmarks; the runner refuses to time while another process owns the GPU.

## Two-stage dynamic-batch experiment

`sab.models.benchmark_rfpose_split` also accepts the version-8 manifest
exported by RF-Pose's `benchmarks.trt.joint_split_export`. This is explicitly
**two engines**, not a single-engine result or a sum of standalone timings.
The default policy captures the detector and leaves pose uncaptured.

The detector performs filtering and stable compaction into fixed GPU
buffers, with an int32 person count. A pinned four-byte count copy is part of
the detector CUDA graph. The host synchronizes, reads that count, and runs
one dynamic-batch pose engine on the exact GPU-resident prefix. Crops,
mixed-aspect pose, GMM decode and scores stay in that second engine. Zero
people skips pose. The profile spans 1–300; uncached counts are never
truncated or silently padded. Common count-specific contexts share one
engine and one activation arena, and must run sequentially.

```bash
uv run --frozen --extra trt python -m sab.benchmark_split_modes \
  /path/to/manifest.json --engine-cache artifacts/engines \
  --output artifacts/split-modes --reference-clocks

uv run --frozen --extra trt python -m sab.check_rfpose_split \
  /path/to/manifest.json --engine-cache artifacts/engines \
  --output artifacts/split-stage-check

uv run --frozen --extra trt python -m sab.evaluate_rfpose_split \
  /path/to/manifest.json --engine-cache artifacts/engines \
  --images /data/coco/val2017 \
  --annotations /data/coco/annotations/person_keypoints_val2017.json \
  --output artifacts/split-validation.json --graphs detector \
  --threshold 0.48 --save-predictions
```

The four-mode diagnostic explicitly opts into pose captures to compare all
policies, and requires identical outputs between graph modes. The ordinary
runner/evaluator does not capture pose by default. CUDA events bracket the
**entire** detector/count-copy/host-wait/dispatch/pose sequence; wall-clock
latency is also recorded by the mode diagnostic. Initial formatting, input
upload, warmup/capture and trivial final formatting remain outside the timer.
Different compiler builds may alter floating-point predictions. Exact-input
stage checks and fresh full-COCO accuracy, not timing or graph-mode equality
alone, determine whether a result is suitable for publication.
