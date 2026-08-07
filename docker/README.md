# SAB hardware images

One Dockerfile per deployment target, each `FROM` the pinned canonical serving
image for that hardware. This makes latency numbers reproducible and measured
in the stack customers deploy to.

| Target | Dockerfile | Base image | Stack |
|---|---|---|---|
| T4 (hosted GPU) | `Dockerfile.t4` | `roboflow/roboflow-inference-server-gpu:1.3.9@sha256:1d36a554...` | CUDA 12.4.1, TensorRT 10.12 |
| AI1 (Jetson Orin NX, JetPack 6.2) | `Dockerfile.ai1` | `repo.roboflow.com/roboflow-edge/roboflow-edge-jp62:787d2134...` | JetPack 6.2, TensorRT 10.7 |

Pinning rules:

- Always pin the base by digest (Docker Hub) or immutable sha tag (`repo.roboflow.com`). Never `:latest`.
- Bump the pin in its own commit. Latency numbers are only comparable within one base pin; the environment block in each results file records which pin produced it.

## Build

```bash
docker/build.sh t4
docker/build.sh ai1     # run on the device, or use an arm64 builder
```

`build.sh` refuses a dirty worktree and bakes `SAB_GIT_SHA=$(git rev-parse HEAD)`
into the image. `sab.environment.collect_environment()` reads it (plus
`SAB_BASE_IMAGE`, `SAB_HARDWARE`) at runtime, so every results file is
self-describing.

## Run

```bash
# T4
docker run --rm --gpus all \
    -v /path/to/coco-val:/data \
    -v "$PWD/results:/results" \
    -v sab-cache:/cache \
    sab-t4:latest \
    python scripts/get_sab_latencies.py --coco-path /data --output-dir /results

# AI1
docker run --rm --runtime nvidia \
    -v /path/to/coco-val:/data \
    -v "$PWD/results:/results" \
    -v sab-cache:/cache \
    sab-ai1:latest \
    python scripts/get_sab_latencies.py --coco-path /data --output-dir /results
```

Mounts:

- `/data` — COCO-format val set (`_annotations.coco.json` inside).
- `/results` — output tables + `.meta.json` sidecars.
- `/cache` (`SAB_CACHE_DIR`) — downloaded ONNX, built engines, TRT timing
  cache. Safe to share across runs of the same image; engines and timing
  caches are keyed by TRT version so an image bump does not reuse stale
  artifacts.

## Clock control

- **T4**: the harness locks GPU clocks via `nvidia-smi` around each benchmark.
  Inside a container this needs the host driver utilities; if locking is not
  permitted in your runtime, pre-lock on the host before `docker run`:

  ```bash
  sudo nvidia-smi -pm 1 && sudo nvidia-smi --lock-gpu-clocks "$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)"
  ```

  The NVML throttle monitor still runs either way and flags any run that
  throttled (`"throttled": true` in results).
- **AI1**: no clock locking or NVML throttle events (`is_jetson=True` no-ops
  the monitor). For stable numbers, set the power model and max clocks on the
  host first: `sudo nvpmodel -m 0 && sudo jetson_clocks`.
