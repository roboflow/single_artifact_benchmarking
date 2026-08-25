# SAB hardware images

There is one Dockerfile per deployment target. Each image builds `FROM` the pinned serving image for that hardware, at one serving release. As a result, latency numbers are reproducible, and they come from the stack that customers deploy to.

| Target | Dockerfile | Base image | Stack |
|---|---|---|---|
| T4 (hosted GPU) | `Dockerfile.t4` | `roboflow/roboflow-inference-server-gpu:1.4.1@sha256:852a3c91...` | CUDA 12.x, TensorRT 10.12 |
| AI1 (Jetson Orin NX, JetPack 6.2) | `Dockerfile.ai1` | `roboflow/roboflow-inference-server-jetson-6.2.0:1.4.1@sha256:da193642...` | JetPack 6.2 / L4T 36.4, TensorRT 10.7 tegra |

Pin rules:

- Pin the base image by digest. Do not use `:latest`.
- Keep both images on one serving release. A T4 number and an AI1 number are only one benchmark when the two images carry the same release.
- Change a pin in its own commit. Latency numbers are only comparable within one base pin. The environment block in each results file records which pin produced the numbers.

## Build

```bash
docker/build.sh t4
docker/build.sh ai1     # run on the device, or use an arm64 builder
```

`build.sh` refuses a dirty worktree. It bakes `SAB_GIT_SHA=$(git rev-parse HEAD)` into the image. At runtime, `sab.environment.collect_environment()` reads this value, with `SAB_BASE_IMAGE` and `SAB_HARDWARE`. As a result, every results file records the stack that produced it.

The AI1 build gates its `tensorrt`/`torch` import smoke on the Tegra host libraries. An off-device arm64 build prints `SMOKE SKIPPED` and passes. Run the deep import once on the device before you trust that image.

## Run

```bash
# T4
docker run --rm --gpus all \
    -v /path/to/coco-val:/data \
    -v "$PWD/results:/results" \
    -v sab-cache:/cache \
    sab-t4:latest \
    python scripts/benchmark_platform_models.py \
        --image_dir /data \
        --annotations_file_path /data/_annotations.coco.json \
        --output_file_name /results/platform_models.json

# AI1
docker run --rm --runtime nvidia \
    -v /path/to/coco-val:/data \
    -v "$PWD/results:/results" \
    -v sab-cache:/cache \
    sab-ai1:latest \
    python scripts/benchmark_platform_models.py \
        --image_dir /data \
        --annotations_file_path /data/_annotations.coco.json \
        --output_file_name /results/platform_models.json
```

Mounts:

- `/data` — the COCO-format val set. The directory holds `_annotations.coco.json`.
- `/results` — the output tables.
- `/cache` (`SAB_CACHE_DIR`) — the TensorRT timing cache. You can share this volume between runs of the same image. The timing cache file name carries the TensorRT version, so a new image does not reuse a stale cache.

## Clock control

- **T4**: the harness locks the GPU clocks with `nvidia-smi` around each benchmark. In a container this needs the host driver utilities. The harness drops `sudo` when it already runs as root, which is the container case. If your runtime does not permit clock locking, lock the clocks on the host before you start the container:

  ```bash
  sudo nvidia-smi -pm 1 && sudo nvidia-smi --lock-gpu-clocks "$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)"
  ```

  The NVML throttle monitor runs in both cases. It sets `"throttled": true` in the results for each run that throttled.
- **AI1**: Jetson has no NVML and no clock lock. `nvpmodel` sets a power mode, not an SM frequency. `sab.tegra_clock_watch.TegraClockWatch` reads the GPU devfreq node and the thermal trip points instead, and it records what happened rather than prevents it. Set the power mode and the maximum clocks on the host first:

  ```bash
  sudo nvpmodel -m 0 && sudo jetson_clocks
  ```

  A watch that cannot read its sources raises. No run publishes a `throttled: false` that nothing observed.
