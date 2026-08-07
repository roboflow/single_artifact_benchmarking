# SAB hardware images

There is one Dockerfile per deployment target. Each image builds `FROM` the pinned canonical serving image for that hardware. As a result, latency numbers are reproducible, and they come from the stack that customers deploy to.

| Target | Dockerfile | Base image | Stack |
|---|---|---|---|
| T4 (hosted GPU) | `Dockerfile.t4` | `roboflow/roboflow-inference-server-gpu:1.3.9@sha256:1d36a554...` | CUDA 12.4.1, TensorRT 10.12 |
| AI1 (Jetson Orin NX, JetPack 6.2) | `Dockerfile.ai1` | `repo.roboflow.com/roboflow-edge/roboflow-edge-jp62:787d2134...` | JetPack 6.2, TensorRT 10.7 |

Pin rules:

- Pin the base image by digest (Docker Hub) or by immutable sha tag (`repo.roboflow.com`). Do not use `:latest`.
- Change a pin in its own commit. Latency numbers are only comparable within one base pin. The environment block in each results file records which pin produced the numbers.

## Build

```bash
docker/build.sh t4
docker/build.sh ai1     # run on the device, or use an arm64 builder
```

`build.sh` refuses a dirty worktree. It bakes `SAB_GIT_SHA=$(git rev-parse HEAD)` into the image. At runtime, `sab.environment.collect_environment()` reads this value, with `SAB_BASE_IMAGE` and `SAB_HARDWARE`. As a result, every results file records the stack that produced it.

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

- `/data` — the COCO-format val set. The directory must contain `_annotations.coco.json`.
- `/results` — the output tables and their `.meta.json` sidecars.
- `/cache` (`SAB_CACHE_DIR`) — downloaded ONNX files, built engines, and the TensorRT timing cache. You can share this volume between runs of the same image. Engine files and timing caches include the TensorRT version in their names. As a result, a new image does not reuse stale artifacts.

## Clock control

- **T4**: the harness locks the GPU clocks with `nvidia-smi` around each benchmark. In a container, this needs the host driver utilities. If your runtime does not permit clock locking, lock the clocks on the host before you run the container:

  ```bash
  sudo nvidia-smi -pm 1 && sudo nvidia-smi --lock-gpu-clocks "$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)"
  ```

  The NVML throttle monitor runs in both cases. It sets `"throttled": true` in the results for each run that throttled.
- **AI1**: Jetson has no clock locking and no NVML throttle events. The monitor does nothing there (`is_jetson=True`). For stable numbers, set the power model and the maximum clocks on the host first: `sudo nvpmodel -m 0 && sudo jetson_clocks`.
