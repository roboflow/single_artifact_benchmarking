# Standardized Object Detection Benchmarking

## Problem

Current object detection benchmarking practices suffer from significant inconsistencies that compromise the reliability of reported performance metrics. Typically, researchers report mAP values from their research code, then export models to ONNX format and compile with fp16 TensorRT to report latency measurements. This approach introduces several sources of error:

1. **Precision Compatibility**: Some models do not function correctly when compiled to fp16 precision
2. **Postprocessing Overhead**: Complex postprocessing operations significantly impact model performance but are inconsistently handled across implementations
3. **Measurement Methodology**: Inconsistent reporting between raw `trtexec` outputs and Python session measurements
4. **Thermal Throttling**: Inadequate control for GPU power throttling due to thermal saturation, leading to unreproducible latency measurements

## Solution

This framework provides an optimized TensorRT Python implementation that translates directly from ONNX graphs to latency/mAP pairs without leveraging complex postprocessing for any model. The implementation addresses the identified issues through:

- **Throttling Monitoring**: Active detection of GPU thermal throttling to determine measurement reliability
- **Thermal Management**: Insertion of cooling buffers between subsequent inference calls to reduce throttling effects
- **Hosted Model Repository**: Centralized hosting of ONNX graphs to ensure model availability and reproducibility
- **Standardized Export**: Consistent model export methodology across architectures

## Model Export Standards

ONNX graphs are obtained directly from the original author repositories for each model type. For YOLO models specifically, export is performed using the command:

```
yolo export format=onnx nms=True conf=0.001
```

## Technical Implementation

A notable distinction from the D-FINE implementation is the inclusion of CUDA graph support. While CUDA graphs are straightforward to implement with `trtexec`, they present additional complexity in Python environments. However, they provide meaningful performance improvements for certain model architectures, justifying their inclusion in this framework.

## Usage

SAB uses [uv](https://docs.astral.sh/uv/). Each host installs the extra for its hardware.

1. On an x86 Linux host with an NVIDIA GPU, install the dependencies:
   ```bash
   uv sync --python 3.12 --extra nvidia
   ```
2. Run all benchmark scripts:
   ```bash
   uv run python -m sab.benchmark_all <path to coco val dir> <path to coco val annotations>
   ```
3. To run one model family, run its script:
   ```bash
   uv run python -m sab.models.benchmark_rfdetr <path to coco val dir> <path to coco val annotations>
   ```

### Options

`benchmark_all` and each script accept these flags:

| Flag | Effect |
|---|---|
| `--runtimes=tensorrt,onnxruntime` | Run only the rows of these runtimes. |
| `--devices=gpu` | Run only the rows on these devices (`cpu`, `gpu`, `npu`). |
| `--max_images=50` | Evaluate on the first N images. The table marks these rows with `*`. Use it for smoke tests. |
| `--rerun` | Run each row again, also when the output file already holds a result for it. |

SAB writes each result to the output file when the row is complete. If a run stops, run the same command again. SAB keeps the complete rows and runs only the missing rows.

SAB skips a row when its runtime or device is not available on the host. A row that SAB cannot support (for example, an artifact that does not compile) shows `unsupported` and the reason.

### What the latency includes

Each runtime times the smallest call that runs the full graph. Preprocessing and postprocessing are outside the timed region.

| Runtime | Timed call | Per-image copy to the device |
|---|---|---|
| TensorRT | CUDA graph replay, or `execute_async_v3` | Not included. The input is already in GPU memory. |
| ONNX Runtime (GPU) | `run_with_iobinding` | Not included. IOBinding binds GPU buffers. |
| ONNX Runtime (CPU) | `run_with_iobinding` | None (CPU). |

### Throttle detection

| Device | Signal |
|---|---|
| NVIDIA GPU | NVML clock events. SAB locks the clocks to their maximum during the run. |
| x86 Linux CPU | Core frequencies from `/sys/devices/system/cpu/cpu*/cpufreq`, polled during the run. |
| Other | None. The `Throttled` column shows `?`. |

The CPU signal is clean only with the `performance` governor. With `schedutil` or `powersave`, idle cores slow down between images, and SAB can report throttling that did not occur. Set the governor before a CPU run:

```bash
sudo cpupower frequency-set --governor performance
```

### Tests

```bash
uv sync --python 3.12 --extra onnx-cpu --group dev
```
```bash
uv run pytest -q
```

## Contributions

Contributions of new models to the benchmark suite are welcome. Please submit model additions by opening a pull request to the repository.
