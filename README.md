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

Other hosts install the extra for each runtime they need:

| Extra | Adds | Host |
|---|---|---|
| `openvino` | `OpenVINORuntime` | Any CPU host. |
| `litert` | `LiteRTRuntime` | Any CPU host. |
| `executorch` | `ExecuTorchRuntime` | Any CPU host. On macOS, it also runs Core ML programs on the `npu` device. |
| `coreml` | `CoreMLRuntime` | macOS. |
| `coreai` | `CoreAIRuntime` | macOS 27 on Apple silicon. |

The `nvidia` extra now includes `openvino`, so `benchmark_all` also runs the OpenVINO CPU rows. These rows take most of the run time. To run only the TensorRT and ONNX Runtime rows, add `--runtimes=tensorrt,onnxruntime`.

A CPU host without NVIDIA GPUs, for example:

```bash
uv sync --python 3.12 --extra onnx-cpu --extra openvino --extra executorch --extra litert
```

The `mac` extra installs ONNX Runtime (CPU), OpenVINO, ExecuTorch and Core ML. It works on any Apple silicon Mac:

```bash
uv sync --python 3.12 --extra mac
```

On macOS 27, add the `coreai` extra for Core AI. The `coreai-core` wheels exist only for macOS 26 and later, so `--extra mac` does not include them:

```bash
uv sync --python 3.12 --extra mac --extra coreai
```

#### Jetson

SAB supports JetPack 6.2 (Python 3.10) and JetPack 7.2 (Python 3.12). PyPI has no Jetson builds of torch with CUDA, ONNX Runtime with CUDA, or TensorRT. Thus, do not use `uv sync` on a Jetson. Use a Python environment that already has these three packages, and install SAB into it. `pip` keeps the installed packages that satisfy the SAB requirements:

```bash
python3 -m pip install ".[openvino]"
python3 -m sab.benchmark_all <path to coco val dir> <path to coco val annotations>
```

On JetPack 7.2, the system Python is externally managed, so add `--break-system-packages` to the `pip` command.

The Roboflow Inference server image for each JetPack is such an environment. For example, on JetPack 6.2:

```bash
docker run --rm -it --runtime nvidia -v "$PWD":/sab -w /sab --entrypoint bash roboflow/roboflow-inference-server-jetson-6.2.0
```

SAB does not set the Jetson clocks. Before a run, set the maximum power mode and the maximum clocks on the host. The ID of the maximum power mode is different for each module. For example, it is 0 on an AGX Orin and 2 on an Orin Nano Super. Find the ID of your module in `/etc/nvpmodel.conf`:

```bash
sudo nvpmodel -m <ID of the maximum mode>
```

```bash
sudo jetson_clocks
```

The results file records the board, the L4T release and the power mode. A container hides the device tree and has no `nvpmodel`. There, SAB detects the Jetson from `/etc/nv_tegra_release` or from the `RUNNING_ON_JETSON` variable that the Inference images set. The board and the power mode are then empty, and the L4T release is empty when the container does not have that file.

The benchmark scripts have rows only for the `.onnx` artifacts in the bucket. TensorRT, ONNX Runtime and OpenVINO read these files directly. The LiteRT, ExecuTorch, Core ML and Core AI runtimes need exported artifacts, so the scripts have no rows for them. To benchmark such an artifact, give its `ArtifactBenchmarkRequest` to `run_benchmark_on_artifacts`.

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
| OpenVINO | `InferRequest.infer()` | None (CPU). |
| LiteRT | `Interpreter.invoke()` | None (CPU). |
| ExecuTorch | `ExecuTorchMethod.execute()` | None (CPU). With Core ML (`npu`), included. |
| Core ML | `MLModel.predict()` | Included, with the conversion of the inputs to `MLMultiArray` (a pixel buffer for an image input) and of the outputs to numpy. |
| Core AI | One call of the inference function | Included. |

A LiteRT artifact can take NHWC input, or quantized int8 input. The runtime converts the input before the timed call, and dequantizes the outputs after it. The conversion is not in the latency.

Core ML and Core AI decide which compute unit runs each op. The `cpu` rows allow only the CPU. The `gpu` and `npu` rows prefer the GPU or the Neural Engine, and the runtime can put the ops that this unit does not support on the CPU.

OpenVINO rows read the existing `.onnx` files and compile them on the host, as TensorRT rows do. They set `INFERENCE_PRECISION_HINT` from the precision of the row. On a CPU with no native fp16, the fp16 rows are skipped, because OpenVINO compiles them in f32.

### Throttle detection

| Device | Signal |
|---|---|
| NVIDIA GPU | NVML clock events. SAB locks the clocks to their maximum during the run. |
| Jetson GPU | GPU frequency from `/sys/class/devfreq`, polled while the model runs. |
| x86 Linux CPU | Core frequencies from `/sys/devices/system/cpu/cpu*/cpufreq`, polled while the model runs. |
| Mac (all devices) | `ProcessInfo.thermalState`, polled during the run. `serious` or higher counts as throttling, because Apple places the performance impact there. |
| Other | None. The `Throttled` column shows `?`. |

On a Jetson, a GPU frequency below the baseline counts as throttling. This signal is reliable only when `jetson_clocks` holds the GPU at its maximum frequency.

SAB reads the core frequencies only while the model runs. Between images, a scaling governor such as `schedutil` slows the idle cores, and SAB does not count that as throttling. The baseline is the first frequency read during an inference.

### Tests

```bash
uv sync --python 3.12 --extra onnx-cpu --group dev
```
```bash
uv run pytest -q
```

## Contributions

Contributions of new models to the benchmark suite are welcome. Please submit model additions by opening a pull request to the repository.
