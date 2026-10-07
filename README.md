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

To run the benchmark:

1. Install dependencies: `pip install -r requirements.txt`
2. Execute: `python3 benchmark_all.py <path to coco val dir> <path to coco val annotations>`

## Contributions

Contributions of new models to the benchmark suite are welcome. Please submit model additions by opening a pull request to the repository.

## Oriented bounding boxes (OBB)

OBB models are evaluated on YOLO-OBB datasets such as DOTA cut into tiles: a directory of images and a directory of
YOLO-OBB labels (`class x1 y1 x2 y2 x3 y3 x4 y4`, corners normalized, one `<stem>.txt` per image; missing means no
objects), passed where the bbox benchmarks take the COCO annotations file. Each labelled polygon is replaced by its
minimum-area rectangle (as Ultralytics validation does), and predictions are matched by exact rotated IoU inside
faster-coco-eval, so the 12 COCO statistics print as for bbox models (equal to the bbox metric when every angle is 0).
`max_dets` defaults to 300: DOTA tiles can hold more than 100 objects.

```
# YOLO26 OBB: yolo export model=yolo26n-obb.pt format=onnx imgsz=1024 nms=True conf=0.001
python -m sab.models.benchmark_yolo26_obb <tiles>/images/val <tiles>/labels/val --sizes nsmlx
# RF-DETR OBB (rf-detr RFDETR.export(format="onnx") of an oriented-box checkpoint)
python -m sab.models.benchmark_rfdetr_obb <tiles>/images/val <tiles>/labels/val --onnx_path rf-detr-nano-obb.onnx
```

Both run TensorRT FP32 and FP16 (`--include_cpu` adds ONNX Runtime CPU). OBB adapters return `(obboxes, class_id,
score)` with `obboxes` `(N, 5)` `(cx, cy, w, h, theta)` in original-image pixels (see `sab/obb_evaluation.py`).
