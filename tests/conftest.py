"""Test setup: make sab importable and stub the GPU-only modules dev machines lack."""

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for gpu_only_module in ("tensorrt", "onnxruntime"):  # only ship on the benchmark boxes
    try:
        __import__(gpu_only_module)
    except ImportError:
        stub = mock.MagicMock()
        # torch._dynamo walks sys.modules and demands a real spec on every entry
        stub.__spec__ = ModuleSpec(gpu_only_module, loader=None)
        sys.modules[gpu_only_module] = stub
