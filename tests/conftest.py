"""Test setup: make sab importable and stub the GPU-only modules dev machines lack."""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # tensorrt only ships on the benchmark boxes
    import tensorrt  # noqa: F401
except ImportError:
    sys.modules["tensorrt"] = mock.MagicMock()
