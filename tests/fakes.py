from typing import ClassVar

import torch

from sab.processors import Processor
from sab.profiler import CPUProfiler
from sab.runtimes.base import InputSpec, Runtime


class FakeRuntime(Runtime):
    """Echoes its inputs back as outputs. Records every call for assertions."""

    name: ClassVar[str] = "fake"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu", "gpu", "npu"})
    available: ClassVar[bool] = True

    def __init__(self, artifact_path: str, device: str, precision: str, shape=(1, 3, 8, 8)):
        super().__init__(artifact_path, device, precision)
        self.profiler = CPUProfiler()
        self.input_device = "cpu"
        self.input_spec = InputSpec(name="images", shape=tuple(shape))
        self.calls: list[dict[str, torch.Tensor]] = []

    @classmethod
    def is_available(cls, device: str) -> bool:
        return cls.available and device in cls.devices

    @classmethod
    def version(cls) -> str:
        return "fake 0.0"

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.calls.append(inputs)
        with self.profiler.profile():
            return dict(inputs)


class FakeProcessor(Processor):
    """Resizes nothing; gives one fixed box so that evaluation has a prediction."""

    def preprocess(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image if self.normalize else image * 255.0, {"normalized": self.normalize}

    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, ...]:
        xyxy = torch.tensor([[[0.1, 0.1, 0.5, 0.5]]])
        class_id = torch.tensor([[1]])
        score = torch.tensor([[0.9]])
        return xyxy, class_id, score
