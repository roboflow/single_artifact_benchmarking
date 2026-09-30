from typing import ClassVar

import onnx
import torch
from onnx import TensorProto, helper

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


def save_model(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def make_two_input_model(path, image_name="images"):
    """boxes = images * 2 and labels = orig_target_sizes + 1, with an int64 second input."""
    images = helper.make_tensor_value_info(image_name, TensorProto.FLOAT, [1, 3, 4, 4])
    sizes = helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, [1, 2])
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3, 4, 4])
    labels = helper.make_tensor_value_info("labels", TensorProto.INT64, [1, 2])
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    one = helper.make_tensor("one", TensorProto.INT64, [], [1])
    nodes = [
        helper.make_node("Mul", [image_name, "two"], ["boxes"]),
        helper.make_node("Add", ["orig_target_sizes", "one"], ["labels"]),
    ]
    graph = helper.make_graph(nodes, "double", [images, sizes], [boxes, labels], [two, one])
    return save_model(graph, path)
