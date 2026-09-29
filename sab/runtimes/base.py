from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar

import torch

from sab.profiler import ProfilerBase

DEVICES = frozenset({"cpu", "gpu", "npu"})
PRECISIONS = frozenset({"fp32", "fp16", "int8"})


@dataclass(frozen=True)
class InputSpec:
    """The image input of an artifact, in SAB's canonical NCHW layout."""

    name: str
    shape: tuple[int, ...]


def pick_image_input(input_names: list[str], image_input_name: str | None) -> str:
    """The name of the image input: the given name, the only input, or the input named "images"."""
    if image_input_name is not None:
        if image_input_name not in input_names:
            raise ValueError(f"Image input name {image_input_name!r} not found in model inputs {input_names}")
        return image_input_name
    if len(input_names) == 1:
        return input_names[0]
    if "images" in input_names:
        return "images"
    raise ValueError(f"Model has several inputs {input_names}. Pass image_input_name.")


class Runtime(ABC):
    """Loads one artifact and times the smallest call that runs the full graph.

    A runtime knows nothing about the model family. `run()` converts inputs to the
    layout and dtype of the artifact, times only the graph execution, and gives the
    outputs back as torch tensors keyed by the output names of the artifact.
    """

    name: ClassVar[str]
    devices: ClassVar[frozenset[str]]

    profiler: ProfilerBase
    input_device: str  # torch device of the tensors that run() accepts: "cpu" or "cuda"
    input_spec: InputSpec

    def __init__(self, artifact_path: str, device: str, precision: str):
        if device not in self.devices:
            raise ValueError(f"{self.name} does not support device {device!r}. Supported: {sorted(self.devices)}")
        if precision not in PRECISIONS:
            raise ValueError(f"Unknown precision {precision!r}. Supported: {sorted(PRECISIONS)}")
        self.artifact_path = artifact_path
        self.device = device
        self.precision = precision

    @classmethod
    @abstractmethod
    def is_available(cls, device: str) -> bool:
        """True when this host can run the runtime on `device`: the import works and the hardware exists."""

    @classmethod
    @abstractmethod
    def version(cls) -> str: ...

    @abstractmethod
    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


RuntimeFactory = Callable[[str, str, str], Runtime]


def runtime_class(factory: RuntimeFactory) -> type[Runtime]:
    """The Runtime class behind a factory: the class itself, or the class wrapped by functools.partial."""
    while isinstance(factory, partial):
        factory = factory.func
    return factory
