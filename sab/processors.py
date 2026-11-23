from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from sab.runtimes.base import InputSpec, Runtime


class Processor(ABC):
    """The input and output contract of one model family: preprocess, postprocess, extra inputs.

    `preprocess` receives a CHW float image in [0, 1] on the runtime's input device, and
    gives the canonical NCHW float32 tensor at `input_spec.shape`. When `normalize` is
    False, the artifact normalizes inside its graph: preprocess then gives 0-255 values
    and skips the family's mean/std normalization.

    `postprocess` gives (xyxy, class_id, score) for "bbox", plus masks for "segm". Boxes
    are normalized to 0-1, in xyxy format.
    """

    prediction_type: ClassVar[str] = "bbox"

    def __init__(self, input_spec: InputSpec, normalize: bool = True):
        self.input_spec = input_spec
        self.normalize = normalize

    @abstractmethod
    def preprocess(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]: ...

    @abstractmethod
    def postprocess(self, outputs: dict[str, torch.Tensor], metadata: dict) -> tuple[torch.Tensor, ...]: ...

    def extra_inputs(self, image: torch.Tensor, metadata: dict) -> dict[str, torch.Tensor]:
        """Inputs other than the image, keyed by input name. `image` is the preprocessed tensor."""
        return {}


class Pipeline:
    """One runtime and one processor, behind the interface that evaluate() and run_timed_pass() use."""

    def __init__(self, runtime: Runtime, processor: Processor, output_names: dict[str, str] | None = None):
        self.runtime = runtime
        self.processor = processor
        self.output_names = output_names or {}

    @property
    def profiler(self):
        return self.runtime.profiler

    @property
    def input_device(self) -> str:
        return self.runtime.input_device

    @property
    def prediction_type(self) -> str:
        return self.processor.prediction_type

    def infer(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, metadata = self.processor.preprocess(image)
        inputs = {self.runtime.input_spec.name: tensor, **self.processor.extra_inputs(tensor, metadata)}
        outputs = self.runtime.run(inputs)
        outputs = {self.output_names.get(name, name): value for name, value in outputs.items()}
        return self.processor.postprocess(outputs, metadata)
