from dataclasses import dataclass
from typing import Callable

from sab.processors import Processor
from sab.runtimes.base import DEVICES, PRECISIONS, RuntimeFactory, runtime_class


@dataclass
class ArtifactBenchmarkRequest:
    """One row of a benchmark: one artifact, run by one runtime on one device."""

    artifact_path: str  # relative to the SAB bucket; a .zip unpacks to a directory
    runtime: RuntimeFactory  # a Runtime class, or functools.partial of one
    processor: type[Processor]
    device: str  # cpu | gpu | npu
    precision: str = "fp32"  # fp32 | fp16 | int8
    normalized_in_graph: bool = False
    output_names: dict[str, str] | None = None  # artifact output name -> name the processor reads
    unsupported: str | None = None  # the reason; SAB records the row with no numbers
    needs_class_remapping: bool = False
    buffer_time: float = 0.0
    max_images: int | None = None
    graph_surgery_func: Callable[[str], str] | None = None
    max_dets: int = 100

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"Unknown device {self.device!r}. Supported: {sorted(DEVICES)}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"Unknown precision {self.precision!r}. Supported: {sorted(PRECISIONS)}")

    @property
    def runtime_name(self) -> str:
        return runtime_class(self.runtime).name

    def key(self) -> tuple:
        """Identifies the row for resume: a result with the same key is the same measurement."""
        return (self.artifact_path, self.runtime_name, self.device, self.precision, self.max_images)

    def dump(self) -> dict:
        return {
            "artifact_path": self.artifact_path,
            "runtime": self.runtime_name,
            "device": self.device,
            "precision": self.precision,
            "processor": self.processor.__name__,
            "normalized_in_graph": self.normalized_in_graph,
            "output_names": self.output_names,
            "unsupported": self.unsupported,
            "needs_class_remapping": self.needs_class_remapping,
            "buffer_time": self.buffer_time,
            "max_images": self.max_images,
            "graph_surgery_func": self.graph_surgery_func.__name__ if self.graph_surgery_func else None,
            "max_dets": self.max_dets,
        }
