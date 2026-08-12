"""The CUDA graph capture call must not contribute a sample to the timing series."""

from contextlib import contextmanager, nullcontext

import pytest
import torch

from sab.profiler import ProfilerBase
from sab.trt_inference import TRTInference


INPUT_SHAPE = (1, 3, 640, 640)


class ScriptedProfiler(ProfilerBase):
    """Stand-in for CUDAProfiler that records the next scripted sample per call.

    It mirrors the real flow: profile() records on exit, while profile_async()
    records only when get_last_timing_async() is called after the stream sync.
    """

    def __init__(self, samples):
        super().__init__()
        self._samples = list(samples)

    def _record_next(self):
        self.timings.append(self._samples.pop(0))

    @contextmanager
    def profile(self, stream=None):
        yield
        self._record_next()

    @contextmanager
    def profile_async(self, stream=None):
        yield

    def get_last_timing_async(self):
        self._record_next()
        return self.timings[-1]


class FakeStream:
    def synchronize(self):
        pass


@pytest.fixture(autouse=True)
def stub_cuda(monkeypatch):
    """infer() drives streams and syncs that a GPU-less machine cannot run."""
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)


def make_inference(profiler, use_cuda_graph=True, input_shapes=None):
    """A TRTInference with every GPU-touching collaborator replaced by a double."""
    inference = TRTInference.__new__(TRTInference)
    inference.profiler = profiler
    inference.use_cuda_graph = use_cuda_graph
    inference.cuda_graph_compatible = True
    inference.graph_cache = {}
    inference.torch_stream = FakeStream()
    inference.executions = []

    shapes = list(input_shapes) if input_shapes is not None else None

    def copy_input_data(image):
        return shapes.pop(0) if shapes else INPUT_SHAPE

    def execute_with_graph(input_shape):
        inference.executions.append(input_shape)
        if input_shape not in inference.graph_cache:
            inference.graph_cache[input_shape] = object()  # stands for the capture

    inference.preprocess = lambda image: (image, {})
    inference.copy_input_data = copy_input_data
    inference._execute_with_graph = execute_with_graph
    inference._execute_standard = lambda: inference.executions.append(INPUT_SHAPE)
    inference.get_outputs = lambda: {}
    inference.postprocess = lambda outputs, metadata: outputs
    return inference


def test_capture_sample_is_absent_from_timings():
    profiler = ScriptedProfiler([420.0, 1.0, 2.0, 3.0])
    inference = make_inference(profiler)

    for _ in range(4):
        inference.infer(object())

    assert profiler.timings == [1.0, 2.0, 3.0]
    assert len(inference.executions) == 4  # the capture call still ran


def test_stats_over_remaining_samples_match_hand_computed():
    # 420.0 is the capture call; the replays are 2, 4, 4, 6, 9.
    profiler = ScriptedProfiler([420.0, 2.0, 4.0, 4.0, 6.0, 9.0])
    inference = make_inference(profiler)

    for _ in range(6):
        inference.infer(object())

    stats = profiler.get_stats()

    assert stats["count"] == 5
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["median"] == pytest.approx(4.0)
    assert stats["min"] == pytest.approx(2.0)
    assert stats["max"] == pytest.approx(9.0)
    assert stats["std"] == pytest.approx(5.6 ** 0.5)  # variance 28/5
    assert stats["p90"] == pytest.approx(7.8)  # 6 + 0.6 * (9 - 6)
    assert stats["p95"] == pytest.approx(8.4)
    assert stats["p99"] == pytest.approx(8.88)


def test_capture_discarded_for_every_new_shape():
    profiler = ScriptedProfiler([420.0, 1.0, 380.0, 5.0])
    small, large = (1, 3, 640, 640), (1, 3, 800, 800)
    inference = make_inference(profiler, input_shapes=[small, small, large, large])

    for _ in range(4):
        inference.infer(object())

    assert profiler.timings == [1.0, 5.0]


def test_standard_execution_keeps_every_sample():
    profiler = ScriptedProfiler([7.0, 1.0, 2.0])
    inference = make_inference(profiler, use_cuda_graph=False)

    for _ in range(3):
        inference.infer(object())

    assert profiler.timings == [7.0, 1.0, 2.0]


def test_discard_keeps_samples_when_the_call_recorded_nothing():
    """get_last_timing_async() returns None when the events are not ready."""
    profiler = ProfilerBase()
    profiler.timings.extend([1.0, 2.0])

    profiler.discard_timings_since(2)

    assert profiler.timings == [1.0, 2.0]
