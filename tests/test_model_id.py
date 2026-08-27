"""model_id names a benchmark row, and the artifact name still names an old one."""

from sab.models.utils import ArtifactBenchmarkRequest, pretty_print_results
from sab.trt_inference import TRTInference


class FakeTRTInference(TRTInference):
    def __init__(self, *args, **kwargs):  # never constructed in these tests
        pass


def request_for(**kwargs) -> ArtifactBenchmarkRequest:
    return ArtifactBenchmarkRequest(inference_class=FakeTRTInference, **kwargs)


def test_model_id_defaults_to_the_artifact_name():
    request = request_for(onnx_path="rf-detr-nano.onnx")
    assert request.model_id == "rf-detr-nano.onnx"
    assert request.dump()["model_id"] == "rf-detr-nano.onnx"


def test_model_id_is_kept_when_given():
    request = request_for(onnx_path="rf-detr-large-new.onnx", model_id="rfdetr-large")
    assert request.dump() == {
        **request.dump(),
        "model_id": "rfdetr-large",
        "onnx_path": "rf-detr-large-new.onnx",
    }


def test_model_id_survives_graph_surgery():
    # Surgery rewrites onnx_path mid-run, so the artifact name cannot name the
    # row afterwards.
    request = request_for(onnx_path="yolo11n_seg_nms_conf_0.01.onnx", model_id="yolov11n-seg")
    request.onnx_path = "yolo11n_seg_nms_conf_0.01.fused.onnx"
    assert request.dump()["model_id"] == "yolov11n-seg"


def result_row(request_dump: dict) -> dict:
    return {
        "artifact_request": request_dump,
        "accuracy_stats": [0.5] * 12,
        "latency_stats": {"median": 1.23},
        "throttled": False,
    }


def test_printed_table_names_rows_by_model_id(capsys):
    pretty_print_results([result_row(request_for(onnx_path="rf-detr-xxlarge.onnx", model_id="rfdetr-2xlarge").dump())])
    printed = capsys.readouterr().out
    assert "rfdetr-2xlarge" in printed
    assert "rf-detr-xxlarge.onnx" not in printed


def test_printed_table_falls_back_to_the_artifact_name(capsys):
    # A results file written before model_id existed still prints.
    legacy = request_for(onnx_path="rf-detr-nano.onnx").dump()
    del legacy["model_id"]
    pretty_print_results([result_row(legacy)])
    assert "rf-detr-nano.onnx" in capsys.readouterr().out


def test_long_model_ids_are_not_truncated(capsys):
    model_id = "rfdetr-nas-pecoret-parent/704_16_2_4_300"
    pretty_print_results([result_row(request_for(onnx_path="child.onnx", model_id=model_id).dump())])
    assert model_id in capsys.readouterr().out
