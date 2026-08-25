"""The two platform benchmark tables: their keys, their artifacts, their ids."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import benchmark_platform_models as named  # noqa: E402
import benchmark_platform_nas_models as nas  # noqa: E402


# --- named models ------------------------------------------------------------


def test_model_ids_are_unique():
    ids = [model.model_id for model in named.PLATFORM_MODELS]
    assert len(ids) == len(set(ids))


def test_every_row_names_its_artifact_and_is_fp16():
    for model in named.PLATFORM_MODELS:
        request = model.to_request(buffer_time=0.2, max_images=None)
        assert request.model_id == model.model_id
        assert request.onnx_path == model.onnx_artifact
        assert request.needs_fp16


def test_yolo11_keys_use_the_platform_spelling():
    # The platform key is "yolov11n" and the artifact is "yolo11n...". A key
    # spelled after the file would not match anything the platform knows.
    yolo11 = [model for model in named.PLATFORM_MODELS if "11" in model.model_id]
    assert yolo11
    for model in yolo11:
        assert model.model_id.startswith("yolov11")
        assert model.onnx_artifact.startswith("yolo11")


def test_rfdetr_large_uses_the_current_export():
    # rf-detr-large.onnx is a stale 560x560 graph.
    large = next(model for model in named.PLATFORM_MODELS if model.model_id == "rfdetr-large")
    assert large.onnx_artifact == "rf-detr-large-new.onnx"


def test_rfdetr_2xlarge_uses_the_xxlarge_artifact():
    model = next(model for model in named.PLATFORM_MODELS if model.model_id == "rfdetr-2xlarge")
    assert model.onnx_artifact == "rf-detr-xxlarge.onnx"


@pytest.mark.parametrize(
    "expected",
    [
        {f"rfdetr-{size}" for size in ("nano", "small", "medium", "large", "xlarge", "2xlarge")},
        {f"yolov8{size}" for size in "nsmlx"},
        {f"yolov8{size}-seg" for size in "nsmlx"},
        {f"yolov11{size}" for size in "nsmlx"},
        {f"yolo26{size}" for size in "nsmlx"},
        {f"yolov11{size}-seg" for size in "nsm"},
    ],
)
def test_families_are_complete(expected):
    assert expected <= {model.model_id for model in named.PLATFORM_MODELS}


@pytest.mark.parametrize(
    "absent",
    [
        # The bucket holds no named rfdetr-seg export.
        "rfdetr-seg-nano",
        "rfdetr-seg-2xlarge",
        # yolo11 seg stops at m.
        "yolov11l-seg",
        "yolov11x-seg",
        # No yolo26 seg export, and no yolo26 seg adapter in SAB yet.
        "yolo26n-seg",
    ],
)
def test_rows_without_an_artifact_are_not_run(absent):
    assert absent not in {model.model_id for model in named.PLATFORM_MODELS}


def test_seg_rows_carry_the_mask_graph_surgery():
    for model in named.PLATFORM_MODELS:
        if model.model_id.endswith("-seg"):
            assert model.graph_surgery is named.fuse_yolo_mask_postprocessing_into_onnx


# --- NAS children ------------------------------------------------------------


def all_nas_requests():
    return [request for family in nas.NAS_FAMILIES for request in family.to_requests(0.2, None)]


def test_nas_families_are_the_four_platform_parents():
    assert [family.family for family in nas.NAS_FAMILIES] == [
        "rfdetr-nas-parent",
        "rfdetr-nas-base-parent",
        "rfdetr-nas-pecoret-parent",
        "rfdetr-nas-seg-parent",
    ]


def test_nas_model_ids_are_unique():
    ids = [request.model_id for request in all_nas_requests()]
    assert len(ids) == len(set(ids))


def test_nas_model_id_is_family_then_tuple():
    family = next(f for f in nas.NAS_FAMILIES if f.family == "rfdetr-nas-seg-parent")
    assert family.model_id((504, 12, 2, 5, 200)) == "rfdetr-nas-seg-parent/504_12_2_5_200"


def test_one_tuple_in_two_families_gives_two_ids():
    # (384, 12, ...) and (384, 16, ...) differ, but a shared tuple must still
    # produce two distinct rows.
    ids = {family.model_id((384, 16, 2, 2, 300)) for family in nas.NAS_FAMILIES}
    assert len(ids) == len(nas.NAS_FAMILIES)


def test_seg_family_runs_the_seg_inference_class():
    seg = next(f for f in nas.NAS_FAMILIES if f.family == "rfdetr-nas-seg-parent")
    assert seg.inference_class is nas.RFDETRSegTRTInference
    assert seg.configs == (
        (312, 12, 1, 4, 100),
        (384, 12, 2, 4, 100),
        (432, 12, 2, 5, 200),
        (504, 12, 2, 5, 200),
        (624, 12, 2, 6, 300),
        (768, 12, 2, 6, 300),
    )


def test_onnx_path_follows_the_family_template():
    family = next(f for f in nas.NAS_FAMILIES if f.family == "rfdetr-nas-parent")
    assert family.onnx_path((512, 16, 2, 3, 300)) == "rfdetr-nas-parent/512_16_2_3_300/inference_model.onnx"


def test_a_wrong_path_template_stops_the_run_before_it_starts(monkeypatch):
    # A 404 downloads an XML error page. Without this guard the run fails hours
    # later, inside the TensorRT parser.
    monkeypatch.setattr(nas, "artifact_in_bucket", lambda artifact: False)
    monkeypatch.setattr(nas.os.path, "exists", lambda path: False)
    missing = nas.missing_artifacts(all_nas_requests())
    assert len(missing) == len(all_nas_requests())
    assert "rfdetr-nas-parent/384_16_2_2_300/inference_model.onnx" in missing


def test_nas_rows_spell_the_tuple_out():
    row = {"model_id": "rfdetr-nas-parent/704_16_2_4_300", "latency_stats": {"T4": {"median": 1.0}}}
    enriched = nas.with_nas_config([row])[0]
    assert enriched["resolution"] == 704
    assert enriched["patch_size"] == 16
    assert enriched["num_windows"] == 2
    assert enriched["dec_layers"] == 4
    assert enriched["num_queries"] == 300
    assert enriched["latency_stats"] == {"T4": {"median": 1.0}}
