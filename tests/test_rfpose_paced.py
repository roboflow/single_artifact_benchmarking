import inspect

import cv2
import numpy as np
import pytest

from sab.benchmark_rfpose_paced import order, assert_equal, summaries, elapsed
from sab.models.benchmark_rfpose_crop import formatted_crop, crop_profiles
from sab.recheck_rfpose_crop import restore_points


def test_paired_order_balances_mode_and_position():
    assert order(0) == [False, True, True, False]
    assert order(1) == [True, False, False, True]
    assert order(0, False) == [False, False]


def test_mode_comparison_is_exact_not_score_masked():
    expected = dict(keypoints=np.zeros((1, 17, 2)), scores=np.zeros(1))
    assert_equal(expected, expected)
    changed = {k: v.copy() for k, v in expected.items()}
    changed['keypoints'][0, 0, 0] = 1e-9
    with pytest.raises(AssertionError):
        assert_equal(expected, changed)
    with pytest.raises(ValueError):
        assert_equal(expected, {})


def test_stage_summary_keeps_empty_images():
    result = summaries([dict(cuda_graph=False, ms=0.), dict(cuda_graph=False, ms=2.)])
    assert result == [dict(cuda_graph=False, count=2, mean_ms=1., median_ms=1., p95_ms=1.9, min_ms=0.)]


def test_pacing_precedes_cuda_event_boundary():
    source = inspect.getsource(elapsed)
    assert source.index('time.sleep(.2)') < source.index('profile_async')


def test_dynamic_crop_profile_accepts_mixed_aspect_batch():
    profiles = crop_profiles(dict(input_shape=[4, 3, 384, 384]))
    assert profiles['image'] == [(1, 3, 384, 384), (1, 3, 384, 384), (45, 3, 384, 384)]
    assert profiles['box_aspect'] == profiles['bucket_id'] == [(1,), (1,), (45,)]


@pytest.mark.parametrize('box', [[20, 30, 80, 160], [-20, -30, 160, 80], [20, 30, 101, 100]])
def test_formatted_crop_is_uint8_udp_detector_geometry(box):
    buckets = [(32, 16), (16, 32), (32, 32)]
    rgb = np.random.default_rng(3).integers(0, 256, (257, 239, 3), dtype=np.uint8)
    canvas, aspect, bucket = formatted_crop(rgb, box, buckets, 1.25)
    h, w = buckets[bucket]
    width = max(box[2], box[3] * w / h) * 1.25
    scale = np.array([width, width * h / w])
    center = np.array(box[:2]) + np.array(box[2:]) / 2
    origin = center - scale / 2
    source = np.array([origin, origin + [scale[0], 0], origin + [0, scale[1]]], np.float32)
    target = np.array([[0, 0], [w-1, 0], [0, h-1]], np.float32)
    expected = cv2.warpAffine(rgb, cv2.getAffineTransform(source, target), (w, h), flags=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(canvas[:, :h, :w].transpose(1, 2, 0), expected)
    assert canvas.dtype == np.uint8 and aspect.dtype == np.float32 and bucket.dtype == np.int32
    assert not canvas[:, h:, :].any() and not canvas[:, :, w:].any()


@pytest.mark.parametrize('box', [[0, 0, 0, 3], [0, 0, 3, -1], [0, 0, np.nan, 3]])
def test_formatted_crop_rejects_bad_boxes(box):
    with pytest.raises(ValueError):
        formatted_crop(np.zeros((32, 32, 3), np.uint8), box, [(16, 16)], 1.25)


@pytest.mark.parametrize('hw', [(320,192), (192,320), (256,240)])
def test_crop_coordinate_restoration_uses_udp_endpoints(hw):
    h, w = hw
    box = [13., 29., 112., 237.]
    coords = np.array([[0.,0.],[(w-1)/2, (h-1)/2],[w-1,h-1]])
    restored = restore_points(coords, box, 0, [hw], 1.25)
    width = max(box[2], box[3]*w/h) * 1.25
    np.testing.assert_allclose(restored[1], [69., 147.5])
    np.testing.assert_allclose(restored[2]-restored[0], [width, width*h/w])
