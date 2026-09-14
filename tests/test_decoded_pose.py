import cv2
import numpy as np
import pytest
from PIL import Image
import torch

from sab.baseline_clocks import MaximumClocks
from sab.models.benchmark_decoded_pose import DecodedPoseTRTInference, format_image, restore_points


@pytest.mark.parametrize('hw', [(37, 81), (81, 37), (640, 640), (426, 640)])
def test_yolo_reference_letterbox_preserves_uint8_and_inverse(hw):
    rgb = np.random.default_rng(4).integers(0, 256, (*hw, 3), dtype=np.uint8)
    out, inverse = format_image(rgb, 640, 640, 'yolo_letterbox')
    gain = 640 / max(hw)
    nh, nw = (round(s * gain) for s in hw)
    top, left = round((640 - nh) / 2 - .1), round((640 - nw) / 2 - .1)
    expected = np.full((640, 640, 3), 114, dtype=np.uint8)
    expected[top:top+nh, left:left+nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(out[0].transpose(1, 2, 0), expected)
    np.testing.assert_allclose(restore_points(np.array([[left, top], [left + gain * 5, top + gain * 7]]), inverse),
                               [[0, 0], [5, 7]], atol=1e-12)
    assert out.dtype == np.uint8 and out.flags.c_contiguous


def test_setpose_matches_pillow_resize_exactly():
    rgb = np.random.default_rng(5).integers(0, 256, (57, 113, 3), dtype=np.uint8)
    out, inverse = format_image(rgb, 64, 64, 'pil_resize')
    expected = np.asarray(Image.fromarray(rgb).resize((64, 64), Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(out[0].transpose(1, 2, 0), expected)
    np.testing.assert_allclose(restore_points([[64, 64]], inverse), [[113, 57]])


def test_topdown_box_center_and_padding_not_gt_information():
    rgb = np.ones((600, 800, 3), dtype=np.uint8)
    box = [40, 20, 180, 360]
    out, inverse = format_image(rgb, 256, 192, 'topdown_affine', box)
    assert out.shape == (1, 3, 256, 192)
    np.testing.assert_allclose(restore_points([[96, 128]], inverse), [[130, 200]])
    endpoints = restore_points([[0, 0], [192, 256]], inverse)
    np.testing.assert_allclose(endpoints[1] - endpoints[0], [337.5, 450])


@pytest.mark.parametrize('hw', [(426, 640), (640, 426), (123, 537)])
def test_rtmo_affine_matches_released_float32_geometry(hw):
    rgb = np.random.default_rng(0).integers(0, 256, (*hw, 3), dtype=np.uint8)
    h, w = hw
    height, width = 640, 640
    src_w = np.float32(w * width / min(width, height * (w / h)))
    center = np.array([w/2, h/2], dtype=np.float32)
    source = np.zeros((3, 2), np.float32)
    source[0] = center
    source[1] = center + np.array([-.5 * src_w, 0])
    delta = source[0] - source[1]
    source[2] = source[1] + np.array([-delta[1], delta[0]])
    target = np.array([[width/2, height/2], [0, height/2], [0, height/2 + width/2]], np.float32)
    matrix = cv2.getAffineTransform(source, target)
    reference = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    out, inverse = format_image(rgb, height, width, 'mmpose_bottomup_fit')
    np.testing.assert_array_equal(out[0].transpose(1, 2, 0), reference)
    np.testing.assert_array_equal(inverse, cv2.invertAffineTransform(matrix))


def test_copy_rejects_silent_float_conversion():
    obj = object.__new__(DecodedPoseTRTInference)
    obj.persistent_tensors = {'image': torch.zeros(1, 3, 4, 4, dtype=torch.uint8)}
    with pytest.raises(ValueError, match='shape and dtype'):
        obj.copy_input_data(torch.zeros(1, 3, 4, 4))


def test_telemetry_does_not_confuse_lock_request_with_sustained_clocks():
    obj = MaximumClocks()
    obj.sm_mhz, obj.memory_mhz = 1590, 5001
    obj.samples = [dict(sm_mhz=1590, throttle_mask=0), dict(sm_mhz=1400, throttle_mask=4),
                   dict(sm_mhz=300, throttle_mask=1)]
    record = obj.report()
    assert record['nonidle_samples'] == 2
    assert record['nonidle_below_target'] == 1
    assert not record['all_sampled_nonidle_at_target']
