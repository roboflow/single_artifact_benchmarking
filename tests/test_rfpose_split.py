import pytest
import inspect

from sab.models.benchmark_rfpose_split import pose_profiles, RFPoseSplitTRTInference
from sab.models.benchmark_rfpose_split import require_fused_attention
import json


def test_default_policy_captures_only_the_detector():
    prepare = inspect.signature(RFPoseSplitTRTInference.prepare).parameters
    execute = inspect.signature(RFPoseSplitTRTInference.execute).parameters
    assert prepare['capture_detector'].default is True
    assert prepare['capture_pose'].default is False
    assert execute['detector_graph'].default is True
    assert execute['pose_graph'].default is False


def test_one_profile_accepts_every_person_count_without_padded_work():
    bounds = pose_profiles()
    assert bounds == {'center': [(1, 2), (1, 2), (300, 2)],
                      'size': [(1, 2), (1, 2), (300, 2)],
                      'selected_scores': [(1,), (1,), (300,)],
                      'selected_valid': [(1,), (1,), (300,)]}
    assert pose_profiles(optimal=4)['center'][1] == (4, 2)


@pytest.mark.parametrize('capacity,optimal', [(0, 1), (301, 1), (300, 0), (16, 17)])
def test_profile_rejects_invalid_count_contracts(capacity, optimal):
    with pytest.raises(ValueError):
        pose_profiles(capacity, optimal)


def test_fused_attention_policy_fails_closed():
    names = [f'_gemm_mha_v2_myl_{i}' for i in range(14)]
    assert require_fused_attention(json.dumps({'Layers': names}), 14) == 14
    for broken in [names[:-1], ['unknown_new_kernel_name']*14, ['QK', 'softmax', 'AV']]:
        with pytest.raises(ValueError, match='requires 14 fused MHA'):
            require_fused_attention(json.dumps({'Layers': broken}), 14)
