from types import SimpleNamespace

import pytest

from sab.benchmark_split_pair import validate_changed_stage


def runner(detector='d', pose='p', tokens=240):
    return SimpleNamespace(detector_receipt=detector, pose_receipt=pose, contract={'tokens': tokens})


def test_pairing_enforces_unchanged_stage():
    validate_changed_stage([runner(), runner(pose='q')], 'pose')
    validate_changed_stage([runner(), runner(detector='e')], 'detector')
    validate_changed_stage([runner(), runner(detector='e', pose='q')], 'both')
    with pytest.raises(ValueError, match='exact detector'):
        validate_changed_stage([runner(), runner(detector='e')], 'pose')
    with pytest.raises(ValueError, match='exact pose'):
        validate_changed_stage([runner(), runner(pose='q')], 'detector')
    with pytest.raises(ValueError, match='recipe'):
        validate_changed_stage([runner(), runner(tokens=576)], 'both')
