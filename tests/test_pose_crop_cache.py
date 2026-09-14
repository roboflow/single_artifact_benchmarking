import gzip
import json

import numpy as np
import pytest

from sab.cache_pose_crops import decoded_row, load_inputs


def inputs(tmp_path, rows, extra=None):
    manifest = tmp_path / 'images.json'
    images = [dict(id=i, file_name=f'{i}.jpg', width=640, height=480, sha256='test') for i in range(5000)]
    manifest.write_text(json.dumps(dict(images=images, **(extra or {}))))
    detections = tmp_path / 'detections.json.gz'
    with gzip.open(detections, 'wt') as handle:
        json.dump(rows, handle)
    return manifest, detections


def test_no_detection_threshold_and_no_annotation_metadata(tmp_path):
    source = [dict(image_id=4, category_id=1, bbox=[10, 20, 30, 40], score=0., area=999999),
              dict(image_id=4, category_id=2, bbox=[10, 20, 30, 40], score=1.),
              dict(image_id=4, category_id=1, bbox=[11, 21, 30, 40], score=.001)]
    images, rows, grouped = load_inputs(*inputs(tmp_path, source))
    assert len(images) == 5000
    assert [r['source_index'] for r in rows] == [0, 2]
    assert rows[0]['score'] == 0.
    assert 'area' not in rows[0]
    assert grouped[4] == [0, 1]


def test_manifest_rejects_gt_annotations(tmp_path):
    with pytest.raises(ValueError, match='annotation fields'):
        load_inputs(*inputs(tmp_path, [], extra=dict(annotations=[])))


@pytest.mark.parametrize('changes', [dict(image_id=5001), dict(score=-1), dict(score=float('nan')),
                                   dict(bbox=[0, 0, 0, 1]), dict(bbox=[0, 0, 1])])
def test_invalid_detector_rows_are_not_silently_skipped(tmp_path, changes):
    row = dict(image_id=0, bbox=[0, 0, 10, 20], score=.5)
    row.update(changes)
    with pytest.raises(ValueError):
        load_inputs(*inputs(tmp_path, [row]))


def test_decoded_coordinates_area_and_score_are_preserved():
    inverse = np.array([[337.5/192, 0, 130-337.5/2], [0, 450/256, 200-450/2]])
    output = dict(keypoints=np.tile([96., 128.], (1, 17, 1)), scores=np.array([2.7]),
                  joint_scores=np.full((1, 17), .7))
    xy, js, score, area = decoded_row(output, inverse, 256, 192, 17)
    np.testing.assert_allclose(xy, np.tile([130, 200], (17, 1)))
    np.testing.assert_array_equal(js, output['joint_scores'][0])
    assert score == 2.7  # RLE is not a probability.
    assert area == pytest.approx(337.5*450)
    output['keypoints'][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match='keypoints'):
        decoded_row(output, inverse, 256, 192, 17)
