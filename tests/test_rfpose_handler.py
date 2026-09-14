import numpy as np
import json
import onnx
import pytest
import torch
import torchvision.transforms.functional as TF

from sab.models.benchmark_rfpose import RFPoseJointTRTInference, build_stock_typed, processed_pose_counts, read_contract


@pytest.mark.parametrize('options', [dict(optimization_level=-1), dict(optimization_level=6),
                                    dict(max_aux_streams=-1)])
def test_invalid_build_options_rejected_before_model_or_gpu_access(tmp_path, options):
    with pytest.raises(ValueError, match='optimization_level'):
        build_stock_typed(tmp_path / 'does-not-exist.onnx', tmp_path / 'engines', **options)


def handler():
    obj = object.__new__(RFPoseJointTRTInference)
    obj.threshold = 0.48
    obj.contract = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    obj.persistent_tensors = {'source_image': torch.empty(1, 3, 640, 640, dtype=torch.uint8)}
    obj.image_input_shape = (1, 3, 384, 384)
    return obj


@pytest.mark.parametrize('contract,expected', [
    ({}, [1, 1, 2, 3, 17, 300]),
    ({'skip_empty': True}, [0, 1, 2, 3, 17, 300]),
    ({'static_person_batches': [1, 2, 4, 300]}, [0, 1, 2, 4, 300, 300]),
    ({'static_person_batches': [1], 'dynamic_person_fallback': True}, [0, 1, 2, 3, 17, 300]),
    ({'static_person_batches': [1, 4], 'dynamic_person_fallback': True}, [0, 1, 4, 4, 17, 300]),
])
def test_processed_counts_include_real_work_but_not_output_storage_padding(contract, expected):
    np.testing.assert_array_equal(processed_pose_counts([0, 1, 2, 3, 17, 300], contract), expected)


def test_fixed_batch_accounting_rejects_truncating_profile():
    with pytest.raises(ValueError, match='exceeds'):
        processed_pose_counts([2], {'static_person_batches': [1]})


def test_preprocess_keeps_original_pixels_and_reference_detector_format():
    obj = handler()
    pixels = torch.randint(0, 256, (3, 426, 640), dtype=torch.uint8)
    image, meta = obj.preprocess(pixels)
    expected = TF.resize(TF.normalize(pixels[None].float() / 255,
        obj.contract['mean'], obj.contract['std']), [384, 384], antialias=True)
    torch.testing.assert_close(image, expected, rtol=0, atol=0)
    torch.testing.assert_close(obj.pending_inputs['source_image'][0, :, :426], pixels, rtol=0, atol=0)
    assert torch.count_nonzero(obj.pending_inputs['source_image'][0, :, 426:]) == 0
    assert meta == dict(height=426, width=640)
    assert obj.pending_inputs['confidence_threshold'].item() == np.float32(0.48)
    assert obj.pending_inputs['source_hw'].tolist() == [[426, 640]]
    # SAB's ordinary evaluator provides float RGB in [0,1]. Roundtrip is exact.
    obj.preprocess(pixels.float() / 255)
    torch.testing.assert_close(obj.pending_inputs['source_image'][0, :, :426], pixels, rtol=0, atol=0)


def test_postprocess_does_not_redecode_rescore_filter_or_clip():
    obj = handler()
    xy = torch.arange(1 * 3 * 17 * 2, dtype=torch.float32).reshape(1, 3, 17, 2) - 20
    outputs = dict(keypoints=xy, boxes=torch.tensor([[[1., 2., 30., 40.]]]*3).transpose(0, 1),
                   valid=torch.tensor([[True, False, True]]), scores=torch.tensor([[0.3, 0.0, 0.2]]))
    boxes, labels, scores, points = obj.postprocess(outputs, dict(height=100, width=200))
    torch.testing.assert_close(scores, torch.tensor([[0.3, 0.2]]), rtol=0, atol=0)
    assert labels.tolist() == [[1, 1]]
    restored = points.reshape(1, 2, 17, 3)[..., :2] * torch.tensor([200., 100.])
    torch.testing.assert_close(restored, xy[:, [0, 2]])
    assert boxes.shape == (1, 2, 4)
    outputs['valid'].fill_(False)
    boxes, labels, scores, points = obj.postprocess(outputs, dict(height=100, width=200))
    assert points.shape == (1, 0, 51)
    assert scores.shape == (1, 0)


def test_raw_cache_preserves_detector_gate_separately_from_pose_score():
    obj = handler()
    obj.cache_outputs, obj.output_cache = True, []
    outputs = dict(keypoints=torch.zeros(1, 2, 17, 2), boxes=torch.zeros(1, 2, 4),
        valid=torch.tensor([[True, False]]), scores=torch.tensor([[.2, .1]]),
        detector_scores=torch.tensor([[.8, .3]]), bucket_ids=torch.tensor([[2, 0]], dtype=torch.int32))
    obj.postprocess(outputs, dict(height=100, width=200))
    row = obj.output_cache[0]
    np.testing.assert_array_equal(row['detector_scores'], np.asarray([.8], dtype=np.float32))
    np.testing.assert_array_equal(row['scores'], np.asarray([.2], dtype=np.float32))
    assert row['keypoints'].shape == (1, 17, 2)


@pytest.mark.parametrize('nested', [False, True])
def test_custom_operators_are_rejected_even_inside_if(tmp_path, nested):
    value = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1])
    custom = onnx.helper.make_node('UntrustedOp', [], ['y'], domain='not.standard')
    branch = onnx.helper.make_graph([custom], 'branch', [], [value])
    node = onnx.helper.make_node('If', ['condition'], ['y'], then_branch=branch, else_branch=branch) if nested else custom
    graph = onnx.helper.make_graph([node], 'test', [], [value])
    model = onnx.helper.make_model(graph)
    onnx.helper.set_model_props(model, {'sab.rfpose': json.dumps(dict(contract_version=7, sab_input=True))})
    path = tmp_path / 'bad.onnx'
    onnx.save(model, path)
    with pytest.raises(ValueError, match='stock ONNX'):
        read_contract(path)
