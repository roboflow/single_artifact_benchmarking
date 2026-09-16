"""CPU-only formatter regression, without importing CUDA/TRT dependencies."""

import ast
from pathlib import Path

import pytest


def formatter():
    path = Path(__file__).resolve().parents[1] / 'sab/models/utils.py'
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('coco_metric_layout', 'pretty_print_results')]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def result(count):
    return dict(artifact_request=dict(onnx_path=f'model{count}', is_trt=True, needs_fp16=True),
                accuracy_stats=[i / 100 for i in range(count)], latency_stats=dict(median=1.))


def table_row(text, heading, model):
    section = text.split(heading, 1)[1].split('\n\n', 1)[0]
    return next(line.split()[1:] for line in section.splitlines() if line.startswith(model))


def test_keypoint_secondary_metrics_use_ten_stat_layout(capsys):
    formatter()['pretty_print_results']([result(10)])
    text = capsys.readouterr().out
    assert table_row(text, 'AP breakdown (COCO keypoints):', 'model10') == ['3.0', '4.0']
    assert table_row(text, 'AR breakdown (COCO keypoints):', 'model10') == ['5.0', '6.0', '7.0', '8.0', '9.0']
    assert 'AP_s' not in text and 'AR@1' not in text and 'AR50' in text


def test_box_layout_and_mixed_task_reporting(capsys):
    formatter()['pretty_print_results']([result(10), result(12)])
    text = capsys.readouterr().out
    assert table_row(text, 'AP breakdown (COCO bbox/segm):', 'model12') == ['3.0', '4.0', '5.0']
    assert table_row(text, 'AR breakdown (COCO bbox/segm):', 'model12') == ['6.0', '7.0', '8.0', '9.0', '10.0', '11.0']
    assert table_row(text, 'AP breakdown (COCO keypoints):', 'model10') == ['3.0', '4.0']


@pytest.mark.parametrize('count', [0, 9, 11, 13])
def test_unknown_layout_is_not_silently_mislabeled(count, capsys):
    with pytest.raises(ValueError, match='statistics length'):
        formatter()['pretty_print_results']([result(count)])
    assert capsys.readouterr().out == ''
