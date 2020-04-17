import pytest
from pyodi.core.clustering import iou
import numpy as np


@pytest.fixture
def get_bboxes_matrices():
    def _get_bboxes_matrices():
        bboxes1 = np.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        bboxes2 = np.array(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]]
        )
        return bboxes1, bboxes2

    return _get_bboxes_matrices


def test_iou(get_bboxes_matrices):
    bboxes1, bboxes2 = get_bboxes_matrices()

    expected_result = np.array(
        [[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]]
    )
    iou_values = iou(bboxes1, bboxes2)

    np.testing.assert_equal(expected_result, iou_values)


def test_iou_works_on_empty_inputs(get_bboxes_matrices):
    bboxes1, bboxes2 = get_bboxes_matrices()
    boxes_empty = np.zeros((0, 4))
    iou_empty_1 = iou(bboxes1, boxes_empty)
    iou_empty_2 = iou(boxes_empty, bboxes2)
    iou_empty_3 = iou(boxes_empty, boxes_empty)

    np.testing.assert_equal(iou_empty_1.shape, (2, 0))
    np.testing.assert_equal(iou_empty_2.shape, (0, 3))
    np.testing.assert_equal(iou_empty_3.shape, (0, 0))
