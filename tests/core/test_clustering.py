import numpy as np
import pytest

from pyodi.core.clustering import (
    get_max_overlap,
    kmeans_euclidean,
    origin_iou,
    pairwise_iou,
)


@pytest.fixture
def get_bboxes_matrices():
    def _get_bboxes_matrices():
        bboxes1 = np.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        bboxes2 = np.array(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]]
        )
        return bboxes1, bboxes2

    return _get_bboxes_matrices


def test_pairwise_iou(get_bboxes_matrices):
    bboxes1, bboxes2 = get_bboxes_matrices()

    expected_result = np.array(
        [[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]]
    )
    iou_values = pairwise_iou(bboxes1, bboxes2)

    np.testing.assert_equal(expected_result, iou_values)


def test_max_overlap(get_bboxes_matrices):
    bboxes1, bboxes2 = get_bboxes_matrices()

    expected_result = np.array([2.0 / 16.0, 1.0 / 16.0])
    iou_values = get_max_overlap(bboxes1.astype(np.float32), bboxes2.astype(np.float32))

    np.testing.assert_equal(expected_result, iou_values)


def test_iou_works_on_empty_inputs(get_bboxes_matrices):
    bboxes1, bboxes2 = get_bboxes_matrices()
    boxes_empty = np.zeros((0, 4))
    iou_empty_1 = pairwise_iou(bboxes1, boxes_empty)
    iou_empty_2 = pairwise_iou(boxes_empty, bboxes2)
    iou_empty_3 = pairwise_iou(boxes_empty, boxes_empty)

    np.testing.assert_equal(iou_empty_1.shape, (2, 0))
    np.testing.assert_equal(iou_empty_2.shape, (0, 3))
    np.testing.assert_equal(iou_empty_3.shape, (0, 0))


def test_origin_iou(get_bboxes_matrices):
    bboxes1, bboxes2 = get_bboxes_matrices()
    orig_iou = origin_iou(bboxes1[:, 2:], bboxes2[:, 2:])
    bboxes1[:, :2] = 0
    bboxes2[:, :2] = 0
    pair_iou = pairwise_iou(bboxes1, bboxes2)
    np.testing.assert_equal(orig_iou, pair_iou)


def test_kmeans_scale_ratio():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    result = kmeans_euclidean(X, n_clusters=2, random_state=0, silhouette_metric=True)
    np.testing.assert_almost_equal(result["silhouette"], 0.713, 3)
