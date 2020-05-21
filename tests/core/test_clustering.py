import numpy as np
import pytest

from pyodi.core.clustering import (
    find_pyramid_level,
    get_max_overlap,
    kmeans_euclidean,
    origin_iou,
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


def test_max_overlap(get_bboxes_matrices):
    bboxes1, bboxes2 = get_bboxes_matrices()

    expected_result = np.array([2.0 / 16.0, 1.0 / 16.0])
    iou_values = get_max_overlap(bboxes1.astype(np.float32), bboxes2.astype(np.float32))

    np.testing.assert_equal(expected_result, iou_values)


def test_origin_iou(get_bboxes_matrices):
    bboxes1, bboxes2 = get_bboxes_matrices()
    orig_iou = origin_iou(bboxes1[:, 2:], bboxes2[:, 2:])
    bboxes1[:, :2] = 0
    bboxes2[:, :2] = 0
    max_overlap = get_max_overlap(
        bboxes1.astype(np.float32), bboxes2.astype(np.float32)
    )
    np.testing.assert_almost_equal(orig_iou.max(1), max_overlap)


def test_kmeans_scale_ratio():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    result = kmeans_euclidean(X, n_clusters=2, silhouette_metric=True)
    np.testing.assert_almost_equal(result["silhouette"], 0.713, 3)


def test_find_levels():
    X = np.array([[1, 1], [10, 10], [64, 64]])
    strides = [4, 8, 16, 32, 64]
    levels = find_pyramid_level(X, strides)
    np.testing.assert_equal(levels, np.array([0, 1, 4]))
