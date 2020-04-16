import numpy as np
import pandas as pd
import pytest

from pyodi.coco.utils import get_area_and_ratio, get_bbox_matrix, scale_bbox_dimensions


@pytest.fixture
def get_simple_annotations_with_img_sizes():
    def _get_fake_data():
        bboxes = np.array([[0, 0, 10, 10, 100, 100], [20, 20, 50, 40, 100, 100]])
        columns = [
            "col_centroid",
            "row_centroid",
            "width",
            "height",
            "img_width",
            "img_height",
        ]
        return pd.DataFrame(bboxes, columns=columns)

    return _get_fake_data


def test_scale_bbox_dimensions(get_simple_annotations_with_img_sizes):
    df_annotations = get_simple_annotations_with_img_sizes()
    df_annotations = scale_bbox_dimensions(df_annotations, (1280, 720))
    bboxes = get_bbox_matrix(df_annotations, prefix="scaled")
    expected_bboxes = np.array([[0, 0, 128, 72], [256, 180, 640, 360]], dtype=np.int32)
    np.testing.assert_equal(bboxes, expected_bboxes)


def test_get_area_and_ratio(get_simple_annotations_with_img_sizes):
    df_annotations = get_simple_annotations_with_img_sizes()
    df_annotations = get_area_and_ratio(df_annotations)
    expected_areas = np.array([100, 2000], dtype=np.int32)
    expected_ratios = np.array([1, 0.8], dtype=np.float)
    np.testing.assert_equal(df_annotations["area"].to_numpy(), expected_areas)
    np.testing.assert_equal(df_annotations["ratio"].to_numpy(), expected_ratios)
