import numpy as np
import pandas as pd
import pytest

from pyodi.core.boxes import (
    add_centroids,
    coco_to_corners,
    corners_to_coco,
    denormalize,
    filter_zero_area_bboxes,
    get_bbox_array,
    get_bbox_column_names,
    get_df_from_bboxes,
    get_scale_and_ratio,
    normalize,
    scale_bbox_dimensions,
)


@pytest.fixture
def get_simple_annotations_with_img_sizes():
    def _get_fake_data(bboxes=None, bbox_format="coco"):
        if bboxes is None:
            bboxes = np.array([[0, 0, 10, 10, 100, 100], [20, 20, 50, 40, 100, 100]])

        columns = get_bbox_column_names(bbox_format) + ["img_width", "img_height"]
        return pd.DataFrame(bboxes, columns=columns)

    return _get_fake_data


def test_scale_bbox_dimensions(get_simple_annotations_with_img_sizes):
    df_annotations = get_simple_annotations_with_img_sizes()
    df_annotations = scale_bbox_dimensions(df_annotations, (1280, 720))
    bboxes = get_bbox_array(df_annotations, prefix="scaled")
    expected_bboxes = np.array([[0, 0, 128, 72], [256, 144, 640, 288]], dtype=np.int32)
    np.testing.assert_equal(bboxes, expected_bboxes)


def test_scale_bbox_dimensions_with_keep_ratio(get_simple_annotations_with_img_sizes):
    df_annotations = get_simple_annotations_with_img_sizes()
    df_annotations = scale_bbox_dimensions(df_annotations, (1280, 720), keep_ratio=True)
    bboxes = get_bbox_array(df_annotations, prefix="scaled")
    expected_bboxes = np.array([[0, 0, 72, 72], [144, 144, 360, 288]], dtype=np.int32)
    np.testing.assert_equal(bboxes, expected_bboxes)


def test_get_area_and_ratio(get_simple_annotations_with_img_sizes):
    df_annotations = get_simple_annotations_with_img_sizes()
    df_annotations = get_scale_and_ratio(df_annotations)
    expected_scales = np.sqrt([100, 2000])
    expected_ratios = np.array([1, 0.8], dtype=float)
    np.testing.assert_equal(df_annotations["scale"].to_numpy(), expected_scales)
    np.testing.assert_equal(df_annotations["ratio"].to_numpy(), expected_ratios)


def test_get_bbox_matrix_corners(get_simple_annotations_with_img_sizes):
    df_annotations = get_simple_annotations_with_img_sizes()
    matrix = get_bbox_array(df_annotations, output_bbox_format="corners")
    expected_result = np.array([[0, 0, 10, 10], [20, 20, 70, 60]])
    np.testing.assert_equal(matrix, expected_result)


@pytest.mark.parametrize("bbox_format", (["corners", "coco"]))
def test_get_df_from_bboxes(get_simple_annotations_with_img_sizes, bbox_format):
    bboxes = np.array([[20, 20, 5, 5, 100, 100], [40, 40, 20, 20, 100, 100]])
    df_annotations = get_simple_annotations_with_img_sizes(
        bboxes, bbox_format=bbox_format
    )
    matrix = get_bbox_array(
        df_annotations, input_bbox_format=bbox_format, output_bbox_format=bbox_format
    )

    df = get_df_from_bboxes(
        matrix, input_bbox_format=bbox_format, output_bbox_format=bbox_format
    )
    expected_result = df_annotations[get_bbox_column_names(bbox_format)]
    pd.testing.assert_frame_equal(df, expected_result)


def test_filter_zero_area_bboxes(get_simple_annotations_with_img_sizes):
    bboxes = np.array([[20, 20, 5, 0, 100, 100], [40, 40, 20, 20, 100, 100]])
    df_annotations = get_simple_annotations_with_img_sizes(bboxes, bbox_format="coco")
    df_annotations = filter_zero_area_bboxes(df_annotations)
    assert len(df_annotations) == 1


def test_bboxes_transforms():
    bboxes_coco = np.array([[0, 0, 10, 10], [0, 6, 3, 9]])
    bboxes_corners = np.array([[0, 0, 10, 10], [0, 6, 3, 15]])
    np.testing.assert_equal(bboxes_coco, corners_to_coco(bboxes_corners))
    np.testing.assert_equal(bboxes_corners, coco_to_corners(bboxes_coco))


def test_add_centroids(get_simple_annotations_with_img_sizes):
    df_annotations = get_simple_annotations_with_img_sizes()
    centroids = add_centroids(df_annotations)[
        ["col_centroid", "row_centroid"]
    ].to_numpy()
    expected_result = np.array([[5, 5], [45, 40]])
    np.testing.assert_equal(centroids, expected_result)


def test_normalization():
    bboxes = np.array([[0, 0, 10, 10], [0, 0, 5, 20]])
    normalized = normalize(bboxes, 100, 100)
    denormalized = denormalize(normalized, 100, 100)
    np.testing.assert_equal(denormalized, bboxes)
