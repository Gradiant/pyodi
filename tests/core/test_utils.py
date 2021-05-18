import json

import numpy as np
import pytest

from pyodi.core.utils import coco_ground_truth_to_df


@pytest.fixture(scope="session")
def annotations_file(tmpdir_factory):
    images = [
        {"id": 1, "file_name": "1.png", "width": 1280, "height": 720},
        {"id": 2, "file_name": "2.png", "width": 1280, "height": 720},
    ]
    annotations = [
        {"id": 1, "image_id": 1, "bbox": [0, 0, 2, 2], "area": 4, "category_id": 1},
        {"id": 1, "image_id": 1, "bbox": [0, 0, 2, 2], "area": 4, "category_id": 2},
        {"id": 1, "image_id": 2, "bbox": [0, 0, 2, 2], "area": 4, "category_id": 3},
    ]
    categories = [
        {"supercategory": "person", "id": 1, "name": "person"},
        {"supercategory": "animal", "id": 2, "name": "cat"},
        {"supercategory": "animal", "id": 3, "name": "dog"},
    ]

    fn = tmpdir_factory.mktemp("data").join("ground_truth.json")
    data = dict(images=images, annotations=annotations, categories=categories)

    with open(str(fn), "w") as f:
        json.dump(data, f)

    return fn


def test_coco_ground_truth_to_df(annotations_file):
    df_annotations = coco_ground_truth_to_df(annotations_file)
    assert len(df_annotations) == 3
    np.testing.assert_array_equal(
        df_annotations["col_left"].to_numpy(), np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        df_annotations["category"].to_numpy(), np.array(["person", "cat", "dog"])
    )


def test_coco_ground_truth_to_df_with_max_images(annotations_file):
    df_annotations = coco_ground_truth_to_df(annotations_file, max_images=1)
    assert len(df_annotations) == 2
