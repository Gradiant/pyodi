import numpy as np
from PIL import Image

from pyodi.core.crops import (
    annotation_inside_crop,
    filter_annotation_by_area,
    get_annotation_in_crop,
    get_crops_corners,
)


def test_get_crop_corners_overllap():
    image_pil = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    no_overlap = get_crops_corners(image_pil, crop_height=10, crop_width=10)
    row_overlap = get_crops_corners(
        image_pil, crop_height=10, crop_width=10, row_overlap=5
    )
    col_overlap = get_crops_corners(
        image_pil, crop_height=10, crop_width=10, col_overlap=5
    )
    row_and_col_overlap = get_crops_corners(
        image_pil, crop_height=10, crop_width=10, row_overlap=5, col_overlap=5
    )

    assert len(no_overlap) < len(row_overlap) < len(row_and_col_overlap)
    assert len(row_overlap) == len(col_overlap)


def test_get_crop_corners_single_crop():
    image_pil = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    no_overlap = get_crops_corners(image_pil, crop_height=100, crop_width=100)

    assert len(no_overlap) == 1
    assert no_overlap[0][0] == 0
    assert no_overlap[0][1] == 0
    assert no_overlap[0][2] == 100
    assert no_overlap[0][3] == 100


def test_get_crop_corners_coordinates():
    image_pil = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    no_overlap = get_crops_corners(image_pil, crop_height=5, crop_width=5)

    assert len(no_overlap) == 4
    assert tuple(no_overlap[0]) == (0, 0, 5, 5)
    assert tuple(no_overlap[1]) == (5, 0, 10, 5)
    assert tuple(no_overlap[2]) == (0, 5, 5, 10)
    assert tuple(no_overlap[3]) == (5, 5, 10, 10)


def test_get_crop_corners_bounds():
    image_pil = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    no_overlap = get_crops_corners(image_pil, crop_height=6, crop_width=6)

    assert len(no_overlap) == 4
    assert tuple(no_overlap[0]) == (0, 0, 6, 6)
    assert tuple(no_overlap[1]) == (4, 0, 10, 6)
    assert tuple(no_overlap[2]) == (0, 4, 6, 10)
    assert tuple(no_overlap[3]) == (4, 4, 10, 10)


def test_annotation_inside_crop():
    annotation = {"bbox": [4, 4, 2, 2]}

    assert annotation_inside_crop(annotation, [0, 0, 5, 5])
    assert annotation_inside_crop(annotation, [5, 0, 10, 5])
    assert annotation_inside_crop(annotation, [0, 5, 5, 10])
    assert annotation_inside_crop(annotation, [5, 5, 10, 10])


def test_annotation_outside_crop():
    annotation = {"bbox": [2, 2, 2, 2]}

    assert annotation_inside_crop(annotation, [0, 0, 5, 5])
    assert not annotation_inside_crop(annotation, [5, 0, 10, 5])
    assert not annotation_inside_crop(annotation, [0, 5, 5, 10])
    assert not annotation_inside_crop(annotation, [5, 5, 10, 10])

    annotation = {"bbox": [6, 2, 2, 2]}

    assert not annotation_inside_crop(annotation, [0, 0, 5, 5])
    assert annotation_inside_crop(annotation, [5, 0, 10, 5])
    assert not annotation_inside_crop(annotation, [0, 5, 5, 10])
    assert not annotation_inside_crop(annotation, [5, 5, 10, 10])

    annotation = {"bbox": [2, 6, 2, 2]}

    assert not annotation_inside_crop(annotation, [0, 0, 5, 5])
    assert not annotation_inside_crop(annotation, [5, 0, 10, 5])
    assert annotation_inside_crop(annotation, [0, 5, 5, 10])
    assert not annotation_inside_crop(annotation, [5, 5, 10, 10])

    annotation = {"bbox": [6, 6, 2, 2]}

    assert not annotation_inside_crop(annotation, [0, 0, 5, 5])
    assert not annotation_inside_crop(annotation, [5, 0, 10, 5])
    assert not annotation_inside_crop(annotation, [0, 5, 5, 10])
    assert annotation_inside_crop(annotation, [5, 5, 10, 10])


def test_get_annotation_in_crop():
    annotation = {"bbox": [2, 2, 2, 2], "iscrowd": 0, "category_id": 0, "score": 1.0}

    new_annotation = get_annotation_in_crop(annotation, [0, 0, 5, 5])
    assert tuple(new_annotation["bbox"]) == (2, 2, 2, 2)

    annotation = {"bbox": [4, 4, 2, 2], "iscrowd": 0, "category_id": 0, "score": 1.0}

    new_annotation = get_annotation_in_crop(annotation, [0, 0, 5, 5])
    assert tuple(new_annotation["bbox"]) == (4, 4, 1, 1)
    new_annotation = get_annotation_in_crop(annotation, [5, 0, 10, 5])
    assert tuple(new_annotation["bbox"]) == (0, 4, 1, 1)
    new_annotation = get_annotation_in_crop(annotation, [0, 5, 5, 10])
    assert tuple(new_annotation["bbox"]) == (4, 0, 1, 1)
    new_annotation = get_annotation_in_crop(annotation, [5, 5, 10, 10])
    assert tuple(new_annotation["bbox"]) == (0, 0, 1, 1)


def test_annotation_larger_than_threshold():
    annotation = {
        "bbox": [2, 2, 4, 5],
        "area": 20,
        "iscrowd": True,
        "score": 1.0,
        "category_id": 1,
    }

    new_annotation_tl = get_annotation_in_crop(annotation, [0, 0, 5, 5])
    new_annotation_tr = get_annotation_in_crop(annotation, [5, 0, 10, 5])
    new_annotation_bl = get_annotation_in_crop(annotation, [0, 5, 5, 10])
    new_annotation_br = get_annotation_in_crop(annotation, [5, 5, 10, 10])

    assert not filter_annotation_by_area(annotation, new_annotation_tl, 0.0)
    assert not filter_annotation_by_area(annotation, new_annotation_tr, 0.0)
    assert not filter_annotation_by_area(annotation, new_annotation_bl, 0.0)
    assert not filter_annotation_by_area(annotation, new_annotation_br, 0.0)

    assert not filter_annotation_by_area(annotation, new_annotation_tl, 0.1)
    assert not filter_annotation_by_area(annotation, new_annotation_tr, 0.1)
    assert not filter_annotation_by_area(annotation, new_annotation_bl, 0.1)
    assert filter_annotation_by_area(annotation, new_annotation_br, 0.1)

    assert not filter_annotation_by_area(annotation, new_annotation_tl, 0.25)
    assert filter_annotation_by_area(annotation, new_annotation_tr, 0.25)
    assert not filter_annotation_by_area(annotation, new_annotation_bl, 0.25)
    assert filter_annotation_by_area(annotation, new_annotation_br, 0.25)

    assert not filter_annotation_by_area(annotation, new_annotation_tl, 0.4)
    assert filter_annotation_by_area(annotation, new_annotation_tr, 0.4)
    assert filter_annotation_by_area(annotation, new_annotation_bl, 0.4)
    assert filter_annotation_by_area(annotation, new_annotation_br, 0.4)

    assert filter_annotation_by_area(annotation, new_annotation_tl, 0.5)
    assert filter_annotation_by_area(annotation, new_annotation_tr, 0.5)
    assert filter_annotation_by_area(annotation, new_annotation_bl, 0.5)
    assert filter_annotation_by_area(annotation, new_annotation_br, 0.5)
