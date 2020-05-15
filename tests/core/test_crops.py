import numpy as np
from PIL import Image

from pyodi.core.crops import (
    annotation_inside_crop,
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


def test_annotation_inside_crop():
    annotation = {"bbox": [5, 5, 5, 5]}

    # Annotation is fully inside
    assert annotation_inside_crop(annotation, [5, 5, 10, 10])

    # Right coordinate is outside
    assert annotation_inside_crop(annotation, [11, 5, 15, 10])
    # Bottom coordinate is outside
    assert annotation_inside_crop(annotation, [5, 11, 10, 15])
    # Left coordinate is outside
    assert annotation_inside_crop(annotation, [0, 5, 4, 10])
    # Top coordinate is outside
    assert annotation_inside_crop(annotation, [5, 0, 10, 4])

    # All coordinates are outside
    assert not annotation_inside_crop(annotation, [11, 11, 15, 15])
    assert not annotation_inside_crop(annotation, [0, 0, 4, 4])


def test_get_annotation_in_crop():
    annotation = {"bbox": [5, 5, 5, 5], "iscrowd": 0, "category_id": 0, "score": 1.0}

    # Annotation is fully inside crop
    new_annotation = get_annotation_in_crop(annotation, [0, 0, 10, 10])
    assert tuple(annotation["bbox"]) == tuple(new_annotation["bbox"])

    # Right coordinate is outside
    new_annotation = get_annotation_in_crop(annotation, [0, 0, 9, 10])
    assert tuple(annotation["bbox"]) != tuple(new_annotation["bbox"])
    assert new_annotation["bbox"][2] < annotation["bbox"][2]

    # Bottom coordinate is outside
    new_annotation = get_annotation_in_crop(annotation, [0, 0, 10, 9])
    assert tuple(annotation["bbox"]) != tuple(new_annotation["bbox"])
    assert new_annotation["bbox"][3] < annotation["bbox"][3]

    # Left coordinate is outside
    new_annotation = get_annotation_in_crop(annotation, [6, 0, 10, 10])
    assert tuple(annotation["bbox"]) != tuple(new_annotation["bbox"])
    assert new_annotation["bbox"][0] == 0

    # Top coordinate is outside
    new_annotation = get_annotation_in_crop(annotation, [0, 6, 10, 10])
    assert tuple(annotation["bbox"]) != tuple(new_annotation["bbox"])
    assert new_annotation["bbox"][1] == 0
