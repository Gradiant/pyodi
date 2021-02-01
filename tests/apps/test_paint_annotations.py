import json
from pathlib import Path

import numpy as np
from matplotlib import cm
from PIL import Image

from pyodi.apps.paint_annotations import paint_annotations


def test_image_annotations(tmp_path):

    images_folder = tmp_path / "images"
    Path(images_folder).mkdir(exist_ok=True, parents=True)

    output_folder = tmp_path / "test_result"
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    images = [
        {"id": 0, "width": 10, "height": 10, "file_name": "test_0.png"},
        {"id": 1, "width": 10, "height": 10, "file_name": "test_1.png"},
        {"id": 2, "width": 10, "height": 10, "file_name": "test_2.png"},
    ]

    annotations = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 2, 2], "score": 1},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 2, 2], "score": 1},
        {"image_id": 1, "category_id": 1, "bbox": [3, 3, 2, 2], "score": 1},
    ]

    categories = [
        {"id": 0, "name": "", "supercategory": "object"},
        {"id": 1, "name": "", "supercategory": "object"},
    ]

    coco_data = dict(images=images, annotations=annotations, categories=categories)

    n_categories = len(categories)
    colormap = cm.rainbow(np.linspace(0, 1, n_categories))

    color_1 = np.round(colormap[0] * 255)
    color_2 = np.round(colormap[1] * 255)

    num_image = 0
    for image_data in images:

        image = np.zeros((image_data["height"], image_data["width"], 3), dtype=np.uint8)
        Image.fromarray(image).save(images_folder / f"test_{num_image}.png")
        num_image += 1

    with open(tmp_path / "test_annotation.json", "w") as json_file:
        json.dump(coco_data, json_file)

    paint_annotations(
        tmp_path / "test_annotation.json",
        images_folder,
        output_folder,
        show_label=False,
    )

    result_image_0 = np.asarray(Image.open(output_folder / "test_0_result.png"))
    result_image_1 = np.asarray(Image.open(output_folder / "test_1_result.png"))

    assert np.array_equal(result_image_0[0, 1], color_1)
    assert np.array_equal(result_image_0[2, 3], color_1)

    assert np.array_equal(result_image_1[0, 1], color_1)
    assert np.array_equal(result_image_1[2, 3], color_1)

    assert np.array_equal(result_image_1[3, 4], color_2)
    assert np.array_equal(result_image_1[5, 6], color_2)
