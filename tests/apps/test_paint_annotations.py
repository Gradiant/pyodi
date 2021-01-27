import json
from pathlib import Path

import numpy as np
from PIL import Image

from pyodi.apps.paint_annotations import paint_annotations


def create_annotations(tmp_path):

    annotations_file = {
        "images": [
            {
                "id": 0,
                "width": 1920,
                "height": 1080,
                "file_name": "test_0.png",
                "license": "",
                "flickr_url": "",
                "coco_url": "",
                "date_captured": "",
                "POV": "",
                "source": "",
                "is_full_sequence": False,
                "owner": "",
            },
            {
                "id": 1,
                "width": 1920,
                "height": 1080,
                "file_name": "test_1.png",
                "license": "",
                "flickr_url": "",
                "coco_url": "",
                "date_captured": "",
                "POV": "",
                "source": "",
                "is_full_sequence": False,
                "owner": "",
            },
            {
                "id": 2,
                "width": 1920,
                "height": 1080,
                "file_name": "test_2.png",
                "license": "",
                "flickr_url": "",
                "coco_url": "",
                "date_captured": "",
                "POV": "",
                "source": "",
                "is_full_sequence": False,
                "owner": "",
            },
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 1,
                "segmentation": [
                    [316.0, 951.0, 316.0, 963.0, 329.0, 963.0, 329.0, 951.0]
                ],
                "area": 156.0,
                "bbox": [653.0, 950.0, 15.0, 10.0],
                "iscrowd": 0,
                "score": 1,
            },
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [
                    [346.0, 957.0, 346.0, 968.0, 359.0, 968.0, 359.0, 957.0]
                ],
                "area": 143.0,
                "bbox": [231.0, 200.0, 10.0, 10.0],
                "iscrowd": 0,
                "score": 1,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [
                    [346.0, 957.0, 346.0, 968.0, 359.0, 968.0, 359.0, 957.0]
                ],
                "area": 143.0,
                "bbox": [600.0, 674.0, 30.0, 30.0],
                "iscrowd": 0,
                "score": 1,
            },
        ],
        "info": {
            "year": "",
            "version": "",
            "description": "",
            "contributor": "",
            "url": "",
            "date_created": "",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "drone", "supercategory": "object"}],
    }
    with open(tmp_path / "test_annotation.json", "w") as json_file:
        json.dump(annotations_file, json_file)


def create_images(tmp_path):

    with open(tmp_path / "test_annotation.json", "r") as json_file:
        annotations = json.loads(json_file.read())

    image_data = annotations["images"]
    bbox_data = []
    for ann in annotations["annotations"]:
        bbox_data.append(ann["bbox"])

    images_folder = tmp_path / "images"
    Path(images_folder).mkdir(exist_ok=True, parents=True)

    output_folder = tmp_path / "test_result"
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    image_0 = np.full(
        (image_data[0]["height"], image_data[0]["width"], 3), 255, dtype=np.uint8
    )

    (
        image_0_bbox_left,
        image_0_bbox_top,
        image_0_bbox_width,
        image_0_bbox_height,
    ) = bbox_data[0]
    image_0[
        int(image_0_bbox_top) : int(image_0_bbox_top + image_0_bbox_height),
        int(image_0_bbox_left) : int(image_0_bbox_left + image_0_bbox_width),
    ] = 0
    Image.fromarray(image_0).save(images_folder / "test_0.png")

    image_1 = np.full(
        (image_data[1]["height"], image_data[1]["width"], 3), 255, dtype=np.uint8
    )

    (
        image_1_bbox_left_a,
        image_1_bbox_top_a,
        image_1_bbox_width_a,
        image_1_bbox_height_a,
    ) = bbox_data[1]

    (
        image_1_bbox_left_b,
        image_1_bbox_top_b,
        image_1_bbox_width_b,
        image_1_bbox_height_b,
    ) = bbox_data[2]
    image_1[
        int(image_1_bbox_top_a) : int(image_1_bbox_top_a + image_1_bbox_height_a),
        int(image_1_bbox_left_a) : int(image_1_bbox_left_a + image_1_bbox_width_a),
    ] = 0
    image_1[
        int(image_1_bbox_top_b) : int(image_1_bbox_top_b + image_1_bbox_height_b),
        int(image_1_bbox_left_b) : int(image_1_bbox_left_b + image_1_bbox_width_b),
    ] = 0
    Image.fromarray(image_1).save(images_folder / "test_1.png")

    image_2 = np.full(
        (image_data[1]["height"], image_data[1]["width"], 3), 255, dtype=np.uint8
    )
    Image.fromarray(image_2).save(images_folder / "test_2.png")


def test_image_annotations(tmp_path):

    create_annotations(tmp_path)

    create_images(tmp_path)

    paint_annotations(
        tmp_path / "test_annotation.json", tmp_path / "images", tmp_path / "test_result"
    )

    assert Path(tmp_path / "test_annotation.json").is_file()

    assert Path(tmp_path / "images").is_dir()

    assert Path(tmp_path / "test_result").is_dir()
