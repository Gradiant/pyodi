import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from pyodi.apps.crops.crops_split import crops_split


def create_image() -> Image:
    array = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)

    # Paint each future crop in a different color
    array[0:720, 0:720] = np.array([0, 0, 0], dtype=np.uint8)
    array[0:720, 720:1440] = np.array([255, 0, 0], dtype=np.uint8)
    array[0:720, 1200:1920] = np.array([0, 255, 0], dtype=np.uint8)
    array[360:1080, 0:720] = np.array([0, 0, 255], dtype=np.uint8)
    array[360:1080, 720:1440] = np.array([255, 255, 0], dtype=np.uint8)
    array[360:1080, 1200:1920] = np.array([255, 0, 255], dtype=np.uint8)

    img = Image.fromarray(array)
    return img


def test_crops_split(tmpdir):
    gt_path = Path(tmpdir) / "gt.json"  # Path to a COCO ground truth file
    img_folder_path = (
        Path(tmpdir) / "img_folder/"
    )  # Path where the images of the ground_truth_file are stored
    output_path = (
        Path(tmpdir) / "output.json"
    )  # Path where the `new_ground_truth_file` will be saved
    output_folder_path = (
        Path(tmpdir) / "output_folder/"
    )  # Path where the crops will be saved
    crop_height = 720
    crop_width = 720

    tmpdir.mkdir("img_folder")  # Create temporary folder to store ground truth images

    gt = {
        "categories": [{"id": 1, "name": "drone", "supercategory": "object"}],
        "images": [{"id": 1, "width": 1920, "height": 1080, "file_name": "img1.png"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "segmentation": [
                    1153.0,
                    986.0,
                    1153.0,
                    999.0,
                    1172.0,
                    999.0,
                    1172.0,
                    986.0,
                ],
                "area": 247.0,
                "category_id": 1,
                "bbox": [1153.0, 986.0, 19.0, 13.0],
                "score": 1,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "segmentation": [
                    433.0,
                    626.0,
                    433.0,
                    639.0,
                    452.0,
                    639.0,
                    452.0,
                    626.0,
                ],
                "area": 247.0,
                "category_id": 1,
                "bbox": [433.0, 626.0, 19.0, 13.0],
                "score": 1,
                "iscrowd": 0,
            },
        ],
    }

    with open(gt_path, "w") as f:
        json.dump(gt, f)

    img = create_image()
    img.save(img_folder_path / "img1.png")

    crops_split(
        gt_path,
        img_folder_path,
        output_path,
        output_folder_path,
        crop_height,
        crop_width,
    )

    number_crops = len(os.listdir(output_folder_path))
    result = json.load(open(output_path))

    assert os.path.isdir(output_folder_path), "Output folder not created"
    assert number_crops == 6, "Error in number of crops in output folder"
    assert (
        len(result["images"]) == 6
    ), "Error in number of crops in crops annotations file"
    assert (
        len(result["old_images"]) == 1
    ), "Error in number of old images in crops annotations file"
    assert [x["id"] for x in result["images"]] == list(range(len(result["images"])))
    assert [x["id"] for x in result["annotations"]] == list(
        range(len(result["annotations"]))
    )
    assert [x["image_id"] for x in result["annotations"]] == [0, 3, 4]


def test_crops_split_path(tmpdir):
    gt_path = Path(tmpdir) / "gt.json"
    img_folder_path = Path(tmpdir) / "img_folder/"
    output_path = Path(tmpdir) / "output.json"
    output_folder_path = Path(tmpdir) / "output_folder/"
    crop_height = 720
    crop_width = 720

    tmpdir.mkdir("img_folder")

    gt = {
        "categories": [{"id": 1, "name": "drone", "supercategory": "object"}],
        "images": [
            {"id": 1, "width": 1920, "height": 1080, "file_name": "images/img1.png"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "segmentation": [
                    1153.0,
                    986.0,
                    1153.0,
                    999.0,
                    1172.0,
                    999.0,
                    1172.0,
                    986.0,
                ],
                "area": 247.0,
                "category_id": 1,
                "bbox": [1153.0, 986.0, 19.0, 13.0],
                "score": 1,
                "iscrowd": 0,
            },
        ],
    }

    with open(gt_path, "w") as f:
        json.dump(gt, f)

    img = create_image()
    img.save(img_folder_path / "img1.png")

    crops_split(
        gt_path,
        img_folder_path,
        output_path,
        output_folder_path,
        crop_height,
        crop_width,
    )

    number_crops = len(os.listdir(output_folder_path))
    result = json.load(open(output_path))

    assert os.path.isdir(output_folder_path), "Output folder not created"
    assert number_crops == 6, "Error in number of crops in output folder"
    assert (
        len(result["images"]) == 6
    ), "Error in number of crops in crops annotations file"
    assert len(result["old_images"]) == 1, "Error in number of old images"


def test_annotation_output_folder_created(tmpdir):
    gt = {
        "categories": [{"id": 1, "name": "drone"}],
        "images": [{"id": 1, "width": 1920, "height": 1080, "file_name": "img1.png"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "area": 1,
                "category_id": 1,
                "bbox": [0, 0, 1, 1],
                "iscrowd": 0,
            },
        ],
    }

    with open(Path(tmpdir) / "gt.json", "w") as f:
        json.dump(gt, f)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    Image.fromarray(img).save(Path(tmpdir) / "img1.png")

    crops_split(
        ground_truth_file=Path(tmpdir) / "gt.json",
        image_folder=tmpdir,
        output_file=Path(tmpdir) / "new_folder/gt.json",
        output_image_folder=Path(tmpdir) / "crops_folder",
        crop_height=5,
        crop_width=5,
    )

    assert (Path(tmpdir) / "new_folder/gt.json").exists()
