import json
from pathlib import Path

import numpy as np

from pyodi.apps.crops_merge import crops_merge


def test_crops_merge_gt(tmpdir):
    # Test for crops merge (only ground_truth file)

    tmpdir = Path(tmpdir)
    gt_path = tmpdir / "gt.json"
    output_path = tmpdir / "output.json"

    gt = {
        "images": [
            {
                "file_name": "gopro_001_102_0_0.png",
                "height": 720,
                "width": 720,
                "id": 0,
            },
            {
                "file_name": "gopro_001_102_720_0.png",
                "height": 720,
                "width": 720,
                "id": 1,
            },
            {
                "file_name": "gopro_001_102_1200_0.png",
                "height": 720,
                "width": 720,
                "id": 2,
            },
            {
                "file_name": "gopro_001_102_0_360.png",
                "height": 720,
                "width": 720,
                "id": 3,
            },
            {
                "file_name": "gopro_001_102_720_360.png",
                "height": 720,
                "width": 720,
                "id": 4,
            },
        ],
        "old_images": [
            {
                "id": 1,
                "width": 1920,
                "height": 1080,
                "file_name": "gopro_001/gopro_001_102.png",
            }
        ],
        "annotations": [
            {
                "bbox": [483.00000000000006, 465.0, 19.999999999999943, 14.0],
                "score": 1,
                "category_id": 1,
                "id": 0,
                "image_id": 3,
            },
            {
                "bbox": [433.0, 626.0, 19.0, 13.0],
                "score": 1,
                "category_id": 1,
                "id": 1,
                "image_id": 4,
            },
        ],
    }

    with open(gt_path, "w") as f:
        json.dump(gt, f)

    output_path = crops_merge(gt_path, output_path)

    result = json.load(open(output_path))

    assert len(result["images"]) == 1, "Error on images length"
    assert len(result["annotations"]) == 2, "Error on annotations length"

    np.testing.assert_almost_equal(
        result["annotations"][0]["bbox"], [1153, 986, 19, 13]
    )
    np.testing.assert_almost_equal(result["annotations"][1]["bbox"], [483, 825, 20, 14])
