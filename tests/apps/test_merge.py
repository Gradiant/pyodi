import json
from pathlib import Path

import numpy as np

from pyodi.apps.crops_merge import crops_merge


def test_crops_merge(tmpdir):
    # Test box is dicarded since iou between preds boxes is >.39 and final box coords are in original image coords
    tmpdir = Path(tmpdir)
    preds_path = tmpdir / "preds.json"
    gt_path = tmpdir / "gt.json"
    output_path = tmpdir / "output.json"
    iou_thr = 0.3

    preds = [
        {"image_id": 0, "bbox": [10, 10, 5, 5], "score": 0.7, "category_id": 1},
        {"image_id": 1, "bbox": [0, 0, 8, 8], "score": 0.9, "category_id": 1},
    ]

    gt = {
        "images": [
            {"id": 0, "file_name": "test_0_0.jpg"},
            {"id": 1, "file_name": "test_10_10.jpg"},
        ],
        "old_images": [{"id": 0, "file_name": "test.jpg", "width": 20, "height": 20}],
    }

    with open(preds_path, "w") as f:
        json.dump(preds, f)

    with open(gt_path, "w") as f:
        json.dump(gt, f)

    output_path = crops_merge(gt_path, output_path, preds_path, iou_thr=iou_thr)

    result = json.load(open(output_path))

    assert len(result) == 1
    np.testing.assert_almost_equal(result[0]["bbox"], [10, 10, 8, 8])
