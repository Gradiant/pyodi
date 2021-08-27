import json
from pathlib import Path

from pyodi.apps.ground_truth import ground_truth


def test_ground_truth_saves_output_to_files(tmpdir):
    output = tmpdir.mkdir("results")

    categories = [{"id": 1, "name": "drone"}]
    images = [{"id": 0, "file_name": "image.jpg", "height": 10, "width": 10}]
    annotations = [
        {"image_id": 0, "category_id": 1, "id": 0, "bbox": [0, 0, 5, 5], "area": 25}
    ]
    coco_data = dict(
        images=images,
        annotations=annotations,
        categories=categories,
        info={},
        licenses={},
    )
    with open(tmpdir / "data.json", "w") as f:
        json.dump(coco_data, f)

    ground_truth(tmpdir / "data.json", show=False, output=output)

    assert len(list(Path(output / "data").iterdir())) == 3
