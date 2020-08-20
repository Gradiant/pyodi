import json
from pathlib import Path

from pyodi.apps.coco_split import coco_split


def test_random_coco_split(tmpdir):
    tmpdir = Path(tmpdir)

    categories = [{"id": 1, "name": "drone"}, {"id": 2, "name": "bird"}]

    images = [
        {"id": 0, "file_name": "vidA-0.jpg", "source": "A"},
        {"id": 1, "file_name": "vidA-1.jpg", "source": "A"},
        {"id": 2, "file_name": "vidB-0.jpg", "source": "B"},
        {"id": 3, "file_name": "vidC-0.jpg", "source": "C"},
    ]

    annotations = [
        {"image_id": 0, "category_id": 1, "id": 0},
        {"image_id": 0, "category_id": 2, "id": 1},
        {"image_id": 1, "category_id": 1, "id": 2},
        {"image_id": 1, "category_id": 1, "id": 3},
        {"image_id": 1, "category_id": 2, "id": 4},
        {"image_id": 2, "category_id": 1, "id": 5},
        {"image_id": 3, "category_id": 2, "id": 6},
        {"image_id": 3, "category_id": 2, "id": 7},
    ]

    coco_data = dict(
        images=images,
        annotations=annotations,
        categories=categories,
        info={},
        licenses={},
    )

    json.dump(coco_data, open(tmpdir / "coco.json", "w"))

    train_path, val_path = coco_split(
        annotations_file=str(tmpdir / "coco.json"),
        output_filename=str(tmpdir / "random_coco_split"),
        mode="random",
        val_percentage=0.25,
        seed=47,
    )

    train_data = json.load(open(train_path))
    val_data = json.load(open(val_path))

    assert train_data["categories"] == categories
    assert val_data["categories"] == categories
    assert len(train_data["images"]) == 3
    assert len(val_data["images"]) == 1
    for x in train_data["images"]:
        assert x["file_name"] == images[x["id"]]["file_name"]
    for x in val_data["images"]:
        assert x["file_name"] == images[x["id"]]["file_name"]
    assert len(train_data["annotations"]) == 6
    assert len(val_data["annotations"]) == 2


def test_property_coco_split(tmpdir):
    pass  # ToDo test
