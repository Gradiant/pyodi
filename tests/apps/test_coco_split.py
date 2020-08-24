import json
from pathlib import Path

from pyodi.apps.coco_split import property_split, random_split


def get_coco_data():

    categories = [{"id": 1, "name": "drone"}, {"id": 2, "name": "bird"}]

    images = [
        {"id": 0, "file_name": "vidA-0.jpg", "source": "A"},
        {"id": 1, "file_name": "vidA-1.jpg", "source": "A"},
        {"id": 2, "file_name": "vidB-0.jpg", "source": "B"},
        {"id": 3, "file_name": "vidC-0.jpg", "source": "C"},
        {"id": 4, "file_name": "vidD-0.jpg", "source": "D"},
        {"id": 5, "file_name": "vidD-66.jpg", "source": "badsrc"},
        {"id": 6, "file_name": "badvid-13.jpg", "source": "E"},
        {"id": 7, "file_name": "errvid-14.jpg", "source": "E"},
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
        {"image_id": 4, "category_id": 1, "id": 8},
        {"image_id": 5, "category_id": 1, "id": 9},
        {"image_id": 5, "category_id": 2, "id": 10},
        {"image_id": 5, "category_id": 2, "id": 11},
    ]

    coco_data = dict(
        images=images,
        annotations=annotations,
        categories=categories,
        info={},
        licenses={},
    )

    return coco_data


def test_random_coco_split(tmpdir):
    tmpdir = Path(tmpdir)

    coco_data = get_coco_data()

    json.dump(coco_data, open(tmpdir / "coco.json", "w"))

    train_path, val_path = random_split(
        annotations_file=str(tmpdir / "coco.json"),
        output_filename=str(tmpdir / "random_coco_split"),
        val_percentage=0.25,
        seed=49,
    )

    train_data = json.load(open(train_path))
    val_data = json.load(open(val_path))

    assert train_data["categories"] == coco_data["categories"]
    assert val_data["categories"] == coco_data["categories"]
    assert len(train_data["images"]) == 6
    assert len(val_data["images"]) == 2
    assert len(train_data["annotations"]) == 9
    assert len(val_data["annotations"]) == 3


def test_property_coco_split(tmpdir):
    tmpdir = Path(tmpdir)

    coco_data = get_coco_data()

    json.dump(coco_data, open(tmpdir / "coco.json", "w"))

    config = {
        "discard": {"file_name": "badvid|errvid", "source": "badsrc"},
        "val": {
            "file_name": {"frame 0": "vidA-0.jpg", "frame 1": "vidA-1.jpg"},
            "source": {"example C": "C", "example D": "D"},
        },
    }

    json.dump(config, open(tmpdir / "split_config.json", "w"))

    train_path, val_path = property_split(
        annotations_file=str(tmpdir / "coco.json"),
        output_filename=str(tmpdir / "property_coco_split"),
        split_config_file=str(tmpdir / "split_config.json"),
    )

    train_data = json.load(open(train_path))
    val_data = json.load(open(val_path))

    assert train_data["categories"] == coco_data["categories"]
    assert val_data["categories"] == coco_data["categories"]
    assert len(train_data["images"]) == 1
    assert len(val_data["images"]) == 4
    assert len(train_data["annotations"]) == 1
    assert len(val_data["annotations"]) == 8
