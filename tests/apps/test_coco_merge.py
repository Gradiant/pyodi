import json
from pathlib import Path

from pyodi.apps.coco_merge import coco_merge


def test_coco_merge(tmpdir):
    tmpdir = Path(tmpdir)
    images = [{"id": 0, "file_name": "0.jpg"}, {"id": 1, "file_name": "1.jpg"}]

    anns1 = [
        {"image_id": 0, "category_id": 1, "id": 0},
        {"image_id": 1, "category_id": 2, "id": 1},
    ]
    anns2 = [
        {"image_id": 0, "category_id": 1, "id": 0},
        {"image_id": 1, "category_id": 2, "id": 1},
    ]
    anns3 = [
        {"image_id": 0, "category_id": 1, "id": 0},
        {"image_id": 1, "category_id": 2, "id": 1},
    ]

    categories1 = [{"id": 1, "name": "drone"}, {"id": 2, "name": "bird"}]
    categories2 = [{"id": 1, "name": "drone"}, {"id": 2, "name": "plane"}]
    categories3 = [{"id": 1, "name": "plane"}, {"id": 2, "name": "drone"}]

    coco1 = dict(images=images, annotations=anns1, categories=categories1)
    coco2 = dict(images=images, annotations=anns2, categories=categories2)
    coco3 = dict(images=images, annotations=anns3, categories=categories3)

    tmp_files = []
    for i, coco_data in enumerate([coco1, coco2, coco3]):
        tmp_files.append(tmpdir / f"{i}.json")
        with open(tmp_files[-1], "w") as f:
            json.dump(coco_data, f)

    result_file = coco_merge(tmpdir / "result.json", tmp_files)

    data = json.load(open(result_file))

    assert data["categories"] == [
        {"id": 1, "name": "drone"},
        {"id": 2, "name": "bird"},
        {"id": 3, "name": "plane"},
    ]
    assert [x["id"] for x in data["images"]] == list(range(len(data["images"])))
    assert [x["id"] for x in data["annotations"]] == list(
        range(len(data["annotations"]))
    )
    assert [i["category_id"] for i in data["annotations"]] == [1, 2, 1, 3, 3, 1]
