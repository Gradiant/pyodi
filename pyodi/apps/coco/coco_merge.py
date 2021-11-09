"""# Coco Merge App.

The [`pyodi coco`][pyodi.apps.coco.coco_merge.coco_merge] app can be used to merge COCO annotation files.

Example usage:

``` bash
pyodi coco merge coco_1.json coco_2.json output.json
```

This app merges COCO annotation files by replacing original image and annotations ids with new ones
and adding all existent categories.

---

# API REFERENCE
"""  # noqa: E501
import json
from typing import Any, Dict

from loguru import logger


@logger.catch
def coco_merge(input_extend: str, input_add: str, output_file: str,) -> str:
    """Merge COCO annotation files.

    Args:
        input_extend: Path to input file to be extended.
        input_add: Path to input file to be added.
        output_file : Path to output file with merged annotations.
    """
    with open(input_extend, "r") as f:
        data_extend = json.load(f)
    with open(input_add, "r") as f:
        data_add = json.load(f)

    output: Dict[str, Any] = {
        k: data_extend[k] for k in data_extend if k not in ("images", "annotations")
    }

    output["images"], output["annotations"] = [], []

    for i, data in enumerate([data_extend, data_add]):

        logger.info(
            "Input {}: {} images, {} annotations".format(
                i + 1, len(data["images"]), len(data["annotations"])
            )
        )

        cat_id_map = {}
        for new_cat in data["categories"]:
            new_id = None
            for output_cat in output["categories"]:
                if new_cat["name"] == output_cat["name"]:
                    new_id = output_cat["id"]
                    break

            if new_id is not None:
                cat_id_map[new_cat["id"]] = new_id
            else:
                new_cat_id = max(c["id"] for c in output["categories"]) + 1
                cat_id_map[new_cat["id"]] = new_cat_id
                new_cat["id"] = new_cat_id
                output["categories"].append(new_cat)

        img_id_map = {}
        for image in data["images"]:
            n_imgs = len(output["images"])
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs

            output["images"].append(image)

        for annotation in data["annotations"]:
            n_anns = len(output["annotations"])
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            output["annotations"].append(annotation)

    logger.info(
        "Result: {} images, {} annotations".format(
            len(output["images"]), len(output["annotations"])
        )
    )

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    return output_file
