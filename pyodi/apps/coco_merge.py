"""# Coco Merge App.

The [`pyodi coco`][pyodi.apps.coco_merge.coco_merge] app can be used to merge COCO annotation files.

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
from collections import defaultdict
from typing import Any, Dict

import typer
from loguru import logger

app = typer.Typer()


@logger.catch
@app.command()
def coco_merge(input_file_1: str, input_file_2: str, output_file: str,) -> str:
    """Merge COCO annotation files.

    Args:
        input_file_1: Path to input file to be extended.
        input_file_2: Path to input file to be added.
        output_file : Path to output file with merged annotations.
    """
    n_imgs, n_anns = 0, 0
    output: Dict[str, Any] = defaultdict()

    with open(input_file_1, "r") as f:
        data_1 = json.load(f)
    with open(input_file_2, "r") as f:
        data_2 = json.load(f)

    output = {k: data_1[k] for k in data_1 if k not in ("images", "annotations")}

    output["images"], output["annotations"] = [], []

    for i, data in enumerate([data_1, data_2]):

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
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs

            n_imgs += 1
            output["images"].append(image)

        for annotation in data["annotations"]:
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            n_anns += 1
            output["annotations"].append(annotation)

    logger.info(
        "Result: {} images, {} annotations".format(
            len(output["images"]), len(output["annotations"])
        )
    )

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    return output_file
