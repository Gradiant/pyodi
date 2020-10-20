"""# Coco Merge App.

The [`pyodi coco`][pyodi.apps.coco_merge.coco_merge] app can be used to merge multiple COCO
annotation files.

Example usage:

``` bash
pyodi coco merge output_file_path coco_1.json coco_2.json
```

This app merges COCO annotation files by replacing original image and annotations ids with new ones
and adding all existent categories.

It is possible to concatenate a different base path for images in each file by using 'base_imgs_folders'
argument. A simple example could be:

``` bash
pyodi coco merge output_file_path coco_1.json coco_2.json --base-imgs-folders path/coco_images_1 --base-imgs-folders path/coco_images_2
```

```

# API REFERENCE
"""  # noqa: E501
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import typer
from loguru import logger

app = typer.Typer()


@logger.catch
@app.command()
def coco_merge(
    output_file: str, annotation_files: List[str], base_imgs_folders: List[str] = None
) -> str:
    """Merge COCO annotation files and concat correspondent base_img_folder to image paths.

    Args:
        output_file : Path to output file with merged annotations
        annotation_files: List of paths of COCO annotation files to be merged
        base_imgs_folders: Base image folder path to concatenate to each annotation
            file images path. If None, annotations maintain its filename. Defaults to None
    """
    n_imgs, n_anns = 0, 0
    result: Dict[str, Any] = defaultdict()

    if base_imgs_folders is None:
        base_imgs_folders = ["" for i in range(len(annotation_files))]

    if len(annotation_files) != len(base_imgs_folders):
        raise ValueError("Base imgs folders must have same length as annotation files")

    for i, (annotation_file, base_img_folder) in enumerate(
        zip(annotation_files, base_imgs_folders)
    ):
        data = json.load(open(annotation_file))
        logger.info(
            "Processing {}: {} images, {} annotations".format(
                Path(annotation_file).name,
                len(data["images"]),
                len(data["annotations"]),
            )
        )

        if not i:
            result = deepcopy(data)
            result["images"], result["annotations"] = [], []
        else:
            cat_id_map = {}  # cat_id_map = {old_id: new_id}
            for data_cat in data["categories"]:
                new_id = None
                for result_cat in result["categories"]:
                    if data_cat["name"] == result_cat["name"]:
                        new_id = result_cat["id"]
                        break

                if new_id is not None:
                    cat_id_map[data_cat["id"]] = new_id
                else:
                    cat_id_map[data_cat["id"]] = (
                        max(c["id"] for c in result["categories"]) + 1
                    )
                    data_cat["id"] = max(c["id"] for c in result["categories"]) + 1
                    result["categories"].append(data_cat)

        img_id_map = {}  # img_id_map = {old_id: new_id}
        for image in data["images"]:
            filename = Path(base_img_folder) / image["file_name"]
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs
            image["file_name"] = str(filename)

            n_imgs += 1
            result["images"].append(image)

            if not (filename).is_file():
                logger.error(f"{filename} not found")

        for annotation in data["annotations"]:
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            n_anns += 1
            result["annotations"].append(annotation)

    logger.info(
        "Result {}: {} images, {} annotations".format(
            Path(output_file).name, len(result["images"]), len(result["annotations"])
        )
    )
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    return output_file
