import json
from collections import defaultdict
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
    start_image_id, start_ann_id = 0, 0
    result: Dict[str, Any] = defaultdict()

    if base_imgs_folders is None:
        base_imgs_folders = ["" for i in range(len(annotation_files))]

    if len(annotation_files) != len(base_imgs_folders):
        raise ValueError("Base imgs folders must have same length as annotation files")

    for i, (annotation_file, base_img_folder) in enumerate(
        zip(annotation_files, base_imgs_folders)
    ):
        data = json.load(open(annotation_file))

        for image in data["images"]:
            filename = Path(base_img_folder) / image["file_name"]
            image["id"] += start_image_id
            image["file_name"] = str(filename)

            if not (filename).is_file():
                logger.error(f"{filename} not found")

        if not i:
            result = data

        else:

            rename_categories = dict()

            for new_category in data["categories"]:
                new_id = None
                for actual_category in result["categories"]:
                    if new_category["name"] == actual_category["name"]:
                        new_id = actual_category["id"]
                        break

                if new_id is not None:
                    rename_categories[new_category["id"]] = new_id
                else:
                    rename_categories[new_category["id"]] = (
                        len(result["categories"]) + 1
                    )
                    new_category["id"] = len(result["categories"]) + 1
                    result["categories"].append(new_category)

            for ann in data["annotations"]:
                ann["id"] += start_ann_id
                ann["image_id"] += start_image_id
                ann["category_id"] = rename_categories[ann["category_id"]]

            result["annotations"] += data["annotations"]
            result["images"] += data["images"]

        start_ann_id = len(result["annotations"])
        start_image_id = len(result["images"])

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    return output_file
