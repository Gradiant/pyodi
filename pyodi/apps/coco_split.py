"""# Coco Split App.

The [`pyodi coco`][pyodi.apps.coco_split.coco_split] app can be used to split COCO
annotation files in train and val annotations files.

There are two modes: 'random' or 'property'. The 'random' mode splits randomly the COCO file, while
the 'property' mode allows to customize the split operation based in the properties of the COCO
annotations file.

Example usage:

``` bash
pyodi coco split coco.json --mode random --output-filename random_coco_split --val-percentage 0.1
```

``` bash
pyodi coco split coco.json --mode property --output-filename property_coco_split --split-config-file config.json
```
"""  # noqa: E501
import json
import re
from copy import copy
from pathlib import Path
from typing import Any, List

import numpy as np
import typer
from loguru import logger

from ..coco.utils import divide_filename

app = typer.Typer()


def property_split(
    annotations_file: str,
    split_config_file: str,
    output_filename: str,
    show_summary: bool = True,
) -> List[str]:
    """Split the annotations file in training and validation subsets by properties.

    Args:
        annotations_file: Path to annotations file.
        split_config_file: Path to configuration file.
        output_filename: Output filename.
        show_summary: Whether to show some information about the results. Defaults to True.

    Returns:
        Output filenames.

    """
    split_config = json.load(open(split_config_file))
    annotations = json.load(open(annotations_file))
    train_images, val_images = [], []
    train_annotations, val_annotations = [], []
    train_img_id, val_img_id = 0, 0
    split_dict = dict()
    checked = dict()

    for i in range(len(annotations["images"])):
        img = annotations["images"][i]
        val_flag = False
        discard_flag = False

        for property_name, properties_to_match in split_config.items():
            if re.match(properties_to_match.get("filename", ""), img["file_name"]):

                if property_name == "discard":
                    discard_flag = True

                else:
                    val_flag = True

                if show_summary:
                    if property_name not in checked:
                        checked[property_name] = {
                            "videos": set([divide_filename(img["file_name"])[0]]),
                            "frames": 1,
                        }
                    else:
                        checked[property_name]["videos"].add(  # type: ignore
                            divide_filename(img["file_name"])[0]
                        )
                        checked[property_name]["frames"] += 1  # type: ignore

                break

        if not discard_flag:
            if val_flag:
                split_dict[img["id"]] = {"val": True, "new_idx": val_img_id}
                img["id"] = val_img_id
                val_images.append(img)
                val_img_id += 1
            else:
                split_dict[img["id"]] = {"val": False, "new_idx": train_img_id}
                img["id"] = train_img_id
                train_images.append(img)
                train_img_id += 1

    n_train_anns, n_val_anns = 0, 0
    for annotation in annotations["annotations"]:
        ann_data = split_dict.get(annotation["image_id"], None)

        if ann_data is None:
            continue
        else:
            annotation["image_id"] = ann_data["new_idx"]
            if ann_data["val"]:
                annotation["id"] = n_val_anns
                val_annotations.append(annotation)
                n_val_anns += 1
            else:
                annotation["id"] = n_train_anns
                train_annotations.append(annotation)
                n_train_anns += 1

    train_split = {
        "images": train_images,
        "annotations": train_annotations,
        "info": annotations["info"],
        "licenses": annotations["licenses"],
        "categories": annotations["categories"],
    }

    val_split = {
        "images": val_images,
        "annotations": val_annotations,
        "info": annotations["info"],
        "licenses": annotations["licenses"],
        "categories": annotations["categories"],
    }

    if show_summary:
        logger.info(f"Validation summary for {Path(annotations_file).stem}")

        for k, summ in checked.items():
            logger.info(f"  {k}: {len(summ['videos'])} videos, {summ['frames']} frames")  # type: ignore

        logger.info(f"Validation -> Images: {val_img_id}   Annotations: {n_val_anns}")
        logger.info(f"Train -> Images: {train_img_id}   Annotations: {n_train_anns}")

    logger.info("Saving splits to file")
    output_files = []
    for split_type, split in zip(["train", "val"], [train_split, val_split]):
        output_files.append(output_filename + f"_{split_type}.json")
        with open(output_files[-1], "w") as f:
            json.dump(split, f, indent=2)

    return output_files


def random_split(
    annotations_file: str,
    output_filename: str,
    val_percentage: float = 0.25,
    seed: int = 47,
    show_summary: bool = True,
) -> List[str]:
    """Split the annotations file in training and validation subsets randomly.

    Args:
        annotations_file: Path to annotations file.
        output_filename: Output filename.
        val_percentage: Percentage of validation images. Defaults to 0.25.
        seed: Seed for the random generator. Defaults to 47.
        show_summary: Whether to show some information about the results. Defaults to True.

    Returns:
        Output filenames.

    """
    annotations = json.load(open(annotations_file))
    train_images, val_images, val_ids = [], [], []

    np.random.seed(seed)
    rand_values = np.random.rand(len(annotations["images"]))

    logger.info("Gathering images...")
    for i, image in enumerate(annotations["images"]):

        if rand_values[i] < val_percentage:
            val_images.append(copy(image))
            val_ids.append(image["id"])
        else:
            train_images.append(copy(image))

    train_annotations, val_annotations = [], []

    logger.info("Gathering annotations...")
    for annotation in annotations["annotations"]:

        if annotation["image_id"] in val_ids:
            val_annotations.append(copy(annotation))
        else:
            train_annotations.append(copy(annotation))

    train_split = {
        "images": train_images,
        "annotations": train_annotations,
        "info": annotations["info"],
        "licenses": annotations["licenses"],
        "categories": annotations["categories"],
    }

    val_split = {
        "images": val_images,
        "annotations": val_annotations,
        "info": annotations["info"],
        "licenses": annotations["licenses"],
        "categories": annotations["categories"],
    }

    if show_summary:
        summary = [
            f"\nValidation summary for {Path(annotations_file).stem}",
            "-> IMAGES",
            f"\t-> Train: {len(train_images)}/{len(annotations['images'])}",
            f"\t-> Val: {len(val_images)}/{len(annotations['images'])}",
            "-> ANNOTATIONS",
            f"\t-> Train: {len(train_annotations)}/{len(annotations['annotations'])}",
            f"\t-> Val: {len(val_annotations)}/{len(annotations['annotations'])}",
        ]
        logger.info("\n".join(summary))

    logger.info("Saving splits to file...")
    output_files = []
    for split_type, split in zip(["train", "val"], [train_split, val_split]):
        output_files.append(output_filename + f"_{split_type}.json")
        with open(output_files[-1], "w") as f:
            json.dump(split, f, indent=2)

    return output_files


@logger.catch
@app.command()
def coco_split(annotations_file: str, mode: str = "random", **kwargs: Any) -> List[str]:
    """Split the annotations file in training and validation subsets.

    Args:
        annotations_file: Path to annotations file.
        mode: Mode used to split.

    Returns:
        Output filenames.

    """
    if mode == "random":
        return random_split(annotations_file, **kwargs)
    elif mode == "property":
        return property_split(annotations_file, **kwargs)
    else:
        raise ValueError(f"Mode {mode} not supported")
