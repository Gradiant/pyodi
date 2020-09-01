"""# Coco Split App.

The [`pyodi coco split`][pyodi.apps.coco_split] app can be used to split COCO
annotation files in train and val annotations files.

There are two modes: 'random' or 'property'. The 'random' mode splits randomly the COCO file, while
the 'property' mode allows to customize the split operation based in the properties of the COCO
annotations file.

Example usage:

``` bash
pyodi coco random-split ./coco.json ./random_coco_split --val-percentage 0.1
```

``` bash
pyodi coco property-split ./coco.json ./property_coco_split ./split_config.json
```

The split config file is a json file that has 2 keys: 'discard' and 'val', both with dictionary values. The keys of the
dictionaries will be the properties of the images that we want to match, and the values can be either the regex string to
match or, for human readability, a dictionary with keys (you can choose whatever you want) and values (the regex string).

Split config example:
``` python
{
    "discard": {
        "file_name": "people_video|crowd_video|multiple_people_video",
        "source": "Youtube People Dataset|Bad Dataset",
    },
    "val": {
        "file_name": {
            "My Val Ground Vehicle Dataset": "val_car_video|val_bus_video|val_moto_video|val_bike_video",
            "My Val Flying Vehicle Dataset": "val_plane_video|val_drone_video|val_helicopter_video",
        },
        "source": "Val Dataset",
    }
}
```
"""  # noqa: E501
import json
import re
from copy import copy
from pathlib import Path
from typing import List

import numpy as np
import typer
from loguru import logger

app = typer.Typer()


def get_summary(train_file: str, val_file: str, full_file: str) -> str:
    """Log summary of the split operation.

    Args:
        train_file: Train annotations file.
        val_file: Val annotations file.
        full_file: Full annotations file.

    Returns:
        A string with the summary.
    """
    logger.info("Computing summary...")
    train_data = json.load(open(train_file))
    val_data = json.load(open(val_file))
    full_data = json.load(open(full_file))

    properties = list(full_data["images"][0].keys())
    properties.remove("id")
    properties.remove("file_name")

    summary_info = {}
    for property_name in properties:
        train_set, val_set, all_set = set(), set(), set()
        for img in train_data["images"]:
            if property_name in img:
                train_set.add(str(img[property_name]))
        for img in val_data["images"]:
            if property_name in img:
                val_set.add(str(img[property_name]))
        for img in full_data["images"]:
            if property_name in img:
                all_set.add(str(img[property_name]))

        summary_info[property_name] = {
            "train": train_set,
            "val": val_set,
            "discard": all_set - (train_set | val_set),
            "all": all_set,
        }

    n_discard_imgs = len(full_data["images"]) - (
        len(train_data["images"]) + len(val_data["images"])
    )
    n_discard_anns = len(full_data["annotations"]) - (
        len(train_data["annotations"]) + len(val_data["annotations"])
    )

    summary = ["\n\nSUMMARY:\n"]
    if (n_discard_imgs + n_discard_anns) == 0:
        sections = ["train", "val"]
        splits = [train_data, val_data]
    else:
        sections = ["train", "val", "discard"]
        splits = [train_data, val_data, None]

    for section, split in zip(sections, splits):
        summary.append(f"-> {section.upper()}")

        if section == "discard":
            summary.append(
                f"Number of frames: {n_discard_imgs}/{len(full_data['images'])}"
            )
            summary.append(
                f"Number of annotations: {n_discard_anns}/{len(full_data['annotations'])}"
            )
        else:
            summary.append(
                f"Number of frames: {len(split['images'])}/{len(full_data['images'])}"
            )
            summary.append(
                f"Number of annotations: {len(split['annotations'])}/{len(full_data['annotations'])}"
            )

        for property_name in summary_info:
            property_set = summary_info[property_name][section]
            summary.append(
                "{} ({}/{}): {}".format(
                    property_name.capitalize(),
                    len(property_set),
                    len(summary_info[property_name]["all"]),
                    ", ".join(list(property_set)),
                )
            )
        summary.append("\n")

    return "\n".join(summary)


@logger.catch  # noqa: C901
@app.command()
def property_split(
    annotations_file: str,
    output_filename: str,
    split_config_file: str,
    show_summary: bool = True,
) -> List[str]:
    """Split the annotations file in training and validation subsets by properties.

    Args:
        annotations_file: Path to annotations file.
        output_filename: Output filename.
        split_config_file: Path to configuration file.
        show_summary: Whether to show some information about the results. Defaults to False.

    Returns:
        Output filenames.

    """
    logger.info("Loading files...")
    split_config = json.load(open(Path(split_config_file)))
    split_list = []

    # Transform split_config from human readable format to a more code efficient format
    for section in split_config:  # sections: val / discard
        for property_name, property_value in split_config[section].items():
            if isinstance(property_value, dict):
                property_value = "|".join(property_value.values())
            split_list.append(
                dict(
                    split=section,
                    property_name=property_name,
                    property_regex=property_value,
                )
            )

    data = json.load(open(annotations_file))

    train_images, val_images = [], []
    train_annotations, val_annotations = [], []

    n_train_imgs, n_val_imgs = 0, 0
    n_train_anns, n_val_anns = 0, 0

    old_to_new_train_ids = dict()
    old_to_new_val_ids = dict()

    logger.info("Gathering images...")
    for img in data["images"]:

        i = 0
        while i < len(split_list) and not re.match(
            split_list[i]["property_regex"], img[split_list[i]["property_name"]]
        ):
            i += 1

        if i < len(split_list):  # discard or val
            if split_list[i]["split"] == "val":
                old_to_new_val_ids[img["id"]] = n_val_imgs
                img["id"] = n_val_imgs
                val_images.append(img)
                n_val_imgs += 1
        else:  # train
            old_to_new_train_ids[img["id"]] = n_train_imgs
            img["id"] = n_train_imgs
            train_images.append(img)
            n_train_imgs += 1

    logger.info("Gathering annotations...")
    for ann in data["annotations"]:

        if ann["image_id"] in old_to_new_val_ids:
            ann["image_id"] = old_to_new_val_ids[ann["image_id"]]
            ann["id"] = n_val_anns
            val_annotations.append(ann)
            n_val_anns += 1
        elif ann["image_id"] in old_to_new_train_ids:
            ann["image_id"] = old_to_new_train_ids[ann["image_id"]]
            ann["id"] = n_train_anns
            train_annotations.append(ann)
            n_train_anns += 1

    logger.info("Spliting data...")
    train_split = {
        "images": train_images,
        "annotations": train_annotations,
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
    }
    val_split = {
        "images": val_images,
        "annotations": val_annotations,
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
    }

    logger.info("Writing splited files...")
    output_files = []
    for split_type, split in zip(["train", "val"], [train_split, val_split]):
        output_files.append(output_filename + f"_{split_type}.json")
        with open(output_files[-1], "w") as f:
            json.dump(split, f, indent=2)

    if show_summary:
        logger.info(get_summary(output_files[0], output_files[1], annotations_file))

    logger.success("Done!")

    return output_files


@logger.catch
@app.command()
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
    data = json.load(open(annotations_file))
    train_images, val_images, val_ids = [], [], []

    np.random.seed(seed)
    rand_values = np.random.rand(len(data["images"]))

    logger.info("Gathering images...")
    for i, image in enumerate(data["images"]):

        if rand_values[i] < val_percentage:
            val_images.append(copy(image))
            val_ids.append(image["id"])
        else:
            train_images.append(copy(image))

    train_annotations, val_annotations = [], []

    logger.info("Gathering annotations...")
    for annotation in data["annotations"]:

        if annotation["image_id"] in val_ids:
            val_annotations.append(copy(annotation))
        else:
            train_annotations.append(copy(annotation))

    train_split = {
        "images": train_images,
        "annotations": train_annotations,
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
    }

    val_split = {
        "images": val_images,
        "annotations": val_annotations,
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
    }

    if show_summary:
        summary = [
            f"\nSummary for {Path(annotations_file).name}",
            "-> TRAIN",
            f"Number of frames: {len(train_images)}/{len(data['images'])}",
            f"Number of annotations: {len(train_annotations)}/{len(data['annotations'])}\n",
            "-> VAL",
            f"Number of frames: {len(val_images)}/{len(data['images'])}",
            f"Number of annotations: {len(val_annotations)}/{len(data['annotations'])}",
        ]
        logger.success("\n".join(summary))

    logger.info("Saving splits to file...")
    output_files = []
    for split_type, split in zip(["train", "val"], [train_split, val_split]):
        output_files.append(output_filename + f"_{split_type}.json")
        with open(output_files[-1], "w") as f:
            json.dump(split, f, indent=2)

    return output_files
