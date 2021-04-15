"""# Coco Split App.

The [`pyodi coco split`][pyodi.apps.coco.coco_split] app can be used to split COCO
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
---

# API REFERENCE
"""  # noqa: E501
import json
import re
from copy import copy
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger


@logger.catch  # noqa: C901
def property_split(
    annotations_file: str, output_filename: str, split_config_file: str,
) -> List[str]:
    """Split the annotations file in training and validation subsets by properties.

    Args:
        annotations_file: Path to annotations file.
        output_filename: Output filename.
        split_config_file: Path to configuration file.

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

    return output_files


@logger.catch
def random_split(
    annotations_file: str,
    output_filename: str,
    val_percentage: float = 0.25,
    seed: int = 47,
) -> List[str]:
    """Split the annotations file in training and validation subsets randomly.

    Args:
        annotations_file: Path to annotations file.
        output_filename: Output filename.
        val_percentage: Percentage of validation images. Defaults to 0.25.
        seed: Seed for the random generator. Defaults to 47.

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

    logger.info("Saving splits to file...")
    output_files = []
    for split_type, split in zip(["train", "val"], [train_split, val_split]):
        output_files.append(output_filename + f"_{split_type}.json")
        with open(output_files[-1], "w") as f:
            json.dump(split, f, indent=2)

    return output_files
