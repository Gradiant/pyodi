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


@logger.catch  # noqa: C901
@app.command()
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
    split_config = json.load(open(split_config_file))

    # Transform split_config from human readable format to a more code efficient format
    for section in split_config:
        for prop, prop_value in split_config[section].items():
            if isinstance(prop_value, dict):
                split_config[section][prop] = "|".join(prop_value.values())

    annotations = json.load(open(annotations_file))
    train_images, val_images = [], []
    train_annotations, val_annotations = [], []
    n_train_imgs, n_val_imgs = 0, 0
    total_imgs, total_anns = 0, 0
    split_dict = dict()

    for i in range(len(annotations["images"])):
        img = annotations["images"][i]
        val_flag = False
        discard_flag = False
        matched_flag = False
        total_imgs += 1

        for section in split_config:
            for property_name, property_match in split_config[section].items():
                if re.match(property_match, img[property_name]):
                    matched_flag = True

                    if section == "discard":
                        discard_flag = True
                    else:
                        val_flag = True
                    break

            if matched_flag:
                break

        if not discard_flag:

            if val_flag:
                split_dict[img["id"]] = {"val": True, "new_idx": n_val_imgs}
                img["id"] = n_val_imgs
                val_images.append(img)
                n_val_imgs += 1

            else:
                split_dict[img["id"]] = {"val": False, "new_idx": n_train_imgs}
                img["id"] = n_train_imgs
                train_images.append(img)
                n_train_imgs += 1

    n_train_anns, n_val_anns = 0, 0
    for annotation in annotations["annotations"]:
        total_anns += 1
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

    logger.info("Saving splits to file")
    output_files = []
    for split_type, split in zip(["train", "val"], [train_split, val_split]):
        output_files.append(output_filename + f"_{split_type}.json")
        with open(output_files[-1], "w") as f:
            json.dump(split, f, indent=2)

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
            f"\nSummary for {Path(annotations_file).name}",
            "-> TRAIN",
            f"Number of frames: {len(train_images)}/{len(annotations['images'])}",
            f"Number of annotations: {len(train_annotations)}/{len(annotations['annotations'])}\n",
            "-> VAL",
            f"Number of frames: {len(val_images)}/{len(annotations['images'])}",
            f"Number of annotations: {len(val_annotations)}/{len(annotations['annotations'])}",
        ]
        logger.success("\n".join(summary))

    logger.info("Saving splits to file...")
    output_files = []
    for split_type, split in zip(["train", "val"], [train_split, val_split]):
        output_files.append(output_filename + f"_{split_type}.json")
        with open(output_files[-1], "w") as f:
            json.dump(split, f, indent=2)

    return output_files
