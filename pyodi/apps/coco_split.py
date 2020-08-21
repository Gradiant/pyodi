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
pyodi coco split coco.json --mode property --output-filename property_coco_split --split-config-file split_config.json
```

The split config file is a json file that has 2 keys: 'discard' and 'val', both with dictionary values. The keys of the
dictionaries will be the properties of the images that we want to match, and the values can be either the regex string to
match or, for human readability, a dictionary with keys (you can choose whatever you want) and values (the regex string).

Split config example:
``` python
{
    "discard": {
        "file_name": "people_video|crowd_video|multiple_people_video",
        "source": "Youtube People Dataset|Bad Dataset"
    },
    "val": {
        "file_name": {
            "My Val Ground Vehicle Dataset": "val_car_video|val_bus_video|val_moto_video|val_bike_video",
            "My Val Flying Vehicle Dataset": "val_plane_video|val_drone_video|val_helicopter_video"
        }
        "source": "Val Dataset"
    }
}
```
"""  # noqa: E501
import json
import re
from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import typer
from loguru import logger

app = typer.Typer()


def property_split(  # noqa: C901
    annotations_file: str,
    split_config_file: str,
    output_filename: str,
    show_summary: bool = True,
    get_video: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """Split the annotations file in training and validation subsets by properties.

    Args:
        annotations_file: Path to annotations file.
        split_config_file: Path to configuration file.
        output_filename: Output filename.
        show_summary: Whether to show some information about the results. Defaults to True.
        get_video: Function that returns video name from filename. If None,
            there will not be information about videos in summary. Defaults to None.

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
    train_img_id, val_img_id = 0, 0
    split_dict = dict()
    properties = set()
    any_discard_flag = False

    for i in range(len(annotations["images"])):
        img = annotations["images"][i]
        val_flag = False
        discard_flag = False
        matched_flag = False

        for section in split_config:
            for property_name, property_match in split_config[section].items():
                properties.add(property_name)
                if re.match(property_match, img[property_name]):
                    matched_flag = True

                    if section == "discard":
                        discard_flag = True
                        any_discard_flag = True
                    else:
                        val_flag = True
                    break

            if matched_flag:
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
        summary_info = dict()
        for property_name in properties:
            if property_name == "file_name":
                if get_video is not None:
                    property_name = "video"
                    train_set, val_set, all_set = set(), set(), set()
                    [
                        train_set.add(get_video(img["file_name"]))  # type: ignore
                        for img in train_split["images"]
                    ]
                    [
                        val_set.add(get_video(img["file_name"]))  # type: ignore
                        for img in val_split["images"]
                    ]
                    [
                        all_set.add(get_video(img["file_name"]))  # type: ignore
                        for img in annotations["images"]
                    ]
                else:
                    continue
            else:
                train_set, val_set, all_set = set(), set(), set()
                [train_set.add(img[property_name]) for img in train_split["images"]]  # type: ignore
                [val_set.add(img[property_name]) for img in val_split["images"]]  # type: ignore
                [all_set.add(img[property_name]) for img in annotations["images"]]  # type: ignore

            summary_info[property_name] = {
                "train": train_set,
                "val": val_set,
                "discard": all_set - (train_set | val_set),
                "all": all_set,
            }

        summary = [f"\nSummary for {Path(annotations_file).name}:"]
        sections = ["train", "val"]
        splits: List[Optional[Dict[str, Any]]] = [train_split, val_split]
        if any_discard_flag:
            sections.append("discard")
            splits.append(None)

        total_imgs = len(annotations["images"])
        total_anns = len(annotations["annotations"])
        for section, split in zip(sections, splits):
            summary.append(f"-> {section.upper()}")

            if split:
                partial_imgs = len(split["images"])
                partial_anns = len(split["annotations"])
            else:
                partial_imgs = len(annotations["images"]) - (
                    len(train_split["images"]) + len(val_split["images"])
                )
                partial_anns = len(annotations["annotations"]) - (
                    len(train_split["annotations"]) + len(val_split["annotations"])
                )

            summary.append(f"Number of frames: {partial_imgs}/{total_imgs}")
            summary.append(f"Number of annotations: {partial_anns}/{total_anns}")

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

        logger.success("\n".join(summary))

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
