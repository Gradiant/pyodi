"""# Coco Split App.

**in progress...**
"""  # noqa: E501
import json
from copy import copy, deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import typer
from loguru import logger

app = typer.Typer()

# ToDo: create functions for reusing code


def load_data(file: str) -> Tuple[Dict, Dict, Dict]:
    logger.info("Loading data...")
    with open(file) as f:
        data = json.load(f)

    train_data = deepcopy(data)
    val_data = deepcopy(data)

    return train_data, val_data, data


def write_summary(
    train_imgs: Dict, val_imgs: Dict, train_anns: Dict, val_anns: Dict
) -> None:
    pass


def random_split(
    annotations_file: str, val_percentage: float = 0.25, seed: int = 47
) -> None:

    train_data, val_data, annotations_data = load_data(annotations_file)

    train_images = []
    val_images = []
    val_ids = []
    summary = {}

    np.random.seed(seed)
    rand_values = np.random.rand(len(annotations_data["images"]))

    logger.info("Gathering images...")
    for i, image in enumerate(annotations_data["images"]):

        if rand_values[i] < val_percentage:
            val_images.append(copy(image))
            val_ids.append(image["id"])
        else:
            train_images.append(copy(image))

    # print(f"Train: {len(train_images)}/{len(train_images)+len(val_images)}, Val: {len(val_images)}/{len(train_images)+len(val_images)}")

    summary["train_imgs"] = len(train_images)
    summary["val_imgs"] = len(val_images)
    summary["total_imgs"] = summary["train_imgs"] + summary["val_images"]

    train_annotations = []
    val_annotations = []

    logger.info("Gathering annotations...")
    for annotation in annotations_data["annotations"]:

        if annotation["image_id"] in val_ids:
            val_annotations.append(copy(annotation))
        else:
            train_annotations.append(copy(annotation))

    # print(f"Train: {len(train_annotations)}/{len(train_annotations)+len(val_annotations)}, Val: {len(val_annotations)}/{len(train_annotations)+len(val_annotations)}")

    summary["train_anns"] = len(train_annotations)
    summary["val_anns"] = len(val_annotations)
    summary["total_anns"] = summary["train_anns"] + summary["val_anns"]

    print("Updating train and val annotations...", end=" ")
    train_data["images"] = train_images
    train_data["annotations"] = train_annotations
    val_data["images"] = val_images
    val_data["annotations"] = val_annotations
    print("Done!")

    print("Writing train and val annotations files...", end=" ")
    output_path = Path(annotations_file).parent
    annotations_name = Path(annotations_file).stem
    train_file = output_path / (annotations_name + "_train.json")
    val_file = output_path / (annotations_name + "_val.json")

    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(val_file, "w") as f:
        json.dump(val_data, f, indent=2)
    print("Done!")


def check_match_prefix(string: str, prefixes: List[str]) -> bool:
    if len(prefixes) > 0:
        for prefix in prefixes:
            if prefix in string.lower():
                return True

    return False


def split_fixed_coco_annotations(
    annotations_file: str,
    val_list: list = None,
    split_json: dict = None,
    val_keys: list = None,
    discard: str = None,
):
    """
    Divide the annotations file in two train and val annotations files.

    Parameters
    ----------
    annotations_file: str
        Path to COCO annotations file
    val_list: list
        If the filename of a given image matches with any substring of this list, it will be
        included in the val annotations file, otherwise it will be included in the train one.
        If None, val_keys and split_json will be used.
    split_json: dict
        Dictionary where the values are the substrings used to include filenames as val.
    val_keys: list
        List of keys from split_json of substrings that will be used as val.
        Must be not None if val_list is None.
    discard: str
        Substrings that will be discarded.
    """

    print("Loading data...", end=" ", flush=True)
    with open(annotations_file) as f:
        annotations_data = json.load(f)

    train_data = deepcopy(annotations_data)
    val_data = deepcopy(annotations_data)
    print("Done!")

    print("Gathering prefixes...", end=" ", flush=True)
    discard_prefixes = []
    if discard is not None:
        discard_prefixes = list(map(lambda x: x.lower(), discard.split("|")))

    val_prefixes = []
    if val_list is not None:
        for substring in val_list:
            val_prefixes += list(map(lambda x: x.lower(), substring.split("|")))
    elif val_keys is not None:
        split_data = split_json
        for k in val_keys:
            val_prefixes += list(map(lambda x: x.lower(), split_data[k].split("|")))
    else:
        raise ValueError("val_list and val_keys cannot be both None.")

    print("Done!")

    train_images = []
    val_images = []
    val_ids = []
    discard_ids = []

    print("Gathering images...", end=" ", flush=True)
    for image in annotations_data["images"]:

        filename = image["file_name"]

        if check_match_prefix(filename, discard_prefixes):
            discard_ids.append(image["id"])
            continue

        if check_match_prefix(filename, val_prefixes):
            val_images.append(copy(image))
            val_ids.append(image["id"])
        else:
            train_images.append(copy(image))

    print("Done!")

    train_annotations = []
    val_annotations = []

    print("Gathering annotations...", end=" ", flush=True)
    for annotation in annotations_data["annotations"]:

        if annotation["image_id"] in discard_ids:
            continue
        elif annotation["image_id"] in val_ids:
            val_annotations.append(copy(annotation))
        else:
            train_annotations.append(copy(annotation))

    print("Done!")

    print("Updating train and val annotations...", end=" ", flush=True)
    train_data["images"] = train_images
    train_data["annotations"] = train_annotations
    val_data["images"] = val_images
    val_data["annotations"] = val_annotations
    print("Done!")

    print("Writing train and val annotations files...", end=" ", flush=True)
    output_path = Path(annotations_file).parent
    annotations_name = Path(annotations_file).stem
    train_file = output_path / (annotations_name + "_train.json")
    val_file = output_path / (annotations_name + "_val.json")

    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(val_file, "w") as f:
        json.dump(val_data, f, indent=2)
    print("Done!")


def split_coco_annotations_by_source(annotations_file: str, source_list: list):
    print("Loading data...", end=" ", flush=True)
    with open(annotations_file) as f:
        annotations_data = json.load(f)

    train_data = deepcopy(annotations_data)
    val_data = deepcopy(annotations_data)
    print("Done!")

    train_images = []
    val_images = []
    val_ids = []
    discard_ids = []

    print("Gathering images...", end=" ", flush=True)
    for image in annotations_data["images"]:

        source = image["source"]

        if check_match_prefix(source, source_list):
            val_images.append(copy(image))
            val_ids.append(image["id"])
        else:
            train_images.append(copy(image))

    print("Done!")

    train_annotations = []
    val_annotations = []

    print("Gathering annotations...", end=" ", flush=True)
    for annotation in annotations_data["annotations"]:

        if annotation["image_id"] in discard_ids:
            continue
        elif annotation["image_id"] in val_ids:
            val_annotations.append(copy(annotation))
        else:
            train_annotations.append(copy(annotation))

    print("Done!")

    print("Updating train and val annotations...", end=" ", flush=True)
    train_data["images"] = train_images
    train_data["annotations"] = train_annotations
    val_data["images"] = val_images
    val_data["annotations"] = val_annotations
    print("Done!")

    print("Writing train and val annotations files...", end=" ", flush=True)
    output_path = Path(annotations_file).parent
    annotations_name = Path(annotations_file).stem
    train_file = output_path / (annotations_name + "_train.json")
    val_file = output_path / (annotations_name + "_val.json")

    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(val_file, "w") as f:
        json.dump(val_data, f, indent=2)
    print("Done!")


@logger.catch
@app.command()
def coco_split(annotations_file, mode="random", **kwargs):
    if mode == "random":
        random_split(annotations_file, **kwargs)
    elif mode == "fixed":
        split_fixed_coco_annotations(annotations_file, **kwargs)
    elif mode == "source":
        split_coco_annotations_by_source(annotations_file, **kwargs)
    else:
        raise ValueError(f"Mode {mode} not supported")
