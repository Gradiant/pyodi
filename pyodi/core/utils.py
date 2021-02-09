import json
from collections import defaultdict
from typing import Any, Dict, TextIO, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pycocotools.coco import COCO


def load_coco_ground_truth_from_StringIO(string_io: TextIO) -> COCO:
    """Returns COCO object from StringIO.

    Args:
        string_io: IO stream in text mode.

    Returns:
        COCO object.

    """
    coco_ground_truth = COCO()
    coco_ground_truth.dataset = json.load(string_io)
    coco_ground_truth.createIndex()
    return coco_ground_truth


def load_ground_truth_file(ground_truth_file: str) -> Dict:
    """Loads ground truth file.

    Args:
        ground_truth_file: Path of ground truth file.

    Returns:
        Dictionary with the ground truth data.

    """
    logger.info("Loading Ground Truth File")
    coco_ground_truth = json.load(open(ground_truth_file))
    return coco_ground_truth


def coco_ground_truth_to_dfs(
    coco_ground_truth: Dict, max_images: int = 200000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Transforms COCO ground truth data to pd.DataFrame objects.

    Args:
        coco_ground_truth: COCO ground truth data.
        max_images: Maximum number of images to process.

    Returns:
        Images and annotations pd.DataFrames.

    """
    logger.info("Converting COCO Ground Truth to pd.DataFrame")
    dict_images: Dict[str, Any] = defaultdict(list)
    categories = {x["id"]: x["name"] for x in coco_ground_truth["categories"]}
    image_id_to_name = {}
    if len(coco_ground_truth["images"]) > max_images:
        logger.warning(
            f"Number of images {len(coco_ground_truth['images'])} exceeds maximum: "
            f"{max_images}.\nAll the exceeding images will be ignored."
        )
    for image in coco_ground_truth["images"][:max_images]:
        for k, v in image.items():
            dict_images[k].append(v)
        image_id_to_name[image["id"]] = image["file_name"]

    df_images = pd.DataFrame(dict_images)

    df_images["ratio"] = df_images["height"] / df_images["width"]
    df_images["scale"] = np.sqrt(df_images["height"] * df_images["width"])

    image_id_to_count = {x: 0 for x in df_images["id"]}
    dict_annotations: Dict[str, Any] = defaultdict(list)
    for annotation in coco_ground_truth["annotations"]:
        if annotation["image_id"] not in image_id_to_name:
            # Annotation of one of the exceeding images
            continue
        image_id_to_count[annotation["image_id"]] += 1
        dict_annotations["file_name"].append(image_id_to_name[annotation["image_id"]])
        dict_annotations["category"].append(categories[annotation["category_id"]])
        dict_annotations["area"].append(annotation["area"])
        dict_annotations["col_left"].append(int(annotation["bbox"][0]))
        dict_annotations["row_top"].append(int(annotation["bbox"][1]))
        dict_annotations["width"].append(int(annotation["bbox"][2]))
        dict_annotations["height"].append(int(annotation["bbox"][3]))

    df_images["bounding_box_count"] = image_id_to_count.values()

    df_annotations = pd.DataFrame(dict_annotations)

    return df_images, df_annotations


def join_annotations_with_image_sizes(
    df_annotations: pd.DataFrame, df_images: pd.DataFrame
) -> pd.DataFrame:
    """Left join between annotations pd.DataFrame and images.

    It only keeps df_annotations keys and image sizes.

    Args:
        df_annotations: pd.DataFrame with COCO annotations.
        df_images: pd.DataFrame with images.

    Returns:
        pd.DataFrame with df_annotations keys and image sizes.

    """
    column_names = list(df_annotations.columns) + ["img_width", "img_height"]
    df_images = df_images.add_prefix("img_")
    return df_annotations.join(df_images.set_index("img_file_name"), on="file_name")[
        column_names
    ]
