import json
from typing import TextIO

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


def coco_ground_truth_to_df(
    ground_truth_file: str, max_images: int = 200000
) -> pd.DataFrame:
    """Load and transforms COCO ground truth data to pd.DataFrame object.

    Args:
        ground_truth_file: Path of ground truth file.
        max_images: Maximum number of images to process.

    Returns:
        pd.DataFrame with df_annotations keys and image sizes.

    """
    logger.info("Loading Ground Truth File")
    with open(ground_truth_file) as gt:
        coco_ground_truth = json.load(gt)

    if len(coco_ground_truth["images"]) > max_images:
        logger.warning(
            f"Number of images {len(coco_ground_truth['images'])} exceeds maximum: "
            f"{max_images}.\nAll the exceeding images will be ignored."
        )

    logger.info("Converting COCO Ground Truth to pd.DataFrame")
    df_images = pd.DataFrame(coco_ground_truth["images"][:max_images])[
        ["id", "file_name", "width", "height"]
    ]
    df_images = df_images.add_prefix("img_")

    df_annotations = pd.DataFrame(coco_ground_truth["annotations"])

    # Replace label with category name
    categories = {x["id"]: x["name"] for x in coco_ground_truth["categories"]}
    df_annotations["category"] = df_annotations["category_id"].replace(categories)

    # Add bbox columns
    bbox_columns = ["col_left", "row_top", "width", "height"]
    df_annotations[bbox_columns] = pd.DataFrame(
        df_annotations.bbox.tolist(), index=df_annotations.index
    )

    # Filter columns by name
    column_names = ["image_id", "area", "id", "category"] + bbox_columns
    if "iscrowd" in df_annotations.columns:
        column_names.append("iscrowd")

    # Join with images
    df_annotations = df_annotations[column_names].join(
        df_images.set_index("img_id"), how="inner", on="image_id"
    )

    return df_annotations
