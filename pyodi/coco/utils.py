import json
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from numpy import ndarray
from pandas.core.frame import DataFrame
from pycocotools.coco import COCO


def load_coco_ground_truth_from_StringIO(string_io):
    coco_ground_truth = COCO()
    coco_ground_truth.dataset = json.load(string_io)
    coco_ground_truth.createIndex()
    return coco_ground_truth


def load_ground_truth_file(ground_truth_file):
    logger.info("Loading Ground Truth File")
    coco_ground_truth = json.load(open(ground_truth_file))
    return coco_ground_truth


def coco_ground_truth_to_dfs(coco_ground_truth, max_images=200000):
    logger.info("Converting COCO Ground Truth to DataFrame")
    df_images = defaultdict(list)
    categories = {x["id"]: x["name"] for x in coco_ground_truth["categories"]}
    image_id_to_name = {}
    if len(coco_ground_truth["images"]) > max_images:
        logger.warning(
            f"Number of images {len(coco_ground_truth['images'])} exceeds maximum: {max_images}.\n"
            "All the exceeding images will be ignored"
        )
    for image in coco_ground_truth["images"][:max_images]:
        for k, v in image.items():
            df_images[k].append(v)
        image_id_to_name[image["id"]] = image["file_name"]

    df_images = pd.DataFrame(df_images)

    df_images["ratio"] = df_images["height"] / df_images["width"]
    df_images["scale"] = np.sqrt(df_images["height"] * df_images["width"])

    image_id_to_count = {x: 0 for x in df_images["id"]}
    df_annotations = defaultdict(list)
    for annotation in coco_ground_truth["annotations"]:
        if annotation["image_id"] not in image_id_to_name:
            # Annotation of one of the exceeding images
            continue
        image_id_to_count[annotation["image_id"]] += 1
        df_annotations["file_name"].append(image_id_to_name[annotation["image_id"]])
        df_annotations["category"].append(categories[annotation["category_id"]])
        df_annotations["area"].append(annotation["area"])
        df_annotations["col_centroid"].append(
            int(annotation["bbox"][0] + (annotation["bbox"][2] // 2))
        )
        df_annotations["row_centroid"].append(
            int(annotation["bbox"][1] + (annotation["bbox"][3] // 2))
        )
        df_annotations["width"].append(int(annotation["bbox"][2]))
        df_annotations["height"].append(int(annotation["bbox"][3]))

    df_images["bounding_box_count"] = image_id_to_count.values()

    df_annotations = pd.DataFrame(df_annotations)

    return df_images, df_annotations


def join_annotations_with_image_sizes(df_annotations, df_images):
    """Left join between annotations dataframe and images keeping only df_annotations keys and image sizes

    Parameters
    ----------
    df_annotations : [type]
        [description]
    df_images : [type]
        [description]
    """
    column_names = list(df_annotations.columns) + ["img_width", "img_height"]
    df_images = df_images.add_prefix("img_")
    return df_annotations.join(df_images.set_index("img_file_name"), on="file_name")[
        column_names
    ]


def check_bbox_formats(*args) -> None:
    for arg in args:
        if not (arg in ["coco", "corners"]):
            raise ValueError(
                f"Invalid format {arg}, only coco and corners format are allowed"
            )


def scale_bbox_dimensions(
    df: DataFrame, input_size: Tuple[int, int] = (1280, 720), keep_ratio: bool = False,
) -> DataFrame:
    """Resizes bboxes dimensions to model input size

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with COCO annotations
    input_size : tuple(int, int)
        Model input size
    keep_ratio : bool
        Whether to keep the aspect ratio
    """
    if keep_ratio:
        scale_factor = pd.concat(
            [
                max(input_size) / df[["img_height", "img_width"]].max(1),
                min(input_size) / df[["img_height", "img_width"]].min(1),
            ],
            axis=1,
        ).min(1)
        w_scale = np.round(df["img_width"] * scale_factor) / df["img_width"]
        h_scale = np.round(df["img_height"] * scale_factor) / df["img_height"]
    else:
        w_scale = input_size[0] / df["img_width"]
        h_scale = input_size[1] / df["img_height"]

    df["scaled_col_centroid"] = np.ceil(df["col_centroid"] * w_scale)
    df["scaled_row_centroid"] = np.ceil(df["row_centroid"] * h_scale)
    df["scaled_width"] = np.ceil(df["width"] * w_scale)
    df["scaled_height"] = np.ceil(df["height"] * h_scale)

    return df


def get_scale_and_ratio(df: DataFrame, prefix: str = None) -> DataFrame:
    """Returns df with area and ratio per bbox measurements

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with COCO annotations
    prefix : str
        Prefix to apply to column names, use for scaled data

    Returns
    -------
    pd.DataFrame
        Dataframe with new columns [prefix_]area/ratio
    """

    columns = ["width", "height", "scale", "ratio"]

    if prefix:
        columns = [f"{prefix}_{col}" for col in columns]

    df[columns[2]] = np.sqrt(df[columns[0]] * df[columns[1]])
    df[columns[3]] = df[columns[1]] / df[columns[0]]

    return df


def corners_to_coco(bboxes):
    """Transforms bboxes array from corners format to coco

    Parameters
    ----------
    bboxes : np.array
        Array with dimension N x 4 with bbox coordinates in corner format [x_min, y_min, x_max, y_max]

    Returns
    -------
    np.array
        Array with dimension N x 4 with bbox coordinates in coco format [x_center, y_center, width, height]
    """
    dimensions = bboxes[..., 2:] - bboxes[..., :2]
    centers = bboxes[..., :2] + dimensions // 2
    bboxes = np.concatenate([centers, dimensions], axis=-1)
    return bboxes


def coco_to_corners(bboxes: ndarray) -> ndarray:
    """Transforms bboxes array from coco format to corners

    Parameters
    ----------
    bboxes : np.array
        Array with dimension N x 4 with bbox coordinates in coco format [x_center, y_center, width, height]

    Returns
    -------
    np.array
        Array with dimension N x 4 with bbox coordinates in corner format  [x_min, y_min, x_max, y_max]
    """
    mins = bboxes[..., :2] - bboxes[..., 2:] // 2
    maxs = mins + bboxes[..., 2:]
    bboxes = np.concatenate([mins, maxs], axis=-1)

    if (bboxes < 0).any():
        logger.warning("Clipping bboxes to min corner 0, found negative value")
        bboxes = np.clip(bboxes, 0, None)
    return bboxes


def get_bbox_column_names(bbox_format: str, prefix: Optional[str] = None) -> List[str]:
    """Returns predefined column names for each format. When bbox_format is 'coco' column names
     are ["col_centroid", "row_centroid", "width", "height"], when 'corners' ["col_left", "row_left", "col_right", "row_right"]

    Parameters
    ----------
    bbox_format : str
        Can be "coco" or "corners"
    prefix : str
        Prefix to apply to column names, use for scaled data

    Returns
    -------
    List
        Column names for specified bbox format
    """
    if bbox_format == "coco":
        columns = ["col_centroid", "row_centroid", "width", "height"]
    elif bbox_format == "corners":
        columns = ["col_left", "row_left", "col_right", "row_right"]
    else:
        raise ValueError(f"Invalid bbox format, {bbox_format} does not exist")

    if prefix:
        columns = [f"{prefix}_{col}" for col in columns]

    return columns


def get_bbox_array(
    df: DataFrame,
    prefix: Optional[str] = None,
    input_bbox_format: str = "coco",
    output_bbox_format: str = "coco",
) -> ndarray:
    """Returns array with bbox coordinates

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with COCO annotations
    prefix : str
        Prefix to apply to column names, use for scaled data
    bbox_format: str, optional
        Can be 'coco' or 'corners'. When 'coco' returned array is[x_center, y_center, width, height],
        when format 'corners' returned array [x_min, y_min, x_max, y_max]

    Returns
    -------
    np.array
        Array with dimension N x 4 with bbox coordinates
    """
    check_bbox_formats(input_bbox_format, output_bbox_format)

    columns = get_bbox_column_names(input_bbox_format, prefix=prefix)
    bboxes = df[columns].to_numpy()

    if input_bbox_format != output_bbox_format:
        convert = globals()[f"{input_bbox_format}_to_{output_bbox_format}"]
        bboxes = convert(bboxes)

    return bboxes


def get_df_from_bboxes(
    bboxes: ndarray,
    input_bbox_format: str = "coco",
    output_bbox_format: str = "corners",
) -> DataFrame:
    """Creates dataframe of annotations in coco format from array of bboxes

    Parameters
    ----------
    bboxes : np.array
        Array of bboxes of shape [n, 4]
    bbox_format: str, optional
        Can be 'coco' or 'corners'. When 'coco' input array follows [x_center, y_center, width, height],
        when format 'corners' input is [x_min, y_min, x_max, y_max]

    Returns
    -------
    pandas.DataFrame
    """

    check_bbox_formats(input_bbox_format, output_bbox_format)

    if input_bbox_format != output_bbox_format:
        convert = locals()[f"{input_bbox_format}_{output_bbox_format}"]
        bboxes = convert(bboxes)

    return pd.DataFrame(bboxes, columns=get_bbox_column_names(output_bbox_format))


def filter_zero_area_bboxes(df: DataFrame) -> DataFrame:
    """Filters those bboxes with height or width equal to zero

    Parameters
    ----------
    df : pd.DataFrame
            DataFrame with COCO annotations
    """
    cols = ["width", "height"]
    all_bboxes = len(df)
    df = df[(df[cols] > 0).all(axis=1)].reset_index()
    filtered_bboxes = len(df)

    n_filtered = all_bboxes - filtered_bboxes

    if n_filtered:
        logger.warning(
            f"A total of {n_filtered} bboxes have been filtered from your data for having area equal to zero."
        )

    return df
