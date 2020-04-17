import json
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger
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

    df_annotations = defaultdict(list)
    for annotation in coco_ground_truth["annotations"]:
        if annotation["image_id"] not in image_id_to_name:
            # Annotation of one of the exceeding images
            continue
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

    df_annotations = pd.DataFrame(df_annotations)
    return df_images, df_annotations


def join_annotations_with_image_sizes(df_annotations, df_images):
    """Left join between annotations dataframe and images keeping only given keys

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


def scale_bbox_dimensions(df_annotations, input_size=(1280, 720)):
    """Resizes bboxes dimensions to model input size

    Parameters
    ----------
    df_annotations : pd.DataFrame
        DataFrame with COCO annotations
    input_size : tuple(int, int)
        Model input size
    """
    # todo: add option to keep aspect ratio
    df_annotations["scaled_col_centroid"] = np.ceil(
        df_annotations["col_centroid"] * input_size[0] / df_annotations["img_width"]
    )
    df_annotations["scaled_row_centroid"] = np.ceil(
        df_annotations["row_centroid"] * input_size[1] / df_annotations["img_height"]
    )
    df_annotations["scaled_width"] = np.ceil(
        df_annotations["width"] * input_size[0] / df_annotations["img_width"]
    )
    df_annotations["scaled_height"] = np.ceil(
        df_annotations["height"] * input_size[1] / df_annotations["img_height"]
    )
    return df_annotations


def get_bbox_matrix(df_annotations, prefix=None):
    """Returns array with bbox coordinates

    Parameters
    ----------
    df_annotations : pd.DataFrame
        DataFrame with COCO annotations
    prefix : str
        Prefix to apply to column names, use for scaled data

    Returns
    -------
    np.array
        Array with dimension N x 4 with bbox coordinates
    """

    columns = ["col_centroid", "row_centroid", "width", "height"]
    if prefix:
        columns = [f"{prefix}_{col}" for col in columns]
    return df_annotations[columns].to_numpy()


def get_area_and_ratio(df, prefix=None):
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
    columns = ["width", "height"]
    if prefix:
        columns = [f"{prefix}_{col}" for col in columns]

    df[f"{prefix}_area"] = (
        df[columns[0]] * df[columns[1]]
    )
    df[f"{prefix}_ratio"] = (
        df[columns[1]] / df[columns[0]]
    )

    return df
