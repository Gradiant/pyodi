import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from numpy import ndarray
from pandas.core.frame import DataFrame
from pycocotools.coco import COCO


def load_coco_ground_truth_from_StringIO(string_io: TextIO) -> COCO:
    """Return COCO object from StringIO.

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
    """Load ground truth file.

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
) -> Tuple[DataFrame, DataFrame]:
    """Transform COCO ground truth data to DataFrame objects.

    Args:
        coco_ground_truth: COCO ground truth data.
        max_images: Maximum number of images to process.

    Returns:
        Images and annotations DataFrames.

    """
    logger.info("Converting COCO Ground Truth to DataFrame")
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
    df_annotations: DataFrame, df_images: DataFrame
) -> DataFrame:
    """Left join between annotations dataframe and images.

    It only keeps df_annotations keys and image sizes.

    Args:
        df_annotations: DataFrame with COCO annotations.
        df_images: DataFrame with images.

    Returns:
        DataFrame with df_annotations keys and image sizes.

    """
    column_names = list(df_annotations.columns) + ["img_width", "img_height"]
    df_images = df_images.add_prefix("img_")
    return df_annotations.join(df_images.set_index("img_file_name"), on="file_name")[
        column_names
    ]


def check_bbox_formats(*args: Any) -> None:
    """Check if bounding boxes are in a valid format."""
    for arg in args:
        if not (arg in ["coco", "corners"]):
            raise ValueError(
                f"Invalid format {arg}, only coco and corners format are allowed"
            )


def scale_bbox_dimensions(
    df: DataFrame, input_size: Tuple[int, int] = (1280, 720), keep_ratio: bool = False,
) -> DataFrame:
    """Resizes bboxes dimensions to model input size.

    Args:
        df: DataFrame with COCO annotations.
        input_size: Model input size. Defaults to (1280, 720).
        keep_ratio: Whether to keep the aspect ratio or not. Defaults to False.

    Returns:
        DataFrame with COCO annotations and scaled image sizes.

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

    df["scaled_col_left"] = np.ceil(df["col_left"] * w_scale)
    df["scaled_row_top"] = np.ceil(df["row_top"] * h_scale)
    df["scaled_width"] = np.ceil(df["width"] * w_scale)
    df["scaled_height"] = np.ceil(df["height"] * h_scale)

    return df


def get_scale_and_ratio(df: DataFrame, prefix: str = None) -> DataFrame:
    """Return df with area and ratio per bbox measurements.

    Args:
        df: DataFrame with COCO annotations.
        prefix: Prefix to apply to column names, use for scaled data.

    Returns:
        DataFrame with new columns [prefix_]area/ratio

    """
    columns = ["width", "height", "scale", "ratio"]

    if prefix:
        columns = [f"{prefix}_{col}" for col in columns]

    df[columns[2]] = np.sqrt(df[columns[0]] * df[columns[1]])
    df[columns[3]] = df[columns[1]] / df[columns[0]]

    return df


def add_centroids(
    df: DataFrame, prefix: str = None, input_bbox_format: str = "coco"
) -> DataFrame:
    """Compute bbox centroids.

    Args:
        df: DataFrame with COCO annotations.
        prefix: Prefix to apply to column names, use for scaled data. Defaults to None.
        input_bbox_format: Input bounding box format. Can be "coco" or "corners".
            "coco" ["col_left", "row_top", "width", "height"]
            "corners" ["col_left", "row_top", "col_right", "row_bottom"]
            Defaults to "coco".

    Returns:
        DataFrame with new columns [prefix_]row_centroid/col_centroid

    """
    columns = ["col_centroid", "row_centroid"]
    bboxes = get_bbox_array(df, prefix=prefix, input_bbox_format=input_bbox_format)

    if prefix:
        columns = [f"{prefix}_{col}" for col in columns]

    df[columns[0]] = bboxes[:, 0] + bboxes[:, 2] // 2
    df[columns[1]] = bboxes[:, 1] + bboxes[:, 3] // 2

    return df


def corners_to_coco(bboxes: ndarray) -> ndarray:
    """Transform bboxes array from corners format to coco.

    Args:
        bboxes: Array with dimension N x 4 with bbox coordinates in corner format
        ["col_left", "row_top", "col_right", "row_bottom"]

    Returns:
        Array with dimension N x 4 with bbox coordinates in coco format
        [col_left, row_top, width, height].

    """
    bboxes = bboxes.copy()
    bboxes[..., 2:] = bboxes[..., 2:] - bboxes[..., :2]
    return bboxes


def coco_to_corners(bboxes: ndarray) -> ndarray:
    """Transform bboxes array from coco format to corners.

    Args:
        bboxes: Array with dimension N x 4 with bbox coordinates in corner format
        [col_left, row_top, width, height].

    Returns:
        Array with dimension N x 4 with bbox coordinates in coco format
        ["col_left", "row_top", "col_right", "row_bottom"]

    """
    bboxes = bboxes.copy()
    bboxes[..., 2:] = bboxes[..., :2] + bboxes[..., 2:]

    if (bboxes < 0).any():
        logger.warning("Clipping bboxes to min corner 0, found negative value")
        bboxes = np.clip(bboxes, 0, None)
    return bboxes


def normalize(bboxes: ndarray, image_width: int, image_height: int) -> ndarray:
    """Transform bboxes array from pixels to (0, 1) range.

    Bboxes can be in both formats:
        "coco" ["col_left", "row_top", "width", "height"]
        "corners" ["col_left", "row_top", "col_right", "row_bottom"]

    Args:
        bboxes: Bounding boxes.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Bounding boxes with coordinates in (0, 1) range.
    """
    norms = np.array([image_width, image_height, image_width, image_height])
    bboxes = bboxes * 1 / norms

    return bboxes


def denormalize(bboxes: ndarray, image_width: int, image_height: int) -> ndarray:
    """Transform bboxes array from (0, 1) range to pixels.

    Bboxes can be in both formats:
        "coco" ["col_left", "row_top", "width", "height"]
        "corners" ["col_left", "row_top", "col_right", "row_bottom"]

    Args:
        bboxes: Bounding boxes.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Bounding boxes with coordinates in pixels.

    """
    norms = np.array([image_width, image_height, image_width, image_height])
    bboxes = bboxes * norms
    return bboxes


def get_bbox_column_names(bbox_format: str, prefix: Optional[str] = None) -> List[str]:
    """Return predefined column names for each format.

    When bbox_format is 'coco' column names are
    ["col_left", "row_top", "width", "height"], when 'corners'
    ["col_left", "row_top", "col_right", "row_bottom"].

    Args:
        bbox_format: Bounding box format. Can be "coco" or "corners".
        prefix: Prefix to apply to column names, use for scaled data. Defaults to None.

    Returns:
        Column names for specified bbox format

    """
    if bbox_format == "coco":
        columns = ["col_left", "row_top", "width", "height"]
    elif bbox_format == "corners":
        columns = ["col_left", "row_top", "col_right", "row_bottom"]
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
    """Return array with bbox coordinates.

    Args:
        df: DataFrame with COCO annotations.
        prefix: Prefix to apply to column names, use for scaled data. Defaults to None.
        input_bbox_format: Input bounding box format. Can be "coco" or "corners".
            Defaults to "coco".
        output_bbox_format: Output bounding box format. Can be "coco" or "corners".
            Defaults to "coco".

    Returns:
        Array with dimension N x 4 with bbox coordinates.

    Examples:
        `coco`:
        >>>[col_left, row_top, width, height]

        `corners`:
        >>>[col_left, row_top, col_right, row_bottom]

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
    """Create DataFrame of annotations in Coco format from array of bboxes.

    Args:
        bboxes: Array of bboxes of shape [n, 4].
        input_bbox_format: Input bounding box format. Can be "coco" or "corners".
            Defaults to "coco".
        output_bbox_format: Output bounding box format. Can be "coco" or "corners".
            Defaults to "corners".

    Returns:
        DataFrame with Coco annotations.

    """
    check_bbox_formats(input_bbox_format, output_bbox_format)

    if input_bbox_format != output_bbox_format:
        convert = globals()[f"{input_bbox_format}_to_{output_bbox_format}"]
        bboxes = convert(bboxes)

    return pd.DataFrame(bboxes, columns=get_bbox_column_names(output_bbox_format))


def filter_zero_area_bboxes(df: DataFrame) -> DataFrame:
    """Filter those bboxes with height or width equal to zero.

    Args:
        df: DataFrame with COCO annotations.

    Returns:
        Filtered DataFrame with COCO annotations.

    """
    cols = ["width", "height"]
    all_bboxes = len(df)
    df = df[(df[cols] > 0).all(axis=1)].reset_index()
    filtered_bboxes = len(df)

    n_filtered = all_bboxes - filtered_bboxes

    if n_filtered:
        logger.warning(
            f"A total of {n_filtered} bboxes have been filtered from your data "
            "for having area equal to zero."
        )

    return df


def divide_filename(filename: str) -> Tuple[str, int, str]:
    """Divide a image filename.

    Args:
        filename: Image filename.

    Returns:
        Name of the video, number of frame and extension.

    """
    extension, frame, video_name = map(
        lambda x: x[::-1], re.match("(\w+)\.(\d+)(.+)", filename[::-1]).groups()  # type: ignore
    )
    video_name = video_name if video_name[-1] not in ("-", "_") else video_name[:-1]
    return video_name, int(frame), extension
