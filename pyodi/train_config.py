import argparse
import json
import os
from pathlib import Path

from coco.utils import (
    coco_ground_truth_to_dfs,
    get_area_and_ratio,
    join_annotations_with_image_sizes,
    load_ground_truth_file,
    scale_bbox_dimensions,
)
from loguru import logger
from plots.annotations import plot_bounding_box_distribution


def ground_truth_app(ground_truth_file, show=True, output=None, input_size=(1280, 720)):
    """[summary]
    Parameters
    ----------
    ground_truth_file : str
        Path to COCO ground truth file
    show : bool, optional
        Show results or not, by default True
    output : str, optional
        Output file where results are saved, by default None
    input_size : tuple, optional
        Model image input size, by default (1280, 720)
    """

    if output is not None:
        output = Path(output) / Path(ground_truth_file).name

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    df_annotations = join_annotations_with_image_sizes(df_annotations, df_images)

    df_annotations = scale_bbox_dimensions(df_annotations, input_size=input_size)

    df_annotations = get_area_and_ratio(df_annotations, prefix="scaled")

    plot_bounding_box_distribution(
        df_annotations,
        x="scaled_area",
        y="scaled_ratio",
        title="Bounding box area vs Aspect ratio",
        show=True,
        histogram=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Object Detection Insights: Ground Truth"
    )

    parser.add_argument("--file", help="COCO Ground Truth File")
    parser.add_argument("--show", default=True, action="store_false")
    parser.add_argument("--output", default=None)

    try:
        args = parser.parse_args()
    except SystemExit as e:
        os._exit(e.code)

    ground_truth_app(args.file)
