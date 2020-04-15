import argparse
import json
import os
from pathlib import Path

from coco.utils import coco_ground_truth_to_dfs, load_ground_truth_file
from loguru import logger
from plots.annotations import plot_bounding_box_distribution
from plots.images import plot_image_shape_distribution


def ground_truth_app(ground_truth_file, show=True, output=None):

    if output is not None:
        output = Path(output) / Path(ground_truth_file).name

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    plot_image_shape_distribution(df_images, show=show, output=output)

    plot_bounding_box_distribution(df_annotations, show=show, output=output)


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
