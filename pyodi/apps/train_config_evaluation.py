import os.path as osp
import sys
from importlib import import_module
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple

import numpy as np
import typer
from loguru import logger

from pyodi.coco.utils import (
    coco_ground_truth_to_dfs,
    filter_zero_area_bboxes,
    get_bbox_array,
    get_scale_and_ratio,
    join_annotations_with_image_sizes,
    load_ground_truth_file,
    scale_bbox_dimensions,
)
from pyodi.core.anchor_generator import AnchorGenerator
from pyodi.core.clustering import get_max_overlap
from pyodi.plots.evaluation import plot_overlap_result

app = typer.Typer()


def load_train_config_file(train_config_file: str) -> dict:
    logger.info("Loading Train Config File")
    with TemporaryDirectory() as temp_config_dir:
        copyfile(train_config_file, osp.join(temp_config_dir, "_tempconfig.py"))
        sys.path.insert(0, temp_config_dir)
        mod = import_module("_tempconfig")
        sys.path.pop(0)
        train_config = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__")
        }
        # delete imported module
        del sys.modules["_tempconfig"]
    return train_config


@logger.catch
@app.command()
def train_config_evaluation(
    ground_truth_file: str,
    train_config_file: str,
    input_size: Tuple[int, int] = (1280, 720),
    show: bool = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
):
    """Evaluates the fitness between `ground_truth_file` and `train_config_file`.

    Parameters
    ----------
    ground_truth_file : str
        Path to COCO ground truth file
    train_config_file: str
        Path to MMDetection-like configuration file.
        Must contain `train_pipeline` and `anchor_generator` sections.
        Example content:

        # faster_rcnn_r50_fpn_gn_ws_config.py

        train_pipeline = [
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Pad', size_divisor=32)
        ]

        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        )
    input_size : Tuple[int, int],
        Model image input size, by default (1280, 720)
    show : bool, optional
        Show results or not, by default True
    output : str, optional
        Output file where results are saved, by default None
    output_size : tuple
        Size of saved images, by default (1600, 900)
    """

    if output is not None:
        output = str(Path(output) / Path(ground_truth_file).stem)
        Path(output).mkdir(parents=True, exist_ok=True)

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    df_annotations = join_annotations_with_image_sizes(df_annotations, df_images)

    df_annotations = filter_zero_area_bboxes(df_annotations)

    df_annotations = scale_bbox_dimensions(df_annotations, input_size=input_size)

    df_annotations = get_scale_and_ratio(df_annotations, prefix="scaled")

    train_config = load_train_config_file(train_config_file)

    del train_config["anchor_generator"]["type"]
    anchor_generator = AnchorGenerator(**train_config["anchor_generator"])
    logger.info(anchor_generator)

    width, height = input_size
    featmap_sizes = [
        (width // stride, height // stride) for stride in anchor_generator.strides
    ]
    anchors_per_level = anchor_generator.grid_anchors(featmap_sizes=featmap_sizes)

    bboxes = get_bbox_array(
        df_annotations, prefix="scaled", output_bbox_format="corners"
    )

    overlaps = np.zeros(bboxes.shape[0])
    max_overlap_level = np.zeros(bboxes.shape[0])

    logger.info("Computing overlaps between anchors and ground truth")
    for i, anchor_level in enumerate(anchors_per_level):
        level_overlaps = get_max_overlap(
            bboxes.astype(np.float32), anchor_level.astype(np.float32)
        )
        max_overlap_level[level_overlaps > overlaps] = i
        overlaps = np.maximum(overlaps, level_overlaps)

    df_annotations["overlaps"] = overlaps
    df_annotations["max_overlap_level"] = max_overlap_level

    logger.info("Plotting results")
    plot_overlap_result(
        df_annotations, show=show, output=output, output_size=output_size
    )


if __name__ == "__main__":
    app()
