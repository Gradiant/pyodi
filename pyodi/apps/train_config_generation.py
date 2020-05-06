from pathlib import Path
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
from pyodi.core.clustering import find_pyramid_level, kmeans_euclidean
from pyodi.plots.clustering import plot_clustering_results

app = typer.Typer()


@logger.catch
@app.command()
def train_config_generation(
    ground_truth_file: str,
    input_size: Tuple[int, int] = (1280, 720),
    n_ratios: int = 3,
    n_scales: int = 3,
    strides: List[int] = [4, 8, 16, 32, 64],
    anchor_base_sizes: List[int] = [32, 64, 128, 256, 512],
    show: bool = True,
    output: Optional[str] = None,
    keep_ratio: bool = False,
):
    """Computes optimal anchors for a given COCO dataset based on iou clustering.

    Parameters
    ----------
    ground_truth_file : str
        Path to COCO ground truth file
    input_size: Tuple[int, int]
        Model image input size, by default (1280, 720)
    n_ratios: int
        Number of desired ratios, by default 3
    n_scales: int
        Number of desired scales, by default 3
    anchor_base_sizes: List[int]
        The basic sizes of anchors in multiple levels.
    show : bool, optional
        Show results or not, by default True
    output : str, optional
        Output file where results are saved, by default None
    input_size : tuple, optional
        Model image input size, by default (1280, 720)
    keep_ratio: bool, optional
        Whether to keep the aspect ratio, by default False
    """

    if output is not None:
        output = str(Path(output) / Path(ground_truth_file).stem)
        output_dir_path = Path(output)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    df_annotations = join_annotations_with_image_sizes(df_annotations, df_images)

    df_annotations = filter_zero_area_bboxes(df_annotations)

    df_annotations = scale_bbox_dimensions(
        df_annotations, input_size=input_size, keep_ratio=keep_ratio
    )

    df_annotations = get_scale_and_ratio(df_annotations, prefix="scaled")

    # Assign fpn level
    df_annotations["fpn_level"] = find_pyramid_level(
        get_bbox_array(df_annotations, prefix="scaled")[:, 2:], anchor_base_sizes
    )

    df_annotations["fpn_level_scale"] = df_annotations["fpn_level"].replace(
        {i: scale for i, scale in enumerate(anchor_base_sizes)}
    )

    df_annotations["level_scale"] = (
        df_annotations["scaled_scale"] / df_annotations["fpn_level_scale"]
    )

    # Normalize ratio to logn scale
    df_annotations["log_ratio"] = np.log(df_annotations["scaled_ratio"])

    # Cluster bboxes by scale and ratio independently
    clustering_results = [
        kmeans_euclidean(df_annotations[value], n_clusters=n_clusters)
        for i, (value, n_clusters) in enumerate(
            zip(["level_scale", "log_ratio"], [n_scales, n_ratios])
        )
    ]

    scales = clustering_results[0]["centroids"]
    ratios = np.e ** clustering_results[1]["centroids"]

    anchor_generator = AnchorGenerator(
        strides=strides, ratios=ratios, scales=scales, base_sizes=anchor_base_sizes
    )
    logger.info(f"Anchor configuration: \n{anchor_generator}")

    # Plot results
    plot_clustering_results(
        df_annotations, anchor_generator, show=show, output=output,
    )

    anchor_config = dict(
        type="AnchorGenerator",
        scales=list(scales),
        ratios=list(ratios),
        strides=list(strides),
        base_sizes=anchor_base_sizes,
    )

    return anchor_config


if __name__ == "__main__":
    app()
