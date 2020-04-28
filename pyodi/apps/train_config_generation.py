from pathlib import Path
from typing import List, Optional, Tuple

import typer

from loguru import logger

from pyodi.coco.utils import (
    coco_ground_truth_to_dfs,
    filter_zero_area_bboxes,
    get_scale_and_ratio,
    join_annotations_with_image_sizes,
    load_ground_truth_file,
    scale_bbox_dimensions,
)

from pyodi.core.clustering import kmeans_euclidean
from pyodi.plots.clustering import plot_clustering_results


app = typer.Typer()


@logger.catch
@app.command()
def train_config_generation(
    ground_truth_file: str,
    input_size: Tuple[int, int] = (1280, 720),
    n_ratios: int = 3,
    n_scales: int = 3,
    strides: List[int] = [8, 16, 32, 64, 128],
    show: bool = True,
    output: Optional[str] = None,
):
    """Computes optimal anchors for a given COCO dataset based on iou clustering.

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
        output = str(Path(output) / Path(ground_truth_file).name)

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    df_annotations = join_annotations_with_image_sizes(df_annotations, df_images)

    df_annotations = filter_zero_area_bboxes(df_annotations)

    df_annotations = scale_bbox_dimensions(df_annotations, input_size=input_size)

    df_annotations = get_scale_and_ratio(df_annotations, prefix="scaled")

    # Cluster bboxes by scale and ratio independently
    clustering_results = [
        kmeans_euclidean(df_annotations[value], n_clusters=n_clusters)
        for i, (value, n_clusters) in enumerate(
            zip(["scaled_scale", "scaled_ratio"], [n_scales, n_ratios])
        )
    ]

    # Plot results
    plot_clustering_results(clustering_results, df_annotations)


if __name__ == "__main__":
    app()
