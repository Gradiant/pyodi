import argparse
import json
import os
from pathlib import Path
import numpy as np
from coco.utils import (
    coco_ground_truth_to_dfs,
    get_area_and_ratio,
    join_annotations_with_image_sizes,
    load_ground_truth_file,
    scale_bbox_dimensions,
    get_bbox_array,
    get_df_from_bboxes,
)
from loguru import logger
from plots.annotations import plot_scatter_with_histograms
from plots.clustering import plot_clustering_results
from core.clustering import kmeans_iou
import plotly.graph_objects as go


def ground_truth_app(
    ground_truth_file, show=True, output=None, input_size=(1280, 720), clusters=None,
):
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
    cluster: list or int, optional
        Number of clusters to compute. If list, a different clustering will be computed per each number of cluster
    """

    if output is not None:
        output = Path(output) / Path(ground_truth_file).name

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    df_annotations = join_annotations_with_image_sizes(df_annotations, df_images)

    df_annotations = scale_bbox_dimensions(df_annotations, input_size=input_size)

    df_annotations = get_area_and_ratio(df_annotations, prefix="scaled")

    plot_scatter_with_histograms(
        df_annotations,
        x="scaled_area",
        y="scaled_ratio",
        title="Bounding box area vs Aspect ratio",
        show=True,
        histogram=True,
    )

    if clusters is not None:
        if isinstance(clusters, int):
            clusters = list(clusters)

        bboxes = get_bbox_array(df_annotations, prefix="scaled", bbox_format="coco")
        centroids, silhouette_metrics, predicted_clusters = kmeans_iou(
            bboxes[:, 2:], k=clusters
        )

        # todo: improve way of getting index & move selection to frontend
        selected = 0
        if len(clusters) > 1 and show:
            fig = go.Figure(
                data=[
                    go.Scattergl(x=clusters, y=silhouette_metrics, mode="lines+markers")
                ]
            )
            fig.update_layout(
                xaxis_title="Number of clusters", yaxis_title="Silhouette Coefficient"
            )
            fig.show()
            selected = int(input("Choose best number of clusters: ")) - clusters[0]

        df_annotations["cluster"] = predicted_clusters[selected]

        # bring coco format back adding centered coords
        centroids = np.concatenate(
            [np.zeros_like(centroids[selected]), centroids[selected]], axis=-1
        )
        centroids = get_df_from_bboxes(
            centroids, input_bbox_format="coco", output_bbox_format="coco"
        )
        centroids = get_area_and_ratio(centroids)
        plot_clustering_results(centroids, df_annotations)


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
