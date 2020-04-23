from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import typer
from loguru import logger

from pyodi.coco.utils import (
    coco_ground_truth_to_dfs,
    get_area_and_ratio,
    get_bbox_array,
    get_df_from_bboxes,
    join_annotations_with_image_sizes,
    load_ground_truth_file,
    scale_bbox_dimensions,
)
from pyodi.core.clustering import kmeans_iou
from pyodi.plots.annotations import plot_scatter_with_histograms
from pyodi.plots.clustering import plot_clustering_results

app = typer.Typer()


@logger.catch
@app.command()
def train_config(
    ground_truth_file: str,
    show: bool = True,
    output: Optional[str] = None,
    input_size: Tuple[int, int] = (1280, 720),
    clusters: List[int] = None,
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
        output = str(Path(output) / Path(ground_truth_file).name)

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

        bboxes = get_bbox_array(
            df_annotations, prefix="scaled", output_bbox_format="coco"
        )
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
    app()
