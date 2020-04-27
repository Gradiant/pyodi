from typing import Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from numpy import float64, ndarray
from pandas.core.frame import DataFrame
from plotly.colors import DEFAULT_PLOTLY_COLORS as COLORS
from plotly.subplots import make_subplots

from pyodi.plots.boxes import plot_scatter_with_histograms


def plot_clustering_results(
    clustering_results: List[Dict[str, Union[ndarray, float64]]],
    df_annotations: DataFrame,
    show: Optional[bool] = True,
    output: Optional[str] = None,
    centroid_color: Optional[tuple] = None,
):
    """Plots cluster results in two different views, width vs heihgt and area vs ratio.

    Parameters
    ----------
    clustering_results : List[dict]
        List of dictionaries with cluster information, see output for `core.clustering.kmeans_euclidean`
    df_annotations : pd.DataFrame
        COCO annotations generated dataframe
    show : bool, optional
        If true plotly figure will be shown, by default True
    output : str, optional
        Output image folder, by default None
    centroid_color: tuple, optional
        Plotly rgb color format for painting centroids, by default None
    """

    if centroid_color is None:
        centroid_color = COLORS[len(df_annotations.category.unique()) % len(COLORS)]

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["Area vs Ratio", "Width vs Height"]
    )

    plot_scatter_with_histograms(
        df_annotations,
        x=f"scaled_scale",
        y=f"scaled_ratio",
        legendgroup="classes",
        show=False,
        colors=COLORS,
        histogram=False,
        fig=fig,
    )

    scale_clusters = clustering_results[0]["centroids"]
    ratio_clusters = clustering_results[1]["centroids"]
    cluster_grid = np.array(np.meshgrid(scale_clusters, ratio_clusters)).T.reshape(
        -1, 2
    )

    fig.append_trace(
        go.Scattergl(
            x=cluster_grid[:, 0],
            y=cluster_grid[:, 1],
            mode="markers",
            legendgroup="centroids",
            name="centroids",
            marker=dict(
                color=centroid_color,
                size=10,
                line=dict(width=2, color="DarkSlateGrey"),
            ),
        ),
        row=1,
        col=1,
    )

    plot_scatter_with_histograms(
        df_annotations,
        x=f"scaled_width",
        y=f"scaled_height",
        show=False,
        colors=COLORS,
        legendgroup="classes",
        histogram=False,
        showlegend=False,
        fig=fig,
        col=2,
    )

    cluster_widths = cluster_grid[:, 0] / np.sqrt(cluster_grid[:, 1])
    cluster_heights = cluster_widths * cluster_grid[:, 1]
    fig.append_trace(
        go.Scattergl(
            x=cluster_widths,
            y=cluster_heights,
            mode="markers",
            legendgroup="centroids",
            name="centroids",
            showlegend=False,
            marker=dict(
                color=centroid_color,
                size=10,
                line=dict(width=2, color="DarkSlateGrey"),
            ),
        ),
        row=1,
        col=2,
    )

    fig["layout"].update(
        title="Anchor cluster visualization",
        xaxis2=dict(title="Scaled width"),
        xaxis=dict(title="Area"),
        yaxis2=dict(title="Scaled height"),
        yaxis=dict(title="Ratio"),
    )

    if show:
        fig.show()

    if output:
        fig.write_image(f"{output}/clusters.png")
