from typing import Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from numpy import float64, ndarray
from pandas.core.frame import DataFrame
from plotly.colors import DEFAULT_PLOTLY_COLORS as COLORS
from plotly.subplots import make_subplots

from pyodi.plots.boxes import plot_scatter_with_histograms
from pyodi.core.anchor_generator import AnchorGenerator
from pyodi.coco.utils import get_df_from_bboxes


def plot_clustering_results(
    df_annotations: DataFrame,
    scales: List[float],
    ratios: List[float],
    strides: List[int],
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
        rows=1, cols=2, subplot_titles=["Relative Scale vs Ratio", "Width vs Height"]
    )

    plot_scatter_with_histograms(
        df_annotations,
        x=f"level_scale",
        y=f"scaled_ratio",
        legendgroup="classes",
        show=False,
        colors=COLORS,
        histogram=False,
        fig=fig,
    )

    cluster_grid = np.array(np.meshgrid(scales, ratios)).T.reshape(-1, 2)

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

    base_anchors = AnchorGenerator(
        strides=strides, ratios=ratios, scales=scales
    ).base_anchors

    for anchor_level in base_anchors:
        anchor_level = get_df_from_bboxes(
            anchor_level, input_bbox_format="corners", output_bbox_format="coco"
        )
        fig.append_trace(
            go.Scattergl(
                x=anchor_level["width"],
                y=anchor_level["height"],
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
        xaxis=dict(title="Relative Scale"),
        yaxis2=dict(title="Scaled height"),
        yaxis=dict(title="Ratio"),
    )

    if show:
        fig.show()

    if output:
        fig.write_image(f"{output}/clusters.png")
