from typing import Optional, Tuple

import numpy as np
from pandas.core.frame import DataFrame
from plotly import graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS as COLORS
from plotly.subplots import make_subplots

from pyodi.core.anchor_generator import AnchorGenerator
from pyodi.core.boxes import get_df_from_bboxes
from pyodi.plots.common import plot_scatter_with_histograms, save_figure


def plot_clustering_results(
    df_annotations: DataFrame,
    anchor_generator: AnchorGenerator,
    show: Optional[bool] = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
    centroid_color: Optional[Tuple] = None,
    title: Optional[str] = None,
) -> None:
    """Plots cluster results in two different views, width vs height and area vs ratio.

    Args:
        df_annotations: COCO annotations generated DataFrame.
        anchor_generator: Anchor generator instance.
        show: Whether to show the figure or not. Defaults to True.
        output: Output path folder. Defaults to None.
        output_size: Size of saved images. Defaults to (1600, 900).
        centroid_color: Plotly rgb color format for painting centroids. Defaults to
            None.
        title: Plot title and filename if output is not None. Defaults to None.

    """
    if centroid_color is None:
        centroid_color = COLORS[len(df_annotations.category.unique()) % len(COLORS)]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Relative Log Scale vs Log Ratio",
            "Scaled Width vs Scaled Height",
        ],
    )

    plot_scatter_with_histograms(
        df_annotations,
        x="log_level_scale",
        y="log_ratio",
        legendgroup="classes",
        show=False,
        colors=COLORS,
        histogram=False,
        fig=fig,
    )

    cluster_grid = np.array(
        np.meshgrid(np.log(anchor_generator.scales), np.log(anchor_generator.ratios))
    ).T.reshape(-1, 2)

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
        x="scaled_width",
        y="scaled_height",
        show=False,
        colors=COLORS,
        legendgroup="classes",
        histogram=False,
        showlegend=False,
        fig=fig,
        col=2,
    )

    for anchor_level in anchor_generator.base_anchors:
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
        title=title,
        xaxis2=dict(title="Scaled Width"),
        xaxis=dict(title="Log Relative Scale"),
        yaxis2=dict(title="Scaled Height"),
        yaxis=dict(title="Log Ratio"),
    )

    if show:
        fig.show()

    if output:
        save_figure(fig, "clusters", output, output_size)
