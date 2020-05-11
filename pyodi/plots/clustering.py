from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from numpy import float64, ndarray
from pandas.core.frame import DataFrame
from plotly.colors import DEFAULT_PLOTLY_COLORS as COLORS
from plotly.subplots import make_subplots

from pyodi.coco.utils import get_df_from_bboxes
from pyodi.core.anchor_generator import AnchorGenerator
from pyodi.plots.common import plot_scatter_with_histograms


def plot_clustering_results(
    df_annotations: DataFrame,
    anchor_generator: AnchorGenerator,
    show: Optional[bool] = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
    centroid_color: Optional[tuple] = None,
    title: Optional[str] = None,
):
    """Plots cluster results in two different views, width vs heihgt and area vs ratio.

    Parameters
    ----------
    df_annotations : pd.DataFrame
        COCO annotations generated dataframe
    anchor_generator: core.AnchorGenerator
        Anchor generator instance
    show : bool, optional
        If true plotly figure will be shown, by default True
    output : str, optional
        Output image folder, by default None
    output_size : tuple
        Size of saved images, by default (1600, 900)
    centroid_color: tuple, optional
        Plotly rgb color format for painting centroids, by default None
    title: str, optional
        Plot title and filename is output is not None, by default None
    """

    if centroid_color is None:
        centroid_color = COLORS[len(df_annotations.category.unique()) % len(COLORS)]

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["Relative Scale vs Ratio", "Width vs Height"]
    )

    plot_scatter_with_histograms(
        df_annotations,
        x=f"log_level_scale",
        y=f"log_ratio",
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
        xaxis2=dict(title="Scaled width"),
        xaxis=dict(title="Relative Scale"),
        yaxis2=dict(title="Scaled height"),
        yaxis=dict(title="Ratio"),
    )

    if show:
        fig.show()

    if output and title:
        title = title.replace(" ", "_")
        fig.update_layout(width=output_size[0], height=output_size[1])
        fig.write_image(f"{output}/{title}.png")
