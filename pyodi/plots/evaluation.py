from typing import Dict, List, Optional, Tuple, Union

import plotly.graph_objects as go
from pandas.core.frame import DataFrame
from plotly.subplots import make_subplots


def plot_overlap_result(
    df: DataFrame,
    max_bins: int = 30,
    show: Optional[bool] = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
):
    """Generates plot for train config evaluation based on overlap

    Parameters
    ----------
    df : DataFrame
        COCO annotations generated dataframe with overlap
    max_bins : int, optional
        Max bins to use in histograms, by default 30
    show : Optional[bool], optional
        If true plotly figure will be shown, by default True
    output : str, optional
        Output image folder, by default None
    output_size : tuple
        Size of the saved images, by default (1600, 900)
    """

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cumulative overlap distribution",
            "Bounding Box Distribution",
            "Scale and mean overlap",
            "Ratio and mean overlap",
        ),
    )

    fig.append_trace(
        go.Histogram(
            x=df["overlaps"],
            histnorm="probability",
            cumulative_enabled=True,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.append_trace(
        go.Scattergl(
            x=df["scaled_width"],
            y=df["scaled_height"],
            mode="markers",
            showlegend=False,
            marker=dict(
                color=df["overlaps"],
                colorscale="Electric",
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title="Overlap value", lenmode="fraction", len=0.5, y=0.8
                ),
            ),
        ),
        row=1,
        col=2,
    )

    for i, column in enumerate(["scaled_scale", "scaled_ratio"], 1):
        fig.append_trace(
            go.Histogram(
                x=df[column],
                y=df["overlaps"],
                histfunc="avg",
                nbinsx=max_bins,
                showlegend=False,
            ),
            row=2,
            col=i,
        )

    fig["layout"].update(
        title="Train config evaluation",
        xaxis=dict(title="Overlap values"),
        xaxis2=dict(title="Scaled width"),
        xaxis3=dict(title="Scale"),
        xaxis4=dict(title="Ratio"),
        yaxis=dict(title="Accumulated percentage"),
        yaxis2=dict(title="Scaled heigh"),
        yaxis3=dict(title="Mean overlap"),
        yaxis4=dict(title="Mean overlap"),
        legend=dict(y=0.5),
    )

    if show:
        fig.show()

    if output:
        fig.update_layout(width=output_size[0], height=output_size[1])
        fig.write_image(f"{output}/overlap.png")
