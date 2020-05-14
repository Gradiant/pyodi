from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame

from pyodi.plots.common import save_figure


def get_centroids_heatmap(
    df: DataFrame, n_rows: int = 9, n_cols: int = 9
) -> np.ndarray:
    rows = df["row_centroid"] / df["img_height"]
    cols = df["col_centroid"] / df["img_width"]
    heatmap = np.zeros((n_rows, n_cols))
    for row, col in zip(rows, cols):
        heatmap[int(row * n_rows), int(col * n_cols)] += 1

    return heatmap


def plot_heatmap(
    heatmap: np.ndarray,
    title: str = "",
    show: bool = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=heatmap))

    fig.update_layout(title_text=title, title_font_size=20)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    if show:
        fig.show()

    if output:
        save_figure(fig, title, output, output_size)

    return fig
