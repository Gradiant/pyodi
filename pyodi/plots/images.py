from typing import Any, Dict, Optional, Tuple

import plotly.graph_objects as go
from loguru import logger
from pandas import DataFrame

from .utils import plot_scatter_with_histograms


def plot_image_shape_distribution(
    df_images,
    x="width",
    y="height",
    title="Image Shape Distribution",
    histogram=True,
    show=True,
    output=None,
    output_size=(1600, 900),
):
    """Image Shape Distribution

    This plot shows the height and width distributions of all the images in the dataset.

    It can serve as an indicator for setting the optimal **input size** and **aspect ratio** of your model.

    """
    return plot_scatter_with_histograms(
        df_images,
        x=x,
        y=y,
        title=title,
        histogram=histogram,
        show=show,
        output=output,
        output_size=output_size,
    )


def plot_histogram(
    df: DataFrame,
    column: str,
    title: Optional[str] = None,
    xrange: Optional[Tuple[int, int]] = None,
    yrange: Optional[Tuple[int, int]] = None,
    xbins: Optional[Dict[str, Any]] = None,
    histnorm: Optional[str] = "percent",
    show: bool = False,
    output: Optional[str] = None,
):

    logger.info(f"Plotting {column} Histogram")
    fig = go.Figure(
        data=[
            go.Histogram(
                x=df[column], histnorm=histnorm, hovertext=df["file_name"], xbins=xbins
            )
        ]
    )

    if xrange is not None:
        fig.update_xaxes(range=xrange)

    if yrange is not None:
        fig.update_yaxes(range=yrange)

    if title is None:
        title = f"{column} histogram"

    fig.update_layout(title_text=title, title_font_size=20)

    if show:
        fig.show()

    if output:
        title = title.replace(" ", "_")
        fig.write_image(f"{output}/{title}.png")

    return fig
