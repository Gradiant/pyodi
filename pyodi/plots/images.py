from typing import Any, Dict, Optional, Tuple

import plotly.graph_objects as go
from loguru import logger
from pandas import DataFrame


def plot_image_shape_distribution(df_images, show=True, output=None):
    """Image Shape Distribution

    This plot shows the height and width distributions of all the images in the dataset.

    It can serve as an indicator for setting the optimal **input size** and **aspect ratio** of your model.

    """
    logger.info("Plotting Image Shape Distribution")
    fig = go.Figure(
        data=[
            go.Box(
                y=df_images[x],
                boxpoints="all",
                boxmean="sd",
                name=x,
                hovertext=df_images["file_name"],
            )
            for x in ["height", "width"]
        ]
    )
    fig.update_yaxes(range=[0, 8192])
    fig.update_layout(title_text="Image Shape Distribution", title_font_size=20)

    if show:
        fig.show()
    if output:
        fig.write_image(f"{output}/Image_Shape_Distribution.png")

    return fig


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
