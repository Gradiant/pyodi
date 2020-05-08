from typing import Any, Dict, Optional, Tuple

import plotly.graph_objects as go
from loguru import logger
from pandas import DataFrame


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
    output_size: Tuple[int, int] = (1600, 900),
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
        fig.update_layout(width=output_size[0], height=output_size[1])
        fig.write_image(f"{output}/{title}.png")

    return fig
