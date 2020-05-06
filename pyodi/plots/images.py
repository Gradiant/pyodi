from typing import Any, Dict, Optional, Tuple

import plotly.graph_objects as go
from loguru import logger
from pandas import DataFrame


def plot_image_shape_distribution(
    df_images,
    x="width",
    y="height",
    title="Image Shape Distribution",
    histogram=True,
    show=True,
    output=None,
):
    """Image Shape Distribution

    This plot shows the height and width distributions of all the images in the dataset.

    It can serve as an indicator for setting the optimal **input size** and **aspect ratio** of your model.

    """
    logger.info("Plotting Image Shape Distribution")
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=df_images[x],
                y=df_images[y],
                mode="markers",
                name="Image Shapes",
                text=df_images["file_name"],
            )
        ]
    )
    if histogram:
        fig.add_histogram(
            x=df_images[x],
            name=f"{x} distribution",
            yaxis="y2",
            marker=dict(color="rgb(246, 207, 113)"),
            histnorm="percent",
        )
        fig.add_histogram(
            y=df_images[y],
            name=f"{y} distribution",
            xaxis="x2",
            marker=dict(color="rgb(102, 197, 204)"),
            histnorm="percent",
        )

        fig.layout = dict(
            xaxis=dict(domain=[0, 0.84], showgrid=False, zeroline=False,),
            yaxis=dict(domain=[0, 0.83], showgrid=False, zeroline=False),
            xaxis2=dict(
                domain=[0.85, 1], showgrid=False, zeroline=False, range=(0, 100)
            ),
            yaxis2=dict(
                domain=[0.85, 1], showgrid=False, zeroline=False, range=(0, 100)
            ),
        )

    fig.update_layout(
        title_text=title,
        title_font_size=20,
        showlegend=False,
        xaxis_title=f"{x}",
        yaxis_title=f"{y}",
    )

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
