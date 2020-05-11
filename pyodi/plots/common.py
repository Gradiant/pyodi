from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import plotly.graph_objects as go
from loguru import logger
from pandas import DataFrame
from plotly.subplots import make_subplots


def plot_scatter_with_histograms(
    df,
    x="width",
    y="height",
    title=None,
    show=True,
    output=None,
    output_size=(1600, 900),
    histogram=True,
    label="category",
    colors=None,
    legendgroup=None,
    fig=None,
    row=1,
    col=1,
    **kwargs,
):
    """This plot allows to compare the relation between two variables of your coco dataset
    Parameters
    ----------
    df : pd.DataFrame
        COCO annotations generated dataframe
    x : str, optional
        name of column that will be represented in x axis, by default "width"
    y : str, optional
        name of column that will be represented in y axis, by default "height"
    title : [type], optional
        plot name, by default None
    show : bool, optional
        if activated figure is shown, by default True
    output : str, optional
        output path folder, by default None
    output_size : tuple
        size of saved images, by default (1600, 900)
    histogram: bool, optional
        when histogram is true a marginal histogram distribution of each axis is drawn, by default False
    label: str, optional
        name of the column with class information in df_annotations, by default 'category'
    colors: list, optional
        list of rgb colors to use, if none default plotly colors are used
    legendgroup: str, optional
        when present legend is grouped by different categories (see https://plotly.com/python/legend/)
    fig: plotly.Figure, optional
        when figure is provided, trace is automatically added on it
    row: int, optional
        subplot row to use when fig is provided, default 1
    col: int, optional
        subplot col to use when fig is provided, default 1
    Returns
    -------
    plotly figure
    """

    logger.info("Plotting Scatter with Histograms")
    if not fig:
        fig = make_subplots(rows=1, cols=1)

    classes = [(0, None)]
    if label in df:
        classes = enumerate(sorted(df[label].unique()))

    for i, c in classes:
        if c:
            filtered_df = df[df[label] == c]
        else:
            filtered_df = df
        scatter = go.Scattergl(
            x=filtered_df[x],
            y=filtered_df[y],
            mode="markers",
            name=str(c or "Images Shape"),
            text=filtered_df["file_name"],
            marker=dict(color=colors[i % len(colors)] if colors else None),
            legendgroup=f"legendgroup_{i}" if legendgroup else None,
            **kwargs,
        )
        fig.add_trace(scatter, row=row, col=col)

    if histogram:
        fig.add_histogram(
            x=df[x],
            name=f"{x} distribution",
            yaxis="y2",
            marker=dict(color="rgb(246, 207, 113)"),
            histnorm="percent",
            xbins=dict(size=10),
        )
        fig.add_histogram(
            y=df[y],
            name=f"{y} distribution",
            xaxis="x2",
            marker=dict(color="rgb(102, 197, 204)"),
            histnorm="percent",
            ybins=dict(size=10),
        )

        fig.layout = dict(
            xaxis=dict(domain=[0, 0.84], showgrid=False, zeroline=False,),
            yaxis=dict(domain=[0, 0.83], showgrid=False, zeroline=False,),
            xaxis2=dict(
                domain=[0.85, 1], showgrid=False, zeroline=False, range=(0, 100)
            ),
            yaxis2=dict(
                domain=[0.85, 1], showgrid=False, zeroline=False, range=(0, 100)
            ),
        )

    if title is None:
        title = f"{x} vs {y}"
    fig.update_layout(
        title_text=title, xaxis_title=f"{x}", yaxis_title=f"{y}", title_font_size=20
    )

    if show:
        fig.show()

    if output:
        save_figure(fig, title, output, output_size)

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
        save_figure(fig, title, output, output_size)

    return fig


def save_figure(
    figure: go.Figure, output_name: str, output_dir: str, output_size: Tuple[int, int]
):
    output = str(Path(output_dir) / (output_name.replace(" ", "_") + ".png"))
    figure.update_layout(width=output_size[0], height=output_size[1])
    figure.write_image(output)
