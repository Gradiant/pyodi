from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pandas import DataFrame
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def plot_scatter_with_histograms(
    df: DataFrame,
    x: str = "width",
    y: str = "height",
    title: Optional[str] = None,
    show: bool = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
    histogram: bool = True,
    label: str = "category",
    colors: Optional[List] = None,
    legendgroup: Optional[str] = None,
    fig: Optional[go.Figure] = None,
    row: int = 1,
    col: int = 1,
    xaxis_range: Optional[Tuple[float, float]] = None,
    yaxis_range: Optional[Tuple[float, float]] = None,
    histogram_xbins: Optional[Dict[str, Any]] = None,
    histogram_ybins: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> go.Figure:
    """Allows to compare the relation between two variables of your COCO dataset.

    Args:
        df: COCO annotations generated DataFrame.
        x: Name of column that will be represented in x axis. Defaults to "width".
        y: Name of column that will be represented in y axis. Defaults to "height".
        title: Plot name. Defaults to None.
        show: Whether to show the figure or not. Defaults to True.
        output: Output path folder. Defaults to None.
        output_size: Size of saved images. Defaults to (1600, 900).
        histogram: Whether to draw a marginal histogram distribution of each axis or
            not. Defaults to True.
        label: Name of the column with class information in df_annotations. Defaults
            to 'category'.
        colors: List of rgb colors to use. If None default plotly colors are used.
            Defaults to None.
        legendgroup: When present legend is grouped by different categories
            (see https://plotly.com/python/legend/).
        fig: When figure is provided, trace is automatically added on it. Defaults to
            None.
        row: Subplot row to use when fig is provided. Defaults to 1.
        col: Subplot col to use when fig is provided. Defaults to 1.
        xaxis_range: range of values for the histogram's horizontal axis
        yaxis_range: range of values for the histogram's vertical axis
        histogram_xbins: number of bins for the histogram's horizontal axis
        histogram_ybins: number of bins for the histogram's vertical axis
    Returns:
        Plotly figure.

    """
    logger.info("Plotting Scatter with Histograms")
    if not fig:
        fig = make_subplots(rows=1, cols=1)

    classes = [(0, None)]
    if label in df:
        classes = list(enumerate(sorted(df[label].unique())))

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
            xbins=histogram_xbins,
        )
        fig.add_histogram(
            y=df[y],
            name=f"{y} distribution",
            xaxis="x2",
            marker=dict(color="rgb(102, 197, 204)"),
            histnorm="percent",
            ybins=histogram_ybins,
        )

        fig.layout = dict(
            xaxis=dict(
                domain=[0, 0.84], showgrid=False, zeroline=False, range=xaxis_range
            ),
            yaxis=dict(
                domain=[0, 0.83], showgrid=False, zeroline=False, range=yaxis_range
            ),
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
) -> go.Figure:
    """Plot histogram figure.

    Args:
        df: Data to plot.
        column: DataFrame column to plot.
        title: Title of figure. Defaults to None.
        xrange: Range in axis X. Defaults to None.
        yrange: Range in axis Y. Defaults to None.
        xbins: Width of X bins. Defaults to None.
        histnorm: Histnorm. Defaults to "percent".
        show: Whether to show the figure or not. Defaults to False.
        output: Output path folder. Defaults to None.
        output_size: Size of saved images. Defaults to (1600, 900).

    Returns:
        Histogram figure.

    """
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
) -> None:
    """Saves figure into png image file.

    Args:
        figure: Figure to save.
        output_name: Output filename.
        output_dir: Output directory.
        output_size: Size of saved image.

    """
    output = str(Path(output_dir) / (output_name.replace(" ", "_") + ".png"))
    figure.update_layout(width=output_size[0], height=output_size[1])
    figure.write_image(output, engine="kaleido")
