import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots


def plot_scatter_with_histograms(
    df_annotations,
    x="width",
    y="height",
    title=None,
    show=True,
    output=None,
    max_values=None,
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
    df_annotations : pd.DataFrame
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
        output path folder , by default None
    max_values : tuple, optional
        x,y max allowed values in represention, by default None
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

    logger.info("Plotting Bounding Box Distribution")

    if not fig:
        fig = make_subplots(rows=1, cols=1)

    for i, c in enumerate(sorted(df_annotations[label].unique())):
        scatter = go.Scattergl(
            x=df_annotations[df_annotations[label] == c][x],
            y=df_annotations[df_annotations[label] == c][y],
            mode="markers",
            name=str(c),
            text=df_annotations[df_annotations[label] == c]["file_name"],
            marker=dict(color=colors[i] if colors else None),
            legendgroup=f"legendgroup_{i}" if legendgroup else None,
            **kwargs,
        )
        fig.add_trace(scatter, row=row, col=col)

    if histogram:
        fig.add_histogram(
            x=df_annotations[x],
            name=f"{x} distribution",
            yaxis="y2",
            marker=dict(color="#17becf"),
            histnorm="percent",
            xbins=dict(size=10),
        )
        fig.add_histogram(
            y=df_annotations[y],
            name=f"{y} distribution",
            xaxis="x2",
            marker=dict(color="#17becf"),
            histnorm="percent",
            ybins=dict(size=10),
        )

        fig.layout = dict(
            xaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
            yaxis=dict(domain=[0, 0.85], showgrid=False, zeroline=False),
            xaxis2=dict(
                domain=[0.85, 1], showgrid=False, zeroline=False, range=(0, 100)
            ),
            yaxis2=dict(
                domain=[0.85, 1], showgrid=False, zeroline=False, range=(0, 100)
            ),
        )

    if max_values:
        fig.update_xaxes(title=x, range=[0, max_values[0]])
        fig.update_yaxes(title=y, range=[0, max_values[1]])

    if title is None:
        title = f"{x} vs {y}"
    fig.update_layout(
        title_text=title, xaxis_title=f"{x}", yaxis_title=f"{y}", title_font_size=20
    )

    if show:
        fig.show()

    if output:
        title = title.replace(" ", "_")
        fig.write_image(f"{output}/{title}.png")

    return fig
