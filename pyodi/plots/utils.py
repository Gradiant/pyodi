import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from plotly.subplots import make_subplots


def plot_with_streamlit(plot_function, inputs):
    info = plot_function.__doc__
    lines = info.split("\n")
    title = lines[0]
    description = "\n".join(lines[1:])
    figure = plot_function(inputs)
    figure.update_layout(title_text=title, title_font_size=20)
    st.write(figure)
    show_info = st.checkbox(f"Show Info: {title}")
    if show_info:
        st.write(description)


def optional_show_df(df, name):
    st.header(f"{name}")
    show_image_dataframe = st.checkbox(f"Show {name} DataFrame")
    if show_image_dataframe:
        logger.info(f"Showing DataFrame: {name}")
        st.write(df)


def plot_scatter_with_histograms(
    df,
    x="width",
    y="height",
    title=None,
    show=True,
    output=None,
    output_size=(1600, 900),
    max_values=(1, 1),
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
    max_values : tuple, optional
        x,y max allowed values in representation, by default None
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
            xaxis=dict(
                domain=[0, 0.84],
                showgrid=False,
                zeroline=False,
                range=[-0.01, max_values[0]],
            ),
            yaxis=dict(
                domain=[0, 0.83],
                showgrid=False,
                zeroline=False,
                range=[-0.01, max_values[1]],
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
        title = title.replace(" ", "_")
        fig.update_layout(width=output_size[0], height=output_size[1])
        fig.write_image(f"{output}/{title}.png")

    return fig
