from .utils import plot_scatter_with_histograms


def plot_boxes_distribution(
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
    return plot_scatter_with_histograms(
        df_annotations,
        x=x,
        y=y,
        title=title,
        show=show,
        output=output,
        max_values=max_values,
        histogram=histogram,
        label=label,
        colors=colors,
        legendgroup=legendgroup,
        fig=fig,
        row=row,
        col=col,
        **kwargs,
    )
