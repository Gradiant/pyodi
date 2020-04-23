import plotly.graph_objects as go
from loguru import logger
from plotly.colors import DEFAULT_PLOTLY_COLORS as COLORS
from plotly.subplots import make_subplots

from pyodi.plots.annotations import plot_scatter_with_histograms


def plot_clustering_results(centroids, df_annotations, show=True, output=None):
    """Plots cluster results in two different views, width vs heihgt and area vs ratio.

    Parameters
    ----------
    centroids : pd.DataFrame
        Coco format dataframe with centroid coordinates, only width, height, area and ratio are used
    df_annotations : pd.DataFrame
        COCO annotations generated dataframe
    show : bool, optional
        If true plotly figure will be shown, by default True
    output : str, optional
        Output image folder, by default None
    """

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["Width vs Height", "Area vs Ratio"]
    )

    for col, (x, y) in enumerate(zip(("width", "area"), ("height", "ratio")), 1):
        # Bboxes distribution
        subplot = plot_scatter_with_histograms(
            df_annotations,
            x=f"scaled_{x}",
            y=f"scaled_{y}",
            show=False,
            label="cluster",
            colors=COLORS,
            legendgroup="Cluster",
        )
        # Centroids
        for i, data in enumerate(subplot.data):
            fig.append_trace(data, row=1, col=col)
            fig.append_trace(
                go.Scattergl(
                    x=[centroids.iloc[i][x]],
                    y=[centroids.iloc[i][y]],
                    mode="markers",
                    legendgroup=f"legendgroup_{i}",
                    name=str(i),
                    showlegend=col == 1,
                    marker=dict(
                        size=15,
                        color=COLORS[i],
                        line=dict(width=2, color="DarkSlateGrey"),
                    ),
                ),
                row=1,
                col=col,
            )

    fig["layout"].update(
        title="Anchor cluster visualization",
        xaxis=dict(title="Scaled width"),
        xaxis2=dict(title="Area"),
        yaxis=dict(title="Scaled height"),
        yaxis2=dict(title="Ratio"),
    )

    if show:
        fig.show()

    if output:
        fig.write_image(f"{output}/clusters.png")
