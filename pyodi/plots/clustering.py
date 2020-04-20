import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots
from plots.annotations import plot_scatter_with_histograms


def plot_clustering_results(centroids, df_annotations, show=True, output=None):
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["Width vs Height", "Area vs Ratio"]
    )

    fig1 = plot_scatter_with_histograms(
        df_annotations, x="scaled_width", y="scaled_height", show=False, label="cluster"
    )
    for data in fig1.data:
        fig.append_trace(data, row=1, col=1)
    fig.append_trace(
        go.Scattergl(
            x=centroids["width"],
            y=centroids["height"],
            mode="markers",
            marker=dict(size=15),
        ),
        row=1,
        col=1,
    )

    fig2 = plot_scatter_with_histograms(
        df_annotations, x="scaled_area", y="scaled_ratio", show=False, label="cluster"
    )
    for data in fig2.data:
        fig.append_trace(data, row=1, col=2)
    fig.append_trace(
        go.Scattergl(
            x=centroids["area"],
            y=centroids["ratio"],
            mode="markers",
            marker=dict(size=15),
        ),
        row=1,
        col=2,
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
