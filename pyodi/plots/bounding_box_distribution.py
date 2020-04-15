import plotly.graph_objects as go

from loguru import logger


def plot_bounding_box_distribution(df_annotations, show=True, output=None):
    """Annotation Shape Distribution
    
    This plot shows the bounding box height and width distributions of all the annotations in the dataset.

    It can serve as an indicator for setting the optimal **anchor configuration** of your model.
    """
    logger.info("Plotting Bounding Box Distribution")
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=df_annotations[df_annotations["category"] == c]['width'],
                y=df_annotations[df_annotations["category"] == c]['height'],
                mode='markers',
                name=c,
                text=df_annotations[df_annotations["category"] == c]['file_name']
            )
            for c in df_annotations["category"].unique()
        ]
    )
    fig.update_xaxes(title="width", range=[0, 4096])
    fig.update_yaxes(title="height", range=[0, 2160])
    fig.update_layout(title_text="Bounding Box Distribution", title_font_size=20)

    if show:
        fig.show()
    if output:
        fig.write_image(f"{output}/Bounding_Box_Distribution.png") 

    return fig
