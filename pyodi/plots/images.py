import plotly.graph_objects as go
from loguru import logger


def plot_image_shape_distribution(df_images, show=True, output=None):
    """Image Shape Distribution

    This plot shows the height and width distributions of all the images in the dataset.

    It can serve as an indicator for setting the optimal **input size** and **aspect ratio** of your model.

    """
    logger.info("Plotting Image Shape Distribution")
    fig = go.Figure(
        data=[
            go.Box(
                y=df_images[x],
                boxpoints="all",
                boxmean="sd",
                name=x,
                hovertext=df_images["file_name"],
            )
            for x in ["height", "width"]
        ]
    )
    fig.update_yaxes(range=[0, 8192])
    fig.update_layout(title_text="Image Shape Distribution", title_font_size=20)

    if show:
        fig.show()
    if output:
        fig.write_image(f"{output}/Image_Shape_Distribution.png")

    return fig
