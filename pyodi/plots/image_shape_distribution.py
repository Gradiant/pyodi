import plotly.graph_objects as go

import streamlit as st


def plot_image_shape_distribution(df_images):
    """Image Shape Distribution

    This plot shows the height and width distributions of all the images in the dataset.

    It can serve as an indicator for setting the optimal **input size** and **aspect ratio** of your model.
    
    """
    fig = go.Figure(
        data=[
            go.Box(
                y=df_images[x], 
                boxpoints='all',
                boxmean='sd',
                name=x,
                hovertext=df_images["file_name"])
            for x in ["height", "width"]
        ]
    )
    fig.update_yaxes(range=[0, 8192])
    return fig