import plotly.graph_objects as go
import streamlit as st


def plot_bounding_box_distribution(df_annotations):
    """Annotation Shape Distribution
    
    This plot shows the bounding box height and width distributions of all the annotations in the dataset.

    It can serve as an indicator for setting the optimal **anchor configuration** of your model.
    """
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

    return fig
