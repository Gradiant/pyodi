from pathlib import Path
from typing import Optional, Tuple

import typer

from loguru import logger

from pyodi.coco.utils import (
    coco_ground_truth_to_dfs,
    get_area_and_ratio,
    join_annotations_with_image_sizes,
    load_ground_truth_file,
    scale_bbox_dimensions,
)
from pyodi.plots.annotations import plot_scatter_with_histograms


app = typer.Typer()


@logger.catch
@app.command()
def train_config(
    ground_truth_file: str,
    show: bool = True,
    output: Optional[str] = None,
    input_size: Tuple[int, int] = (1280, 720)):
    """[summary]
    Parameters
    ----------
    ground_truth_file : str
        Path to COCO ground truth file
    show : bool, optional
        Show results or not, by default True
    output : str, optional
        Output file where results are saved, by default None
    input_size : tuple, optional
        Model image input size, by default (1280, 720)
    """

    if output is not None:
        output = Path(output) / Path(ground_truth_file).name

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    df_annotations = join_annotations_with_image_sizes(df_annotations, df_images)

    df_annotations = scale_bbox_dimensions(df_annotations, input_size=input_size)

    df_annotations = get_area_and_ratio(df_annotations, prefix="scaled")

    plot_scatter_with_histograms(
        df_annotations,
        x="scaled_area",
        y="scaled_ratio",
        title="Bounding box area vs Aspect ratio",
        show=True,
        histogram=True,
    )


if __name__ == "__main__":
    app()
