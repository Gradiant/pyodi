from pathlib import Path
from typing import Optional

import typer

from loguru import logger

from pyodi.coco.utils import coco_ground_truth_to_dfs, load_ground_truth_file
from pyodi.plots.annotations import plot_scatter_with_histograms
from pyodi.plots.images import plot_image_shape_distribution


app = typer.Typer()


@logger.catch
@app.command()
def ground_truth(
    ground_truth_file: str, 
    show: bool = True,
    output: Optional[str] = None):

    if output is not None:
        output = Path(output) / Path(ground_truth_file).name

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    plot_image_shape_distribution(df_images, show=show, output=output)

    plot_scatter_with_histograms(
        df_annotations, x="width", y="height", show=show, output=output,
    )

if __name__ == "__main__":
    app()
