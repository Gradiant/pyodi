"""
# Ground Truth App

The [`pyodi ground-truth`][pyodi.apps.ground_truth.ground_truth] app can be used to explore the images and bounding boxes that compose an object detection dataset.

The shape distribution of the images and bounding boxes is one of the key aspects to take in account when setting your training
configuration.

Example usage:

``` bash
pyodi ground-truth "data/COCO/COCO_train2017.json"
```

The app is divided in two sections:

## Images

[`Images Plots Reference`][pyodi.plots.images]

![COCO Image Shapes](../../images/COCO_image_shapes.png)
![Drone-vs-Bird Image Shapes](../../images/Drone-vs-Bird_image_shapes.png)
![TinyPerson Image Shapes](../../images/TinyPerson_image_shapes.png)


## Bounding Boxes

[`Bounding Boxes Plots Reference`][pyodi.plots.boxes]

![COCO Bounding Box Shapes](../../images/COCO_bounding_box_shapes.png)
![Drone-vs-Bird Bounding Box Shapes](../../images/Drone-vs-Bird_bounding_box_shapes.png)
![TinyPerson Boudning Box Shapes](../../images/TinyPerson_bounding_box_shapes.png)



# API REFERENCE
"""
from pathlib import Path
from typing import Optional, Tuple

import typer
from loguru import logger

from pyodi.coco.utils import (
    coco_ground_truth_to_dfs,
    join_annotations_with_image_sizes,
    load_ground_truth_file,
)
from pyodi.plots.boxes import get_centroids_heatmap, plot_heatmap
from pyodi.plots.utils import plot_scatter_with_histograms

app = typer.Typer()


@logger.catch
@app.command()
def ground_truth(
    ground_truth_file: str,
    show: bool = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
) -> None:
    """Explore the images and bounding boxes of a dataset.

    Parameters
    ----------
    ground_truth_file : str
        Path to COCO ground truth file

    show : bool, optional
        Default: True.
        Whether to show results or not.

    output : str, optional
        Default: None
        If not None, results will be saved under `output` dir.

    output_size : tuple
        Default: (1600, 900)
        Size of the saved images when output is defined.
    """
    if output is not None:
        output = str(Path(output) / Path(ground_truth_file).stem)
        Path(output).mkdir(parents=True, exist_ok=True)

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    plot_scatter_with_histograms(
        df_images,
        title=f"{Path(ground_truth_file).stem}: Image Shapes",
        show=show,
        output=output,
        output_size=output_size,
    )

    df_annotations = join_annotations_with_image_sizes(df_annotations, df_images)
    df_annotations["absolute_height"] = (
        df_annotations["height"] / df_annotations["img_height"]
    )
    df_annotations["absolute_width"] = (
        df_annotations["width"] / df_annotations["img_width"]
    )

    plot_heatmap(
        get_centroids_heatmap(df_annotations),
        title=f"{Path(ground_truth_file).stem}: Bounding Box Centers",
        show=show,
        output=output,
        output_size=output_size,
    )

    plot_scatter_with_histograms(
        df_annotations,
        x="absolute_width",
        y="absolute_height",
        max_values=(1.01, 1.01),
        title=f"{Path(ground_truth_file).stem}: Bounding Box Shapes",
        show=show,
        output=output,
        output_size=output_size,
    )


if __name__ == "__main__":
    app()
