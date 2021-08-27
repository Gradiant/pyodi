r"""# Ground Truth App.

The [`pyodi ground-truth`][pyodi.apps.ground_truth.ground_truth] app can be used to
explore the images and bounding boxes that compose an object detection dataset.

The shape distribution of the images and bounding boxes and their locations are the key
aspects to take in account when setting your training configuration.

Example usage:

```bash
pyodi ground-truth \\
$TINY_COCO_ANIMAL/annotations/train.json
```

The app is divided in three different sections:

## Images shape distribution

Shows information related with the shape of the images present in the dataset.
In this case we can clearly identify two main patterns in this dataset and if
we have a look at the histogram, we can see how most of images have 640 pixels
width, while as height is more distributed between different values.

![COCO Animals Image Shapes](../../images/ground_truth/gt_img_shapes.png)

## Bounding Boxes shape distribution

We observe bounding box distribution, with the possibility of enabling
filters by class or sets of classes. This dataset shows a tendency to  rectangular
bounding boxes with larger width than height and where most of them embrace
areas below the 20% of the total image.

![Bbox distribution](../../images/ground_truth/gt_bb_shapes.png)

## Bounding Boxes center locations

It is possible to check where centers of bounding boxes are most commonly
found with respect to the image. This can help us distinguish ROIs in input images.
In this case we observe that the objects usually appear in the center of the image.

![Bbox center distribution](../../images/ground_truth/gt_bb_centers.png)

---

# API REFERENCE
"""  # noqa: E501
from pathlib import Path
from typing import Optional, Tuple

from pyodi.core.boxes import add_centroids
from pyodi.core.utils import coco_ground_truth_to_df
from pyodi.plots.boxes import get_centroids_heatmap, plot_heatmap
from pyodi.plots.common import plot_scatter_with_histograms


def ground_truth(
    ground_truth_file: str,
    show: bool = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
) -> None:
    """Explore the images and bounding boxes of a dataset.

    Args:
        ground_truth_file: Path to COCO ground truth file.
        show: Whether to show results or not. Defaults to True.
        output: Results will be saved under `output` dir. Defaults to None.
        output_size: Size of the saved images when output is defined. Defaults to
            (1600, 900).

    """
    if output is not None:
        output = str(Path(output) / Path(ground_truth_file).stem)
        Path(output).mkdir(parents=True, exist_ok=True)

    df_annotations = coco_ground_truth_to_df(ground_truth_file)

    df_images = df_annotations.loc[
        :, df_annotations.columns.str.startswith("img_")
    ].drop_duplicates()

    plot_scatter_with_histograms(
        df_images,
        x="img_width",
        y="img_height",
        title="Image_Shapes",
        show=show,
        output=output,
        output_size=output_size,
        histogram_xbins=dict(size=10),
        histogram_ybins=dict(size=10),
    )

    df_annotations = add_centroids(df_annotations)

    df_annotations["absolute_height"] = (
        df_annotations["height"] / df_annotations["img_height"]
    )
    df_annotations["absolute_width"] = (
        df_annotations["width"] / df_annotations["img_width"]
    )

    plot_scatter_with_histograms(
        df_annotations,
        x="absolute_width",
        y="absolute_height",
        title="Bounding_Box_Shapes",
        show=show,
        output=output,
        output_size=output_size,
        xaxis_range=(-0.01, 1.01),
        yaxis_range=(-0.01, 1.01),
        histogram_xbins=dict(size=0.05),
        histogram_ybins=dict(size=0.05),
    )

    plot_heatmap(
        get_centroids_heatmap(df_annotations),
        title="Bounding_Box_Centers",
        show=show,
        output=output,
        output_size=output_size,
    )
