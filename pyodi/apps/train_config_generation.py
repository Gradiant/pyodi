"""# Train Config Generation App.

The [`pyodi train-config generation`][pyodi.apps.train_config_generation.train_config_generation]
app can be used to automatically generate a [mmdetection](https://github.com/open-mmlab/mmdetection)
anchor configuration to train your model.

## Procedure
Ground truth boxes are assigned to the anchor base size that has highest Intersection
over Union (IoU) score with them. This step, allow us to locate each ground truth
bounding box in a feature level of the FPN pyramid.

Once this is done, a ratio between the scales of ground truth boxes and the scales of
their associated anchors is computed. A log transform is applied to this ratio between
scales and they are clustered using kmeans algorithm, where the number of obtained
clusters depends on `n_scales` input parameter.

Then a similar procedure is followed to obtain the reference scale ratios of the
dataset, computing log scales ratios of each box and clustering them with number of
clusters equal to `n_ratios`.

Example usage:
``` bash
pyodi train-config generation "data/COCO/COCO_train2017.json"
```

The app allows you to observe two different plots:

## Scale vs Ratio

In this graphic you can distinguish how your bounding boxes log scales and ratios are
distributed. The x axis represent the log scale of the ratio between the bounding box
scales and the scale of their matched anchor base size. The y axis contains the bounding
box log ratios. Centroids are the result of combinating the obtained scales and ratios
obtained with kmeans.


![COCO scale_ratio](../../images/train-config-generation/COCO_scale_vs_ratio.png#center)

See how clusters appear in those areas where box distribution is more dense. For COCO
dataset, most of objects have log relative scales between (-.5, .5). Nevertheless,
aspect ratio log distribution looks quite different and their values are more spread.

We could increase the value of `n_ratios` from three to four, having into account that
this would result in a larger number of anchors that would result in an increase of the
training computational cost.

``` bash
pyodi train-config generation "data/COCO/COCO_train2017.json" --n-ratios 4
```

In plot below we can observe the result for `n_ratios` equal to four.

![COCO scale_ratio](../../images/train-config-generation/COCO_scale_vs_ratio_4.png#center)

## Bounding Box Distribution

This plot is very useful to observe how the generated anchors fit you bounding box
distribution. The number of anchors depends on:

 - The length of `anchor_base_sizes` which determines the number of FPN pyramid levels.
 - A total of `n_ratios` x `n_scales` anchors is generated per level

Therefore the total amount of anchors will be `anchor_base_sizes` x `n_ratios` x `n_scales` .

In figure below, we show how the anchors we previously generated fit COCO bounding box distribution.

![COCO width_height](../../images/train-config-generation/COCO_width_vs_height.png#center)

Note that, although most anchors follow boxes distribution, there is one that lies
outside the allowed sizes. This is due to the meshgrid we create after applying kmeans,
the combination of scale and ratio for that pyramid level result in anchor that is too
large for our actual image input size. This could be solved of multiple ways like
playing with the actual parameter configuration, changing the input image size or using
diferent combination of scales and ratios per pyramid level, which is still not supported.

If we increase once again the number of `n_ratios` we would see how the number of
anchors increase and adapts to the bounding box distribution.

![COCO width_height](../../images/train-config-generation/COCO_width_vs_height.png#center)

By default, [`pyodi train-config evaluation`][pyodi.apps.train_config_evaluation.train_config_evaluation] is
used after the generation of anchors in order to compare which generated anchor config suits better your data.
You can disable this evaluation by setting to False the `evaluate` argument, but it is strongly advised to
use the anchor evaluation module.

"""  # noqa: E501
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import typer
from loguru import logger

from pyodi.apps.train_config_evaluation import train_config_evaluation
from pyodi.core.anchor_generator import AnchorGenerator
from pyodi.core.boxes import (
    filter_zero_area_bboxes,
    get_bbox_array,
    get_scale_and_ratio,
    scale_bbox_dimensions,
)
from pyodi.core.clustering import find_pyramid_level, kmeans_euclidean
from pyodi.core.utils import (
    coco_ground_truth_to_dfs,
    join_annotations_with_image_sizes,
    load_ground_truth_file,
)
from pyodi.plots.clustering import plot_clustering_results

app = typer.Typer()


@logger.catch
@app.command()
def train_config_generation(
    ground_truth_file: str,
    input_size: Tuple[int, int] = (1280, 720),
    n_ratios: int = 3,
    n_scales: int = 3,
    strides: List[int] = [4, 8, 16, 32, 64],
    anchor_base_sizes: List[int] = [32, 64, 128, 256, 512],
    show: bool = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
    keep_ratio: bool = False,
    evaluate: bool = True,
) -> AnchorGenerator:
    """Computes optimal anchors for a given COCO dataset based on iou clustering.

    Args:
        ground_truth_file: Path to COCO ground truth file.
        input_size: Model image input size. Defaults to (1280, 720).
        n_ratios: Number of ratios. Defaults to 3.
        n_scales: Number of scales. Defaults to 3.
        strides: List of strides. Defatults to [4, 8, 16, 32, 64].
        anchor_base_sizes: Basic sizes of anchors in multiple levels. Defaults to
            [32, 64, 128, 256, 512].
        show: Show results or not. Defaults to True.
        output: Output file where results are saved. Defaults to None.
        output_size: Size of saved images. Defaults to (1600, 900).
        keep_ratio: Whether to keep the aspect ratio or not. Defaults to False.
        evaluate: Whether to evaluate or not the anchors. Check
            [`pyodi train-config evaluation`][pyodi.apps.train_config_evaluation.train_config_evaluation]
            for more information.

    Returns:
        Anchor generator instance.
    """
    if output is not None:
        output = str(Path(output) / Path(ground_truth_file).stem)
        Path(output).mkdir(parents=True, exist_ok=True)

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    df_annotations = join_annotations_with_image_sizes(df_annotations, df_images)

    df_annotations = filter_zero_area_bboxes(df_annotations)

    df_annotations = scale_bbox_dimensions(
        df_annotations, input_size=input_size, keep_ratio=keep_ratio
    )

    df_annotations = get_scale_and_ratio(df_annotations, prefix="scaled")

    # Assign fpn level
    df_annotations["fpn_level"] = find_pyramid_level(
        get_bbox_array(df_annotations, prefix="scaled")[:, 2:], anchor_base_sizes
    )

    df_annotations["fpn_level_scale"] = df_annotations["fpn_level"].replace(
        {i: scale for i, scale in enumerate(anchor_base_sizes)}
    )

    df_annotations["level_scale"] = (
        df_annotations["scaled_scale"] / df_annotations["fpn_level_scale"]
    )

    # Normalize to log scale
    df_annotations["log_ratio"] = np.log(df_annotations["scaled_ratio"])
    df_annotations["log_level_scale"] = np.log(df_annotations["level_scale"])

    # Cluster bboxes by scale and ratio independently
    clustering_results = [
        kmeans_euclidean(df_annotations[value], n_clusters=n_clusters)
        for i, (value, n_clusters) in enumerate(
            zip(["log_level_scale", "log_ratio"], [n_scales, n_ratios])
        )
    ]

    # Bring back
    scales = np.e ** clustering_results[0]["centroids"]
    ratios = np.e ** clustering_results[1]["centroids"]

    anchor_generator = AnchorGenerator(
        strides=strides, ratios=ratios, scales=scales, base_sizes=anchor_base_sizes,
    )
    logger.info(f"Anchor configuration: \n{anchor_generator.to_string()}")

    # Plot results
    plot_clustering_results(
        df_annotations,
        anchor_generator,
        show=show,
        output=output,
        output_size=output_size,
        title="COCO_anchor_generation",
    )

    if output:
        output_file = Path(output) / "result.json"
        with open(output_file, "w") as f:
            f.write(anchor_generator.to_string())

    if evaluate:

        train_config = dict(anchor_generator=anchor_generator.to_dict())
        train_config_evaluation(
            ground_truth_file=ground_truth_file,
            train_config=train_config,  # type: ignore
            input_size=input_size,
            show=show,
            output=output,
            output_size=output_size,
        )

    return anchor_generator


if __name__ == "__main__":
    app()
