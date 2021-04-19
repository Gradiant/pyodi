r"""# Train Config Generation App.

The [`pyodi train-config generation`][pyodi.apps.train_config.train_config_generation.train_config_generation]
app can be used to automatically generate a [mmdetection](https://github.com/open-mmlab/mmdetection)
anchor configuration to train your model.

The design of anchors is critical for the performance of one-stage detectors. Usually, published models
such [Faster R-CNN](https://arxiv.org/abs/1506.01497) or [RetinaNet](https://arxiv.org/abs/1708.02002)
include default anchors which has been designed to work with general object detection purpose as COCO dataset.
Nevertheless, you might be envolved in different problems which data contains only a few different classes that
share similar properties, as the object sizes or shapes, this would be the case for a drone detection dataset
such [Drone vs Bird](https://wosdetc2020.wordpress.com/). You can exploit this knowledge by designing anchors
that specially fit the distribution of your data, optimizing the probability of matching ground truth bounding
boxes with generated anchors, which can result in an increase in the performance of your model. At the same time,
you can reduce the number of anchors you use to boost inference and training time.

## Procedure

The input size parameter determines the model input size and automatically reshapes images and annotations sizes to it.
Ground truth boxes are assigned to the anchor base size that has highest Intersection
over Union (IoU) score with them. This step, allow us to locate each ground truth
bounding box in a feature level of the FPN pyramid.

Once this is done, the ratio between the scales of ground truth boxes and the scales of
their associated anchors is computed. A log transform is applied to it and they are clustered
using kmeans algorithm, where the number of obtained clusters depends on `n_scales` input parameter.

After this step, a similar procedure is followed to obtain the reference scale ratios of the
dataset, computing log scales ratios of each box and clustering them with number of
clusters equal to `n_ratios`.

Example usage:
```bash
pyodi train-config generation \\
$TINY_COCO_ANIMAL/annotations/train.json \\
--input-size [1280,720] \\
--n-ratios 3 --n-scales 3
```

The app shows two different plots:

![Anchor clustering plot](../../images/train-config-generation/clusters.png#center)


## Log Relative Scale vs Log Ratio

In this graphic you can distinguish how your bounding boxes scales and ratios are
distributed. The x axis represent the log scale of the ratio between the bounding box
scales and the scale of their matched anchor base size. The y axis contains the bounding
box log ratios. Centroids are the result of combinating the obtained scales and ratios
obtained with kmeans. We can see how clusters appear in those areas where box distribution is more dense.

We could increase the value of `n_ratios` from three to four, having into account that
the number of anchors is goint to increase, which will influence training computational cost.

```bash
pyodi train-config generation annotations/train.json --input-size [1280,720] --n-ratios 4 --n-scales 3
```

In plot below we can observe the result for `n_ratios` equal to four.

![Anchor clustering plot 4 ratios](../../images/train-config-generation/clusters_4_ratios.png#center)

## Bounding Box Distribution

This plot is very useful to observe how the generated anchors fit you bounding box
distribution. The number of anchors depends on:

 - The length of `base_sizes` which determines the number of FPN pyramid levels.
 - A total of `n_ratios` x `n_scales` anchors is generated per level

We can now increase the number of `n_scales` and observe the effect on the bounding box distribution plot.

![Anchor clustering plot 4 scales](../../images/train-config-generation/clusters_4_scales.png#center)


Proposed anchors are also attached in a Json file that follows
[mmdetection anchors](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L10) format:

```python
anchor_generator=dict(
    type='AnchorGenerator',
    scales=[1.12, 3.13, 8.0],
    ratios=[0.33, 0.67, 1.4],
    strides=[4, 8, 16, 32, 64],
    base_sizes=[4, 8, 16, 32, 64],
)
```

By default, [`pyodi train-config evaluation`][pyodi.apps.train_config.train_config_evaluation.train_config_evaluation] is
used after the generation of anchors in order to compare which generated anchor config suits better your data.
You can disable this evaluation by setting to False the `evaluate` argument, but it is strongly advised to
use the anchor evaluation module.


---

# API REFERENCE
"""  # noqa: E501
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

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


@logger.catch
def train_config_generation(
    ground_truth_file: str,
    input_size: Tuple[int, int] = (1280, 720),
    n_ratios: int = 3,
    n_scales: int = 3,
    strides: Optional[List[int]] = None,
    base_sizes: Optional[List[int]] = None,
    show: bool = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
    keep_ratio: bool = False,
) -> AnchorGenerator:
    """Computes optimal anchors for a given COCO dataset based on iou clustering.

    Args:
        ground_truth_file: Path to COCO ground truth file.
        input_size: Model image input size. Defaults to (1280, 720).
        n_ratios: Number of ratios. Defaults to 3.
        n_scales: Number of scales. Defaults to 3.
        strides: List of strides. Defatults to [4, 8, 16, 32, 64].
        base_sizes: The basic sizes of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
        show: Show results or not. Defaults to True.
        output: Output file where results are saved. Defaults to None.
        output_size: Size of saved images. Defaults to (1600, 900).
        keep_ratio: Whether to keep the aspect ratio or not. Defaults to False.

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

    if strides is None:
        strides = [4, 8, 16, 32, 64]
    if base_sizes is None:
        base_sizes = strides

    # Assign fpn level
    df_annotations["fpn_level"] = find_pyramid_level(
        get_bbox_array(df_annotations, prefix="scaled")[:, 2:], base_sizes
    )

    df_annotations["fpn_level_scale"] = df_annotations["fpn_level"].replace(
        {i: scale for i, scale in enumerate(base_sizes)}
    )

    df_annotations["level_scale"] = (
        df_annotations["scaled_scale"] / df_annotations["fpn_level_scale"]
    )

    # Normalize to log scale
    df_annotations["log_ratio"] = np.log(df_annotations["scaled_ratio"])
    df_annotations["log_level_scale"] = np.log(df_annotations["level_scale"])

    # Cluster bboxes by scale and ratio independently
    clustering_results = [
        kmeans_euclidean(df_annotations[value].to_numpy(), n_clusters=n_clusters)
        for i, (value, n_clusters) in enumerate(
            zip(["log_level_scale", "log_ratio"], [n_scales, n_ratios])
        )
    ]

    # Bring back from log scale
    scales = np.e ** clustering_results[0]["centroids"]
    ratios = np.e ** clustering_results[1]["centroids"]

    anchor_generator = AnchorGenerator(
        strides=strides, ratios=ratios, scales=scales, base_sizes=base_sizes,
    )
    logger.info(f"Anchor configuration: \n{anchor_generator.to_string()}")

    plot_clustering_results(
        df_annotations,
        anchor_generator,
        show=show,
        output=output,
        output_size=output_size,
        title="COCO_anchor_generation",
    )

    if output:
        output_file = Path(output) / "anchor_config.py"
        with open(output_file, "w") as f:
            f.write(anchor_generator.to_string())

    return anchor_generator
