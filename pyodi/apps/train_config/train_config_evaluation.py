r"""# Train Config Evaluation App.

The [`pyodi train-config evaluation`][pyodi.apps.train_config.train_config_evaluation.train_config_evaluation]
app can be used to evaluate a given [mmdetection](https://github.com/open-mmlab/mmdetection)
Anchor Generator Configuration to train your model using a specific training pipeline.

## Procedure

Training performance of object detection model depends on how well generated anchors
match with ground truth bounding boxes. This simple application provides intuitions
about this, by recreating train preprocessing conditions such as image resizing or
padding, and computing different metrics based on the largest Intersection over Union
(IoU) between ground truth boxes and the provided anchors.

Each bounding box is assigned with the anchor that shares a largest IoU with it. We call
overlap, to the maximum IoU each ground truth box has with the generated anchor set.

Example usage:

```bash
pyodi train-config evaluation \\
$TINY_COCO_ANIMAL/annotations/train.json \\
$TINY_COCO_ANIMAL/resources/anchor_config.py \\
--input-size [1280,720]
```

The app provides four different plots:

![COCO scale_ratio](../../images/train-config-evaluation/overlap.png#center)


## Cumulative Overlap

It shows a cumulative distribution function for the overlap distribution. This view
helps to distinguish which percentage of bounding boxes have a very low overlap with
generated anchors and viceversa.

It can be very useful to determine positive and negative thresholds for your training,
these are the values that determine is a ground truth bounding box will is going to be
taken into account in the loss function or discarded and considered as background.

## Bounding Box Distribution

It shows a scatter plot of bounding box width vs height. The color of each point
represent the overlap value assigned to that bounding box. Thanks to this plot we
can easily observe pattern such low overlap values for large bounding boxes.
We could have this into account and generate larger anchors to improve this matching.

## Scale and Mean Overlap

This plot contains a simple histogram with bins of similar scales and its mean overlap
value. It help us to visualize how overlap decays when scale increases, as we said
before.

## Log Ratio and Mean Overlap

Similarly to previous plot, it shows an histogram of bounding box log ratios and its
mean overlap values. It is useful to visualize this relation and see how certain box
ratios might be having problems to match with generated anchors. In this example, boxes
with negative log ratios, where width is much larger than height, overlaps are very
small. See how this matches with patterns observed in bounding box distribution plot,
where all boxes placed near to x axis, have low overlaps.

---

# API REFERENCE
"""  # noqa: E501
import sys
from importlib import import_module
from os import path as osp
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

from pyodi.core.anchor_generator import AnchorGenerator
from pyodi.core.boxes import (
    filter_zero_area_bboxes,
    get_bbox_array,
    get_scale_and_ratio,
    scale_bbox_dimensions,
)
from pyodi.core.clustering import get_max_overlap
from pyodi.core.utils import (
    coco_ground_truth_to_dfs,
    join_annotations_with_image_sizes,
    load_ground_truth_file,
)
from pyodi.plots.evaluation import plot_overlap_result


def load_anchor_config_file(anchor_config_file: str) -> Dict[str, Any]:
    """Loads the `anchor_config_file`.

    Args:
        anchor_config_file: File with the anchor configuration.

    Returns:
        Dictionary with the training configuration.

    """
    logger.info("Loading Train Config File")
    with TemporaryDirectory() as temp_config_dir:
        copyfile(anchor_config_file, osp.join(temp_config_dir, "_tempconfig.py"))
        sys.path.insert(0, temp_config_dir)
        mod = import_module("_tempconfig")
        sys.path.pop(0)
        train_config = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__")
        }
        # delete imported module
        del sys.modules["_tempconfig"]
    return train_config


@logger.catch
def train_config_evaluation(
    ground_truth_file: str,
    anchor_config: str,
    input_size: Tuple[int, int] = (1280, 720),
    show: bool = True,
    output: Optional[str] = None,
    output_size: Tuple[int, int] = (1600, 900),
) -> None:
    """Evaluates the fitness between `ground_truth_file` and `anchor_config_file`.

    Args:
        ground_truth_file: Path to COCO ground truth file.
        anchor_config: Path to MMDetection-like `anchor_generator` section. It can also be a
            dictionary with the required data.
        input_size: Model image input size. Defaults to (1333, 800).
        show: Show results or not. Defaults to True.
        output: Output file where results are saved. Defaults to None.
        output_size: Size of saved images. Defaults to (1600, 900).

    Examples:
        ```python
        # faster_rcnn_r50_fpn.py:
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        )
        ```
    """
    if output is not None:
        output = str(Path(output) / Path(ground_truth_file).stem)
        Path(output).mkdir(parents=True, exist_ok=True)

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    df_annotations = join_annotations_with_image_sizes(df_annotations, df_images)

    df_annotations = filter_zero_area_bboxes(df_annotations)

    df_annotations = scale_bbox_dimensions(df_annotations, input_size=input_size)

    df_annotations = get_scale_and_ratio(df_annotations, prefix="scaled")

    df_annotations["log_scaled_ratio"] = np.log(df_annotations["scaled_ratio"])

    if isinstance(anchor_config, str):
        anchor_config_data = load_anchor_config_file(anchor_config)
    elif isinstance(anchor_config, dict):
        anchor_config_data = anchor_config
    else:
        raise ValueError("anchor_config must be string or dictionary.")

    anchor_config_data["anchor_generator"].pop("type", None)
    anchor_generator = AnchorGenerator(**anchor_config_data["anchor_generator"])

    if isinstance(anchor_config, str):
        logger.info(anchor_generator.to_string())

    width, height = input_size
    featmap_sizes = [
        (width // stride, height // stride) for stride in anchor_generator.strides
    ]
    anchors_per_level = anchor_generator.grid_anchors(featmap_sizes=featmap_sizes)

    bboxes = get_bbox_array(
        df_annotations, prefix="scaled", output_bbox_format="corners"
    )

    overlaps = np.zeros(bboxes.shape[0])
    max_overlap_level = np.zeros(bboxes.shape[0])

    logger.info("Computing overlaps between anchors and ground truth ...")
    for i, anchor_level in enumerate(anchors_per_level):
        level_overlaps = get_max_overlap(
            bboxes.astype(np.float32), anchor_level.astype(np.float32)
        )
        max_overlap_level[level_overlaps > overlaps] = i
        overlaps = np.maximum(overlaps, level_overlaps)

    df_annotations["overlaps"] = overlaps
    df_annotations["max_overlap_level"] = max_overlap_level

    logger.info("Plotting results ...")
    plot_overlap_result(
        df_annotations, show=show, output=output, output_size=output_size
    )
