r"""# Paint Annotations App.

The [`pyodi paint-annotations`][pyodi.apps.paint_annotations.paint_annotations]
helps you to easily visualize in a beautiful format your object detection dataset.
You can also use this function to visualize model predictions if they are in COCO predictions format.

Example usage:

```bash
pyodi paint-annotations \\
$TINY_COCO_ANIMAL/annotations/train.json \\
$TINY_COCO_ANIMAL/sample_images/ \\
$TINY_COCO_ANIMAL/painted_images/
```

![COCO image with painted annotations](../../images/coco_sample_82680.jpg)

---

# API REFERENCE
"""  # noqa: E501

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image


@logger.catch
def paint_annotations(
    ground_truth_file: str,
    image_folder: str,
    output_folder: str,
    predictions_file: Optional[str] = None,
    score_thr: float = 0.0,
    color_key: str = "category_id",
    show_label: bool = True,
    filter_crowd: bool = True,
) -> None:
    """Paint `ground_truth_file` or `predictions_file` annotations on `image_folder` images.

    Args:
        ground_truth_file: Path to COCO ground truth file.
        image_folder: Path to root folder where the images of `ground_truth_file` are.
        output_folder: Path to the folder where painted images will be saved.
            It will be created if it does not exist.
        predictions_file: Path to COCO predictions file.
            If not None, the annotations of predictions_file will be painted instead of ground_truth_file's.
        score_thr: Detections bellow this threshold will not be painted.
            Default 0.0.
        color_key: Choose the key in annotations on which the color will depend. Defaults to 'category_id'.
        show_label: Choose whether to show label and score threshold on image. Default True.
        filter_crowd: Filter out crowd annotations or not. Default True.
    """
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    ground_truth = json.load(open(ground_truth_file))
    image_name_to_id = {
        Path(x["file_name"]).stem: x["id"] for x in ground_truth["images"]
    }

    category_id_to_label = {
        cat["id"]: cat["name"] for cat in ground_truth["categories"]
    }
    image_id_to_annotations: Dict = defaultdict(list)
    if predictions_file is not None:
        with open(predictions_file) as pred:
            annotations = json.load(pred)
    else:
        annotations = ground_truth["annotations"]

    n_colors = len(set(ann[color_key] for ann in annotations))
    colormap = cm.rainbow(np.linspace(0, 1, n_colors))

    for annotation in annotations:
        if not (filter_crowd and annotation.get("iscrowd", False)):
            image_id_to_annotations[annotation["image_id"]].append(annotation)

    image_data = ground_truth["images"]

    for image in image_data:

        image_filename = image["file_name"]
        image_id = image["id"]
        image_path = Path(image_folder) / image_filename

        logger.info(f"Loading {image_filename}")

        if Path(image_filename).stem not in image_name_to_id:
            logger.warning(f"{image_filename} not in ground_truth_file")

        if image_path.is_file():
            image_pil = Image.open(image_path)

            width, height = image_pil.size
            fig = plt.figure(frameon=False, figsize=(width / 96, height / 96))

            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(image_pil, aspect="auto")

            polygons = []
            colors = []

            for annotation in image_id_to_annotations[image_id]:

                score = annotation.get("score", 1)
                if score < score_thr:
                    continue
                bbox_left, bbox_top, bbox_width, bbox_height = annotation["bbox"]

                cat_id = annotation["category_id"]
                label = category_id_to_label[cat_id]
                color_id = annotation[color_key]
                color = colormap[color_id % len(colormap)]

                poly = [
                    [bbox_left, bbox_top],
                    [bbox_left, bbox_top + bbox_height],
                    [bbox_left + bbox_width, bbox_top + bbox_height],
                    [bbox_left + bbox_width, bbox_top],
                ]
                polygons.append(Polygon(poly))
                colors.append(color)

                if show_label:
                    label_text = f"{label}"
                    if predictions_file is not None:
                        label_text += f": {score:.2f}"

                    ax.text(
                        bbox_left,
                        bbox_top,
                        label_text,
                        va="top",
                        ha="left",
                        bbox=dict(facecolor="white", edgecolor=color, alpha=0.5, pad=0),
                    )

            p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.3)
            ax.add_collection(p)

            p = PatchCollection(
                polygons, facecolor="none", edgecolors=colors, linewidths=1
            )
            ax.add_collection(p)

            filename = Path(image_filename).stem
            file_extension = Path(image_filename).suffix
            output_file = Path(output_folder) / f"{filename}_result{file_extension}"
            logger.info(f"Saving {output_file}")

            plt.savefig(output_file)
            plt.close()
