import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import typer
from loguru import logger
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image

app = typer.Typer()


@logger.catch
@app.command()
def paint_annotations(
    ground_truth_file: str,
    image_folder: str,
    output_folder: str,
    predictions_file: Optional[str] = None,
    show_score_thr: float = 0.0,
    color_key: str = "category_id",
) -> None:
    """Paint `ground_truth_file` or `predictions_file` annotations on `image_folder` images.

    Args:
        ground_truth_file: Path to COCO ground truth file.
        image_folder: Path to root folder where the images of `ground_truth_file` are.
        output_folder: Path to the folder where painted images will be saved.
            It will be created if it does not exist.
        predictions_file: Path to COCO predictions file.
            If not None, the annotations of predictions_file will be painted instead of ground_truth_file's.
        show_score_thr: Detections bellow this threshold will not be painted.
            Default 0.0.
        color_key: Choose the key in annotations on which the color will depend. Defaults to 'category_id'.
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
        annotations = json.load(open(predictions_file))
    else:
        annotations = ground_truth["annotations"]

    n_colors = len(set(ann[color_key] for ann in annotations))
    colormap = cm.rainbow(np.linspace(0, 1, n_colors))

    for annotation in annotations:
        image_id_to_annotations[annotation["image_id"]].append(annotation)

    img_data = ground_truth["images"]
    bbox_data = ground_truth["annotations"]

    for img in img_data:

        image = img.get("file_name")
        logger.info(f"Loading {image}")

        if image not in image_name_to_id:
            logger.warning(f"{image} not in ground_truth_file")

        if os.path.isfile(image_folder + "/" + image):

            image_pil = Image.open(image_folder + "/" + image)

            width, height = image_pil.size
            fig = plt.figure(frameon=False, figsize=(width / 80, height / 80))
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(image_pil, aspect="auto")

            polygons = []
            colors = []

            # Filter the list of annotations and receive a list with only the desired element
            bbox_dict = list(
                filter(lambda bbox: bbox["image_id"] == img.get("id"), bbox_data)
            )

            if not bbox_dict:
                logger.warning(f"No bbox found at {image}")
                continue
            bbox_left, bbox_top, bbox_width, bbox_height = bbox_dict[0].get("bbox")
            cat_id = bbox_dict[0].get("category_id")
            label = category_id_to_label[cat_id]
            color_id = annotation[color_key]
            color = colormap[color_id % len(colormap)]
            score = bbox_dict[0].get("score")

            poly = [
                [bbox_left, bbox_top],
                [bbox_left, bbox_top + bbox_height],
                [bbox_left + bbox_width, bbox_top + bbox_height],
                [bbox_left + bbox_width, bbox_top],
            ]

            ax.text(
                bbox_left,
                bbox_top,
                f"{label}: {score:.2f}",
                va="top",
                ha="left",
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.5, pad=0),
            )

            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
            colors.append(color)

            p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.3)
            ax.add_collection(p)

            p = PatchCollection(
                polygons, facecolor="none", edgecolors=colors, linewidths=1
            )
            ax.add_collection(p)

            filename, file_extension = os.path.splitext(image)
            filename = filename.split("/")
            output_file = Path(output_folder) / f"{filename[1]}_result{file_extension}"
            logger.info(f"Saving {output_file}")
            plt.savefig(output_file)
            fig.close()
            plt.close()


if __name__ == "__main__":
    app()
