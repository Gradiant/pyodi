import json
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
    annotations_data = ground_truth["annotations"]

    bbox_data: Dict = defaultdict(list)
    for data in annotations_data:
        bbox_data[data["image_id"]].append(data)

    for img in img_data:

        image = img["file_name"]
        logger.info(f"Loading {image}")

        if image not in image_name_to_id:
            logger.warning(f"{image} not in ground_truth_file")

        if (Path(image_folder) / image).is_file():
            image_pil = Image.open((Path(image_folder) / image))

            width, height = image_pil.size
            fig = plt.figure(frameon=False, figsize=(width / 80, height / 80))
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(image_pil, aspect="auto")

            polygons = []
            colors = []

            if img["id"] not in bbox_data.keys():
                logger.warning(f"No bbox found at {image}")
                continue

            for v in range(len(bbox_data[img["id"]])):

                bbox_left, bbox_top, bbox_width, bbox_height = bbox_data[img["id"]][v][
                    "bbox"
                ]

                cat_id = bbox_data[img["id"]][v]["category_id"]
                label = category_id_to_label[cat_id]
                color_id = annotation[color_key]
                color = colormap[color_id % len(colormap)]
                score = bbox_data[img["id"]][v]["score"]

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

            filename = Path(image).stem
            file_extension = Path(image).suffix
            output_file = Path(output_folder) / f"{filename}_result{file_extension}"
            logger.info(f"Saving {output_file}")
            plt.savefig(output_file)
            plt.close()


if __name__ == "__main__":
    app()
