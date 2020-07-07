import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import typer
from loguru import logger
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

    """
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    ground_truth = json.load(open(ground_truth_file))
    image_name_to_id = {
        Path(x["file_name"]).stem: x["id"] for x in ground_truth["images"]
    }
    image_id_to_annotations: Dict = defaultdict(list)

    if predictions_file is not None:
        annotations = json.load(open(predictions_file))
    else:
        annotations = ground_truth["annotations"]

    for annotation in annotations:
        image_id_to_annotations[annotation["image_id"]].append(annotation)

    for image in Path(image_folder).iterdir():
        logger.info(f"Loading {image}")

        if image.stem not in image_name_to_id:
            logger.warning(f"{image.stem} not in ground_truth_file")
            continue

        image_pil = Image.open(image)

        width, height = image_pil.size
        fig = plt.figure(frameon=False, figsize=(width / 80, height / 80))

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(image_pil, aspect="auto")

        polygons = []
        colors = []
        image_id = image_name_to_id[image.stem]
        annotations = image_id_to_annotations[image_id]
        for annotation in annotations:
            if annotation["score"] < show_score_thr:
                continue

            bbox_left, bbox_top, bbox_width, bbox_height = annotation["bbox"]

            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = [
                [bbox_left, bbox_top],
                [bbox_left, bbox_top + bbox_height],
                [bbox_left + bbox_width, bbox_top + bbox_height],
                [bbox_left + bbox_width, bbox_top],
            ]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
            colors.append(c)

        p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.3)
        ax.add_collection(p)

        p = PatchCollection(polygons, facecolor="none", edgecolors=colors, linewidths=1)
        ax.add_collection(p)

        output_file = Path(output_folder) / f"{image.stem}_result{image.suffix}"
        logger.info(f"Saving {output_file}")
        plt.savefig(output_file)


if __name__ == "__main__":
    app()
