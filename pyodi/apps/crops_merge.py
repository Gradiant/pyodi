import json
from pathlib import Path

import numpy as np
import typer
from loguru import logger

from pyodi.core.nms import nms_predictions

app = typer.Typer()


@logger.catch
@app.command()
def crops_merge(
    ground_truth_file: str,
    predictions_file: str,
    output_file: str,
    apply_nms: bool = True,
    score_thr: float = 0.0,
    iou_thr: float = 0.5,
):
    """Merge and translate `predictions` to `ground_truth`'s `old_images` coordinates.

    Parameters
    ----------
    ground_truth_file : str
        Path to COCO ground truth file of crops.
        Generated with `crops_split`.

    predictions_file: str
        Path to COCO predictions file over `ground_truth_file`.

    output_file: str
        Path where the merged predictions will be saved.

    apply_nms: bool, optional
        Default: True.
        Whether to apply Non Maximum Supression to the merged predictions of each image.

    score_thr: float, optional
        Default: 0.0.
        Only used if `apply_nms`.
        Predictions bellow `score_thr` will be filtered.

    iou_thr: float, optional
        Default 0.5.
        Only used if `apply_nms`.
        None of the filtered predictions will have an iou above `iou_thr` to any other.
    """
    ground_truth = json.load(open(ground_truth_file))

    crop_id_to_filename = {x["id"]: x["file_name"] for x in ground_truth["images"]}

    stem_to_original_id = {
        Path(x["file_name"]).stem: x["id"] for x in ground_truth["old_images"]
    }

    predictions = json.load(open(predictions_file))

    for n, crop in enumerate(predictions):
        if not n % 10:
            logger.info(n)
        filename = crop_id_to_filename[crop["image_id"]]
        parts = Path(filename).stem.split("_")

        stem = "_".join(parts[:-2])
        original_id = stem_to_original_id[stem]
        crop["image_id"] = original_id

        # Corners are encoded in crop's filename
        # See crops_split.py
        crop_row = int(parts[-1])
        crop_col = int(parts[-2])
        crop["bbox"][0] += crop_col
        crop["bbox"][1] += crop_row

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    if apply_nms:
        new_predictions = nms_predictions(predictions)

        with open(f"{Path(output_file).stem}_{score_thr}_{iou_thr}.json", "w") as f:
            json.dump(new_predictions, f, indent=2)


if __name__ == "__main__":
    app()
