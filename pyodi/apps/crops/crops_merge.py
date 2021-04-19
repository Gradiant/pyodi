"""# Crops Merge App.

---

# API REFERENCE
"""
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from pyodi.core.nms import nms_predictions


@logger.catch
def crops_merge(
    ground_truth_file: str,
    output_file: str,
    predictions_file: Optional[str] = None,
    apply_nms: bool = True,
    nms_mode: str = "nms",
    score_thr: float = 0.0,
    iou_thr: float = 0.5,
) -> str:
    """Merge and translate `ground_truth_file` or `predictions` to `ground_truth`'s `old_images` coordinates.

    Args:
        ground_truth_file: Path to COCO ground truth file of crops. Generated with
            `crops_split`.
        output_file: Path where the merged annotations will be saved.
        predictions_file: Path to COCO predictions file over `ground_truth_file`.
            If not None, the annotations of predictions_file will be merged instead of ground_truth_file's.
        apply_nms: Whether to apply Non Maximum Supression to the merged predictions of
            each image. Defaults to True.
        nms_mode: Non Maximum Supression mode. Defaults to "nms".
        score_thr: Predictions bellow `score_thr` will be filtered. Only used if
            `apply_nms`. Defaults to 0.0.
        iou_thr: None of the filtered predictions will have an iou above `iou_thr` to
            any other. Only used if `apply_nms`. Defaults to 0.5.

    """
    ground_truth = json.load(open(ground_truth_file))

    crop_id_to_filename = {x["id"]: x["file_name"] for x in ground_truth["images"]}

    stem_to_original_id = {
        Path(x["file_name"]).stem: x["id"] for x in ground_truth["old_images"]
    }
    stem_to_original_shape = {
        Path(x["file_name"]).stem: (x["width"], x["height"])
        for x in ground_truth["old_images"]
    }

    if predictions_file is not None:
        annotations = json.load(open(predictions_file))
    else:
        annotations = ground_truth["annotations"]

    for n, crop in enumerate(annotations):
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
        crop["original_image_shape"] = stem_to_original_shape[stem]

    if apply_nms:
        annotations = nms_predictions(
            annotations, score_thr=score_thr, nms_mode=nms_mode, iou_thr=iou_thr
        )
        output_file = str(
            Path(output_file).parent
            / f"{Path(output_file).stem}_{nms_mode}_{score_thr}_{iou_thr}.json"
        )

    if predictions_file is not None:
        new_ground_truth = annotations
    else:
        new_ground_truth = {
            "info": ground_truth.get("info", []),
            "licenses": ground_truth.get("licenses", []),
            "categories": ground_truth.get("categories", []),
            "images": ground_truth.get("old_images"),
            "annotations": annotations,
        }

    with open(output_file, "w") as f:
        json.dump(new_ground_truth, f, indent=2)

    return output_file
