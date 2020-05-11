import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer
from loguru import logger

app = typer.Typer()


def nms(dets, iou_thr):
    """
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = x1 + dets[:, 2]
    y2 = y1 + dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_idx = scores.argsort()[::-1]

    keep = []
    while sorted_idx.size > 0:
        i = sorted_idx[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[sorted_idx[1:]])
        yy1 = np.maximum(y1[i], y1[sorted_idx[1:]])
        xx2 = np.minimum(x2[i], x2[sorted_idx[1:]])
        yy2 = np.minimum(y2[i], y2[sorted_idx[1:]])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[sorted_idx[1:]] - inter)

        retained_idx = np.where(iou <= iou_thr)[0]
        sorted_idx = sorted_idx[retained_idx + 1]

    return keep


@logger.catch
@app.command()
def crops_join(ground_truth, predictions, output, apply_nms=True):
    ground_truth = json.load(open(ground_truth))

    crop_id_to_filename = {x["id"]: x["file_name"] for x in ground_truth["images"]}

    stem_to_original_id = {
        Path(x["file_name"]).stem: x["id"] for x in ground_truth["old_images"]
    }

    predictions = json.load(open(predictions))

    for n, crop in enumerate(predictions):
        if not n % 10:
            logger.info(n)
        filename = crop_id_to_filename[crop["image_id"]]
        # logger.info(filename)
        parts = Path(filename).stem.split("_")

        stem = "_".join(parts[:-2])
        original_id = stem_to_original_id[stem]
        crop["image_id"] = original_id

        crop_row = int(parts[-1])
        crop_col = int(parts[-2])
        # logger.info(f"CROP ROW: {crop_row}")
        # logger.info(f"CROP COL: {crop_col}")
        crop["bbox"][0] += crop_col
        crop["bbox"][1] += crop_row

    with open(output, "w") as f:
        json.dump(predictions, f, indent=2)

    if apply_nms:
        new_predictions = []
        image_id_to_all_boxes = defaultdict(list)
        for n, prediction in enumerate(predictions):
            image_id_to_all_boxes[prediction["image_id"]].append(
                prediction["bbox"] + [prediction["score"]] + [prediction["category_id"]]
            )

        for image_id, all_boxes in image_id_to_all_boxes.items():
            logger.info(image_id)
            categories = np.array([box[-1] for box in all_boxes])
            boxes = np.vstack([box[:-1] for box in all_boxes])

            above_score = boxes[:, -1] > 0.0

            boxes = boxes[above_score]
            categories = categories[above_score]

            logger.info(boxes.shape)
            indices_to_keep = nms(boxes, iou_thr=0.5)

            categories = categories[indices_to_keep]
            boxes = boxes[indices_to_keep]
            logger.info(f" After nms: {boxes.shape}")
            for box, category in zip(boxes, categories):
                new_predictions.append(
                    {
                        "image_id": image_id,
                        "bbox": list(box[:-1]),
                        "category_id": int(category),
                        "score": box[-1],
                    }
                )

        with open(f"nms_{output}", "w") as f:
            json.dump(new_predictions, f, indent=2)


if __name__ == "__main__":
    app()
