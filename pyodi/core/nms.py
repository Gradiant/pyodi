from collections import defaultdict
from typing import Any, Dict, List

import ensemble_boxes
import numpy as np
from loguru import logger

from pyodi.coco.utils import coco_to_corners, corners_to_coco, denormalize, normalize


def nms_predictions(
    predictions: List[Dict[Any, Any]],
    nms_mode: str = "nms",
    score_thr: float = 0.0,
    iou_thr: float = 0.5,
) -> List[Dict[Any, Any]]:
    """Apply Non Maximum supression to all the images in a COCO `predictions` list.

    Parameters
    ----------
    predictions: List[Dict[Any, Any]]
        List of predictions in COCO format.

    score_thr: float, optional
        Default: 0.0.
        Only used if `apply_nms`.
        Predictions bellow `score_thr` will be filtered.

    iou_thr: float, optional
        Default 0.5.
        Only used if `apply_nms`.
        None of the filtered predictions will have an iou above `iou_thr` to any other.

    Returns
    -------
    List[Dict[Any, Any]]
        List of filtered predictions in COCO format.
    """
    nms = getattr(ensemble_boxes, nms_mode)
    new_predictions = []
    image_id_to_all_boxes: Dict[str, List[List[float]]] = defaultdict(list)
    image_id_to_width: Dict[str, int] = dict()
    image_id_to_height: Dict[str, int] = dict()

    for prediction in predictions:
        image_id_to_all_boxes[prediction["image_id"]].append(
            [*prediction["bbox"], prediction["score"], prediction["category_id"]]
        )
        if prediction["image_id"] not in image_id_to_width:
            image_id_to_width[prediction["image_id"]] = int(
                prediction["original_image_width"]
            )
            image_id_to_height[prediction["image_id"]] = int(
                prediction["original_image_height"]
            )

    for image_id, all_boxes in image_id_to_all_boxes.items():
        categories = np.array([box[-1] for box in all_boxes])
        scores = np.array([box[-2] for box in all_boxes])
        boxes = np.vstack([box[:-2] for box in all_boxes])

        image_width = image_id_to_width[image_id]
        image_height = image_id_to_height[image_id]

        boxes = normalize(coco_to_corners(boxes), image_width, image_height)

        logger.info(f"Before nms: {boxes.shape}")
        boxes, scores, categories = nms(
            [boxes], [scores], [categories], iou_thr=iou_thr
        )
        logger.info(f"After nms: {boxes.shape}")

        logger.info(f"Before score threshold: {boxes.shape}")
        above_thr = scores > score_thr
        boxes = boxes[above_thr]
        scores = scores[above_thr]
        categories = categories[above_thr]
        logger.info(f"After score threshold: {boxes.shape}")

        boxes = denormalize(corners_to_coco(boxes), image_width, image_height)

        for box, score, category in zip(boxes, scores, categories):
            new_predictions.append(
                {
                    "image_id": image_id,
                    "bbox": list(box),
                    "score": float(score),
                    "category_id": int(category),
                }
            )

    return new_predictions