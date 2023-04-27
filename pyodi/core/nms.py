from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger
from numba import jit

from pyodi.core.boxes import coco_to_corners, corners_to_coco, denormalize, normalize


@jit(nopython=True)
def nms(dets: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Non Maximum supression algorithm from https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nms.py.

    Args:
        dets: Array of predictions in corner format.
        scores: Array of scores for each prediction.
        iou_thr: None of the filtered predictions will have an iou above `iou_thr`
            to any other.

    Returns:
        List of filtered predictions in COCO format.

    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]

    return keep


def nms_predictions(
    predictions: List[Dict[Any, Any]],
    score_thr: float = 0.0,
    iou_thr: float = 0.5,
) -> List[Dict[Any, Any]]:
    """Apply Non Maximum supression to all the images in a COCO `predictions` list.

    Args:
        predictions: List of predictions in COCO format.
        score_thr: Predictions below `score_thr` will be filtered. Defaults to 0.0.
        iou_thr: None of the filtered predictions will have an iou above `iou_thr`
            to any other. Defaults to 0.5.

    Returns:
        List of filtered predictions in COCO format.

    """
    new_predictions = []
    image_id_to_all_boxes: Dict[str, List[List[float]]] = defaultdict(list)
    image_id_to_shape: Dict[str, Tuple[int, int]] = dict()

    for prediction in predictions:
        image_id_to_all_boxes[prediction["image_id"]].append(
            [*prediction["bbox"], prediction["score"], prediction["category_id"]]
        )
        if prediction["image_id"] not in image_id_to_shape:
            image_id_to_shape[prediction["image_id"]] = prediction[
                "original_image_shape"
            ]

    for image_id, all_boxes in image_id_to_all_boxes.items():
        categories = np.array([box[-1] for box in all_boxes])
        scores = np.array([box[-2] for box in all_boxes])
        boxes = np.vstack([box[:-2] for box in all_boxes])

        image_width, image_height = image_id_to_shape[image_id]

        boxes = normalize(coco_to_corners(boxes), image_width, image_height)

        logger.info(f"Before nms: {boxes.shape}")
        keep = nms(boxes, scores, iou_thr=iou_thr)
        # Filter out predictions
        boxes, categories, scores = boxes[keep], categories[keep], scores[keep]

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
                    "bbox": box.tolist(),
                    "score": float(score),
                    "category_id": int(category),
                }
            )

    return new_predictions
