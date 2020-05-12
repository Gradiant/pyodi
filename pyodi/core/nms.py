from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from loguru import logger


def nms(boxes: np.ndarray, iou_thr: float = 0.5) -> List[bool]:
    """Apply Non Maximum Supression to `boxes`.

    Parameters
    ----------
    boxes: np.ndarray
        Shape (N, 5).
        In format: (left, top, width, height, score)

    iou_thr: float, optional
        Default 0.5
        None of the boxes to keep ill have an iou above `iou_thr` to any other.

    Returns
    -------
    List[bool]
        Indices to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    scores = boxes[:, 4]

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


def nms_predictions(
    predictions: List[Dict[Any, Any]], score_thr: float = 0.0, iou_thr: float = 0.5
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
    new_predictions = []
    image_id_to_all_boxes: Dict[str, List[List[float]]] = defaultdict(list)
    for prediction in predictions:
        image_id_to_all_boxes[prediction["image_id"]].append(
            [*prediction["bbox"], prediction["score"], prediction["category_id"]]
        )

    for image_id, all_boxes in image_id_to_all_boxes.items():
        categories = np.array([box[-1] for box in all_boxes])
        boxes = np.vstack([box[:-1] for box in all_boxes])

        above_score = boxes[:, -1] > score_thr

        boxes = boxes[above_score]
        categories = categories[above_score]

        indices_to_keep = nms(boxes, iou_thr=iou_thr)

        logger.info(f"Before nms: {boxes.shape}")
        categories = categories[indices_to_keep]
        boxes = boxes[indices_to_keep]
        logger.info(f"After nms: {boxes.shape}")
        for box, category in zip(boxes, categories):
            new_predictions.append(
                {
                    "image_id": image_id,
                    "bbox": list(box[:-1]),
                    "category_id": int(category),
                    "score": box[-1],
                }
            )

    return new_predictions
