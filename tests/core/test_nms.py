import numpy as np

from pyodi.core.nms import nms, nms_predictions


def get_random_boxes(score_low=0.0, score_hight=1.0):
    tops = np.random.randint(0, 99, size=100)
    lefts = np.random.randint(0, 99, size=100)
    heights = [np.random.randint(1, 100 - left) for left in lefts]
    widths = [np.random.randint(1, 100 - top) for top in tops]
    scores = np.random.uniform(low=score_low, high=score_hight, size=100)
    boxes = np.c_[lefts, tops, widths, heights, scores]
    return boxes


def get_random_predictions(image_id):
    boxes = get_random_boxes()
    predictions = []

    for box in boxes:
        predictions.append(
            {
                "image_id": image_id,
                "bbox": box[:-1],
                "score": box[-1],
                "iscrowd": 0,
                "category_id": 1,
            }
        )
    return predictions


def test_nms_iou_thr():
    boxes = get_random_boxes()
    keep = []
    for iou_thr in np.arange(0, 1.1, 0.1):
        keep.append(nms(boxes, iou_thr=iou_thr))

    for n in range(1, len(keep)):
        # As iou_thr increases, more boxes pass the filter
        assert len(keep[n - 1]) <= len(keep[n])

    # At iou_thr=1.0 no box is filtered
    assert len(keep[-1]) == len(boxes)


def test_nms_predictions_score_thr():
    predictions = []
    for image_id in range(2):
        predictions.extend(get_random_predictions(image_id))

    filtered = []
    for score_thr in np.arange(0, 1.1, 0.1):
        filtered.append(nms_predictions(predictions, score_thr=score_thr, iou_thr=1.0))

    for n in range(1, len(filtered)):
        # As score_thr increases, less boxes pass the filter
        assert len(filtered[n - 1]) >= len(filtered[n])

    # At score_thr=0.0 no box is filtered
    assert len(filtered[0]) == len(predictions)
