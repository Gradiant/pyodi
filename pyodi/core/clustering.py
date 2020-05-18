from typing import Dict, List, Union

import numpy as np
from loguru import logger
from numba import float32, njit, prange
from numpy import float64, ndarray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def origin_iou(bboxes: ndarray, clusters: ndarray) -> ndarray:
    """Calculates the Intersection over Union (IoU) between a box and k clusters in coco format
     shifted to origin.

    Parameters
    ----------
    box : np.array
        Bbox array with dimension [n, 2] in widht, height order
    clusters : np.array
        Bbox array with dimension [n, 2] in widht, height order

    Returns
    -------
    np.array
        BBox array with centroids with dimensions [k, 2]
    """
    col = np.minimum(bboxes[:, None, 0], clusters[:, 0])
    row = np.minimum(bboxes[:, None, 1], clusters[:, 1])

    if np.count_nonzero(col == 0) > 0 or np.count_nonzero(row == 0) > 0:
        raise ValueError("Box has no area")

    intersection = col * row
    box_area = bboxes[:, 0] * bboxes[:, 1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area[:, None] + cluster_area - intersection)

    return iou_


@njit(float32[:](float32[:, :], float32[:, :]), parallel=True)
def get_max_overlap(boxes: np.array, anchors: np.array):
    """Computes max intersection-over-union between box and anchors.

    Parameters
    ----------
    boxes : np.array
        Array of bboxes with shape [n, 4].
        In corner format
    anchors : np.array
        Array of bboxes with shape [m, 4]
        In corner format

    Returns
    -------
    np.array
        Max iou between box and anchors with shape [n, 1]
    """
    rows = boxes.shape[0]
    cols = anchors.shape[0]
    overlap = np.zeros(rows, dtype=np.float32)
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    anchors_areas = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])

    for row in prange(rows):

        for col in range(cols):
            ymin = max(boxes[row, 0], anchors[col, 0])
            xmin = max(boxes[row, 1], anchors[col, 1])
            ymax = min(boxes[row, 2], anchors[col, 2])
            xmax = min(boxes[row, 3], anchors[col, 3])

            intersection = max(0, ymax - ymin) * max(0, xmax - xmin)
            union = box_areas[row] + anchors_areas[col] - intersection

            overlap[row] = max(intersection / union, overlap[row])

    return overlap


def kmeans_euclidean(
    values: ndarray,
    n_clusters: int = 3,
    silhouette_metric: bool = False,
    random_state: int = 0,
) -> Dict[str, Union[ndarray, float64]]:
    if len(values.shape) == 1:
        values = values[:, None]

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(values)
    result = dict(centroids=kmeans.cluster_centers_, labels=kmeans.labels_)

    if silhouette_metric:
        result["silhouette"] = silhouette_score(values, labels=kmeans.labels_)

    return result


def find_pyramid_level(bboxes: ndarray, anchor_base_sizes: List[int]) -> ndarray:
    """Matches bboxes with pyramid levels given their stride

    Parameters
    ----------
    bboxes : ndarray
        Bbox array with dimension [n, 2] in widht, height order
    anchor_base_sizes : List[int]
        List with anchor base sizes
    Returns
    -------
    ndarray
        Best match per bbox correponding with index of stride
    """
    anchor_base_sizes = sorted(anchor_base_sizes)
    levels = np.tile(anchor_base_sizes, (2, 1)).T
    ious = origin_iou(bboxes, levels)
    return np.argmax(ious, axis=1)
