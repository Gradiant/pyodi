from typing import Dict, List, Union

import numpy as np
from numba import float32, njit, prange
from numpy import float64, ndarray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def origin_iou(bboxes: ndarray, clusters: ndarray) -> ndarray:
    """Calculates the Intersection over Union (IoU) between a box and k clusters.

    Note: COCO format shifted to origin.

    Args:
        bboxes: Bboxes array with dimension [n, 2] in width-height order.
        clusters: Bbox array with dimension [n, 2] in width-height order.

    Returns:
        BBox array with centroids with dimensions [k, 2].

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
def get_max_overlap(boxes: ndarray, anchors: ndarray) -> ndarray:
    """Computes max intersection-over-union between box and anchors.

    Args:
        boxes: Array of bboxes with shape [n, 4]. In corner format.
        anchors: Array of bboxes with shape [m, 4]. In corner format.

    Returns:
        Max iou between box and anchors with shape [n, 1].

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
    values: ndarray, n_clusters: int = 3, silhouette_metric: bool = False,
) -> Dict[str, Union[ndarray, float64]]:
    """Computes k-means clustering with euclidean distance.

    Args:
        values: Data for the k-means algorithm.
        n_clusters: Number of clusters.
        silhouette_metric: Whether to compute the silhouette metric or not. Defaults
            to False.

    Returns:
        Clustering results.

    """
    if len(values.shape) == 1:
        values = values[:, None]

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(values)
    result = dict(centroids=kmeans.cluster_centers_, labels=kmeans.labels_)

    if silhouette_metric:
        result["silhouette"] = silhouette_score(values, labels=kmeans.labels_)

    return result


def find_pyramid_level(bboxes: ndarray, base_sizes: List[int]) -> ndarray:
    """Matches bboxes with pyramid levels given their stride.

    Args:
        bboxes: Bbox array with dimension [n, 2] in width-height order.
        base_sizes: The basic sizes of anchors in multiple levels.

    Returns:
        Best match per bbox corresponding with index of stride.

    """
    base_sizes = sorted(base_sizes)
    levels = np.tile(base_sizes, (2, 1)).T
    ious = origin_iou(bboxes, levels)
    return np.argmax(ious, axis=1)
