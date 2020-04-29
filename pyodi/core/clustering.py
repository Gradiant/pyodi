from typing import Dict, List, Union

import numpy as np
from loguru import logger
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


def pairwise_iou(bboxes1: ndarray, bboxes2: ndarray) -> ndarray:
    """Calculates the pairwise Intersection over Union (IoU) between two sets of bboxes

    Parameters
    ----------
    boxes1 : np.array
        Array of bboxes with shape [n, 4].
        In corner format
    boxes2 : np.array
        Array of bboxes with shape [m, 4]
        In corner format

    Returns
    -------
    np.array
        Pairwise iou array with shape [n, m]
    """

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    left_corner = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    right_corner = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

    intersection = np.clip(right_corner - left_corner, a_min=0, a_max=None)
    intersection_area = intersection[..., 0] * intersection[..., 1]

    ious = intersection_area / (area1[:, None] + area2 - intersection_area)

    return ious


def kmeans_iou(bboxes, k, silhouette_metric=False, distance_metric=np.median):
    """Calculates k-means clustering with the Intersection over Union (IoU) metric for different number of clusters.
    Silhouette average metric is returned for each different k value

    Parameters
    ----------
    boxes : np.array
       shape (n, 2), where r is the number of rows
    k : list of int
        list with different number of clusters, different k-means will be computed per each value.
    dist : fn, optional
        average function used to compute cluster centroids, by default np.median

    Returns
    -------
    [type]
        [description]
    """

    n_bboxes = bboxes.shape[0]

    clustering_results = []

    for n_clusters in k:
        logger.info(f"Computing cluster for k = {n_clusters}")
        distances = np.ones((n_bboxes, n_clusters))
        last_clusters = np.zeros((n_bboxes,))
        clusters = bboxes[np.random.choice(n_bboxes, n_clusters, replace=False)]

        while True:

            distances = 1 - origin_iou(bboxes, clusters)

            nearest_clusters = np.argmin(distances, axis=1)

            if (last_clusters == nearest_clusters).all():
                break

            for cluster in range(n_clusters):
                cluster_elements = nearest_clusters == cluster

                if cluster_elements.any() > 0:
                    clusters[cluster] = distance_metric(
                        bboxes[cluster_elements], axis=0
                    )

            last_clusters = nearest_clusters

        result = dict(centroids=clusters, labels=last_clusters)

        if silhouette_metric:
            # todo: improve and compute only upper triangular matrix
            iou_bboxes_pairwise_distance = 1 - origin_iou(bboxes, bboxes)
            silhouette = silhouette_score(
                iou_bboxes_pairwise_distance,
                labels=nearest_clusters,
                metric="precomputed",
            )
            result["silhouette"] = np.mean(silhouette)
            logger.info(
                f"Mean silhouette coefficient for {n_clusters}: {result['silhouette']}"
            )

        clustering_results.append(result)

    return clustering_results


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


def find_pyramid_level(bboxes: ndarray, strides: List[int]) -> ndarray:
    """Matches bboxes with pyramid levels given their stride

    Parameters
    ----------
    bboxes : ndarray
        Bbox array with dimension [n, 2] in widht, height order
    strides : List[int]
        List with strides

    Returns
    -------
    ndarray
        Best match per bbox correponding with index of stride
    """
    strides = sorted(strides)
    levels = np.tile(strides, (2, 1)).T
    ious = origin_iou(bboxes, levels)
    return np.argmax(ious, axis=1)
