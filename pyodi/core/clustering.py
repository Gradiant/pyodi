import numpy as np
from loguru import logger


def iou(bboxes1, bboxes2):
    """Calculates the pairwise Intersection over Union (IoU) between two sets of bboxes

    Parameters
    ----------
    boxes1 : np.array
        [description]
    boxes2 : np.array
        [description]

    Returns
    -------
    np.array
        [description]

    Raises
    ------
    ValueError
        [description]
    """

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    left_corner = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    right_corner = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

    intersection = np.clip(right_corner - left_corner, a_min=0, a_max=None)
    intersection_area = intersection[..., 0] * intersection[..., 1]

    ious = intersection_area / (area1[:, None] + area2 - intersection_area)

    return ious


def kmeans_iou(bboxes, k, mean_distance_threshold=0.3, distance_metric=np.median):
    """Calculates k-means clustering with the Intersection over Union (IoU) metric.

    Parameters
    ----------
    boxes : np.array
       shape (n, 2), where r is the number of rows
    k : int or list
        number of desired clusters, if list multiple combinations are computed
    mean_distance_threshold : float, optional
        [description], by default .3
    dist : [type], optional
        [description], by default np.median

    Returns
    -------
    [type]
        [description]
    """

    if isinstance(k, int):
        k = [k]

    n_bboxes = bboxes.shape[0]

    silhouette_metrics = []

    for n_clusters in k:
        logger.info(f"Computing cluster for k = {n_clusters}")
        distances = np.ones((n_bboxes, n_clusters))
        last_clusters = np.zeros((n_bboxes,))
        clusters = bboxes[np.random.choice(n_bboxes, n_clusters, replace=False)]

        while True:

            for i in range(n_bboxes):
                distances[i] = 1 - iou(bboxes[i], clusters)

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

        nearest_distances = distances.argsort(axis=-1)[:, -2:]
        silhouette = (
            nearest_distances[:, 0] - nearest_distances[:, 1]
        ) / nearest_distances[:, 0]
        silhouette_metrics.append(np.mean(silhouette))
        logger.info(
            f"Mean silhouette coefficient for {n_clusters}: {silhouette_metrics[-1]}"
        )
