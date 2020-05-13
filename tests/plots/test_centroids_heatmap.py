import pandas as pd
import pytest

from pyodi.plots.boxes import get_centroids_heatmap


def test_centroids_heatmap_default():
    df = pd.DataFrame(
        {
            "row_centroid": [5, 7],
            "col_centroid": [5, 7],
            "img_height": [10, 10],
            "img_width": [10, 10],
        }
    )
    heatmap = get_centroids_heatmap(df)
    assert heatmap.sum() == 2
    assert heatmap[4, 4] == 1
    assert heatmap[6, 6] == 1


@pytest.mark.parametrize("n_rows,n_cols", [(3, 3), (5, 5), (7, 7), (3, 7), (7, 3)])
def test_centroids_heatmap_n_rows_n_cols(n_rows, n_cols):
    df = pd.DataFrame(
        {
            "row_centroid": [0, 5, 9],
            "col_centroid": [0, 5, 9],
            "img_height": [10, 10, 10],
            "img_width": [10, 10, 10],
        }
    )
    heatmap = get_centroids_heatmap(df, n_rows, n_cols)
    assert heatmap.shape == (n_rows, n_cols)
    assert heatmap[0, 0] == 1
    assert heatmap[n_rows // 2, n_cols // 2] == 1
    assert heatmap[-1, -1] == 1
