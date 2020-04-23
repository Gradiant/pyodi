import pytest
from pyodi.core.anchor_generator import AnchorGenerator
import numpy as np


@pytest.mark.parametrize("base_sizes", [[1], [9], [16]])
def test_anchor_base_generation(base_sizes):
    expected = np.array([-0.5, -0.5, 0.5, 0.5])[None, None, :] * base_sizes
    anchor_generator = AnchorGenerator(
        strides=[16], ratios=[1.0], scales=[1.0], base_sizes=base_sizes
    )
    np.testing.assert_equal(anchor_generator.base_anchors, expected)


@pytest.mark.parametrize("feature_maps", [[(4, 4)], [(9, 9)]])
def test_anchor_grid_for_different_feature_maps(feature_maps):
    """Creates anchors of size 2 over a feature map with size 'feature_maps' computed with stride 4
    Each pixel from the feature map comes from applying a stride 4 to the original size
    One anchor computed for each pixel"""
    anchor_sizes = [2]
    strides = [4]
    anchor_generator = AnchorGenerator(
        strides=[4], ratios=[1.0], scales=[1.0], base_sizes=anchor_sizes
    )
    base_anchor = np.array([-0.5, -0.5, 0.5, 0.5]) * anchor_sizes

    expected_anchors = []
    for i in range(feature_maps[0][0]):
        for j in range(feature_maps[0][1]):
            new_anchor = [
                base_anchor[0] + strides[0] * j,
                base_anchor[1] + strides[0] * i,
                base_anchor[2] + strides[0] * j,
                base_anchor[3] + strides[0] * i,
            ]
            expected_anchors.append(new_anchor)

    multi_level_anchors = anchor_generator.grid_anchors(feature_maps)[0]

    assert len(multi_level_anchors) == feature_maps[0][0] * feature_maps[0][1]
    np.testing.assert_equal(multi_level_anchors, np.stack(expected_anchors))
