import numpy as np
import pytest

from pyodi.core.anchor_generator import AnchorGenerator
from pyodi.core.clustering import pairwise_iou


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
        strides=strides, ratios=[1.0], scales=[1.0], base_sizes=anchor_sizes
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


def test_iou_with_different_size_anchors():
    """Create two grids of anchors of different sizes but same stride and check overlap
    """
    strides = [2, 2]
    feature_maps = [(2, 2), (2, 2)]
    anchor_generator = AnchorGenerator(
        strides=strides, ratios=[1.0], scales=[1.0], base_sizes=[2, 4]
    )
    multi_level_anchors = anchor_generator.grid_anchors(feature_maps)
    assert len(multi_level_anchors) == 2

    iou = pairwise_iou(multi_level_anchors[0], multi_level_anchors[1])
    np.testing.assert_equal(np.diag(iou), np.ones(len(iou)) * 0.25)


def test_anchors_with_octave_scales():
    anchor_generator = AnchorGenerator(
        strides=[2],
        ratios=[1.0],
        base_sizes=[1],
        octave_base_scale=4,
        scales_per_octave=3,
    )

    np.testing.assert_equal(anchor_generator.base_anchors, 1)
