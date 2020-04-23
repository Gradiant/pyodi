import pytest
from pyodi.core.anchor_generator import AnchorGenerator
import numpy as np


@pytest.mark.parametrize("base_sizes", [[1], [9], [16]])
def test_anchor_base_generation(base_sizes):
    expected = np.array([-0.5, -0.5, 0.5, 0.5])[None, None, :] * base_sizes
    anch_gen = AnchorGenerator(
        strides=[16], ratios=[1.0], scales=[1.0], base_sizes=base_sizes
    )
    np.testing.assert_equal(anch_gen.base_anchors, expected)
