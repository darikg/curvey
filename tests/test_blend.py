from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from shapely import hausdorff_distance

from curvey import Curve
from curvey.blend import CurvatureShapeBlending, LinearBlending


@pytest.fixture()
def endpoints() -> tuple[Curve, Curve]:
    n = 128
    src = Curve.circle(n=n)
    tgt = Curve.star(6, r0=1, r1=2).interpolate_n(n).scale(1.5)
    return src, tgt


def test_linear_blending(endpoints):
    lb = LinearBlending(*endpoints)
    t = np.linspace(0, 1, 5)
    curves = lb.interpolate(t=t)
    assert curves.n == len(t)

    assert_array_equal(curves[0].points, endpoints[0].points)
    assert_array_equal(curves[-1].points, endpoints[1].points)


def test_curvature_shape_blending(endpoints):
    csb = CurvatureShapeBlending.preprocess(*endpoints)
    t = np.linspace(0, 1, 5)
    curves = csb.interpolate(t=t, stop_tol=1e-3, interp_size="area")
    assert curves.n == len(t)

    o0, o1 = (c.to_shapely("polygon") for c in endpoints)
    i0, i1 = (curves[i].to_shapely("polygon") for i in (0, -1))

    assert hausdorff_distance(o0, i0) < 0.1
    assert hausdorff_distance(o1, i1) < 0.1
