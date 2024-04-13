from __future__ import annotations

import pytest

from curvey import Curve, Polygon


@pytest.mark.parametrize("letter", ["c", "d"])
def test_medial_axis(letter: str):
    p = Polygon.from_text(letter)[0]
    p = p.apply(Curve.interpolate_thresh, thresh=1)
    ama = p.approximate_medial_axis(
        dist_thresh=0.25,
        abs_err=0.1,
    )
    assert ama.n_points > 0
    assert ama.n_edges > 0
