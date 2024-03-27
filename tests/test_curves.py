from __future__ import annotations

import pytest
from numpy.testing import assert_array_equal

from curvey import Curve, Curves
from curvey.curves import _SubplotsBuilder


@pytest.fixture()
def curve() -> Curve:
    return Curve.circle(n=3)


def curves(n: int) -> Curves:
    c = Curve.circle(n=3)
    return Curves([c for _ in range(n)])


def test_empty():
    cs = Curves()
    assert cs.n == 0

    cs = Curves([])
    assert cs.n == 0


def test_from_list(curve):
    n = 3
    cs = Curves([curve for _ in range(n)])
    assert cs.n == n


def test_from_generator(curve):
    n = 3
    cs = Curves(curve for _ in range(n))
    assert cs.n == n


def test_subplots_builder_get_dims():
    assert _SubplotsBuilder.get_dims(20, 1, 5) == (1, 5)


def test_subplots_builder_from_dims():
    cs = curves(n=20)
    b = _SubplotsBuilder.from_dims(cs, nr=1, nc=5)
    assert b.n_axs == 5
    assert b.n_plots == 5
    assert_array_equal(b.plot_idxs, [0, 4, 8, 12, 16])
