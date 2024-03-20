import pytest
from numpy import array_equal, sqrt, pi
from numpy.testing import assert_approx_equal, assert_array_almost_equal, assert_array_equal

from cemetery.util.curvey import Curve


@pytest.fixture
def tri():
    return Curve([[0, 0], [1, 0], [1, 1]])


def test_n(tri):
    assert tri.n == 3


def test_edge(tri):
    assert_array_almost_equal(
        tri.edge,
        [[1, 0], [0, 1], [-1, -1]],
    )


def test_edge_length(tri):
    assert_array_almost_equal(tri.edge_length, [1, 1, sqrt(2)])


def test_tangent(tri):
    assert_array_almost_equal(tri.tangent, [
        [1, 0], [0, 1], [-sqrt(2), -sqrt(2)]
    ])


def test_cum_edge_length(tri):
    assert_array_almost_equal(tri.cum_edge_length, [1, 2, 2 + sqrt(2)])


def test_arclength(tri):
    assert_array_almost_equal(tri.arclength, [0, 1, 2])


def test_turning_angle(tri):
    assert_array_almost_equal(tri.turning_angle, [3 * pi / 4, pi / 2, 3 * pi / 4])


@pytest.mark.parametrize(
    'tri_repeated', [
        # Repeated once
        [[0, 0], [0, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 1], [1, 1]],
        # # Repeated twice
        [[0, 0], [0, 0], [0, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 1], [1, 1], [1, 1]],
    ]
)
def test_drop_repeated_pts(tri, tri_repeated):
    c = Curve(tri_repeated).drop_repeated_points()
    assert c.n == 3
    assert array_equal(tri.pts, c.pts)


def test_to_length(tri):
    assert_approx_equal(tri.to_length(7).length, 7)


def test_to_area(tri):
    assert_approx_equal(tri.to_area(9).area, 9)


def test_signed_area_and_orientation(tri):
    assert_approx_equal(tri.signed_area, 0.5)
    assert tri.orientation is 1
    rev = tri.reverse()
    assert_approx_equal(rev.signed_area, -0.5)
    assert rev.orientation is -1


def test_centroid(tri):
    assert_array_almost_equal(tri.centroid, tri.pts.mean(axis=0))
    assert_array_almost_equal(Curve.dumbbell(n=20).centroid, [0, 0])


def test_reverse(tri):
    # [[0, 0], [1, 0], [1, 1]]
    assert array_equal(tri.reverse(keep_first=False).pts, [[1, 1], [1, 0], [0, 0]])
    assert array_equal(tri.reverse(keep_first=True).pts, [[0, 0], [1, 1], [1, 0]])


def test_subdivide(tri: Curve):
    tri2 = tri.subdivide(1)
    assert tri2.n == tri.n * 2


def test_joint_sample(tri: Curve):
    tri2 = tri.subdivide(1)
    assert_array_equal(tri.joint_sample(tri2), tri2.arclength)
    assert_array_equal(tri.joint_sample(tri, min_edge_length=1000), tri.arclength)





