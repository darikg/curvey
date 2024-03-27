from __future__ import annotations

import pytest
from numpy import cumsum, ones, pi, sqrt, zeros
from numpy.testing import (
    assert_approx_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from curvey.curve import Curve


@pytest.fixture()
def tri():
    return Curve([[0, 0], [1, 0], [1, 1]])


@pytest.fixture()
def star():
    return Curve.star(n=6, r0=1, r1=1.5)


@pytest.fixture()
def circle():
    return Curve.circle(n=100)


def test_n(tri):
    assert tri.n == 3


def test_edge(tri):
    assert_array_almost_equal(
        tri.edge,
        [[1, 0], [0, 1], [-1, -1]],
    )


def test_metadata():
    c0 = Curve.star(n=6, r0=1, r1=1.5)
    assert not c0.data
    c1 = c0.with_data(name="mystar", max_radius=1.5)
    assert c1["name"] == "mystar"
    assert c1["max_radius"] == 1.5
    c2 = c1.scale(2)
    assert c1["name"] == "mystar"
    assert c1["max_radius"] == 1.5
    c3 = c2.with_data(max_radius=3)
    assert c3["name"] == "mystar"
    assert c3["max_radius"] == 3
    c4 = c3.drop_data("max_radius")
    assert "max_radius" not in c4.data
    assert "name" in c4.data


def test_edge_length(tri):
    assert_array_almost_equal(tri.edge_length, [1, 1, sqrt(2)])


def test_length(tri):
    assert tri.length == sum([1, 1, sqrt(2)])


def test_cum_edge_length(tri):
    assert_array_almost_equal(tri.cum_edge_length, [1, 2, 2 + sqrt(2)])
    assert_array_equal(tri.cum_edge_length, cumsum(tri.edge_length))


def test_arclength(tri):
    assert_array_almost_equal(tri.arclength, [0, 1, 2])
    assert tri.arclength[0] == 0
    assert_array_equal(tri.arclength[1:], cumsum(tri.edge_length[:-1]))


def test_closed_arclength(tri):
    assert len(tri.closed_arclength) == (tri.n + 1)
    assert_array_equal(tri.closed_arclength[:-1], tri.arclength)
    assert tri.closed_arclength[-1] == tri.length


def test_unit_edge(tri):
    assert_array_almost_equal(tri.unit_edge, [[1, 0], [0, 1], [-1 / sqrt(2), -1 / sqrt(2)]])


def test_turning_angle(tri):
    assert_array_almost_equal(tri.turning_angle, [3 * pi / 4, pi / 2, 3 * pi / 4])


@pytest.mark.parametrize(
    "tri_repeated",
    [
        # Repeated once
        [[0, 0], [0, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 1], [1, 1]],
        # # Repeated twice
        [[0, 0], [0, 0], [0, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 1], [1, 1], [1, 1]],
    ],
)
def test_drop_repeated_pts(tri, tri_repeated):
    c = Curve(tri_repeated).drop_repeated_points()
    assert c.n == 3
    assert_array_equal(tri.points, c.points)


def test_to_length(tri):
    assert_approx_equal(tri.to_length(7).length, 7)


def test_to_area(tri):
    assert_approx_equal(tri.to_area(9).area, 9)


def test_signed_area_and_orientation(tri):
    assert_approx_equal(tri.signed_area, 0.5)
    assert tri.orientation == 1
    rev = tri.reverse()
    assert_approx_equal(rev.signed_area, -0.5)
    assert rev.orientation == -1


def test_centroid(tri):
    assert_array_almost_equal(tri.centroid, tri.points.mean(axis=0))
    assert_array_almost_equal(Curve.dumbbell(n=20).centroid, [0, 0])


def test_reverse(tri):
    # [[0, 0], [1, 0], [1, 1]]
    assert_array_equal(tri.reverse(keep_first=False).points, [[1, 1], [1, 0], [0, 0]])
    assert_array_equal(tri.reverse(keep_first=True).points, [[0, 0], [1, 1], [1, 0]])


def test_subdivide(tri: Curve):
    tri2 = tri.subdivide(1)
    assert tri2.n == tri.n * 2


@pytest.mark.parametrize("kwargs", [{"n": 1}, {"min_edge_length": 0.1}])
def test_collapse_shortest_edges(kwargs):
    c = Curve([[0, 0], [0.98, 0], [1, 0], [1, 1]]).collapse_shortest_edges(**kwargs)
    assert_array_equal(c.points, [[0, 0], [1, 0], [1, 1]])


def test_is_simple():
    assert Curve([(0, 0), (2, 0), (1, 1)]).is_simple
    assert not Curve([(0, 0), (2, 0), (1, 1), (1, -1)]).is_simple


def test_edge_intersections():
    c = Curve([(0, 0), (2, 0), (1, 1)])
    assert_array_equal(c.edge_intersections(), zeros((0, 2)))
    c = Curve([(0, 0), (2, 0), (1, 1), (1, -1)])
    assert_array_equal(c.edge_intersections(), [[1.0, 0.0]])


def test_align_to(star):
    c0 = star.translate([0.1, 0.1]).rotate(0.3)

    c1 = c0.align_to(star)
    assert_array_almost_equal(c1.points, star.points)

    t = c0.align_to(star, return_transform=True)
    c1 = c0.transform(t)
    assert_array_almost_equal(c1.points, star.points)


def test_roll(star):
    star1 = star.roll(-3)
    assert_array_equal(star1.points[0], star.points[3])


def test_roll_to(star):
    star1 = star.roll(-3)
    star1 = star1.roll_to(star)
    assert_array_equal(star.points, star1.points)


def test_register_to(star):
    star1 = star.subdivide(1)

    c0 = star1.translate([0.1, 0.1]).rotate(0.3)
    c1 = c0.register_to(star)
    assert_array_almost_equal(c1.points, star1.points)

    t = c0.register_to(star, return_transform=True)
    c1 = c0.transform(t)
    assert_array_almost_equal(c1.points, star1.points)


def test_curvature(circle):
    assert_array_almost_equal(circle.curvature, ones(circle.n), decimal=3)
    assert_array_almost_equal(circle.scale(2).curvature, ones(circle.n) / 2, decimal=3)

    assert_array_almost_equal(circle.reverse().curvature, -ones(circle.n), decimal=3)
    assert_array_almost_equal(circle.reverse().scale(2).curvature, -ones(circle.n) / 2, decimal=3)


def test_optimize_edge_lengths_to(circle, star):
    star = star.interpolate_n(circle.n).to_length(circle.length)
    err0 = ((star.edge_length - circle.edge_length) ** 2).sum()
    star = star.optimize_edge_lengths_to(circle)
    err1 = ((star.edge_length - circle.edge_length) ** 2).sum()
    assert err1 < err0
