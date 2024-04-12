from __future__ import annotations

import pytest
from numpy.testing import assert_array_equal

from curvey.triangulation import Triangulation


@pytest.fixture()
def tri3() -> Triangulation:
    pts = [
        [0, 0],
        [1, 0],
        [0, 1],
    ]
    faces = [[0, 1, 2]]
    return Triangulation(pts, faces)


@pytest.fixture()
def tri4() -> Triangulation:
    pts = [
        [0, 0],
        [1, 0],
        [0.5, 1],
        [0.5, 0.5],
    ]
    faces = [
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ]
    return Triangulation(pts, faces)


def test_edges3(tri3):
    assert_array_equal(
        tri3.edges,
        [
            [0, 1],
            [1, 2],
            [2, 0],
        ],
    )


def test_edges4(tri4):
    assert_array_equal(
        tri4.edges,
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [1, 3],
            [2, 3],
            [0, 3],
            [3, 0],
            [3, 1],
            [3, 2],
        ],
    )


def test_is_boundary_vertex(tri4):
    assert_array_equal(tri4.is_boundary_vertex, [True, True, True, False])


def test_boundary_loops(tri4):
    loops = list(tri4.boundary_loops())
    assert len(loops) == 1
    assert_array_equal(
        loops[0].points,
        [
            [0, 0],
            [1, 0],
            [0.5, 1],
        ],
    )
