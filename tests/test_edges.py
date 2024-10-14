from __future__ import annotations

import pytest
import numpy as np
from numpy import array
from numpy.testing import assert_array_equal

from curvey import Curve, Edges


@pytest.fixture()
def tri() -> Edges:
    return Curve([[0, 0], [1, 0], [1, 1]]).to_edges()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ([0.1, 0], ([0], [0])),  # Allow single point query
        ([[0.1, 0]], ([0], [0])),
        ([[0.1, 0], [0.2, 0]], ([0, 0], [0, 0])),
        ([[0.5, -1]], ([0], [1])),
        ([[0.5, 0.1]], ([0], [0.1])),
    ],
)
def test_closest_edge(tri, query, expected):
    edge_idx, dist = tri.closest_edge(query)
    expected_edge_idx, expected_dist = expected
    assert_array_equal(edge_idx, expected_edge_idx)
    assert_array_equal(dist, expected_dist)


@pytest.mark.parametrize("edge", [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [3, 1],
    [3, 2],
    [3, 0],
])
def test_drop_unreferenced_verts(edge):
    points = array([[0, 0], [0, 1], [1, 1], [5, 5]])
    point_data = array([0, 1, 2, 3])

    es0 = Edges(
        points=points,
        edges=array([edge])
    ).with_point_data(
        foo=point_data
    ).with_edge_data(bar=[999])
    es1 = es0.drop_unreferenced_verts()
    assert_array_equal(es1.points, es0.points[np.sort(edge)])
    assert_array_equal(es1.edges, [np.argsort(edge)])
    assert_array_equal(es1.point_data["foo"], point_data[np.sort(edge)])
    assert_array_equal(es1.edge_data["bar"], [999])
