from __future__ import annotations

import pytest
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


def test_drop_unreferenced_verts():
    es0 = (
        Edges(
            points=[[0, 0], [0, 1], [1, 1]],
            edges=[[0, 2]],
        )
        .with_point_data(pd=[0, 1, 2])
        .with_edge_data(ed=[999])
    )
    es1 = es0.drop_unreferenced_verts()
    assert_array_equal(es1.points, [[0, 0], [1, 1]])
    assert_array_equal(es1.edges, [[0, 1]])
    assert_array_equal(es1.point_data["pd"], [0, 2])
    assert_array_equal(es1.edge_data["ed"], [999])
