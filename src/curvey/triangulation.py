from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from types import MappingProxyType
from typing import Any, Dict, TYPE_CHECKING
from typing_extensions import Self

import numpy as np
import shapely
from numpy import arange, array, asanyarray, concatenate, cross, diff, isin, ndarray, newaxis

import curvey
from curvey._typing import PointsLike, TrisLike
from curvey.plot import _get_ax


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D


class Triangulation:
    """A set of triangles defined by vertex coordinates and connectivity array

    Parameters
    ----------
    points
        `(n_points, 2)` array of vertex coordinates.

    faces
        `(n_faces, 3)` integer array of vertex indices.

    point_data
        A `dict` of keys to arrays of length `n_points`

    face_data
        A `dict` of keys to arrays of length `n_faces`

    """

    def __init__(
            self,
            points: PointsLike,
            faces: TrisLike,
            point_data: Dict[str, ndarray] | None = None,
            face_data: Dict[str, ndarray] | None = None,
    ):
        self.points: ndarray = asanyarray(points)
        """`(n_points, 2)` array of vertex coordinates."""

        self.faces: ndarray = asanyarray(faces)
        """`(n_faces, 3)` integer array of vertex indices."""

        self._point_data = {} if point_data is None else point_data
        self._face_data = {} if face_data is None else face_data

    @property
    def point_data(self) -> MappingProxyType[str, Any]:
        """A read-only view of the point metadata"""
        return MappingProxyType(self._point_data)

    def with_(
            self,
            points: ndarray | None = None,
            faces: ndarray | None = None,
            point_data: dict[str, ndarray] | None = None,
            face_data: Dict[str, ndarray] | None = None,
    ) -> Self:
        """Copy of self replacing some subset of properties"""
        return self.__class__(
            points=self.points if points is None else points,
            faces=self.faces if faces is None else faces,
            point_data=self._point_data if point_data is None else point_data,
        )

    def with_point_data(self, **kwargs) -> Triangulation:
        return self.with_(point_data={**self._point_data, **kwargs})

    def with_face_data(self, **kwargs) -> Triangulation:
        return self.with_(face_data={**self._face_data, **kwargs})

    @property
    def n_points(self) -> int:
        """Number of points"""
        return len(self.points)

    @property
    def n_faces(self) -> int:
        """Number of triangles"""
        return len(self.faces)

    @cached_property
    def edge(self) -> ndarray:
        """A `(n_tris, 2, 2)` array of edge vectors

        `self.edge[i, 0]` and `self.edge[i, 1]` are the unnormalized edge vectors of the first
        two edges in triangle `i`.
        """
        return diff(self.points[self.faces], axis=1)

    def reverse(self) -> Triangulation:
        """Reverse the orientation of each triangle"""
        return Triangulation(points=self.points, faces=self.faces[:, ::-1])

    @cached_property
    def signed_area(self) -> ndarray:
        """A length `n_tris` vector of signed triangle areas"""
        return cross(self.edge[:, 0], self.edge[:, 1]) / 2

    @cached_property
    def centroids(self) -> ndarray:
        return self.points[self.faces].mean(axis=1)

    def plot_tris(self, ax: Axes | None = None, **kwargs) -> tuple[Line2D, Line2D]:
        """Plot the triangulation

        Parameters
        ----------
        ax
            Axes to plot in, default current axes.

        **kwargs
            Passed to `matplotlib.pyplot.triplot`
        """
        ax = _get_ax(ax)
        x, y = self.points.T
        return ax.triplot(x, y, self.faces, **kwargs)

    def to_edges(self) -> curvey.edges.Edges:
        """An edge soup

        Returns
        -------
        edges :
            An edge set with points `self.points` and `n_edges=3 * self.n_faces`

        """
        return curvey.edges.Edges(points=self.points, edges=self.edges)

    @cached_property
    def edges(self) -> ndarray:
        """A `(n_faces * 3, 2)` array of edge connectivity"""
        f = self.faces
        return concatenate(
            [
                f[:, [0, 1]],
                f[:, [1, 2]],
                f[:, [2, 0]],
            ],
            axis=0,
        )

    @cached_property
    def is_boundary_vertex(self) -> ndarray:
        """`(n_points,) vector of booleans True if the vertex is on the triangulation boundary"""
        return isin(arange(self.n_points), self.boundary_edges.edges)

    @cached_property
    def shapely(self) -> shapely.MultiPolygon:
        """A `shapely.MultiPolygon` of trangles"""
        return shapely.MultiPolygon([shapely.Polygon(self.points[tri]) for tri in self.faces])

    @cached_property
    def tree(self) -> shapely.STRtree:
        """A `shapely.STRtree` of the triangles"""
        return shapely.STRtree(self.shapely.geoms)

    def signed_distance(self, points: PointsLike) -> ndarray:
        """Signed distance from the boundary

        Signed distance is positive if the point is outside the triangulation, negative inside,
        and zero if the point is on the boundary.
        """
        points = asanyarray(points)
        if points.ndim == 1:
            points = points[newaxis]

        pts = shapely.MultiPoint(points).geoms
        (_pts_idx, _edge_idx), boundary_dist = self.boundary_edges.tree.query_nearest(
            pts, return_distance=True, all_matches=False
        )

        (_pts_idx, _tri_idx), tri_dist = self.tree.query_nearest(
            pts, return_distance=True, all_matches=False
        )
        boundary_dist[tri_dist == 0] *= -1
        return boundary_dist

    @cached_property
    def boundary_edges(self) -> curvey.edges.Edges:
        """Edges on the boundary of the triangulation

        Boundary edges are edges that only belong to one triangle.
        """
        # Border edges only appear once; internal edges twice; once in each orientation
        directed_edges = self.edges
        undirected_edges = np.sort(directed_edges, axis=1)
        _unq_edges, unq_idx, n = np.unique(
            undirected_edges, axis=0, return_index=True, return_counts=True
        )
        directed_border_edges = directed_edges[unq_idx[n == 1]]
        return curvey.edges.Edges(
            points=self.points,
            edges=directed_border_edges,
            point_data=self._point_data,
        )

    def boundary_loops(self, idx_name: str | None = None) -> Iterator[curvey.curve.Curve]:
        """Iterate over directed boundary loops

        Parameters
        ----------
        idx_name
            If supplied, the original point index is stored in the curve metadata parameter
            named `idx`.

        Yields
        ------
        loop :
            Boundary loop as a `curvey.curve.Curve`. Orientation is maintained.

        """
        from networkx import DiGraph, simple_cycles
        g = DiGraph()
        g.add_edges_from(self.boundary_edges.edges)
        for loop in simple_cycles(g):
            c = curvey.curve.Curve(self.points[loop])
            if idx_name is not None:
                c = c.with_data(**{idx_name: array(loop)})
            yield c
