from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from types import MappingProxyType
from typing import Any

import numpy as np
import scipy.sparse
import shapely
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy import (
    arange,
    array,
    asanyarray,
    block,
    concatenate,
    cross,
    diff,
    full,
    isscalar,
    ndarray,
    ones,
    searchsorted,
    zeros,
)
from numpy.linalg import norm
from typing_extensions import Self

from curvey.plot import _get_ax, segments
from curvey.util import _rescale


class Edges:
    """A collection of line segments defined by their vertex coordinates and connectivity

    Parameters
    ----------
    points
        `(n, 2)` array of vertex coordinates

    edges
        `(n, 2)` integer array of vertex indices

    point_data
        Point data in key => value format. Values are `(n_points,)` or `(n_points, ndims)`
        arrays.

    edge_data
        Edge data in key => value format. Values are `(n_edges,)` or `(n_points, ndims)`
        arrays.
    """

    def __init__(
        self,
        points: ndarray,
        edges: ndarray,
        point_data: dict[str, ndarray] | None = None,
        edge_data: dict[str, ndarray] | None = None,
    ):
        self.points = points
        self.edges = edges
        self._point_data: dict[str, ndarray] = {} if point_data is None else point_data
        self._edge_data: dict[str, ndarray] = {} if edge_data is None else edge_data

    @property
    def point_data(self) -> MappingProxyType[str, ndarray]:
        """A read-only view of the point data"""
        return MappingProxyType(self._point_data)

    @property
    def edge_data(self) -> MappingProxyType[str, ndarray]:
        """A read-only view of the edge data"""
        return MappingProxyType(self._edge_data)

    def with_(
        self,
        points: ndarray | None = None,
        edges: ndarray | None = None,
        point_data: dict[str, ndarray] | None = None,
        edge_data: dict[str, ndarray] | None = None,
    ) -> Self:
        """Copy of self overwriting some subset of properties"""
        return self.__class__(
            points=self.points if points is None else points,
            edges=self.edges if edges is None else edges,
            point_data=self._point_data if point_data is None else point_data,
            edge_data=self._edge_data if edge_data is None else edge_data,
        )

    def _data_with(
        self,
        n_name: str,
        data: dict[str, ndarray],
        kwargs: dict[str, Any],
    ) -> dict[str, ndarray]:
        n = getattr(self, n_name)
        data = data.copy()
        for k, v in kwargs.items():
            if isscalar(v):
                val = full(shape=n, fill_value=v)
            else:
                val = asanyarray(v)
                if val.shape[0] != n:
                    msg = f"Data '{k}' has length {val.shape[0]}, expected {n_name}={n}"
                    raise ValueError(msg)
            data[k] = val
        return data

    def with_point_data(self, **kwargs) -> Self:
        """Attach point data in key=value format"""
        return self.with_(point_data=self._data_with("n_points", self._point_data, kwargs))

    def with_edge_data(self, **kwargs) -> Self:
        """Attach edge data in key=value format"""
        return self.with_(edge_data=self._data_with("n_edges", self._edge_data, kwargs))

    def drop_edges(self) -> Self:
        """An `Edges` with only points and point data"""
        return self.with_(edges=zeros((0, 2), dtype="int"), edge_data={})

    def reverse(self) -> Edges:
        """Flip edge direction"""
        return Edges(points=self.points, edges=self.edges[:, ::-1])

    @property
    def n_points(self) -> int:
        """Number of vertices

        This includes points not referenced by the edge array
        """
        return len(self.points)

    @cached_property
    def edge_length(self) -> ndarray:
        """A `n_edges` length vector of edge lengths"""
        dedge = self.points[self.edges[:, 1]] - self.points[self.edges[:, 0]]
        return norm(dedge, axis=1)

    @property
    def n_edges(self) -> int:
        """Number of edges"""
        return len(self.edges)

    @classmethod
    def empty(cls) -> Self:
        """An empty edge set"""
        return cls(points=zeros((0, 2)), edges=zeros((0, 2), dtype="int"))

    @classmethod
    def concatenate(cls, *edge_sets: Edges) -> Self:
        """Concatenate multiple edge sets into one"""
        if len(edge_sets) == 0:
            return cls.empty()

        if len(edge_sets) == 1:
            return edge_sets[0]

        idx_offset, points, edges = 0, [], []

        point_keys = set.intersection(*(set(e._point_data.keys()) for e in edge_sets))
        point_data: dict[str, list[ndarray]] = {k: [] for k in point_keys}

        edge_keys = set.intersection(*(set(e._edge_data.keys()) for e in edge_sets))
        edge_data: dict[str, list[ndarray]] = {k: [] for k in edge_keys}

        for e in edge_sets:
            points.append(e.points)
            edges.append(idx_offset + e.edges)

            for k in point_data:
                point_data[k].append(e.point_data[k])

            for k in edge_data:
                edge_data[k].append(e.edge_data[k])

            idx_offset += e.n_points

        return cls(
            points=concatenate(points, axis=0),
            edges=concatenate(edges, axis=0),
            point_data={k: concatenate(v, axis=0) for k, v in point_data.items()},
            edge_data={k: concatenate(v, axis=0) for k, v in edge_data.items()},
        )

    @cached_property
    def shapely(self) -> shapely.MultiLineString:
        """Representation of the edge set as a `shapely.MultiLineString"""
        return shapely.MultiLineString(list(self.points[self.edges]))

    @cached_property
    def tree(self) -> shapely.STRtree:
        """A shapely.STRtree of edges for fast distance queries"""
        return shapely.STRtree(self.shapely.geoms)

    def plot_points(
        self,
        color: str | ndarray | Any | None = None,
        size: str | ndarray | float | None = None,
        scale_sz: tuple[float, float] | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> PathCollection:
        """Plot a scalar quantity on edge vertices

        Parameters
        -----------
        color
            If a string, assumed to be a name of a `self.point_data` array. Otherwise, either a
            matplotlib scalar colorlike or length `n` array of scalar vertex
            quantities.

        size
            Name of a `point_data` property, or length `n` scalar vertex quantity to size markers
            by, or a fixed size for all vertices.

        scale_sz
            Min and max sizes to scale the vertex quantity `size` to.

        ax
            Matplotlib axes to plot in. Defaults to the current axes.

        **kwargs
            additional kwargs passed to `matplotlib.pyplot.scatter`

        """
        ax = _get_ax(ax)
        if isinstance(color, str) and (color in self.point_data):
            color = self.point_data[color]

        size = _rescale(size, scale_sz)
        return ax.scatter(self.points[:, 0], self.points[:, 1], s=size, c=color, **kwargs)

    def plot_edges(self, **kwargs):
        return segments(
            points=self.points,
            edges=self.edges,
            **kwargs,
        )

    def point_labels(
        self, labels: Iterable[str] | None = None, ax: Axes | None = None, clip=True, **kwargs
    ) -> list[Text]:
        ax = _get_ax(ax)
        txts = []
        if labels is None:
            labels = (str(i) for i in range(self.n_points))

        for (x, y), label in zip(self.points, labels):
            txt = ax.text(x, y, label, **kwargs)
            if clip:
                txt.set_clip_on(True)
                txts.append(txt)

        return txts

    def edge_labels(self, ax: Axes | None = None, **kwargs):
        midpoints = self.points[self.edges].mean(axis=1)
        labels = [str(i) for i in range(self.n_edges)]
        ax = _get_ax(ax)
        for (x, y), label in zip(midpoints, labels):
            txt = ax.text(x, y, label, **kwargs)
            txt.set_clip_on(True)

    def triangulate(
        self,
        max_tri_area: float | None = None,
        min_angle: float | None = None,
        holes: ndarray | None = None,
        interior_points: ndarray | None = None,
        extra_params: str | None = None,
    ) -> Triangulation:
        """Triangulate the polygon enclosed by the edges with Shewchuck's triangulation library

        The python bindings [triangle](https://rufat.be/triangle/index.html) must be importable.
        They can be installed with `pip install triangle`.

        Parameters
        ----------
        max_tri_area
            A global maximum triangle area constraint.

        min_angle
            Minimum angle constraint, in degrees.

        holes
            If this edge set includes edges clockwise bounding an exterior hole, specify a point
            interior to that hole to discard triangles inside that hole.

        interior_points
            Additional vertex constraints in addition to `self.points`

        extra_params
            See the [API documentation](https://rufat.be/triangle/API.html).
            E.g. `extra_params='S10X' specifies a maximum number of 10 Steiner points and suppresses
            exact arithmetic.

        Returns
        -------
        points :
            The `(n, 2)` vertex coordinates of the triangulation. If there are no additional
            constraints on the the triangulation, this is probably just equal to self.points.

        tris :
            `(n_tris, 3)` array of integer vertex indices.

        is_border :
            Length `n` vector of booleans, true for vertices on the border of the triangulation.

        """
        try:
            import triangle
        except ImportError as e:
            msg = "Cannot import `triangle`. Use `pip install triangle` to install."
            raise ValueError(msg) from e

        params = "p"  # Constrained polygon triangulation
        if max_tri_area is not None:
            params += f"a{max_tri_area:.17f}"

        if min_angle is not None:
            params += f"q{min_angle:.17f}"

        if extra_params is not None:
            params += extra_params

        if interior_points is None:
            # Pretty sure triangle expects to be able to write into this array?
            points = self.points.copy()
        else:
            points = concatenate([self.points, interior_points])

        constraints = {"vertices": points, "segments": self.edges}
        if holes is not None:
            constraints["holes"] = holes

        d = triangle.triangulate(constraints, params)
        is_boundary = d["vertex_markers"].astype(bool).squeeze()

        return Triangulation(
            boundary=self,
            points=d["vertices"],
            tris=d["triangles"],
            is_boundary_vertex=is_boundary,
        )

    def closest_edge(self, points: ndarray) -> tuple[ndarray, ndarray]:
        """The edge index and distance to the corresponding closest points to the input

        Parameters
        ----------
        points
            `(n, 2)` array of query points

        Returns
        -------
        edge_idx :
            `(n,)` vector of edge indices

        distance :
            `(n,)` vector of euclidean distances to the closest point on the edge

        """
        pts = shapely.MultiPoint(points).geoms
        (_pts_idx, edge_idx), dist = self.tree.query_nearest(
            pts, return_distance=True, all_matches=False
        )

        return edge_idx, dist

    def closest_point(self, points: ndarray):
        """The closest points on the closest edge

        Parameters
        ----------
        points
            `(n, 2)` array of query points

        Returns
        -------
        edge_idx :
            `(n,)` vector of edge indices

        distance :
            `(n,)` vector of euclidean distances to the closest point on the edge

        closest :
            `(n, 2)` vector of the closest point on that edge

        """
        pts = shapely.MultiPoint(points).geoms
        (pts_idx, edge_idx), dist = self.tree.query_nearest(
            pts, return_distance=True, all_matches=False
        )

        edges = self.tree.geometries[edge_idx]
        closest = array([e.interpolate(e.project(pt)).coords[0] for (e, pt) in zip(edges, pts)])
        return edge_idx, dist, closest

    def to_csgraph(self, weighted=True, directed=True) -> scipy.sparse.coo_array:
        edge_weights = 1 / self.edge_length if weighted else ones(self.n_edges)
        n = self.n_points
        u, v = self.edges.T
        out = scipy.sparse.coo_array((edge_weights, (u, v)), shape=(n, n))
        if directed:
            return out
        return out + out.T

    def drop_degenerate_edges(self) -> Self:
        return self.with_(edges=self.edges[self.edge_length == 0])

    def drop_unreferenced_verts(self) -> Self:
        # Throw out degenerate edges
        orig_verts = arange(self.n_points)
        unq_verts = np.unique(self.edges)
        i = searchsorted(unq_verts, orig_verts)
        is_referenced = orig_verts == unq_verts[i]

        points = self.points[is_referenced]
        # Remap edges to updated points
        edges = i[self.edges]

        # TODO index point, edge data
        return self.with_(points=points, edges=edges)


class Triangulation:
    """Wrapper around the triangulation returned by `curvey.Edges.triangulate`"""

    def __init__(
        self,
        boundary: Edges,
        points: ndarray,
        tris: ndarray,
        is_boundary_vertex: ndarray | None = None,
    ):
        self.boundary = boundary
        self.points = points
        self.tris = tris
        if is_boundary_vertex is not None:
            # Overwrite the @cached_property defn
            self.is_boundary_vertex = is_boundary_vertex

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_tris(self) -> int:
        return len(self.tris)

    @cached_property
    def edge(self) -> ndarray:
        """A `(n_tris, 2, 2)` array of edge vectors

        `self.edge[i, 0]` and `self.edge[i, 1]` are the unnormalized edge vectors of the first
        two edges in triangle `i`.
        """
        return diff(self.points[self.tris], axis=1)

    def reverse(self) -> Triangulation:
        return Triangulation(
            boundary=self.boundary.reverse(),
            points=self.points,
            tris=self.tris[:, ::-1],
        )

    @cached_property
    def signed_area(self) -> ndarray:
        """A length `n_tris` vector of signed triangle areas"""
        return cross(self.edge[:, 0], self.edge[:, 1]) / 2

    def plot_tris(self, ax: Axes | None = None, **kwargs) -> tuple[Line2D, Line2D]:
        ax = _get_ax(ax)
        x, y = self.points.T
        return ax.triplot(x, y, self.tris, **kwargs)

    def to_edges(self) -> Edges:
        return Edges(points=self.points, edges=self._get_edges())

    def _get_edges(self, sort=False, unique=False) -> ndarray:
        v0, v1, v2 = (self.tris[:, [i]] for i in range(3))
        edges = block([[v0, v1], [v1, v2], [v2, v0]])
        if sort:
            edges.sort(axis=1)

        if unique:
            return np.unique(edges, axis=0)

        return edges

    @cached_property
    def is_boundary_vertex(self) -> ndarray:
        edges = self._get_edges(sort=True, unique=True)
        vidx, n = np.unique(edges, return_counts=True)
        if len(vidx) > self.n_points:
            is_boundary = zeros(self.n_points, dtype=int)
            is_boundary[vidx] = n == 2
            return is_boundary

        return n == 2

    @cached_property
    def shapely(self) -> shapely.MultiPolygon:
        return shapely.MultiPolygon([shapely.Polygon(self.points[tri]) for tri in self.tris])

    @cached_property
    def tree(self) -> shapely.STRtree:
        return shapely.STRtree(self.shapely.geoms)

    def signed_distance(self, points: ndarray) -> ndarray:
        pts = shapely.MultiPoint(points).geoms
        (_pts_idx, _edge_idx), boundary_dist = self.boundary.tree.query_nearest(
            pts, return_distance=True, all_matches=False
        )

        (_pts_idx, _tri_idx), tri_dist = self.tree.query_nearest(
            pts, return_distance=True, all_matches=False
        )
        boundary_dist[tri_dist == 0] *= -1
        return boundary_dist
