from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from types import MappingProxyType
from typing import Any

import numpy as np
import scipy.sparse
import shapely
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from numpy import (
    arange,
    asanyarray,
    concatenate,
    full,
    isscalar,
    ndarray,
    newaxis,
    ones,
    searchsorted,
    zeros,
)
from numpy.linalg import norm
from typing_extensions import Self

from curvey._typing import EdgesLike, PointsLike
from curvey.plot import _get_ax, segments, text
from curvey.triangulation import Triangulation
from curvey.util import _rescale


class Edges:
    """A 'edge soup' of directed line segments defined by their vertex coordinates and connectivity

    Parameters
    ----------
    points
        `(n, 2)` array of vertex coordinates.

    edges
        `(n, 2)` integer array of vertex indices. Can also be `None` for a pure point set.

    point_data
        Point data in key => value format. Values are `(n_points,)` or `(n_points, ndims)`
        arrays.

    edge_data
        Edge data in key => value format. Values are `(n_edges,)` or `(n_points, ndims)`
        arrays.
    """

    def __init__(
        self,
        points: PointsLike,
        edges: EdgesLike | None,
        point_data: dict[str, ndarray] | None = None,
        edge_data: dict[str, ndarray] | None = None,
    ):
        self.points: ndarray = asanyarray(points)
        """`(n, 2)` array of vertex coordinates."""

        self.edges: ndarray = asanyarray(edges) if edges is not None else zeros((0, 2), dtype="int")
        """`(n, 2)` integer array of vertex indices."""

        self._point_data: dict[str, ndarray] = {} if point_data is None else point_data
        self._edge_data: dict[str, ndarray] = {} if edge_data is None else edge_data

    def __repr__(self) -> str:
        items: dict[str, str] = {
            "n_points": str(self.n_points),
            "n_edges": str(self.n_edges),
        }
        if pd := self._point_data:
            items["point_data"] = f"{{{', '.join(pd.keys())}}}"

        if ed := self._edge_data:
            items["edge_data"] = f"{{{', '.join(ed.keys())}}}"

        item_list = ", ".join(f"{k}={v}" for k, v in items.items())
        return f"{self.__class__.__name__}({item_list})"

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
        """Copy of self replacing some subset of properties"""
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
        """Used for copying e.g. point_data or edge_data with extra data added

        Parameters
        ----------
        n_name
            The expected data size variable, e.g. 'n_points' or 'n_edges'.

        data
             E.g. self.point_data or self.edge_data

        **kwargs
            Data to add in key => value format.

        """
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
        """Attach point data in key=value format

        Values must be `(n_points,)` or `(n_points, n_dims)` arrays, *or* a scalar value, in which
        case the scalar is broadcast to a `(n_points,)` array.
        """
        return self.with_(point_data=self._data_with("n_points", self._point_data, kwargs))

    def with_edge_data(self, **kwargs) -> Self:
        """Attach edge data in key=value format

        Values must be `(n_edges,)` or `(n_edges, n_dims)` arrays, *or* a scalar value, in which
        case the scalar is broadcast to a `(n_edges,)` array.
        """
        return self.with_(edge_data=self._data_with("n_edges", self._edge_data, kwargs))

    def drop_edges(self) -> Self:
        """An `Edges` with only points and point data"""
        return self.with_(edges=zeros((0, 2), dtype="int"), edge_data={})

    def reverse(self) -> Edges:
        """Flip edge direction"""
        return self.with_(edges=self.edges[:, ::-1])

    @property
    def n_points(self) -> int:
        """Number of vertices

        This includes points not referenced by the edges array.
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
        """An `Edges` with zero points and zero edges"""
        return cls(points=zeros((0, 2)), edges=zeros((0, 2), dtype="int"))

    @classmethod
    def concatenate(cls, *es: Self) -> Self:
        """Concatenate multiple edge sets into one

        Parameters
        ----------
        *es
            Multiple `Edges` to concatenate into a single `Edges`.
        """
        if len(es) == 0:
            return cls.empty()

        if len(es) == 1:
            return es[0]

        idx_offset, points, edges = 0, [], []

        point_keys = set.intersection(*(set(e._point_data.keys()) for e in es))
        point_data: dict[str, list[ndarray]] = {k: [] for k in point_keys}

        edge_keys = set.intersection(*(set(e._edge_data.keys()) for e in es))
        edge_data: dict[str, list[ndarray]] = {k: [] for k in edge_keys}

        for e in es:
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
        """Representation of the edges as a `shapely.MultiLineString"""
        # return shapely.MultiLineString(list(self.points[self.edges]))
        return shapely.multilinestrings(self.points[self.edges])

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
        """Plot a scalar quantity on vertices

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

    def plot_edges(self, **kwargs) -> LineCollection | Quiver:
        """Plot edges

        See `curvey.plot.segments` for additional kwargs descriptions.
        """
        return segments(
            points=self.points,
            edges=self.edges,
            **kwargs,
        )

    def point_labels(
        self, labels: Iterable[str] | None = None, ax: Axes | None = None, clip=True, **kwargs
    ) -> list[Text]:
        """Draw labels on points"""
        return text(points=self.points, labels=labels, ax=ax, clip=clip, **kwargs)

    def edge_labels(
        self, labels: Iterable[str] | None = None, ax: Axes | None = None, clip=True, **kwargs
    ) -> list[Text]:
        """Draw labels on edge midpoints"""
        midpoints = self.points[self.edges].mean(axis=1)
        return text(points=midpoints, labels=labels, ax=ax, clip=clip, **kwargs)

    def triangulate(
        self,
        max_tri_area: float | None = None,
        min_angle: float | None = None,
        polygon: bool = False,
        holes: ndarray | None = None,
        interior_points: ndarray | None = None,
        extra_params: str | None = None,
    ) -> Triangulation:
        """
        Triangulate the polygon enclosed by the edges with Jonathan Shewchuck's
        `triangle` library.

        The python bindings [triangle](https://rufat.be/triangle/index.html) must be importable.
        They can be installed with `pip install triangle`.

        Note
        ----
        This assumes, but does not enforce, no repeated points. `triangle` will often segfault
        with repeated points.

        Parameters
        ----------
        polygon
            If true, perform constrained polygon triangulation. This is equivalent
            to including `'p'` in `extra_params`.

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

        """
        import triangle

        params = ""
        if polygon:
            params += "p"

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
            holes = np.asarray(holes)
            # triangle doesn't like empty arrays
            if len(holes):
                constraints["holes"] = holes

        d = triangle.triangulate(constraints, params)
        is_boundary = d["vertex_markers"].astype(bool).squeeze()

        out = Triangulation(
            points=d["vertices"],
            faces=d["triangles"],
        )
        out.is_boundary_vertex = is_boundary  # overwrite cached property definition
        return out

    def closest_edge(self, points: PointsLike) -> tuple[ndarray, ndarray]:
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
        points = asanyarray(points)
        if points.ndim == 1:
            points = points[newaxis]

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
        points = asanyarray(points)
        if points.ndim == 1:
            points = points[newaxis]

        pts = shapely.MultiPoint(points).geoms
        (pts_idx, edge_idx), dist = self.tree.query_nearest(
            pts, return_distance=True, all_matches=False
        )

        edges = self.tree.geometries[edge_idx]
        closest = zeros((len(points), 2))
        for i, (e, pt) in enumerate(zip(edges, pts)):
            closest[i, :] = e.interpolate(e.project(pt)).coords[0]

        return edge_idx, dist, closest

    def to_csgraph(self, weighted=True, directed=True) -> scipy.sparse.coo_array:
        """Vertex adjacency array for use with scipy's sparse groph routines

        Parameters
        ----------
        weighted
            If true, edge weights are set to inverse edge lengths. Otherwise, edge weights are
            set to `1`.

        directed
            If true, the adjacency matrix `adj[i, j]` is non-zero only if the
            directed edge `(i, j)` is in `self.edges`. If false, `adj[i, j] == adj[j, i]` is
            non-zero if `(i, j)` or `(j, i)` is in `self.edges`.

        Returns
        -------
        adj :
            A `(n_points, n_points)` adjacency matrix.

        """
        edge_weights = 1 / self.edge_length if weighted else ones(self.n_edges)
        n = self.n_points
        u, v = self.edges.T
        out = scipy.sparse.coo_array((edge_weights, (u, v)), shape=(n, n))
        if directed:
            return out
        return out + out.T

    def drop_degenerate_edges(self) -> Self:
        """Drop edges with zero length"""
        eidx = self.edge_length != 0
        return self.with_(
            edges=self.edges[eidx],
            edge_data={k: v[eidx] for k, v in self._edge_data.items()},
        )

    def drop_unreferenced_verts(self) -> Self:
        """Drop points that aren't referenced by the edge array"""
        orig_verts = arange(self.n_points)
        unq_verts = np.unique(self.edges)
        i = searchsorted(unq_verts, orig_verts)
        is_referenced = orig_verts == unq_verts[i]

        # Keep referenced points
        points = self.points[is_referenced]

        # Remap edges to updated points
        edges = i[self.edges]

        return self.with_(
            points=points,
            edges=edges,
            point_data={k: v[is_referenced] for k, v in self._point_data.items()},
            # Don't need to do anything about edge data
            # edge_data=self._edge_data,
        )
