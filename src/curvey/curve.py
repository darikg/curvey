"""A curve"""

from __future__ import annotations

import collections.abc
import itertools
import warnings
from collections.abc import Sequence
from functools import cached_property
from types import MappingProxyType
from typing import Any, Callable, Literal, NamedTuple, cast, overload

import numpy as np
import scipy
import shapely
import sortedcontainers
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from numpy import (
    append,
    arange,
    arctan2,
    argmin,
    array,
    asanyarray,
    asarray,
    ceil,
    concatenate,
    cos,
    cross,
    cumsum,
    diff,
    gradient,
    linspace,
    ndarray,
    newaxis,
    ones,
    pi,
    repeat,
    roll,
    sign,
    sin,
    sqrt,
    stack,
    where,
    zeros,
)
from numpy.linalg import norm
from numpy.typing import ArrayLike
from typing_extensions import Self

from .util import (
    InterpType,
    _get_ax,
    _rescale,
    _VariableColorSpec,
    align_edges,
    angle_to_points,
    periodic_interpolator,
    reflection_matrix,
    rotation_matrix,
)


class Curve:
    """A discrete planar closed curve

    Parameters
    ----------
    pts
        `(n, 2)` array of vertex coordinates.

    **kwargs
        Metadata parameters in key=value format.
    """

    def __init__(
        self,
        pts: ndarray | Sequence[Sequence[float] | tuple[float, float]] | ArrayLike,
        _data: dict[str, Any] | None = None,
        **kwargs,
    ):
        # Use `asanyarray` here to allow the user to pass in whatever they want as long as it obeys
        # the numpy array protocol; in particular thinking of arrays of dual numbers for automatic
        # differentiation
        pts = asanyarray(pts)

        if pts.ndim != 2:
            msg = f"Points array must be 2-dimensional, got pts.ndims={pts.ndim}"
            raise WrongDimensions(msg)

        if pts.shape[1] != 2:
            msg = f"Points array must have 2 columns, got pts.shape={pts.shape}"
            raise WrongDimensions(msg)

        if pts.shape[0] < 3:
            msg = f"Need at least 3 points for a curve, got {pts.shape[0]}"
            raise NotEnoughPoints(msg)

        self._pts: ndarray = pts

        if kwargs and _data:
            self._data = {**_data, **kwargs}
        elif kwargs:
            self._data = kwargs
        elif _data:
            self._data = _data
        else:
            self._data = {}

    @cached_property
    def points(self) -> ndarray:
        """A `(n, 2)` array of curve vertex coordinates"""
        # Because we rely so heavily on `cached_property`s, prevent confusion due to stuff
        # like `curve.points *= 2` modifying the points array in place
        # Could just set this flag on `Curve` construction but
        #   1) The user owns the original array, not us, so don't set flag in place
        #   2) Don't want to copy the points array on construction, that's needlessly wasteful
        #   3) Assume a lot of curve construction happens from fluent chaining like
        #      `curve.scale(2).translate([3, 3]).to_length(1) where the public `points` array is
        #       never touched
        pts = self._pts.view()
        pts.flags["WRITEABLE"] = False
        return pts

    def __getitem__(self, item: str) -> Any:
        """Get curve metadata value by key name

        `curve['foo']` returns the value of the metadata parameter 'foo'.

        """
        return self._data[item]

    @property
    def data(self) -> MappingProxyType[str, Any]:
        """A read-only view of the curve's metadata"""
        return MappingProxyType(self._data)

    def with_points(self, pts: ndarray) -> Curve:
        """A curve with the newly supplied points array, but same metadata values"""
        # We can share the data dict here without a copy because it's publicly read-only
        return self.__class__(pts=pts, _data=self._data)

    def with_data(self, **kwargs) -> Curve:
        """A new curve with the same points and metadata appended with the supplied metadata

        E.g. `curve.with_data(foo=1, bar=2).with_data(baz=3)` has metadata parameters
        'foo', 'bar', and 'baz'.

        This allows without complaint overwriting previous metadata.

        Parameters
        ----------
        **kwargs
            New metadata in key=value format

        """
        return self.__class__(self._pts, _data={**self._data, **kwargs})

    def drop_data(self, *args: str) -> Curve:
        """Copy of the curve without the listed metadata parameters

        Use `curve.drop_data(*curve.data.keys())` to drop all data.
        """
        to_drop = set(args)
        data = {k: v for k, v in self._data.items() if k not in to_drop}
        return self.__class__(self._pts, _data=data)

    @property
    def n(self) -> int:
        """The number of vertices

        (or the number of edges, since this is a closed curve)

        """
        return len(self._pts)

    def __repr__(self):
        typ = self.__class__.__name__

        if self._data:
            metadata = ", ".join(f"{k}={v}" for k, v in self._data.items())
            return f"{typ}(n={self.n}, {metadata})"

        return f"{typ}(n={self.n})"

    @property
    def x(self) -> ndarray:
        """The x-component of the curve vertices"""
        return self._pts[:, 0]

    @property
    def y(self) -> ndarray:
        """The y-component of the curve vertices"""
        return self._pts[:, 1]

    @property
    def closed_points(self) -> ndarray:
        """A `(n+1, 2)` array of the vertex coordinates where the last row is equal to the first

        Curvey uses an implicitly closed representation, assuming an edge exists between the last
        and first point, i.e. in general `curve.points[0] != curve.points[-1]`. Sometimes
        it's useful to have an explicit representation.

        """
        return concatenate([self._pts, self._pts[[0]]], axis=0)

    def reverse(self, keep_first=False) -> Curve:
        """Reverse the curve orientation

        Flips between clockwise and counter-clockwise orientation

        Parameters
        ----------
        keep_first
            By default, the list of vertices is simply flipped. This changes which point is first.
            If `keep_first` is True, the points are also rolled so the first point is maintained.
        """
        pts = self._pts[::-1]

        if keep_first:
            pts = roll(pts, 1, axis=0)

        return self.with_points(pts)

    def scale(self, scale: float) -> Curve:
        """Scale vertex positions by a constant"""
        return self.with_points(scale * self._pts)

    def translate(self, offset: ndarray | Literal["center", "centroid"]) -> Curve:
        """Translate the curve

        Parameters
        ----------
        offset
            One of
                - A 2 element vector `(dx, dy)`
                - A `(n, 2)` array of `(dx, dy)` translations
                - The string 'center' or `centroid`, in which case the curve is translated so that
                  point sits on the origin.
        """
        if isinstance(offset, str):
            if offset == "center":
                offset = -self.center
            elif offset == "centroid":
                offset = -self.centroid
            else:
                raise ValueError(offset)
        else:
            offset = asarray(offset)

        return self.with_points(self._pts + offset.reshape((-1, 2)))

    def roll(self, shift: int) -> Curve:
        """Circular permutation of the vertex order

        To make vertex `i` the first vertex, use `curve.roll(-i)`.
        """
        return self.with_points(roll(self._pts, shift, axis=0))

    def rotate(self, theta: float) -> Curve:
        """Rotate the curve about the origin

        Parameters
        ----------
        theta
            Angle in radians to rotate the curve. Positive angles are counter-clockwise.
        """
        return self.transform(rotation_matrix(theta))

    def reflect(self, theta: float | Literal["x", "X", "y", "Y"]) -> Curve:
        """Reflect the curve over a line through the origin

        Parameters
        ----------
        theta
            Angle in radians of the reflection line through the origin.
            If `theta` is the string 'x' or 'y', reflect over that axis.
        """
        if isinstance(theta, str):
            if theta in ("x", "X"):
                transform = reflection_matrix(0)
            elif theta in ("y", "Y"):
                transform = reflection_matrix(pi / 2)
            else:
                msg = "Theta can only 'x', 'y', or an angle in radians"
                raise ValueError(msg)
        else:
            transform = reflection_matrix(theta)

        return self.transform(transform)

    @cached_property
    def center(self) -> ndarray:
        """The average vertex position"""
        return self._pts.mean(axis=0)

    @cached_property
    def centroid(self) -> ndarray:
        """The center of mass of the uniformly weighted polygon enclosed by the curve"""
        # https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        pts0, pts1 = self._pts, roll(self._pts, -1, axis=0)
        # xy = (x0 * y1 - x1 * y0)
        xy = (pts0 * pts1[:, ::-1] * array([[1, -1]])).sum(axis=1, keepdims=True)
        return ((pts0 + pts1) * xy).sum(axis=0) / 6 / self.signed_area

    @cached_property
    def edge(self) -> ndarray:
        """The vectors from vertex `i` to `i+1`

        See also
        --------
        [Curve.unit_edge][curvey.curve.Curve.unit_edge]
            For unit edge vectors.
        """
        pts0, pts1 = self._pts, roll(self._pts, -1, axis=0)
        return pts1 - pts0

    def to_edge_midpoints(self) -> Curve:
        """The curve whose vertices are the midpoints of this curve's edges

        Mostly just useful for plotting scalar quantities on edge midpoints.
        """
        pts = self._pts + self.edge / 2
        return self.with_points(pts)

    @cached_property
    def edge_length(self) -> ndarray:
        """Curve edge lengths

        `edge_length[i]` is the length of the edge from vertex `i` to vertex `i+1`.

        See also
        --------
        [Curve.cum_edge_length][curvey.curve.Curve.cum_edge_length]
            Cumulative egde lengths.
        """
        return norm(self.edge, axis=1)

    @cached_property
    def length(self) -> float:
        """Total arclength; the sum of edge lengths"""
        return self.edge_length.sum()

    @cached_property
    def unit_edge(self) -> ndarray:
        """The unit edge vectors from vertex `i` to `i+1`

        See also
        --------
        [Curve.tangent][curvey.curve.Curve.tangent]
            For unit tangent vectors calculated from second-order finite differences.
        """
        return self.edge / self.edge_length[:, newaxis]

    @cached_property
    def edge_normal(self) -> ndarray:
        r"""Unit edge normals

        `edge_normal[i]` is the unit vector normal to the edge from vertex `i` to `i+1`.

        Normals are computed by rotating the unit edge vectors 90 degrees counter-clockwise.

        For a counter-clockwise-oriented curve, this means that normals point inwards.

        See also
        --------
        [Curve.normal][curvey.curve.Curve.normal]
            Vertex normals calculated from 2nd order finite differences.
        """
        dx, dy = self.unit_edge.T
        # (x, y) -> (-y, x) for 90 degrees CCW rotation
        return stack([-dy, dx], axis=1)

    @cached_property
    def cum_edge_length(self) -> ndarray:
        """Cumulative edge lengths

        Simply equal to `np.cumsum(self.edge_length)`.

        `cum_edge_length` is a length `n` vector, and does not include zero,
        i.e. `curve.cum_edge_lengths[0]` is the length of the first edge, and
        `curve.cum_edge_length[-1]` == `curve.length`.

        See also
        --------
        [Curve.arclength][curvey.curve.Curve.arclength]
            Vertex arclength, like `cum_edge_length` but starts at 0.
        """
        return cumsum(self.edge_length)

    @property
    def arclength(self) -> ndarray:
        """Vertex arclengths

        `arclength` is a length `n` vector, where `arclength[i]` is the arclength
        of the `i`th vertex. `arclength[0]` is always zero. `arclength[i]` for `i>0` is equal to
        `cum_edge_length[i-1].

        See also
        --------
        [Curve.closed_arclength][curvey.curve.Curve.closed_arclength]
            Like `arclength`, but also includes `self.length` as the final element.
        """
        return append(0, self.cum_edge_length[:-1])

    @property
    def closed_arclength(self) -> ndarray:
        """Cumulative edge lengths with zero prepended

        `closed_arclength` is a length `n+1` vector, where the first element is `0`, the
        second element is the length of the first edge, and the last element is the cumulative
        length of all edges, `curve.length`.
        """
        return append(0, self.cum_edge_length)

    @cached_property
    def dual_edge_length(self) -> ndarray:
        """Vertex dual edge lengths

        `curve.dual_edge_length[i]` is the average length of the two edges incident on vertex $i$,
        i.e. $(L_{i-1, i} + L_{i, i+1})/2$  where $L_{ij}$ is the edge length between vertex
        $i$ and vertex $j$.
        """
        l_next = self.edge_length  # from vertex i to i+1
        l_prev = roll(l_next, 1)  # from i-1 to i
        return (l_prev + l_next) / 2

    @cached_property
    def tangent(self) -> ndarray:
        """Unit length tangent vectors

        `tangent[i]` is the curve unit tangent vector at vertex `i`. This is constructed from
        second order finite differences; use `Curve.unit_edge` for the exact vector from
        vertex `i` to vertex `i+1`.
        """
        # return self.edge / self.edge_length[:, newaxis]
        df_ds = self.deriv()
        df_ds /= norm(df_ds, axis=1, keepdims=True)
        return df_ds

    @cached_property
    def normal(self) -> ndarray:
        r"""Vertex unit normal vectors

        Normals are computed by rotating the unit tangents 90 degrees counter-clockwise, so that
        $\left[ T_i, N_i, 1 \right]$ forms a right-handed frame at vertex $i$ with tangent $T_i$ and
        normal $N_i$.

        For a counter-clockwise-oriented curve, this means that normals point inwards.

        See also
        --------
        [Curve.edge_normal][curvey.curve.Curve.edge_normal]
            The exact unit normals for each edge.
        """
        dx, dy = self.tangent.T
        # (x, y) -> (-y, x) for 90 degrees CCW rotation
        return stack([-dy, dx], axis=1)

    @cached_property
    def turning_angle(self) -> ndarray:
        r"""Turning angle (a.k.a exterior angle), in radians between adjacent edges

        `curve.turning_angle[i]` is the angle between the vectors $T_{i-1, i}$ and $T_{i, i+1}$
        where $T_{ij}$ is the vector from vertex $i$ to vertex $j$.

        Angles are in the range $\pm \pi$.
        """
        e_next = self.edge  # from vertex i to i+1
        e_prev = roll(e_next, 1, axis=0)  # from i-1 to i
        cos_theta = (e_prev * e_next).sum(axis=1)
        sin_theta = cross(e_prev, e_next)
        return arctan2(sin_theta, cos_theta)

    @cached_property
    def curvature(self) -> ndarray:
        r"""Length `n` vector of signed vertex curvatures

        Computed as
        $$
        \kappa_i = \frac {2 \psi_i}{L_{i-1, i} + L_{i, i+1}}
        $$
        Where $\psi_i$ is the turning angle between the two edges adjacent to vertex $i$,
        and $L_{ij}$ is the length of the edge between vertex $i$ and $j$.

        Note
        ----
        There are multiple ways to reasonably define discrete curvature. (See Table 1 in
        Vouga 2014 for a summary of their tradeoffs.) One is chosen here for
        convenience; most of the curvature flows in `curvey.flow` accept a `curvature_fn` to allow
        this choice to be overridden.
        """
        # NB self.dual_edge_length already includes the factor 2 term in the equation above
        return self.turning_angle / self.dual_edge_length

    def deriv(self, f: ndarray | None = None) -> ndarray:
        """Second order finite differences approximations of arclength-parametrized derivatives

        Derivatives are calculated by circularly padding the arclength `s` and function values
        `f(s)`, passing those to `numpy.gradient` to calculate second order finite differences,
        and then dropping the padded values.

        `f` is the function values to derivate. By default, this is the curve points, so
        `curve.deriv()` computes the curve tangent. Repeated application will compute the second
        derivative, e.g.

        ```python
        from curvey import Curve

        c = Curve.circle(n=20)
        df_ds = c.deriv()
        d2f_ds2 = c.deriv(f=df_ds)
        ```

        Parameters
        ----------
        f
            The `(n,)` or `(n, ndim)` array of function values. Defaults to `self.points`

        Returns
        -------
        deriv :
            The `(n,)` or `(n, ndim)` array of function derivature values.

        """
        if f is None:
            f = self._pts

        # Circularly pad arrays so that the derivatives of the first and last actual points are
        # calculated in the same way as the interior points
        f_periodic = concatenate([f[[-1]], f, f[[0]]], axis=0)
        s = self.cum_edge_length
        s = concatenate([[-self.edge_length[-1], 0], s])
        df_ds = gradient(f_periodic, s, axis=0)
        return df_ds[1:-1]  # Drop padded points

    @cached_property
    def signed_area(self) -> float:
        """Signed area of the polygon enclosed by the curve

        Signed area is positive if the curve is oriented counter-clockwise.

        Calculated by the [shoelace formula](https://en.wikipedia.org/wiki/Shoelace_formula).
        """
        x0, y0 = self._pts.T
        x1, y1 = roll(self._pts, -1, axis=0).T
        return 0.5 * (x0 * y1 - x1 * y0).sum()

    @cached_property
    def area(self) -> float:
        """Absolute area of the polygon enclosed by the curve"""
        return abs(self.signed_area)

    @cached_property
    def orientation(self) -> int:
        """Orientation of the curve

        Integer-valued; `+1` if curve is oriented counterclockwise, `-1` if clockwise, 0 if zero
        area

        """
        return int(sign(self.signed_area))

    def to_ccw(self) -> Curve:
        """A counterclockwise-oriented curve"""
        return self.reverse() if self.signed_area < 0 else self

    def to_cw(self) -> Curve:
        """A clockwise-oriented curve"""
        return self.reverse() if self.signed_area > 0 else self

    def to_orientation(self, orientation: int) -> Curve:
        """A curve with the specified orientation

        Parameters
        ----------
        orientation
            Must be either 1 or -1.
        """
        if orientation not in (1, -1):
            msg = f"orientation must be either 1 or -1, got {orientation}"
            raise ValueError(msg)

        return self.reverse() if self.orientation != orientation else self

    def orient_to(self, other: Curve) -> Curve:
        """A curve with the same orientation as `other`"""
        return self.to_orientation(other.orientation)

    @cached_property
    def roundness(self) -> float:
        r"""The [roundness](https://en.wikipedia.org/wiki/Roundness) of the curve

        Defined here as $P^2 / 4 \pi A$ for perimeter $P$ and area $A$.

        Equal to 1.0 for a circle and larger otherwise.
        """
        return self.length**2 / 4 / pi / self.area

    @classmethod
    def circle(cls, n: int, r: float = 1.0) -> Self:
        """Construct a regular polygon

        Parameters
        ----------
        n
            Number of vertices.

        r
            The radius.
        """
        theta = linspace(0, 2 * pi, n, endpoint=False)
        return cls(r * angle_to_points(theta))

    @classmethod
    def ellipse(cls, n: int, ra: float, rb: float) -> Self:
        """Construct an ellipse

        Parameters
        ----------
        n
            Number of vertices.

        ra
            Major radius.

        rb
            Minor radius.

        """
        theta = linspace(0, 2 * pi, n, endpoint=False)
        pts = array([[ra, rb]]) * angle_to_points(theta)
        return cls(pts)

    @classmethod
    def star(cls, n: int, r0: float, r1: float) -> Self:
        """Construct a (isotoxal) star polygon with `n` corner vertices

        Parameters
        ----------
        n
            The number of corner vertices. The returned curve has `2n` vertices.

        r0
            Radius of the even vertices.

        r1
            Radius of the odd vertices.

        """
        c = Curve.circle(n=2 * n, r=1)
        r = where(arange(c.n) % 2, r1, r0)
        return cls(r[:, newaxis] * c._pts)

    @classmethod
    def dumbbell(cls, n: int, rx: float = 2, ry: float = 2, neck: float = 0.2) -> Self:
        """Construct a dumbbell shape

        Parameters
        ----------
        n
            Number of points

        rx
            Width parameter

        ry
            Height parameter

        neck
            Height of the pinched neck
        """
        t = np.linspace(0, 1, n, endpoint=False)
        z = 2 * pi * t
        x = rx * cos(z)
        y = ry * sin(z) - (ry - neck) * sin(z) ** 3
        return cls(np.stack([x, y], axis=1))

    def drop_repeated_points(self) -> Curve:
        """Drop points that are equal to their predecessor(s)

        Repeated points result in edges with zero length and are probably bad. This will also drop
        the last point if it's equal to the first.
        """
        # NB closed_points adds the first point to the end,
        # and `diff` returns a vector one element shorter than its argument, so it all works out
        distinct = diff(self.closed_points, axis=0).any(axis=1)
        return self.with_points(self._pts[distinct])

    def interpolator(
        self,
        typ: InterpType = "cubic",
        f: ndarray | None = None,
    ) -> Callable[[ndarray], ndarray]:
        """Construct a function interpolator on curve arclength

        Parameters
        ----------
        typ
            The class of spline to use for interpolation. One of 'linear', 'cubic', or
            'pchip'.

        f
            The (n_verts,) or (n_verts, ndim) array of function values to interpolate. By default,
            this is just the vertex positions.

        Returns
        -------
        interpolator :
            A function g(s) `ndarray -> ndarray` that interpolates values of f at the arclengths s.
        """
        f = self._pts if f is None else f
        return periodic_interpolator(self.closed_arclength, f, typ=typ)

    def interpolate(self, s: ndarray, typ: InterpType = "cubic") -> Curve:
        """Construct a new curve by interpolating vertex coordinates at the supplied arclengths

        Parameters
        ----------
        s
            Arclength values to interpolate at.

        typ
            Type of interpolation, one of ('linear', 'cubic', 'pchip').
        """
        pts = self.interpolator(typ=typ, f=self._pts)(s)
        return self.with_points(pts)

    def interpolate_n(self, n: int, typ: InterpType = "cubic") -> Curve:
        """Interpolate `n` evenly spaced sampled

        Simple convenience wrapper around `Curve.interpolate`.

        Parameters
        ----------
        n
            The number of points in the new curve

        typ
            Interpolation type.
        """
        s_new = linspace(0, self.length, n, endpoint=False)
        return self.interpolate(s=s_new, typ=typ)

    def interpolate_thresh(self, thresh: float, typ: InterpType = "cubic") -> Curve:
        """Interpolate with `thresh` spacing between samples.

        Simple convenience wrapper around `Curve.interpolate`.

        Parameters
        ----------
        thresh
            Maximum arclength spacing.

        typ
            Interpolation type.
        """
        n = int(ceil(self.length / thresh))
        n = max(n, 3)
        return self.interpolate_n(n=n, typ=typ)

    @overload
    def align_to(self, target: Curve, *, return_transform: Literal[False] = False) -> Curve: ...

    @overload
    def align_to(self, target: Curve, *, return_transform: Literal[True]) -> ndarray: ...

    def align_to(self, target: Curve, *, return_transform: bool = False) -> Curve | ndarray:
        """Align to another curve by removing mean change in position and edge orientation

        Parameters
        ----------
        target
            The target curve to align to. It must have the same number of vertices as `self`. The
            edges of the other curve are assumed to be in one-to-one correspondance to the edges in
            `self`.

        return_transform
            If true, return the 3x3 transformation matrix. Otherwise, return a `Curve`

        See also
        --------
        [Curve.register_to][curvey.curve.Curve.register_to]
            Iterative closest point registration, which doesn't require corresponding vertices.

        """
        _ = self.check_same_n_vertices(target)

        if return_transform:
            return align_edges(
                self._pts, self.edge, target._pts, target.edge, return_transform=True
            )

        pts = align_edges(self._pts, self.edge, target._pts, target.edge, return_transform=False)

        return self.with_points(pts)

    def plot(self, color="black", ax: Axes | None = None, **kwargs) -> Line2D:
        """Plot the curve as a closed contour

        For more sophisticated plotting see methods `plot_points`, `plot_edges`, and `plot_vectors`.

        Parameters
        ----------
        color
            A matplotlib colorlike.

        ax
            Defaults to the current axes.

        **kwargs
            additional kwargs passed to `matplotlib.pyplot.plot`

        """
        ax = _get_ax(ax)
        (line,) = ax.plot(*self.closed_points.T, color=color, **kwargs)
        return line

    def plot_edges(
        self,
        color: ndarray | None = None,
        directed: bool = True,
        width: float | ndarray | None = None,
        scale_width: tuple[float, float] | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Quiver | LineCollection:
        """Plot a scalar quantity on curve edges

        Parameters
        ----------
        color
            The color to plot each edge. Defaults to curve arc length.

        directed
            If True, plot edges as arrows between vertices. Otherwise, edges are line segments.

        width
            The thickness of each edge segment, scalar or edge quantity vector.

        scale_width
            Min and max widths to scale the edge quantity to.

        ax
            The matplotlib axes to plot in. Defaults to current axes.

        **kwargs
            Aadditional kwargs passed to `plt.quiver` or `LineCollection` depending on `directed`.

        Returns
        -------
        : matplotlib.quiver.Quiver
            If `directed` is True.

        : matplotlib.collections.LineCollection
            If `directed` is False.
        """
        ax = _get_ax(ax)
        width = _rescale(width, scale_width)
        colorspec = _VariableColorSpec.parse(
            n_data=self.n, supplied=color, default_varied=self.arclength
        )

        if directed:
            if isinstance(width, (collections.abc.Sequence, ndarray)):
                msg = (
                    "`width` was supplied as a sequence but "
                    "matplotlib.pyplot.Quiver doesn't support scaling individual arrows. "
                    "Use directed=False to scale edge width."
                )
                raise ValueError(msg)

            x, y = self._pts.T
            dx, dy = self.edge.T
            return ax.quiver(
                x,
                y,
                dx,
                dy,
                *colorspec.maybe_varied,
                color=colorspec.fixed,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=width,
                **kwargs,
            )

        # not directed
        pts = self.closed_points.reshape((-1, 1, 2))
        segments = concatenate([pts[:-1], pts[1:]], axis=1)

        if colorspec.varied is not None:
            lc = LineCollection(
                segments=segments,
                cmap="viridis",
                norm=plt.Normalize(),
                linewidths=width,
                **kwargs,
            )
            lc.set_array(colorspec.varied)
        else:
            lc = LineCollection(segments=segments, color=colorspec.fixed, linewidths=width)
        ax.add_collection(lc)

        # Adding a line collection doesn't update limits so do it here
        ax.update_datalim(self._pts)
        ax.autoscale_view()
        return lc

    def plot_vectors(
        self,
        vectors: ndarray | None = None,
        scale: ndarray | None = None,
        color=None,
        scale_length: tuple[float, float] | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Quiver:
        """Plot vector quantities on curve vertices

        To plot vector quantities on edges use `curve.to_edge_midpoints.plot_vectors(...)`.

        Parameters
        ----------
        vectors
            A `(n, 2)` array of vectors. Defaults to curve normals.

        scale
            A length `n` vector of length scalars to apply to the vectors.

        color
            Length `n` vector of scalar vertex quantities to color by, or a
            constant color for all edges.

        scale_length
            Limits to scale vector length to, after applying `scale`.

        ax
            The axes to plot in. Defaults to the current axes.

        **kwargs
            additional kwargs passed to `matplotlib.pyplot.quiver`
        """
        ax = _get_ax(ax)

        _vectors = self.normal if vectors is None else vectors

        if scale is not None:
            _vectors = scale.reshape(-1, 1) * _vectors

        if scale_length is not None:
            length = norm(_vectors, axis=1, keepdims=True)
            scaled_length = _rescale(length, scale_length)
            _vectors = _vectors / length * scaled_length

        colorspec = _VariableColorSpec.parse(self.n, color, default_fixed="black")

        # By default quiver doesn't include vector endpoints in x/y lim calculations
        ax.update_datalim(self._pts + _vectors)

        x, y = self._pts.T
        dx, dy = _vectors.T
        return ax.quiver(
            x,
            y,
            dx,
            dy,
            *colorspec.maybe_varied,
            color=colorspec.fixed,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            **kwargs,
        )

    def plot_points(
        self,
        color: ndarray | Any | None = None,
        size: ndarray | float | None = None,
        scale_sz: tuple[float, float] | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> PathCollection:
        """Plot a scalar quantity on curve vertices

        Parameters
        -----------
        color
            Either a matplotlib scalar colorlike or length `n` array of scalar vertex
            quantities. Defaults to `self.dual_edge_length`.

        size
            Length `n` scalar vertex quantity to size markers by, or a fixed size
            for all vertices.

        scale_sz
            Min and max sizes to scale the vertex quantity `size` to.

        ax
            Matplotlib axes to plot in. Defaults to the current axes.

        **kwargs
            additional kwargs passed to `matplotlib.pyplot.scatter`

        """
        ax = _get_ax(ax)

        if color is None:
            color = self.dual_edge_length

        size = _rescale(size, scale_sz)

        return ax.scatter(self.x, self.y, s=size, c=color, **kwargs)

    def triangulate(
        self,
        max_tri_area: float | None = None,
        min_angle: float | None = None,
        extra_params: str | None = None,
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Triangulate the polygon enclosed by the curve with Shewchuck's triangulation library

        The python bindings [triangle](https://rufat.be/triangle/index.html) must be importable.
        They can be installed with `pip install triangle`.

        Parameters
        ----------
        max_tri_area: float, optional
            A global maximum triangle area constraint.

        min_angle: float, optional
            Minimum angle constraint, in degrees.

        extra_params: str, optional
            See the [API documentation](https://rufat.be/triangle/API.html).
            E.g. `extra_params='S10X' specifies a maximum number of 10 Steiner points and suppresses
            exact arithmetic.

        Returns
        -------
        points :
            The `(n, 2)` vertex coordinates of the triangulation. If `max_tri_area=None`,
            this is probably equal to `self.points`

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

        idx = arange(self.n)
        segments = stack([idx, roll(idx, -1)], axis=1)
        params = "p"  # Constrained polygon triangulation
        if max_tri_area is not None:
            params += f"a{max_tri_area: f}"
        if min_angle is not None:
            params += f"q{min_angle: f}"
        if extra_params is not None:
            params += extra_params

        d = triangle.triangulate({"vertices": self._pts, "segments": segments}, params)
        verts, tris, is_border = d["vertices"], d["triangles"], d["vertex_markers"]
        is_border = is_border.astype(bool).squeeze()
        return verts, tris, is_border

    def transform(self, transform: ndarray) -> Curve:
        """Apply a 2x2 or 3x3 transform matrix to the vertex positions"""
        pts = self._pts
        if transform.shape not in ((2, 2), (3, 3)):
            msg = f"Expected transform to have shape (2, 2) or (3, 3), got {transform.shape}"
            raise ValueError(msg)

        sz = transform.shape[0]
        if sz == 3:
            pts = concatenate([pts, ones((self.n, 1))], axis=1)
        pts = pts @ transform.T
        pts = pts[:, :2] if sz == 3 else pts
        return self.with_points(pts)

    def to_length(self, length: float = 1.0) -> Curve:
        """A new curve scaled to the supplied length"""
        return self.scale(length / self.length)

    def to_area(self, area: float = 1.0) -> Curve:
        """A new curve scaled to the supplied area"""
        return self.scale(sqrt(area / self.area))

    def subdivide(self, n: int = 1) -> Curve:
        """Create a new curve by evenly subdividing each edge

        Parameters
        ----------
        n
            Number of new points to add to each edge. For `n = 1`, new points are added at the
            edge midpoint; for `n = 2`, points are added at the one-thirds and two-thirds
            points, etc. If `n = 0`, an identical curve is returned.
        """
        if n < 0:
            msg = "n must be >= 0"
            raise ValueError(msg)

        if n == 0:
            return self

        return self.split_edges(1 + n * ones(self.n, dtype="int"))

    def split_edges(self, n: ndarray) -> Curve:
        """Sample uniformly within edge segments

        Parameters
        ----------
        n
            A integer-valued vector of length `self.n` indicating the number of points to sample
            from each edge.

            When `n[i] == 1`, simply sample vertex `i', i.e.
            `curve.split_edges(ones(curve.n, dtype='int'))` returns an identical curve.

            When `n[i] == 2`, sample at vertex `i` *and* the midpoint between vertex `i` and
            `i+1`.

            When `n[i] == 3` sample at vertex `i` and the one-third and two-thirds point, and so on.

            When `n[i] == 0`, vertex `i` is dropped from the output.

        Returns
        -------
        :
            A curve with `sum(n)` vertices.
        """

        edge_idx = repeat(arange(self.n), n)
        edge_frac = concatenate([arange(ni) / ni for ni in n])
        pts = self._pts[edge_idx] + edge_frac[:, newaxis] * self.edge[edge_idx]
        return self.with_points(pts)

    def split_long_edges(self, thresh: float) -> Curve:
        """Split edges evenly so all edge lengths are below `thresh`"""
        n_split = ceil(self.edge_length / thresh)
        return self.split_edges(n_split)

    def split_longest_edges(self, n: int) -> Curve:
        """Insert `n` new vertices by uniform edge subdivision

        Edges are split in priority of their length, so very long edges may be split into
        thirds, fourths, etc. before very short edges are split in half.
        """
        if n == 0:
            return self

        if n < 0:
            msg = "`n` must be >= 0"
            raise ValueError(msg)

        class Edge(NamedTuple):
            split_length: float
            n_subdivide: int
            idx: int
            orig_length: float

        orig_edges = (
            Edge(split_length=length, n_subdivide=1, idx=i, orig_length=length)
            for (i, length) in enumerate(self.edge_length)
        )

        # Priority queue -- break length ties by the less split edge
        queue = sortedcontainers.SortedList(orig_edges, key=lambda e: (e.length, -e.n_subdivide))

        for _ in range(n):
            edge = queue.pop()
            edge.n_subdivide += 1
            edge.split_length = edge.orig_length / edge.n_subdivide
            queue.add(edge)

        edges = sorted(queue, key=lambda e: e.idx)
        n_subdivide = array([e.n_subdivide for e in edges])
        return self.split_edges(n_subdivide)

    def collapse_shortest_edges(
        self,
        n: int | None = None,
        min_edge_length: float | None = None,
    ) -> Curve:
        """Remove vertices belonging to the shortest edges until a stopping criterion is met

        Note
        ----
        No attempt is made to prevent self-intersection.

        Parameters
        ----------
        n
            Stop after collapsing this many edges.

        min_edge_length
            Stop when the shortest edge is longer than this.

        """
        if n is None:
            n = self.n - 3

        if min_edge_length is None:
            # noinspection PyArgumentList
            min_edge_length = self.edge_length.max()

        class Vertex(NamedTuple):
            idx: int
            prev: int
            next: int
            edge_length: float  # length from vertex i to i+1

        # A doubly-linked list of vertices
        verts = {
            i: Vertex(
                idx=i,
                prev=(i - 1) % self.n,
                next=(i + 1) % self.n,
                edge_length=edge_length,
            )
            for i, edge_length in enumerate(self.edge_length)
        }

        # Priority queue by edge length
        queue = sortedcontainers.SortedSet(verts.values(), key=lambda v: -v.edge_length)

        for n_removed in itertools.count(1):
            shortest = queue.pop(-1)
            del verts[shortest.idx]

            if (n_removed == n) or (queue[-1] >= min_edge_length):
                break

            # Remove previous and next vertices so we can update them
            queue.discard(v_prev := verts[shortest.prev])
            queue.discard(v_next := verts[shortest.next])
            updated_prev_edge_length = norm(self._pts[v_next.idx] - self._pts[v_prev.idx])

            v_prev = verts[v_prev.idx] = Vertex(
                idx=v_prev.idx,
                prev=v_prev.prev,
                next=v_next.idx,
                edge_length=cast(float, updated_prev_edge_length),
            )
            v_next = verts[v_next.idx] = Vertex(
                idx=v_next.idx,
                prev=v_prev.idx,
                next=v_next.next,
                edge_length=v_next.edge_length,
            )
            queue.add(v_prev)
            queue.add(v_next)

        # Put the remaining vertices back in order
        vert_idx = array(sorted(verts.keys()))
        return self.with_points(self._pts[vert_idx])

    @cached_property
    def laplacian(self) -> scipy.sparse.dia_matrix:
        r"""The discrete Laplacian

        The Laplacian here is the graph Laplacian of a weighted graph with edge weights
        $1 / d_{i, j}$, where $d_{i, j}$ is the distance (edge length) between adjacent vertices
        $i$ and $j$.

        Returns a sparse matrix $L$ of size `(n, n)` with diagonal entries

        $$L_{i,i} = 1 / d_{i-1, i} + 1 / d_{i, i+1} $$

        and off-diagonal entries

        $$L_{i,j} = -1 / d_{i, j}$$.

        """
        return Curve._construct_laplacian(self.edge_length)

    @staticmethod
    def _construct_laplacian(edge_lengths: ndarray) -> scipy.sparse.dia_matrix:
        n = len(edge_lengths)
        l_next = 1 / edge_lengths  # l_next[i] is the inverse edge length from vertex i to i+1
        l_prev = roll(l_next, 1)  # l_prev[i] is the inverse edge length from vertex i-1 to i
        return scipy.sparse.diags(
            [
                -l_next[[-1]],  # lower corner = the single edge length (0, n-1)
                -l_next[:-1],  # lower diag = edge lengths (1, 0), (2, 1), (3, 2), ...
                l_prev + l_next,  # diagonal
                -l_next[:-1],  # upper diag = edge lengths (0, 1), (1, 2), (2, 3), ...
                -l_next[[-1]],  # upper corner = the single edge length (n-1, 0)
            ],
            offsets=(-(n - 1), -1, 0, 1, n - 1),
        ).tocsc()

    @classmethod
    def from_curvature(
        cls,
        curvature: ndarray,
        edge_lengths: ndarray,
        solve_vertices: bool = True,
        theta0: float | None = None,
        pt0: ndarray | Sequence[float] | None = None,
        dual_edge_lengths: ndarray | None = None,
        laplacian: ndarray | scipy.sparse.base.spmatrix | None = None,
    ) -> Self:
        """Construct a curve with the supplied new curvatures and edge lengths

        As explained in

        [*Robust Fairing via Conformal Curvature Flow.* Keenan Crane, Ulrich Pinkall, and
        Peter Schröder. 2014](https://www.cs.cmu.edu/~kmcrane/Projects/ConformalWillmoreFlow/paper.pdf)

        The product (curvature * edge_lengths) is integrated to obtain tangent vectors, and then
        tangent vectors are integrated to obtain vertex positions. This reconstructs the curve
        up to rotation and translation. Supply `theta0` and `pt0` to fix the orientation of the
        first edge and the location of the first point.

        This may result in an improperly closed curve. If `solve_vertices` is True, vertex
        positions are found by a linear projection to the closest closed curve, as described
        in Crane et al.

        Parameters
        ----------
        curvature
            A length `n` vector of signed vertex curvatures.

        edge_lengths
            A length `n`  vector of edge lengths. `edge_length[i]` is the distance between
            vertex `i` and `i+1`.

        theta0
            The constant of integration defining the angle of the first edge and the x-axis,
            in radians.

        pt0
            A 2-element array. The constant of integration defining the absolute position of
            the first vertex.

        solve_vertices
            If True, length discretization errors are resolved by solving
            ∆f = ▽ · T as the discrete Poisson equation Lf = b for the vertex positions f,
            as per Crane §5.2. Otherwise, vertex positions are found by simply integrating tangent
            vectors, which may result in an improperly closed contour.

        laplacian
            The `(n, n)` Laplacian array. This is constructed automatically if not supplied.

        dual_edge_lengths
            The length `n` vector of dual edge lengths. This is constructed automatically
            if not supplied.

        Examples
        --------
        Construct a circle from its expected intrinsic parameters.

        ```python
        import numpy as np
        from curvey import Curve

        n = 20
        curvatures = np.ones(n)
        edge_lengths = 2 * np.pi / n * np.ones(n)
        c = Curve.from_curvature(curvatures, edge_lengths)
        _ = c.plot_edges()
        ```

        Construct a circle from noisy parameters, using `solve_vertices` to ensure the curve
        is closed.

        ```python
        curvatures = np.random.normal(1, 0.1, n)
        edge_lengths = 2 * np.pi / n * np.random.normal(1, 0.1, n)
        c0 = Curve.from_curvature(curvatures, edge_lengths, solve_vertices=False)
        c1 = Curve.from_curvature(curvatures, edge_lengths, solve_vertices=True)
        _ = c0.plot(color='black')
        _ = c1.plot(color='red')
        ```

        """
        n = len(curvature)
        l_next = edge_lengths  # l_next[i] is the edge length from vertex i to i+1

        if dual_edge_lengths is None:
            l_prev = roll(l_next, 1)  # from i-1 to i
            # length of the edge dual to vertex i
            dual_edge_lengths = (l_next + l_prev) / 2

        # Integrate curvatures to get edge angles
        theta = zeros(n)
        if theta0 is not None:
            theta[0] = theta0
        theta[1:] = theta[0] + cumsum(curvature[1:] * dual_edge_lengths[1:])

        # Unnormalized tangent vectors from vertex i to i+1
        t_next = l_next.reshape((-1, 1)) * angle_to_points(theta)

        if solve_vertices:
            l_prev = roll(l_next, 1)  # l_prev[i] is the edge length from vertex i-1 to i
            t_prev = roll(t_next, 1, axis=0)  # Vector from vertex i-1 to i

            # `b` is the discrete divergence of the new tangent field
            b = t_prev / l_prev[:, newaxis] - t_next / l_next[:, newaxis]
            if laplacian is None:
                laplacian = Curve._construct_laplacian(edge_lengths=l_next)

            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=scipy.sparse.linalg.MatrixRankWarning)

                try:
                    if scipy.sparse.issparse(laplacian):
                        pts = scipy.sparse.linalg.spsolve(laplacian, b)
                    else:
                        pts = np.linalg.solve(laplacian, b)
                # NB scipy.sparse raises numpy linalg errors
                except (np.linalg.LinAlgError, scipy.sparse.linalg.MatrixRankWarning):
                    pts = zeros((n, 2))
                    if pt0 is not None:
                        pts[0] = pt0
                    pts[1:] = pts[0] + cumsum(t_next[:-1], axis=0)

            if theta0 is not None:
                # Rotate to match requested first edge angle
                dx, dy = pts[1] - pts[0]
                theta1 = arctan2(dy, dx)
                pts @= rotation_matrix(theta1 - theta0)

            if pt0 is not None:
                pts -= pts[0] - pt0

        else:
            # Just integrate the tangent vectors
            pts = zeros((n, 2))
            if pt0 is not None:
                pts[0] = pt0
            pts[1:] = pts[0] + cumsum(t_next[:-1], axis=0)

        return cls(pts)

    def with_curvature(
        self,
        curvature: ndarray,
        solve_vertices=True,
        realign=False,
    ) -> Curve:
        """Construct a curve with the (approx.) same edge lengths and the supplied new curvatures.

        See method `Curve.from_curvature` for more details.

        Parameters
        ----------
        curvature
            A length `n` vector of signed curvatures.

        solve_vertices
            See `Curve.from_curvature`

        realign
            If True, the mean change in edge angle and vertex position is removed.
        """
        if realign:
            # Going to rotate the curve anyway, initial edge angle irrelevant
            theta0 = None
        else:
            dx, dy = self._pts[1] - self._pts[0]
            theta0 = arctan2(dy, dx)

        out = self.__class__.from_curvature(
            curvature=curvature,
            edge_lengths=self.edge_length,
            solve_vertices=solve_vertices,
            theta0=theta0,
            pt0=self._pts[0],
            dual_edge_lengths=self.dual_edge_length,
            laplacian=self.laplacian,
        )

        if realign:
            out = out.align_to(self)

        return out

    @overload
    def to_shapely(self, mode: Literal["ring"]) -> shapely.LinearRing: ...

    @overload
    def to_shapely(self, mode: Literal["edges"]) -> shapely.MultiLineString: ...

    @overload
    def to_shapely(self, mode: Literal["polygon"]) -> shapely.Polygon: ...

    @overload
    def to_shapely(self, mode: Literal["points"]) -> shapely.MultiPoint: ...

    def to_shapely(
        self,
        mode: Literal["ring", "edges", "polygon", "points"] = "ring",
    ):
        """Convenience converter to `shapely` object

        Parameters
        ----------
        mode
            Which type of `shapely` geometry to return.

              - 'ring': a `LinearRing` corresponding to the closed curve.
              - 'edges': a `MultiLineString` containing `n_edges` 2-point line segments.
              - 'polygon': a `Polygon` enclosed by the curve.
              - 'points': a `MultiPoint` containing the vertices.
        """
        if mode == "ring":
            return shapely.LinearRing(self._pts)

        if mode == "edges":
            pts0 = self._pts
            pts1 = roll(pts0, 1, axis=0)
            return shapely.MultiLineString(list(zip(pts0, pts1)))

        if mode == "polygon":
            return shapely.Polygon(self._pts)

        if mode == "points":
            return shapely.MultiPoint(self._pts)

        modes = ", ".join(("ring", "edges", "polygon", "points"))  # type: ignore [unreachable]
        msg = f"mode must be one of ({modes}), got {mode}"
        raise ValueError(msg)

    @cached_property
    def is_simple(self) -> bool:
        """False if the curve intersects or touches itself, including having repeated points

        Uses `shapely.LinearRing.is_simple`
        """
        return self.to_shapely("ring").is_simple

    def edge_intersections(self) -> ndarray:
        """An `(n_intersect, 2)` array of points where two edges cross

        This does not include two co-incident vertices or an edge coincident on
        a non-adjacent vertex.
        """
        # Use unary union to resolve all self-intersections
        mls: shapely.MultiLineString = shapely.unary_union(self.to_shapely("ring"))
        all_pts = shapely.extract_unique_points(mls)
        intersections = all_pts - self.to_shapely("points")  # Set difference

        if intersections.is_empty:
            return zeros((0, 2))

        if isinstance(intersections, shapely.Point):
            return array(intersections.coords)  # (1, 2)

        return concatenate([pt.coords for pt in intersections.geoms])

    @overload
    def register_to(
        self, target: Curve, allow_scale: bool, return_transform: Literal[False]
    ) -> Curve: ...

    @overload
    def register_to(
        self, target: Curve, allow_scale: bool, return_transform: Literal[True]
    ) -> ndarray: ...

    @overload
    def register_to(
        self, target: Curve, allow_scale: bool, return_transform: bool
    ) -> Curve | ndarray: ...

    def register_to(
        self,
        target: Curve,
        allow_scale: bool = False,
        return_transform=False,
    ) -> Curve | ndarray:
        r"""Iterative closest point registration

        Minimizes

        $$
        \sum_i \min_j d(v_i, e_j)^2
        $$

        where $d(v_i, e_j)$ is the euclidean distance btween vertices $v_i$ in `self`
        and edges $e_j$ in `target`.

        Parameters
        ----------
        target
            The `Curve` to register to/

        allow_scale
            If True, allow uniform scaling.

        return_transform
            If True, return a 3x3 transform matrix. Otherwise, return the transformed `Curve`.

        See also
        --------
        [Curve.align_to][curvey.curve.Curve.align_to]
            When `source` and `target` have the same number of vertices in 1-to-1 correspondance.
        """
        tree = shapely.STRtree(target.to_shapely("edges").geoms)

        def get_transform(params: ndarray) -> ndarray:
            """3x3 transform matrix"""
            if len(params) == 4:
                theta, dx, dy, scale_factor = params
            else:
                theta, dx, dy = params
                scale_factor = 1

            cos_theta, sin_theta = scale_factor * cos(theta), scale_factor * sin(theta)
            return array(
                [
                    [cos_theta, -sin_theta, dx],
                    [sin_theta, cos_theta, dy],
                    [0, 0, 1],
                ]
            )

        def sum_sq_dist_closest_pt(params: ndarray) -> float:
            transformed = self.transform(get_transform(params))
            (_self_idx, _other_idx), dists = tree.query_nearest(
                geometry=transformed.to_shapely("points").geoms,
                return_distance=True,
                all_matches=False,
            )
            return (dists**2).sum()

        opt = scipy.optimize.minimize(
            fun=sum_sq_dist_closest_pt,
            x0=array([0, 0, 0, 1] if allow_scale else [0, 0, 0]),
        )
        if not opt.success:
            msg = f"Optimization failed: {opt.message}"
            warnings.warn(msg, category=OptimizationFailed, stacklevel=2)

        transform = get_transform(opt.x)
        return transform if return_transform else self.transform(transform)

    def check_same_n_vertices(self, other: Curve) -> int:
        """Raises a `ValueError` if vertex counts don't match

        Otherwise, returns the common vertex count.
        """
        if self.n != other.n:
            msg = (
                "Curve pair must have the same number of vertices. "
                f"Got self.n = {self.n} and other.n = {other.n}"
            )
            raise MismatchedVertexCounts(msg)
        return self.n

    def roll_to(self, other: Curve) -> Curve:
        """Cyclicly permute points to minimize the distance between corresponding points

        `other` must have the same number of vertices as `self`
        """
        n = self.check_same_n_vertices(other)

        # (n, n) array of pairwise square distances
        dist = scipy.spatial.distance.cdist(other._pts, self._pts, "sqeuclidean")
        i0 = arange(n)

        # (n, n) cyclic permutation matrix
        #   [[0, 1, 2, 3, ..., n-1]
        #    [1, 2, 3, ..., n-1, 0]
        #    [2, 3, ..., n-1, 0, 1] ... ]
        i1 = (i0[:, newaxis] + i0[newaxis, :]) % n

        # The permutation index that minimizes sum of sq. dists
        i_min = cast(int, argmin(dist[i0, i1].sum(axis=1)))
        return self.roll(-i_min)

    def optimize_edge_lengths_to(
        self,
        other: Curve,
        interp_typ: InterpType = "cubic",
    ) -> Curve:
        """Optimize partitioning of vertex arclength positions to match edge_lengths in other

        `self` and `other` must have the same number of vertices.

        This assumes `self` and `other` have already been processed to have the same length!

        Parameters
        ----------
        other
            The curve to optimize against

        interp_typ
            Passed to `Curve.interpolator`
        """
        n = self.check_same_n_vertices(other)

        # So we don't need to refit each iteration
        interpolator = self.interpolator(typ=interp_typ)

        def _resample(ds: ndarray) -> Curve:
            arclength = append(0, cumsum(ds[:-1]))
            return self.with_points(interpolator(arclength))

        def _objective(ds: ndarray) -> float:
            resampled = _resample(ds)
            return ((other.closed_arclength - resampled.closed_arclength) ** 2).sum()

        # noinspection PyTypeChecker
        opt = scipy.optimize.minimize(
            fun=_objective,
            x0=other.edge_length,
            # Edge lengths must sum to total length
            constraints=scipy.optimize.LinearConstraint(ones(n), lb=other.length, ub=other.length),
            # Edge lengths must be positive
            bounds=scipy.optimize.Bounds(lb=0),
        )
        if not opt.success:
            msg = "Optimization failed: " + opt.message
            warnings.warn(msg, OptimizationFailed, stacklevel=2)

        return self.with_points(_resample(opt.x)._pts)


class NotEnoughPoints(Exception):
    """Raised if fewer than 3 points are passed to the `Curve` constructor"""


class WrongDimensions(Exception):
    """Raised if the points array is not 2d"""


class MismatchedVertexCounts(Exception):
    """Raised by methods expecting a pair of curves to have the same number of points"""


class OptimizationFailed(RuntimeWarning):
    """Raised by methods when optimization fails to converge"""
