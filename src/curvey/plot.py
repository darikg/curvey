from __future__ import annotations

from typing import Any, NamedTuple, cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.quiver import Quiver
from numpy import asarray, ndarray, newaxis
from numpy.linalg import norm
from numpy.typing import ArrayLike

from curvey.util import _rescale


class _VariableColorSpec(NamedTuple):
    """Interpret a colorspec that could be either a constant color or an array of scalars

    The reason we return both variables instead of an enum is for convenience with e.g.
    matplotlibs quiver, which takes two disjoint arguments, `c` (varied) and `color` (fixed)

    """

    fixed: Any = None
    varied: ArrayLike | None = None

    @property
    def maybe_varied(self) -> tuple[()] | tuple[ArrayLike]:
        # Quiver doesn't allow named parameter for the varied color argument,
        # so we need a little splatting magic
        if self.varied is None:
            return ()

        return (self.varied,)

    @staticmethod
    def parse(
        n_data: int | None,
        supplied: Any,
        default_varied=None,
        default_fixed=None,
    ) -> _VariableColorSpec:
        if supplied is None:
            return _VariableColorSpec(fixed=default_fixed, varied=default_varied)

        if isinstance(supplied, str):  # e.g. color='black'
            return _VariableColorSpec(fixed=supplied, varied=None)

        try:
            supplied_len = len(supplied)
        except TypeError as e:
            msg = "Expected `color` to have a length"
            raise NotImplementedError(msg) from e

        if n_data == supplied_len == 3:
            # This is ambiguous and hopefully rare
            # Could raise a warning or something but whatever
            return _VariableColorSpec(fixed=None, varied=asarray(supplied))

        if n_data == supplied_len:
            return _VariableColorSpec(fixed=None, varied=asarray(supplied))

        msg = f"Expected color to be of length {n_data}, got {supplied_len}"
        raise ValueError(msg)


def _get_ax(ax: Axes | None) -> Axes:
    if ax:
        return ax

    if len(plt.get_fignums()) == 0:
        # making a new axes so don't feel bad about setting some defaults
        _fig, ax = plt.subplots()
        ax = cast(Axes, ax)
        ax.axis("equal")
        return ax

    return plt.gca()


def quiver(
    points: ndarray,
    vectors: ndarray,
    scale: ndarray | None = None,
    color=None,
    scale_length: tuple[float, float] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Quiver:
    """Convenience wrapper around `matplotlib.pyplot.quiver`

    Differences:

    - Points and vectors are both (n, 2) arrays instead of four vectors or scalars X, Y, U, V
    - Color specification combines both the fixed scalar kwarg `color` and the generic
      array `C` into a single argument.
    - Angles and units for the vectors are always the same as for the points
    - Axes datalimits are updated to include vector heads.

    Parameters
    ----------
    points
        A `(n, 2)` array of vector heads.

    vectors
        A `(n, 2)` array of vectors.

    scale
        A length `n` vector of length scalars to apply to the vectors.

    color
        Length `n` vector of values to color by, or a
        constant color for all edges.

    scale_length
        Limits to scale vector length to, after applying `scale`.

    ax
        The axes to plot in. Defaults to the current axes.

    **kwargs
        additional kwargs passed to `matplotlib.pyplot.quiver`
    """
    ax = _get_ax(ax)

    points, vectors = asarray(points), asarray(vectors)
    if points.ndim == 1:
        points = points[newaxis]
    if vectors.ndim == 1:
        vectors = vectors[newaxis]

    n = len(points)

    if scale is not None:
        vectors = scale.reshape(-1, 1) * vectors

    if scale_length is not None:
        length = norm(vectors, axis=1, keepdims=True)
        scaled_length = _rescale(length, scale_length)
        vectors = vectors / length * scaled_length

    colorspec = _VariableColorSpec.parse(n, color, default_fixed="black")

    # By default quiver doesn't include vector endpoints in x/y lim calculations
    ax.update_datalim(points + vectors)

    # TODO if width is None, set it to some fraction of the bounding box

    (x, y), (dx, dy) = points.T, vectors.T
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


def segments(
    points: ndarray,
    edges: ndarray,
    directed: bool = False,
    color=None,
    width: float | ndarray | None = None,
    scale_width: tuple[float, float] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> LineCollection | Quiver:
    """Plot line segments supplied as separate points and edge arrays

    Parameters
    ----------
    points
        `(m, 2)` array of vertex coordinates

    edges
        `(n, 2)` integer array of indices into `points` defining the start and end of each segment.

    directed
        If true, plot as a quiver plot.

    color
        The color to plot each edge. Can be a constant color or vector of length `n` specifying
        a color for each edge.

    width
        The thickness of each edge segment, scalar or edge quantity vector.

    scale_width
        Min and max widths to scale the edge quantity to.

    ax
        The matplotlib axes to plot in. Defaults to current axes.

    **kwargs
        Aadditional kwargs passed to `matplotlib.collections.LineCollection`.
    """
    ax = _get_ax(ax)

    if directed:
        return quiver(
            points=points[edges[:, 0]],
            vectors=points[edges[:, 1]] - points[edges[:, 0]],
            color=color,
            ax=ax,
            width=width,
            **kwargs,
        )

    colorspec = _VariableColorSpec.parse(len(edges), supplied=color)
    if colorspec.fixed is not None:
        kwargs["color"] = colorspec.fixed

    width = _rescale(width, scale_width)

    lc = LineCollection(
        segments=points[edges],
        linewidths=width,
        **kwargs,
    )

    if colorspec.varied is not None:
        lc.set_array(colorspec.varied)

    ax.add_collection(lc)

    # Adding a line collection doesn't update limits so do it here
    ax.update_datalim(points)
    ax.autoscale_view()
    return lc
