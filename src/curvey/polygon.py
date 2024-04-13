"""A polygon bounded by curves"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cached_property, partial
from typing import Protocol, cast

import numpy as np
import shapely
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy import argmax, array, ndarray, pi, stack, zeros
from typing_extensions import Self

from .approx_medial_axis import ApproxMedialAxisBuilder
from .curve import Curve
from .edges import Edges
from .plot import _get_ax
from .triangulation import Triangulation


class CurveFn(Protocol):
    def __call__(self, _c: Curve, *args, **kwargs) -> Curve: ...


class Polygon:
    """A polygon defined by its boundary curves

    It's assumed that the interior curves have opposite orientation to the exterior,
    but this is not enforced.

    Parameters
    ----------
    exterior
        The exterior boundary `Curve`.

    interiors
        A (possibly) empty sequence of `Curve`s bounding holes in the polygon.

    """

    def __init__(self, exterior: Curve, interiors: Iterable[Curve] | None = None):
        self.exterior: Curve = exterior
        self.interiors: list[Curve] = []
        if interiors is not None:
            self.interiors.extend(interiors)

    def __repr__(self) -> str:
        interiors = ", ".join(repr(c) for c in self.interiors)
        return f"{self.__class__.__name__}(exterior={self.exterior}, interiors=({interiors}))"

    def boundaries(self) -> Iterator[Curve]:
        """Iterate over boundary curves"""
        yield self.exterior
        yield from self.interiors

    @classmethod
    def from_shapely(cls, poly: shapely.Polygon) -> Self:
        """Convert a `shapely.Polygon` to a `curvey.Polygon`"""
        exterior = Curve.from_shapely(poly.exterior)
        interiors = (Curve.from_shapely(c) for c in poly.interiors)
        return cls(exterior=exterior, interiors=interiors)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> list[Self]:
        """Construct a set of polygons from matplotlib's text rendering engine

        ```python
        from curvey import Polygon

        polys = Polygon.from_text("curvey", family="arial", size="18")
        for p in polys:
            p.plot()
        ```

        Parameters
        ----------
        text
            The string to render

        **kwargs
            Remaining kwargs passed to `matplotlib.font_manager.FontProperties`
        """
        from matplotlib.font_manager import FontProperties
        from matplotlib.path import Path
        from matplotlib.textpath import TextToPath

        ttp = TextToPath()
        verts, codes = ttp.get_text_path(FontProperties(**kwargs), text)
        poly_pts: list[ndarray] = cast(list[ndarray], Path(verts, codes).to_polygons())

        # Find interior and exterior boundaries
        # Convert to polygons first so that we can use the 'contains_properly' predicate
        polys = [shapely.Polygon(pts) for pts in poly_pts]
        tree = shapely.STRtree(polys)
        exterior_idxs, interior_idxs = tree.query(polys, "contains_properly")

        # Map interior boundaries to exteriors
        poly_idxs: dict[int, list[int]] = {
            i: [] for i in (set(range(len(polys))) - set(interior_idxs))
        }
        for e_idx, i_idx in zip(exterior_idxs, interior_idxs):
            poly_idxs[e_idx].append(i_idx)

        out: list[Self] = []
        for e_idx, i_idxs in poly_idxs.items():
            exterior = Curve(poly_pts[e_idx]).drop_repeated_points().to_ccw()
            interiors = (Curve(poly_pts[i]).drop_repeated_points().to_cw() for i in i_idxs)
            out.append(cls(exterior, interiors))

        return out

    def to_shapely(self) -> shapely.Polygon:
        """Convert a `curvey.Polygon` to a `shapely.Polygon`"""
        return shapely.Polygon(
            self.exterior.to_shapely("ring"), [c.to_shapely("ring") for c in self.interiors]
        )

    def to_edges(self) -> Edges:
        """An edge soup representation of all the edges in the polygon boundaries"""
        return Edges.concatenate(*(c.to_edges() for c in self.boundaries()))

    def apply(self, fn: CurveFn, *args, **kwargs) -> Self:
        """Apply a curve function to boundary curves

        ```python
        from curvey import Curve, Polygon
        poly = Polygon.from_text("e", family='arial')[0]
        poly = poly.apply(Curve.split_long_edges, thresh=1)
        poly.plot()
        ```

        Parameters
        ----------
        fn
            A function `Curve -> Curve`

        *args
        **kwargs
            Additional arguments passed to the function
        """
        fn = partial(fn, *args, **kwargs)
        exterior = fn(self.exterior)
        interiors = (fn(c) for c in self.interiors)
        return self.with_(exterior=exterior, interiors=interiors)

    def with_(self, exterior: Curve, interiors: Iterable[Curve]) -> Self:
        return self.__class__(exterior=exterior, interiors=interiors)

    @cached_property
    def signed_area(self) -> float:
        """Area enclosed by the polygon

        This is simply equal to the sum of the signed areas enclosed by the polygon boundaries.
        Signed area is positive if the polygon is positively (counter-clockwise) oriented.
        """
        return sum(c.signed_area for c in self.boundaries())

    @cached_property
    def area(self) -> float:
        """Absolute area"""
        return abs(self.signed_area)

    def plot(self, **kwargs):
        """Plot polygon boundary

        All kwargs are passed to [Curve.plot][curvey.curve.Curve.plot].
        """
        for c in self.boundaries():
            c.plot(**kwargs)

    def plot_edges(self, **kwargs):
        """Plot polygon boundary edges

        All kwargs are passed to [Curve.plot_edges][curvey.curve.Curve.plot_edges].
        """
        for c in self.boundaries():
            c.plot_edges(**kwargs)

    def to_orientation(self, orientation: int = 1) -> Self:
        """A polygon with the specified orientation

        Parameters
        ----------
        orientation
            Must be 1 or -1. `orientation=1` is a polygon whose exterior boundary is oriented
            counter-clockwise, and whose internal boundaries (if any) have clockwise orientation.
        """
        return self.with_(
            exterior=self.exterior.to_orientation(orientation),
            interiors=(c.to_orientation(-orientation) for c in self.interiors),
        )

    def to_ccw(self) -> Self:
        """A positively-oriented (counter-clockwise) polygon"""
        return self.to_orientation(1)

    def to_cw(self) -> Self:
        """A negatively-oriented (clockwise) polygon"""
        return self.to_orientation(-1)

    def _iter_hole_points(self) -> Iterator[ndarray]:
        """Yield points inside interior holes"""
        for c in self.interiors:
            tris = c.to_ccw().to_edges().triangulate()
            i = argmax(tris.signed_area)
            centroid = tris.points[tris.faces[i]].mean(axis=0)
            yield centroid

    @cached_property
    def hole_points(self) -> ndarray:
        """A `(len(self.interiors), 2)` array of points inside interior holes

        This is probably only useful for triangulation.
        """
        if pts := list(self._iter_hole_points()):
            return stack(pts, axis=0)

        return zeros((0, 2))

    @cached_property
    def boundary(self) -> Edges:
        """Boundary edges as an edge soup"""
        return Edges.concatenate(*(c.to_edges() for c in self.boundaries()))

    def triangulate(
        self,
        max_tri_area: float | None = None,
        min_angle: float | None = None,
        extra_params: str | None = None,
        **kwargs,
    ) -> Triangulation:
        """Triangulate the polygon

        Parameters
        ----------
        max_tri_area
            A global maximum triangle area constraint.

        min_angle
            Minimum angle constraint, in degrees.

        extra_params
            See the [API documentation](https://rufat.be/triangle/API.html).
            E.g. `extra_params='S10X' specifies a maximum number of 10 Steiner points and suppresses
            exact arithmetic.

        **kwargs
            Remaining kwargs passed to `Edges.triangulate`.

        """
        tris = self.boundary.triangulate(
            polygon=True,
            max_tri_area=max_tri_area,
            min_angle=min_angle,
            holes=self.hole_points,
            extra_params=extra_params,
            **kwargs,
        )
        tris.boundary_edges = self.boundary  # overwrite cached property defn
        return tris

    def approximate_medial_axis(
        self,
        dist_thresh: float,
        abs_err: float,
        angle_thresh: float = pi / 3,
        min_edge_length: float | None = None,
        pt0: ndarray | None = None,
        close_loops: bool = True,
        **kwargs,
    ) -> Edges:
        """Construct the approximate medial axis of the polygon

        Implementation of [*Efficient and Robust Computation of an Approximated Medial Axis.*
        Yuandong Yang, Oliver Brock, and Robert N. Moll. 2004.](
        https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cfc187181ce85d983843c4a184651dbd2a07e7e5)

        The algorithm operates as follows:

        1. Locate an initial point on the medial axis.
        2. Construct a maximally inscribed disk at that point
        3. Uniformly sample points on the boundary of that disk
        4. For each of the sampled points, construct direction vectors pointing at their
        corresponding closest points on the polygon boundary.
        5. Compare the angles between the direction vectors of adjacent points sampled on the
        disk boundary. If the vectors diverge, i.e. the difference in angle exceeds a threshold
        `angle_thresh`, the disk is assumed to intersect the medial axis at midpoint between those
        two adjacent points.
        6. Points found in the previous step are added to the medial axis, and also added to a queue
        to repeated sample maximally inscribed disks as per steps 2-5.

        Parameters
        ----------
        dist_thresh
            Distance from the boundary to stop propagating the medial axis.

        abs_err
            The error allowed in the MA vertex positions. Smaller numbers sample inscribed disks
            more finely.

        angle_thresh
            Angle discreprancy (in radians) to count as a medial axis intersection. Default
            is $pi / 3$.

        min_edge_length
            Prevent adding new vertices if they're within this distance of other vertices

        pt0
            A arbitrary starting point interior to the polygon to begin searching for the medial
            axis. If not supplied, this is chosen automatically by choosing the centroid of the
            largest triangle of the triangulated polygon.

        close_loops
            The standard AMA algorithm produces medial axes in the form of a tree graph. As a final
            post-processing step, look for pairs of leaf vertices within eachother's disks
            and add edges connecting them.

        Returns
        -------
        ama :
            The approximate medial axis as an `curvey.edge.Edges` object. The distance of each
            vertex in the medial axis from the polygon boundary is stored in the `distance`
            point data property.

        """
        if pt0 is None:
            tris = self.triangulate()
            tri = max(tris.shapely.geoms, key=lambda t: t.area)
            pt0 = array(tri.centroid.coords[0])

        b = ApproxMedialAxisBuilder(
            boundary=self.boundary,
            dist_thresh=dist_thresh,
            angle_thresh=angle_thresh,
            abs_err=abs_err,
            min_edge_length=min_edge_length,
            pt0=pt0,
            **kwargs,
        )
        b.run()
        return b.finalize(close_loops=close_loops)

    def to_matplotlib(self) -> Path:
        """Convert a `Polygon` to a `matplotlib.path.Path`"""
        from matplotlib.path import Path

        paths = [c.to_matplotlib() for c in self.boundaries()]
        vertices = np.concatenate([p.vertices for p in paths], axis=0)
        codes = np.concatenate([p.codes for p in paths])
        return Path(vertices, codes)

    def plot_polygon(self, ax: Axes | None = None) -> PathPatch:
        """Plot a filled polygon"""
        ax = _get_ax(ax)
        patch = PathPatch(self.to_matplotlib())
        ax.add_patch(patch)
        ax.update_datalim(self.exterior.points)
        ax.autoscale_view()
        return patch
