from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cached_property, partial
from typing import Protocol

import shapely
from numpy import argmax, array, ndarray, pi, stack
from typing_extensions import Self

from ._approx_medial_axis import ApproxMedialAxisBuilder
from .curve import Curve
from .edge import Edges, Triangulation


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
    ---------
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

    def iter_boundaries(self) -> Iterator[Curve]:
        """Iterate over boundary curves"""
        yield self.exterior
        yield from self.interiors

    @classmethod
    def from_shapely(cls, poly: shapely.Polygon) -> Self:
        """Convert a `shapely.Polygon` to a `curvey.Polygon`"""
        exterior = Curve.from_shapely(poly.exterior)
        interiors = (Curve.from_shapely(c) for c in poly.interiors)
        return cls(exterior=exterior, interiors=interiors)

    def to_shapely(self) -> shapely.Polygon:
        """Convert a `curvey.Polygon` to a `shapely.Polygon`"""
        return shapely.Polygon(
            self.exterior.to_shapely("ring"), [c.to_shapely("ring") for c in self.interiors]
        )

    def to_edges(self) -> Edges:
        """All edges in the polygon boundaries"""
        return Edges.concatenate(*(c.to_edges() for c in self.iter_boundaries()))

    def apply(self, fn: CurveFn, *args, **kwargs) -> Polygon:
        """Apply a curve function to boundary curves

        ```python
        from curvey import Curve, Polygon
        poly = Polygon.from_char("e", family='arial')
        poly = poly.apply(Curve.split_longest_edges, thresh=1)
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
        return Polygon(exterior=exterior, interiors=interiors)

    def with_boundaries(self, exterior: Curve, interiors: Iterable[Curve]) -> Self:
        return self.__class__(exterior=exterior, interiors=interiors)

    def plot(self, **kwargs):
        for c in self.iter_boundaries():
            c.plot(**kwargs)

    def plot_edges(self, **kwargs):
        for c in self.iter_boundaries():
            c.plot_edges(**kwargs)

    def to_orientation(self, orientation: int = 1) -> Self:
        return self.with_boundaries(
            exterior=self.exterior.to_orientation(orientation),
            interiors=(c.to_orientation(-orientation) for c in self.interiors),
        )

    def to_ccw(self) -> Self:
        return self.to_orientation(1)

    def to_cw(self) -> Self:
        return self.to_orientation(-1)

    @classmethod
    def from_char(cls, char: str, **kwargs) -> Self:
        """Construct a polygon by drawing a character

        ```python
        from curvey import Polygon
        poly = Polygon.from_char("e", family='arial')
        poly.plot()
        ```
        """
        from matplotlib.font_manager import FontProperties
        from matplotlib.path import Path
        from matplotlib.text import TextToPath

        ttp = TextToPath()
        verts, codes = ttp.get_text_path(FontProperties(**kwargs), char)
        polys = Path(verts, codes).to_polygons()
        exterior = Curve(polys[0]).drop_repeated_points()
        interiors = (Curve(p).drop_repeated_points() for p in polys[1:])
        return cls(exterior, interiors)

    def _iter_hole_points(self) -> Iterator[ndarray]:
        """Yield points inside interior holes"""
        for c in self.interiors:
            tris = c.to_ccw().to_edges().triangulate()
            i = argmax(tris.signed_area)
            centroid = tris.points[tris.tris[i]].mean(axis=0)
            yield centroid

    @cached_property
    def hole_points(self) -> ndarray:
        """A `(len(self.interiors), 2)` array of points inside interior holes"""
        return stack(list(self._iter_hole_points()), axis=0)

    def triangulate(self, holes: ndarray | None = None, **kwargs) -> Triangulation:
        """Triangulate the polygon

        Parameters
        ----------
        holes
            Points inside interior holes. These are constructed automatically if not supplied.

        **kwargs
            Remaining kwargs passed to `Edges.triangulate`.

        """
        if holes is None and self.interiors:
            holes = self.hole_points

        edges = self.to_edges()
        return edges.triangulate(holes=holes, **kwargs)

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
        4. For each of the sampled points, construct vectors pointing at their closest points on the
        boundary.
        5. Compare the angles between the direction vectors of adjacent points
        6. If the difference in angle exceeds a threshold `angle_thresh`, the disc is assumed to
        intersect the medial axis at that point, and maximally inscribed disks at those points
        are added to a queue to be subsequently processed in turn.

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
            The AMA algorithm produces tree graph medial axis structures. As a final
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
            boundary=self.to_edges(),
            dist_thresh=dist_thresh,
            angle_thresh=angle_thresh,
            abs_err=abs_err,
            min_edge_length=min_edge_length,
            pt0=pt0,
            **kwargs,
        )
        b.run()
        return b.finalize(close_loops=close_loops)
