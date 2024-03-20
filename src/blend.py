from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable, Iterator

from numpy import ndarray, newaxis
from typing_extensions import Literal

from .curve import Curve
from .curves import Curves
from .flow import WillmoreFlow, Solver
from .util import InterpType


@dataclass
class Processed:
    """Convenience class to store a curve in both its original and processed version"""
    orig: Curve
    """The original untouched curve"""

    processed: Curve
    """The curve after resampling and rescaling, etc"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(orig={self.orig}, processed={self.processed})"


@dataclass
class Pair:
    """Convenience class to a store a pair of curves both processed to some common structure

    A `Pair` can be indexed, i.e. `pair[0]` and `pair[1]` return the first and second
    `Processed` object.

    A `Pair` can be iterated over.

    `Pair.orig` returns a 2-curve `Curves` object of the original curves.
    `Pair.processed` returns a 2-curve `Curves` object of the processed curves.

    `Pair`s are constructed from one of the staticmethods
    `Pair.to_common_vertex_count` and `Pair.to_common_edge_length`.
    """
    curve0: Processed
    curve1: Processed

    def __getitem__(self, item: int) -> Processed:
        if item == 0:
            return self.curve0
        elif item == 1:
            return self.curve1
        else:
            raise ValueError(item)

    def __iter__(self) -> Iterator[Processed]:
        yield self.curve0
        yield self.curve1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n\t{self.curve0},\n\t{self.curve1},\n)"

    @property
    def orig(self) -> Curves:
        return Curves([p.orig for p in self])

    @property
    def processed(self) -> Curves:
        return Curves([p.processed for p in self])

    @staticmethod
    def to_common_vertex_count(c0: Curve, c1: Curve) -> Pair:
        """Linear edge subdivision of the curve with fewer vertices until the count matches"""
        r0, r1 = c0, c1
        if c0.n > c1.n:
            r1 = c1.split_longest_edges(c0.n - c1.n)
        elif c1.n > c0.n:
            r0 = c0.split_longest_edges(c1.n - c0.n)
        return Pair(curve0=Processed(c0, r0), curve1=Processed(c1, r1))

    @staticmethod
    def to_common_edge_length(
            orig0: Curve,
            orig1: Curve,
            interp_typ: InterpType = 'cubic',
    ) -> Pair:
        orig0, orig1 = orig0, orig1
        orig0, orig1 = orig0.to_length(1.0), orig1.to_length(1.0)

        orig1 = orig1.optimize_edge_lengths_againt(orig0, interp_typ=interp_typ)
        orig0 = orig0.optimize_edge_lengths_againt(orig1, interp_typ=interp_typ)

        return Pair(
            curve0=Processed(orig0, orig0),
            curve1=Processed(orig1, orig1),
        )


class LinearBlending:
    """Linear vertex position interpolation

    Source and target curves must have the same number of vertices.
    """
    def __init__(self, src: Curve, tgt: Curve):
        _ = src.check_same_n_vertices(tgt)
        self.src = src
        self.tgt = tgt

    def interpolate(self, t: ndarray) -> Curves:
        """Interpolate curves at the requested times

        Parameters
        ----------
        t
            Vector of length `nt` with 0 <= t <= 1 of times to interpolate at.

        Returns
        -------
        curves
            A `Curves` object with `nt` curves. The value of `t` at each point
            is stored in the curve metadata parameter 'time'.

        """
        t_ = t.reshape((-1, 1, 1))  # (nt, 1, 1)
        src_pts = self.src.pts[newaxis, :, :]  # (1, n, 2)
        tgt_pts = self.tgt.pts[newaxis, :, :]  # (1, n, 2)
        pts = (1 - t_) * src_pts + t_ * tgt_pts  # (nt, n, 2)
        return Curves(curves=[Curve(p, time=ti) for (p, ti) in zip(pts, t)])


@dataclass
class CurvatureShapeBlending:
    def __init__(
            self,
            pair: Pair,
            initial: Curve,
            flow: Optional[WillmoreFlow] = None,
            exact_endpoints: Tuple[bool, bool] = (False, False),
            history: bool = False,
    ):
        self.pair = pair
        self.initial = initial
        self.flow = flow or WillmoreFlow(realign=False)
        self.exact_endpoints = exact_endpoints
        self._current_solver: Optional[Solver] = None
        self.history: Optional[List[Curves]] = [] if history else None

        self.curvature0 = self.curvature_fn(pair[0].processed)
        self.curvature1 = self.curvature_fn(pair[1].processed)

    @property
    def curvature_fn(self) -> Callable[[Curve], ndarray]:
        return self.flow.curvature_fn

    def interpolate_curvature(self, t: float) -> ndarray:
        return (1 - t) * self.curvature0 + t * self.curvature1

    def interpolate_length(self, t: float) -> float:
        return (1 - t) * self.pair[0].orig.length + t * self.pair[1].orig.length

    def interpolate_area(self, t: float) -> float:
        return (1 - t) * self.pair[0].orig.area + t * self.pair[1].orig.area

    def interpolate_once(
            self,
            t: float,
            initial: Optional[Curve] = None,
            interp_size: Optional[Literal['length', 'area']] = 'length',
            **kwargs
    ) -> Curve:
        if (t == 0 or t == 1) and self.exact_endpoints[int(t)]:
            return self.pair[int(t)].processed

        k_interp = self.interpolate_curvature(t)
        initial = initial or self.initial

        solver = self._current_solver = self.flow.solver(
            initial=initial,
            tgt_curvatures=k_interp,
            history=self.history is not None,
            **kwargs,
        )

        solver.log('Interpolating t = {}', t)
        solver.run()
        curve = solver.current

        # Get rid of flow-specific data
        curve = curve.drop_data('willmore_energy', 'step', 'timestep').with_data(
            time=t,
            src_error=self.flow.energy(curve, self.curvature0),
            tgt_error=self.flow.energy(curve, self.curvature1),
            interp_error=self.flow.energy(curve, k_interp)
        )

        if interp_size == 'length':
            curve = curve.to_length(self.interpolate_length(t))
        elif interp_size == 'area':
            curve = curve.to_area(self.interpolate_area(t))

        return curve

    def interpolate(
            self,
            t: ndarray,
            path_dependent: bool = False,
            realign: bool = True,
            interp_size: Optional[Literal['length', 'area']] = 'length',
            **kwargs,
    ) -> Curves:
        """Interpolate at multiple timepoints

        Additional kwargs passed to `Flow.solver`
        """
        curves = Curves()
        self.history = [] if self.history is not None else None
        curve = self.initial

        for i, t_interp in enumerate(t):
            interpolated = self.interpolate_once(
                t=t_interp,
                initial=curve,
                interp_size=interp_size,
                **kwargs,
            )

            if i > 0 and realign:
                interpolated = interpolated.align_to(curves[-1])

            curves.append(interpolated)

            if path_dependent:
                curve = interpolated

            if self.history is not None:
                self.history.append(self._current_solver.history)

        return curves
