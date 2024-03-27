"""Curve blending and shape interpolation"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from numpy import arctan2, eye, ndarray, newaxis, sqrt
from typing_extensions import Literal

from .curve import Curve
from .curves import Curves
from .flow import WillmoreFlow
from .util import rotation_matrix


@dataclass
class Processed:
    """Convenience class to store a curve in both its original and processed version"""

    original: Curve
    """The original untouched curve"""

    processed: Curve
    """The curve after resampling and rescaling, etc"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(original={self.original}, processed={self.processed})"


class LinearBlending:
    """Linear vertex position interpolation

    Source and target curves must have the same number of vertices.
    """

    def __init__(self, src: Curve, tgt: Curve):
        _ = src.check_same_n_vertices(tgt)
        self.src = src
        self.tgt = tgt

    def interpolate(self, t: ndarray) -> Curves:
        r"""Interpolate curves at the requested times

        Parameters
        ----------
        t
            Vector of length `nt` with $0 \le t \le 1$ of times to interpolate at.

        Returns
        -------
        curves
            A `Curves` object with `nt` curves. The value of `t` at each point
            is stored in the curve metadata parameter 'time'.

        """
        t_ = t.reshape((-1, 1, 1))  # (nt, 1, 1)
        src_pts = self.src.points[newaxis, :, :]  # (1, n, 2)
        tgt_pts = self.tgt.points[newaxis, :, :]  # (1, n, 2)
        pts = (1 - t_) * src_pts + t_ * tgt_pts  # (nt, n, 2)
        return Curves(self.src.with_points(p).with_data(time=ti) for (p, ti) in zip(pts, t))


class CurvatureShapeBlending:
    """Curvature-based shape blending

    Based on

    [*Rapid blending of closed curves based on curvature flow.*
    Masahiro Hirano, Yoshihiro Watanabe, and Masatoshi Ishikawa. 2017.](
        https://www.sciencedirect.com/science/article/pii/S016783961730016X)

    Parameters
    ----------
    processed0
    processed1
        The processed pair storing the original curves and processed curves. Curvature flow
        interpolation requires the processed curves to have

        1. The same number of vertices
        2. The same total length
        3. Reasonably similar edge lengths between corresponding edge pairs.

    initial
        The initial curve to start targeted curvature flow from. If not supplied this
        defaults to `pair[0].processed`, but I find curvature blending to work best when
        the initial processed curve is Willmore-flowed to a circle.

    flow
        The `WillmoreFlow` object to use for curvature flows. This can be supplied if some default
        parameters need to be overridden, but is usally left `None` by default for automatic
        construction

    Examples
    --------
    ```python
    import numpy as np
    from curvey import Curve
    from curvey.blend import CurvatureShapeBlending

    n = 128
    src = Curve.circle(n=n)
    tgt = Curve.star(6, r0=1, r1=2).interpolate_n(n).scale(1.5)
    csb = CurvatureShapeBlending.preprocess(src, tgt)
    t_interp = np.linspace(0, 1, 5)
    curves = csb.interpolate(t=t_interp, stop_tol=1e-3, interp_size='area')
    curves.subplots()
    ```

    """

    def __init__(
        self,
        processed0: Processed,
        processed1: Processed,
        initial: Curve,
        flow: WillmoreFlow | None = None,
    ):
        self.processed0 = processed0
        self.processed1 = processed1
        self.initial = initial
        self.flow = flow or WillmoreFlow(realign=False)
        self.history: list[Curves | None] | None = None
        self.curvature0 = self.flow.curvature_fn(processed0.processed)
        self.curvature1 = self.flow.curvature_fn(processed1.processed)

    @property
    def processed(self) -> tuple[Curve, Curve]:
        """The pair of processed `Curve`s"""
        return self.processed0.processed, self.processed1.processed

    @property
    def original(self) -> tuple[Curve, Curve]:
        """The pair of original `Curve`s"""
        return self.processed0.original, self.processed1.original

    def _interpolate_curvature(self, t: float) -> ndarray:
        return (1 - t) * self.curvature0 + t * self.curvature1

    def _interpolate_length(self, t: float) -> float:
        c0, c1 = self.original
        return (1 - t) * c0.length + t * c1.length

    def _interpolate_area(self, t: float) -> float:
        c0, c1 = self.original
        return (1 - t) * c0.area + t * c1.area

    def _interpolate_position(self, t: float, mode: Literal["center", "centroid"]) -> ndarray:
        c0, c1 = self.original
        return (1 - t) * getattr(c0, mode) + t * getattr(c1, mode)

    def _interpolate_once(
        self, t: float, initial: Curve | None, exact_endpoints: tuple[bool, bool], **kwargs
    ) -> tuple[Curves | None, Curve]:
        k_interp = self._interpolate_curvature(t)

        flow = self.flow.with_tgt_curvature(k_interp)

        if (t in (0, 1)) and exact_endpoints[int(t)]:
            return None, self.processed[int(t)]

        initial = initial or self.initial

        solver = self._current_solver = flow.solver(
            initial=initial,
            history=self.history is not None,
            **kwargs,
        )

        solver.log("Interpolating t = {}", t)
        solver.run()
        curve = solver.current

        # Get rid of flow-specific metadata
        curve = curve.drop_data("willmore_energy", "step", "timestep")

        # Add interpolation-specific metadata
        curve = curve.with_data(
            time=t,
            src_error=sqrt(flow.energy(curve=curve, tgt_curvature=self.curvature0)),
            tgt_error=sqrt(flow.energy(curve=curve, tgt_curvature=self.curvature1)),
            interp_error=sqrt(flow.energy(curve=curve, tgt_curvature=k_interp)),
        )

        return solver.history, curve

    def interpolate(
        self,
        t: ndarray,
        path_dependent: bool = False,
        realign: bool = True,
        interp_size: Literal["length", "area"] | None = "length",
        interp_pos: Literal["center", "centroid"] | None = "center",
        reregister: bool = False,
        exact_endpoints: tuple[bool, bool] = (False, False),
        post_process: Callable[[Curve], Curve] | None = None,
        history: bool = False,
        **kwargs,
    ) -> Curves:
        """Interpolate shape at the supplied time-points

        Parameters
        ----------
        t
            Vector of interpolation parameter values in [0, 1]

        path_dependent
            If True, the resulting interpolated curve from each timepoint is used as the starting
            point for the next interpolation. Otherwise, each timepoint's flow is independently
            started from the `self.initial` curve.

        realign
            If True, the curve after each interpolation step is aligned to the preceeding step.
            This can account for changes in position and orientation. Realignment is performed
            before transforming by the interpolated size or position.

        interp_size
            If not None, scale each interpolated curve to the area or length obtained by
            interpolating the `self.pair.orig` curves' length or area.

        interp_pos
            If not None, translate the interpolated curves' center or centroid to the position
            interpolated between the `self.pair.orig` curves' center or centroid.

        reregister
            If True, perform a final post-processing step as follows: register the final
            interpolated curve against the original curve using the `Curve.register_to`
            iterative closest point algorithm. The total change in position, orientation, and
            scale is then interpolated by `t` for each interpolated curve. This generally assumes
            that 1) t=0 and t=1 are included in the interpolation points, and 2) `interp_size`
            and `interp_pos` are enabled so that the interpolated curve is close enough to the
            original curve for ICP to be successful.

        exact_endpoints
            By default, the curves at t=0 and t=1 are found by the same targeted curvature flow
            as for the interior time points. This skips that step and simply uses
            `self.pair[t].processed` if `exact_endpoints[t] == True`, for `t = 0` and `t = 1`.

        history
            If true, the history of each interpolating flow is stored in `self.history`
            as a length(t) `list[Curves]`.

        post_process
            An option function `Curve -> Curve` to apply to each interpolated curve.

        **kwargs
            Additional kwargs passed to `WillmoreFlow.solver` every interpolation step.

        Returns
        -------
        Curves
            A `Curves` of length `len(t)`.
        """
        curves = Curves()
        self.history = [] if history else None
        initial = self.initial
        raw = []  # Before interpolating size, position, etc

        for i, t_interp in enumerate(t):
            history_i, c_interp = self._interpolate_once(
                t=t_interp,
                initial=initial,
                exact_endpoints=exact_endpoints,
                **kwargs,
            )
            raw.append(c_interp)

            if i > 0 and realign:
                c_interp = c_interp.align_to(raw[-1])

            if interp_size == "length":
                c_interp = c_interp.to_length(self._interpolate_length(t_interp))
            elif interp_size == "area":
                c_interp = c_interp.to_area(self._interpolate_area(t_interp))

            if interp_pos:
                pos = self._interpolate_position(t_interp, mode=interp_pos)
                c_interp = c_interp.translate(pos)

            if post_process:
                c_interp = post_process(c_interp)

            curves.append(c_interp)

            if path_dependent:
                initial = raw[-1]

            if self.history is not None:
                self.history.append(history_i)

        if reregister:
            curves = self._reregister(curves, t=t)

        return curves

    def _reregister(self, curves: Curves, t: ndarray) -> Curves:
        # Register the final interpolated curve to the original curve
        # A 3x3 transformation matrix
        transform = curves[-1].register_to(
            target=self.processed1.original,
            allow_scale=True,
            return_transform=True,
        )

        # Extract rotation, scale, and translation from the transform matrix
        r_cos_theta, r_sin_theta = transform[:2, 0]
        theta = arctan2(r_sin_theta, r_cos_theta)
        scale_factor = sqrt(r_cos_theta**2 + r_sin_theta**2)
        offset = transform[:2, 2]

        out = Curves()
        transform = eye(3)

        for c, ti in zip(curves, t):
            transform[:2, :2] = ((1 - ti) + scale_factor * ti) * rotation_matrix(theta * ti)
            transform[:2, 2] = offset * ti
            out.append(c.transform(transform))

        return out

    @staticmethod
    def preprocess(
        c0: Curve,
        c1: Curve,
        flow: WillmoreFlow | None = None,
        circle_stop_tol=1e-3,
    ) -> CurvatureShapeBlending:
        """Preprocess a pair of curves for curvature shape blending

        Scales both curves to a common length and optimizes their edge length distributions.
        Also flows the first curve to a circle for use as the `initial` curve for flow-based
        interpolation.
        """
        _ = c0.check_same_n_vertices(c1)
        o0, o1 = c0, c1
        c0, c1 = c0.to_length(1.0), c1.to_length(1.0)
        c1 = c1.optimize_edge_lengths_to(c0)
        c0 = c0.optimize_edge_lengths_to(c1)
        p0 = Processed(original=o0, processed=c0)
        p1 = Processed(original=o1, processed=c1)

        flow = flow or WillmoreFlow()
        solver = flow.solver(initial=c1, stop_tol=circle_stop_tol, history=False)
        initial = solver.run().current

        return CurvatureShapeBlending(
            processed0=p0,
            processed1=p1,
            initial=initial,
            flow=flow,
        )
