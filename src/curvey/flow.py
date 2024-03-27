"""Definitions of flow infrastructure and implementations of some common flows"""

from __future__ import annotations

import enum
import logging
from abc import abstractmethod
from copy import copy
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    TypedDict,
    TypeVar,
    cast,
)

import numpy as np
import scipy
from numpy import clip, eye, nan, ndarray, newaxis, ones, sqrt
from typing_extensions import Self

from .curve import Curve
from .curves import Curves
from .util import InterpType

logger = logging.getLogger(__name__)


class _BraceMessage:
    """For lazy {} style log formatting"""

    def __init__(self, msg: str | None = None, *args: Any, **kwargs: Any):
        self.msg = [msg] if msg else []
        self.args = list(args)
        self.kwargs = kwargs
        self._str: str | None = None

    def __str__(self):
        if self._str is None:
            msg = "".join(self.msg)
            self._str = msg.format(*self.args, **self.kwargs)
        return self._str

    def append(self, msg: str, *args, **kwargs):
        self.msg.append(msg)
        self.args.extend(args)
        self.kwargs.update(kwargs)


TData = TypeVar("TData")
TData.__doc__ = "Flow-specific solver data type"


class AbstractFlow(Generic[TData]):
    """Abstract superclass for curve flow

    The basic contract is that `Flow` objects don't maintain any state specific
    to the solution of a flow. All state is stored in the curve metadata
    or in the `Solver.data` class, which is generic over the `TData` type.

    Two methods for subclasses to implement: `step`, which steps the curve by the
    supplied timestep, and `solver`, which constructs the auxillary `Solver` object with
    flow-specific `data: TData`.

    """

    @abstractmethod
    def solver(self, initial: Curve, **kwargs) -> Solver[TData]:
        """Construct a `Solver` to solve curve flow over time

        **kwargs are all passed to the `Solver` constructor.
        """
        ...

    @abstractmethod
    def step(self, curve: Curve, timestep: float, solver: Solver[TData]) -> Curve:
        """Step the curve by `timestep`"""
        ...

    def poststep(self, curve: Curve, solver: Solver[TData]) -> Curve:  # noqa: ARG002
        """Called after stepping the curve, but before logging it

        This is called after attaching additional curve metadata requested by curve loggers.

        Subclasses can raise `RetryStep` or `StopEarly` here if necessary, or further process the
        curve.
        """
        return curve


class RetryStep(Exception):
    """This can be raised in a custom `Solver.step_fn` to retry the current step

    Usually after adjusting the timestep or some other state.
    """


class StopEarly(Exception):
    """This can be raised in a custom `Solver.step_fn` to stop the current run

    Usually after reaching some stopping criterion.
    """


class Solver(Generic[TData]):
    """Auxillary class for solving curve `Flow`s

    Parameters
    ----------
    flow
        The `Flow` object we're solving

    initial
        The initial `Curve` to start solving from

    timestep
        For fixed timesteps

    timestep_fn
        A function `Solver -> float` that can adaptively decide a timestep on each iteration.

    history
        If true, the `Curve` after each iteration is stored in `Solver.history`, a `Curves` object.

    max_step
        Maximum number of iterations to run.

    stop_on_non_simple
        A step whose `Curve.is_simple == False` stops the run, discarding the non-simple
        curve.

    verbose
        If true, curve state information and stopping messages are printed to stdout on each
        iteration.

    log
        If true, the printed log messages as in `verbose` are saved as a list of `str`s in
        `Solver.log`

    data
        Flow specific data.

    step_fn
        A function `Solver -> Curve` that steps the curve forward at each iteration. This just
        defaults to `Solver.step`.

    """

    def __init__(
        self,
        *,
        flow: AbstractFlow,
        initial: Curve,
        data: TData,
        timestep: float | None = None,
        timestep_fn: Callable[[Solver], float] | None = None,
        history: bool = True,
        max_step: int | None = None,
        stop_on_non_simple: bool = False,
        verbose: bool = False,
        log: bool = False,
        step_fn: Callable[[Solver], Curve] | None = None,
    ):
        self.flow = flow
        self.initial = initial
        self.current = initial.with_data(time=0, step=0)
        self.previous: Curve | None = None
        self.timestep = timestep
        self.timestep_fn = timestep_fn
        self.history: Curves | None = None
        if history:
            # Note that we don't log the first curve until
            # self.run() in case some extra initialization needs to happen somewhere
            self.history = Curves([])
        self.log_history: list[str] | None = [] if log else None
        self.max_step = max_step
        self.stop_on_non_simple = stop_on_non_simple
        self.verbose = verbose

        self._stop_fns: list[Callable[[Solver], bool]] = []
        self._curve_loggers: dict[str, Callable[[Curve], Any]] = {}
        self.data = data
        self.step_fn = step_fn

    def __repr__(self) -> str:
        solver_name = self.__class__.__name__
        flow_name = self.flow.__class__.__name__
        return f"{solver_name}(flow={flow_name}, current={self.current})"

    def log(self, msg: str, *args, **kwargs):
        """Log a message

        This is always sent to the module `logging.logger` at debug level.
        If `self.verbose` is true, it's also printed to stdout.
        If `self.log` is true, it's saved as a str in self.log_history
        """
        self._log(_BraceMessage(msg, *args, **kwargs))

    def _log(self, bm: _BraceMessage):
        logger.debug(bm)
        if self.verbose:
            pass
        if self.log_history is not None:
            self.log_history.append(str(bm))

    def _log_state(self, msg: str):
        m = _BraceMessage(msg=msg)

        for k, v in self.current.data.items():
            if k != "step":
                m.append(", {} = {}", k, v)

        self._log(m)

    def _log_step(self, c1: Curve):
        m = _BraceMessage("Step {}", c1["step"])
        c0 = self.current
        ks = (c0.data.keys() & c1.data.keys()) - {"step"}
        for k in ks:
            m.append(", {}: {} => {}", k, c0[k], c1[k])
        self._log(m)

    def _stop_fn(self) -> bool:
        """Returns True if run should stop"""
        if self.max_step is not None and self.current["step"] == self.max_step:
            self.log("Stopping at max step {}", self.current["step"])
            return True

        for fn in self._stop_fns:
            if fn(self):
                self.log("Stopping due to stop fn {}", fn)
                return True

        return False

    def add_stop_fn(self, fn: Callable[[Solver], bool]) -> Self:
        """Add a custom stop function. The run is stopped early if `fn(curve)` returns True

        Returns
        -------
        self

        """
        self._stop_fns.append(fn)
        return self

    def add_curve_loggers(self, **kwargs: Callable[[Curve], Any]) -> Self:
        """Log additional information as curve metadata

        e.g. `solver.add_curve_loggers(foo=foo_fn, bar=bar_fn) will store the results of
        the function calls `foo_fn(curve)` and `bar_fn(curve)` in the curve metadata 'foo' and 'bar'
        properties.

        Returns
        -------
        self

        """
        self._curve_loggers.update(kwargs)
        return self

    def stop_on_param_limits(
        self,
        param: str,
        min_val=None,
        max_val=None,
        param_fn: Callable[[Curve], Any] | None = None,
    ) -> Self:
        """Add a custom stop function based on a parameter value

        Parameters
        ----------
        param
            The name of the parameter. This is usually a curve metadata object, e.g. one logged
            via `Solver.add_curve_loggers`. If `param_fn` is supplied, `param` is only used as the
            parameter name for logging purposes.

        min_val
            The run is stopped if the parameter value < `min_val`. `None` means no lower limit.

        max_val
            The run is stopped if the parameter value > `max_val`. `None` means no upper limit.

        param_fn
            An optional function `Curve -> value`; if supplied `param` doesn't need to be available
            as `Curve` metadata.

        Returns
        -------
        self

        """
        if param_fn is None:

            def param_fn(c: Curve) -> Any:
                return c[param]

        def param_limits_stop_fn(solver: Solver) -> bool:
            val = param_fn(solver.current)

            if min_val is not None and val < min_val:
                solver.log("Parameter {} value {} < min value {}, stopping", param, val, min_val)
                return True

            if max_val is not None and val > max_val:
                solver.log("Parameter {} value {} > max value {}, stopping", param, val, max_val)
                return True

            return False

        self.add_stop_fn(param_limits_stop_fn)
        return self

    def run(self):
        """Solve the flow by stepping the curve through time

        If no stop criterion are specified by `max_step`, `add_stop_fn`, or
        `stop_on_param_limits`, this might run forever.

        Returns
        -------
        self

        """
        self.current = self.attach_metadata(self.initial, time=0, step=0)
        if self.history is not None:
            self.history.append(self.current)

        self._log_state("Initial state")
        step_fn = self.step_fn or Solver.step

        while not self._stop_fn():
            try:
                next_curve = step_fn(self)
                next_curve = self.flow.poststep(curve=next_curve, solver=self)
            except RetryStep:
                continue
            except StopEarly:
                break

            if self.stop_on_non_simple and not next_curve.is_simple:
                self.log("Curve is non-simple, stopping")
                break

            self._log_step(next_curve)
            self.previous = self.current
            self.current = next_curve
            if self.history is not None:
                self.history.append(next_curve)

        self._log_state("Final state")
        return self

    def step(self) -> Curve:
        """Call `Flow.step` with the current state and timestep"""

        if self.timestep_fn is not None:
            timestep = self.timestep_fn(self)
        elif self.timestep is not None:
            timestep = self.timestep
        else:
            msg = "Neither of `timestep` or `timestep_fn` were provided."
            raise ValueError(msg)

        # The `Flow` does the actual work here
        curve = self.flow.step(
            curve=self.current,
            timestep=timestep,
            solver=self,
        )

        return self.attach_metadata(
            curve=curve,
            time=self.current["time"] + timestep,
            timestep=timestep,
        )

    def attach_metadata(
        self, curve: Curve, time: float, step: int | None = None, **kwargs
    ) -> Curve:
        """Store requested metadata on the curve

        Parameters
        ----------
        curve
            The curve after the most recent step.

        time
            The time of the curve in the solution.

        step
            Which step this curve belongs to. This is almost always left None; it defaults
            to `solver.current['step'] + 1`.

        **kwargs
            Additional metadata to store as key=value pairs.

        Returns
        -------
        curve
            The curve with metadata attached.
        """
        if step is None:
            step = self.current["step"] + 1

        params = dict(time=time, step=step, **kwargs)
        for k, fn in self._curve_loggers.items():
            params[k] = fn(curve)
        return curve.with_data(**params)


class AbstractCurvatureFlow(AbstractFlow[TData], Generic[TData]):
    """Abstract superclass for curvature flows

    Parameters
    ----------
    curvature_fn
        A function `Curve -> ndarray` that returns the signed curvature values at each vertex.

    rescale
        If this is 'length' or 'area', the recurve length/area is rescaled to the original's
        length or area, preventing the usual curvature flow shrinkage.
    """

    def __init__(
        self,
        curvature_fn: Callable[[Curve], ndarray] | None = None,
        rescale: Literal["length", "area"] | None = None,
    ):
        super().__init__()
        self.curvature_fn = curvature_fn or self.default_curvature_fn
        self.rescale = rescale

    @abstractmethod
    def step(self, curve: Curve, timestep: float, solver: Solver[TData]) -> Curve: ...

    @abstractmethod
    def solver(self, initial: Curve, **kwargs) -> Solver[TData]: ...

    def _postprocess(self, curve: Curve, solver: Solver[TData]) -> Curve:
        if self.rescale == "area":
            curve = curve.scale(sqrt(solver.initial.area / curve.area))
        elif self.rescale == "length":
            curve = curve.scale(solver.initial.length / curve.length)

        return curve

    @staticmethod
    def default_curvature_fn(curve: Curve) -> ndarray:
        """Simply return the value of the `Curve.curvature` property"""
        return curve.curvature


class _CsfData(TypedDict):
    orig_thresh: float


class CurveShorteningFlow(AbstractCurvatureFlow[_CsfData]):
    r"""Basic curve shortening flow

    At each iteration, vertices coordinates are moved by $\Delta t \kappa_i N_i$, for
    timestep $\Delta t$ and vertex curvatures $\kappa_i$ and normal $N_i$.

    Parameters
    ----------
    resample_mode
        Type of interpolation to use when resampling, one of ('linear', 'cubic', 'pchip').

    **kwargs
        Remaining kwargs are passed to the `AbstractCurvatureFlow` constructor.
    """

    def __init__(
        self,
        resample_mode: InterpType | None = "cubic",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resample_mode = resample_mode

    def solver(self, initial: Curve, **kwargs) -> Solver[_CsfData]:
        """Construct a `CurveShorteningFlow` `Solver`

        **kwargs are all passed to the `Solver` constructor.
        """
        data = _CsfData(orig_thresh=initial.edge_length.mean())
        return Solver(flow=self, initial=initial, data=data, **kwargs)

    def step(self, curve: Curve, timestep: float, solver: Solver[_CsfData]) -> Curve:
        curve = curve.translate(timestep * self.curvature_fn(curve)[:, newaxis] * curve.normal)
        curve = super()._postprocess(curve=curve, solver=solver)

        if self.resample_mode:
            curve = curve.interpolate_thresh(
                thresh=solver.data["orig_thresh"],
                typ=self.resample_mode,
            )

        return curve


class _Sentinel(enum.Enum):
    DEFAULT = object()


class _WillmoreFlowData(TypedDict):
    stop_on_energy_increase: bool


class WillmoreFlow(AbstractCurvatureFlow[_WillmoreFlowData]):
    r"""Willmore Flow

    As explained in [*Robust Fairing via Conformal Curvature Flow.* Keenan Crane, Ulrich Pinkall,
    and Peter Schröder. 2014.](
    https://www.cs.cmu.edu/~kmcrane/Projects/ConformalWillmoreFlow/paper.pdf)

    Parameters
    ----------
    filter_width
    filter_shape
        The $\theta$ and $k$ parameters in Crane §4. These filter the curvature flow direction and
        can be used to prioritize high or low frequency smoothing.

    constrain
        Whether to apply the closed curve constraints on the curvature flow direction at each
        timestep. See method `WillmoreFlow.constrain_flow` for more details.

    solve_vertices
        Whether to distribute length discretization errors.
        See method `Curve.with_curvatures` for more details.

    realign
        Whether to realign the curve at each timestep to the preceeding one. Because flipping
        back and forth between extrinsic and intrinsic representations loses rotation and
        translation information, this helps visually align the curve at each step, but may be
        an unnecessary computation each iteration if alignment isn't important. See method
        `Curve.with_curvatures` for more details.

    tgt_curvature
        Vector of target vertex curvatures to flow towards.

    """

    def __init__(
        self,
        constrain: bool = True,
        filter_width: float | None = None,
        filter_shape: int | None = None,
        solve_vertices: bool = True,
        realign: bool = True,
        tgt_curvature: ndarray | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.constrain = constrain
        self.filter_width = filter_width
        self.filter_shape = filter_shape
        _filter_params_specified = {filter_shape is not None, filter_width is not None}
        if len(_filter_params_specified) != 1:
            msg = (
                "Both `filter_width` and `filter_shape` must be specified to filter curvature flow"
            )
            raise ValueError(msg)
        self._do_filter: bool = _filter_params_specified.pop()

        self.solve_vertices = solve_vertices
        self.realign = realign
        self.tgt_curvature = tgt_curvature

    @staticmethod
    def constrain_flow(curve: Curve, dk: ndarray) -> ndarray:
        """Constrain curvature flow as per Crane §5

        Constraints are

        1. end points must meet: $f(0) = f(L)$
        2. tangents must agree at endpoints: $T(0) = T(L)$.

        Parameters
        ----------
        curve
            The curve to constrain flow for.

        dk
            A `n_vertices` length vector indicating the curvature flow.

        Returns
        -------
        dk_constrained
            the curvature flow direction after applying the constraints.

        """
        mass = curve.dual_edge_length

        def inner_product(f: ndarray, g: ndarray) -> float:
            """The L2 inner product ⟨⟨F G⟩⟩"""
            return (f * mass * g).sum()

        def proj(f: ndarray, g: ndarray) -> ndarray:
            """Projection of f onto g"""
            return inner_product(f, g) / inner_product(g, g) * g

        # Construct orthogonal constraint basis (Crane §4, the `c_i` terms) via Gram-Schmidt
        x, y = curve.points.T
        c0 = ones(curve.n)
        c1 = x - proj(x, c0)
        c2 = y - proj(y, c1) - proj(y, c0)

        # Subtract flow along the constraint basis
        return dk - proj(dk, c0) - proj(dk, c1) - proj(dk, c2)

    def filter_flow_direction(self, curve: Curve, dk: ndarray) -> ndarray:
        """Filter curvature flow gradient"""
        sigma, order = self.filter_width, self.filter_shape
        if sigma is None:
            msg = "filter_width is None"
            raise ValueError(msg)
        if order is None:
            msg = "filter_shape is None"
            raise ValueError(msg)

        n = curve.n

        # Square matrix `a` here is the term `id - σ∆^k` in Crane §4
        # `filtered` is inv(a) @ dk
        if order == 0:
            # can't be a sparse array
            a = eye(n, n) - sigma * ones((n, n))
            filtered = np.linalg.solve(a, dk)
        else:
            # lb = curve.laplacian
            lb = scipy.sparse.diags_array(-1 / curve.dual_edge_length) @ curve.laplacian
            a = scipy.sparse.eye(n, n) - sigma * (lb**order)
            filtered = scipy.sparse.linalg.spsolve(a, dk)

        return dk - filtered  # v ← v - inv(a)v

    def solver(
        self,
        initial: Curve,
        stop_tol: float | None = None,
        stop_on_energy_increase: bool = False,
        **kwargs,
    ) -> Solver[_WillmoreFlowData]:
        """Construct a `Solver` for the flow

        Parameters
        ----------
        initial
            The initial `Curve`.

        stop_tol
            Optional stopping tolerance. See `WillmoreFlow.stop_on_gradient_tolerance`
            for more details.

        stop_on_energy_increase
            Stop the first time energy is increased. The step with increased energy is discarded.

        **kwargs
            Remaining kwargs passed to the `Solver` constructor.

        Notes
        -----
        If neither `timestep` nor `timestep_fn` are supplied to the solver, sets the solver
        `timestep_fn` to `self.autotimestep_fn` for adaptive timestep selection. When
        `tgt_curvatures` is None, it's probably safe to just use a reasonably large timestep < 1,
        but an adaptive timestep seems to be safer for targeted curvature flow.
        """

        data = _WillmoreFlowData(stop_on_energy_increase=stop_on_energy_increase)
        solver = Solver(flow=self, initial=initial, data=data, **kwargs)
        solver.add_curve_loggers(willmore_energy=self.energy)

        if solver.timestep is solver.timestep_fn is None:
            solver.timestep_fn = self.autotimestep_fn()

        if stop_tol is not None:
            solver.add_stop_fn(self.stop_on_gradient_tolerance(stop_tol))

        return solver

    def step(self, curve: Curve, timestep: float, solver: Solver[_WillmoreFlowData]) -> Curve:
        """Step the curve along its Willmore energy gradient"""
        k0 = self.curvature_fn(curve)

        # Calculate curvature gradient, i.e. the derivative of E(k) = ||k||^2
        dk = -2 * k0 if self.tgt_curvature is None else -2 * (k0 - self.tgt_curvature)

        if self._do_filter:
            dk = self.filter_flow_direction(curve, dk)

        if self.constrain:
            dk = self.constrain_flow(curve, dk)

        k1 = k0 + timestep * dk
        curve = curve.with_curvature(
            curvature=k1,
            solve_vertices=self.solve_vertices,
            realign=self.realign,
        )
        return self._postprocess(curve, solver=solver)

    def poststep(self, curve: Curve, solver: Solver[_WillmoreFlowData]) -> Curve:
        if solver.data["stop_on_energy_increase"]:
            e1 = curve["willmore_energy"]
            e0 = solver.current["willmore_energy"]
            if e1 > e0:
                solver.log("Willmore energy increased {} => {}, stopping", e0, e1)
                raise StopEarly()

        return curve

    def energy(
        self,
        curve: Curve,
        tgt_curvature: ndarray | _Sentinel | None = _Sentinel.DEFAULT,
    ) -> float:
        r"""Calculate curve energy

        By default uses `self.tgt_curvature`, but can be overridden with the supplied
        `tgt_curvature`.

        If `tgt_curvature` is None, calculates the Willmore energy

        $$
            E(c) = \sum_i^n \kappa_i^2 l_i$
        $$

        for vertex curvatures $\kappa_i$ and dual edge lengths $l_i$.

        If `tgt_curvature` is not None, calculates

        $$
            E(c) = \sum_i^n ( \kappa_i - \hat \kappa_i)^2 l_i
        $$

        for target vertex curvatures $\hat \kappa_i$.

        """
        if tgt_curvature is _Sentinel.DEFAULT:
            tgt_curvature = self.tgt_curvature

        if tgt_curvature is None:
            return (self.curvature_fn(curve) ** 2 * curve.dual_edge_length).sum()

        dk = self.curvature_fn(curve) - tgt_curvature
        return (dk**2 * curve.dual_edge_length).sum()

    @staticmethod
    def autotimestep_fn(
        min_step: float | None = 1e-5,
        max_step: float | None = 0.9,
    ) -> Callable[[Solver], float]:
        r"""Construct an adaptive timestep function

        For curve $c$, calculates the timestep as $1 / \sqrt {E(c)}$, for energy $E$, defined
        in `WillmoreFlow.energy`. This value is then clamped to `min_step` and `max_step`,
        if supplied.

        Parameters
        ----------
        min_step
            Minimum timestep.

        max_step
            Maximum timestep

        Returns
        -------
        timestep_fn
            A function `Solver -> timestep`.
        """

        def timestep_fn(solver: Solver) -> float:
            # NB WillmoreFlow.solver adds an energy logger
            e = solver.current["willmore_energy"]
            if e == 0:
                # Doesn't matter what the stepsize is
                dt = solver.current.data.get("timestep", nan)
                return cast(float, np.nan_to_num(dt))

            return float(clip(1 / sqrt(e), min_step, max_step))

        return timestep_fn

    @staticmethod
    def stop_on_gradient_tolerance(tol: float) -> Callable[[Solver], bool]:
        """Construct a `Solver` stopping function for the supplied tolerance"""

        def stop_fn(solver: Solver):
            if solver.previous is None:
                return False

            e1 = solver.current["willmore_energy"]
            e0 = solver.previous["willmore_energy"]
            dt = solver.current["timestep"]
            gradient = abs(e1 - e0) / dt
            if gradient < tol:
                solver.log("Willmore energy gradient {} < tol {}, stopping", gradient, tol)
                return True

            return False

        return stop_fn

    def with_tgt_curvature(self, tgt_curvature: ndarray) -> WillmoreFlow:
        """Replace the target curvatures"""
        out = copy(self)
        out.tgt_curvature = tgt_curvature
        return out


class _SfmcfData(TypedDict):
    stiffness: scipy.sparse.sparray


class SingularityFreeMeanCurvatureFlow(AbstractCurvatureFlow):
    """Singularity free mean curvature flow

    As defined in

    [*Can Mean-Curvature Flow Be Made Non-Singular?* Michael Kazhdan, Jake Solomon, and Mirela Ben-Chen.
    2012.](https://arxiv.org/abs/1203.6819)

    That paper suggests this shouldn't really be necessary in the planar curve case, as curves in
    the continuous case can't form singularities anyway, but it does seem to be much more
    numerically stable than the traditional approach, and doesn't require resampling the curve.

    See also the explanation in

    [*Mean Curvature Flow and Applications*. Maria Eduarda Duarte and Leonardo Sacht. 2017.](
    https://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2017/09.04.18.39/doc/Mean%20Curvature%20Flow%20and%20Applications.pdf)

    Parameters
    ----------
    **kwargs
        All kwargs passed to `AbstractCurvatureFlow` constructor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def solver(self, initial: Curve, **kwargs) -> Solver[_SfmcfData]:
        """Construct a solver for this flow

        All **kwargs are passed to the `Solver` constructor.
        """
        stiffness = scipy.sparse.diags(-1 / initial.dual_edge_length) @ initial.laplacian
        data = _SfmcfData(stiffness=stiffness)
        return Solver(flow=self, initial=initial, data=data, **kwargs)

    def step(self, curve: Curve, timestep: float, solver: Solver[_SfmcfData]) -> Curve:
        inv_mass = scipy.sparse.diags(1 / curve.dual_edge_length)  # The $D_t^-1$ matrix
        stiffness = solver.data["stiffness"]  # The stiffness $L_0$
        # noinspection PyTypeChecker
        pts = scipy.sparse.linalg.spsolve(inv_mass - timestep * stiffness, inv_mass @ curve.points)
        curve = solver.initial.with_points(pts)
        return super()._postprocess(curve=curve, solver=solver)
