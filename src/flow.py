from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Optional, Callable, Dict, Literal, Any, List, TypedDict, Tuple, Protocol

import scipy
from numpy import ndarray, ones, sqrt, newaxis, clip
from typing_extensions import Self

from .curve import Curve
from .curves import Curves

logger = logging.getLogger(__name__)


class _SupportsCmp(Protocol):
    def __lt__(self, other: Self) -> bool: ...

    def __gt__(self, other: Self) -> bool: ...


class _BraceMessage:
    """For lazy {} style log formatting"""
    def __init__(self, msg: Optional[str] = None, *args: Any, **kwargs: Any):
        self.msg = [msg] if msg else []
        self.args = list(args)
        self.kwargs = kwargs
        self._str: Optional[str] = None

    def __str__(self):
        if self._str is None:
            msg = ''.join(self.msg)
            self._str = msg.format(*self.args, **self.kwargs)
        return self._str

    def append(self, msg: str, *args, **kwargs):
        self.msg.append(msg)
        self.args.extend(args)
        self.kwargs.update(kwargs)


class AbstractFlow(ABC):
    """Abstract superclass for curve flow

    The main method that subclasses implement is `Flow.step`, which steps the curve by the
    supplied timestep.

    The basic contract is that `Flow` objects don't maintain any state specific
    to the solution of a flow. All state is stored in the curve metadata
    or in the `Solver.state` dictionary. If state is needed, store it in the `state` object
    in `Flow._init_state` method, which can be extended by subclasses.
    """

    class State(TypedDict):
        """Flow-specific state stored in the `Solver`, supplied to `Flow.step`"""
        solver: Solver

    def solver(self, initial: Curve, **kwargs) -> Solver:
        """Construct a `Solver` to solve curve flow over time"""
        solver_kwargs, state_kwargs = Solver.split_solver_kwargs(kwargs)
        solver = Solver(flow=self, initial=initial, **solver_kwargs)
        self._init_state(initial=initial, state=solver.state, **state_kwargs)
        return solver

    @abstractmethod
    def step(self, curve: Curve, timestep: float, state: State) -> Curve:
        """Step the curve by `timestep`"""
        ...

    def _init_state(self, initial: Curve, state: State, **kwargs) -> None:
        if kwargs:
            # Subclasses should have stripped off their specific kwargs by now
            kws = ', '.join(kwargs.keys())
            raise ValueError(f"Unrecognized keyword arguments {kws}")


class RetryStep(Exception):
    """This can be raised in a custom `Solver.step_fn` to retry the current step

    Usually after adjusting the timestep or some other state.
    """
    pass


class StopEarly(Exception):
    """This can be raise in a custom `Solver.step_fn` to stop the current run

    Usually after reaching some stopping criterion.
    """
    pass


class Solver:
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

    verbose
        If true, curve state information and stopping messages are printed to stdout on each
        iteration.

    log
        If true, the printed log messages as in `verbose` are saved as a list of `str`s in
        `Solver.log`

    state
        A dict[str, Any] where `Flow`s can store run-specific state. For static typing reasons,
        `Flow`s maintain an inner `TypedDict` class `Flow.State` representing the data expected
        to be stored there.

    step_fn
        A function `Solver -> Curve` that steps the curve forward at each iteration. This just
        defaults to `Solver.step`.

    """
    def __init__(
            self,
            *,
            flow: AbstractFlow,
            initial: Curve,
            timestep: Optional[float] = None,
            timestep_fn: Optional[Callable[[Solver], float]] = None,
            history: bool = True,
            max_step: Optional[int] = None,
            verbose: bool = False,
            log: bool = False,
            state: Optional[Dict[str, Any]] = None,
            step_fn: Optional[Callable[[Solver], Curve]] = None,
    ):
        self.flow = flow
        self.initial = initial
        self.current = initial.with_data(time=0, step=0)
        self.previous: Optional[Curve] = None
        self.timestep = timestep
        self.timestep_fn = timestep_fn
        self.history: Optional[Curves] = None
        if history:
            # Note that we don't log the first curve until
            # self.run() in case some extra initialization needs to happen somewhere
            self.history = Curves([])
        self.log_history: Optional[List[str]] = [] if log else None
        self.max_step = max_step
        self.verbose = verbose

        self._stop_fns: List[Callable[[Solver], bool]] = []
        self._curve_loggers: Dict[str, Callable[[Curve], Any]] = dict()

        self.state = state or {}

        # Stash self in `state`, which allows `Flow`s to use the log method if necessary
        self.state['solver'] = self
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
            print(bm)
        if self.log_history is not None:
            self.log_history.append(str(bm))

    def _log_state(self, msg: str):
        m = _BraceMessage(msg=msg)

        for k, v in self.current.data.items():
            if k != 'step':
                m.append(', {} = {}', k, v)

        self._log(m)

    def _log_step(self, c1: Curve):
        m = _BraceMessage('Step {}', c1['step'])
        c0 = self.current
        ks = (c0.data.keys() & c1.data.keys()) - {'step'}
        for k in ks:
            m.append(', {}: {} => {}', k, c0[k], c1[k])
        self._log(m)

    def _stop_fn(self) -> bool:
        """Returns True if run should stop"""
        if self.max_step is not None:
            if self.current['step'] == self.max_step:
                self.log('Stopping at max step {}', self.current['step'])
                return True

        for fn in self._stop_fns:
            if fn(self):
                self.log('Stopping due to stop fn {}', fn)
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
            param_fn: Optional[Callable[[Curve], _SupportsCmp]] = None,
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
            param_fn = itemgetter(param)

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

        self._log_state('Initial state')
        step_fn = self.step_fn or Solver.step

        while not self._stop_fn():
            try:
                next_state = step_fn(self)
            except RetryStep:
                continue
            except StopEarly:
                break

            self._log_step(next_state)
            self.previous = self.current
            self.current = next_state
            if self.history is not None:
                self.history.append(next_state)

        self._log_state('Final state')
        return self

    def step(self) -> Curve:
        """Call `Flow.step` with the current state and timestep"""

        if self.timestep_fn is not None:
            timestep = self.timestep_fn(self)
        elif self.timestep is not None:
            timestep = self.timestep
        else:
            raise ValueError("Neither of `timestep` or `timestep_fn` were provided.")

        # The `Flow` does the actual work here
        curve = self.flow.step(
            curve=self.current,
            timestep=timestep,
            state=self.state,
        )

        return self.attach_metadata(
            curve=curve,
            time=self.current['time'] + timestep,
            timestep=timestep,
        )

    def attach_metadata(
            self,
            curve: Curve,
            time: float,
            step: Optional[int] = None,
            **kwargs
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
            step = self.current['step'] + 1

        params = dict(time=time, step=step, **kwargs)
        for k, fn in self._curve_loggers.items():
            params[k] = fn(curve)
        return curve.with_data(**params)

    @classmethod
    def split_solver_kwargs(cls, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split `kwargs` into `Solver` constructor kwargs and `Flow` state-specific kwargs

        Parameters
        ----------
        kwargs
            The kwargs dict.

        Returns
        -------
        solver_kwargs
            Kwargs to pass to the `Solver` constructor.

        flow_kwargs
            Left-over kwargs the `Flow` should process.

        """
        params = kwargs.keys() & inspect.signature(cls).parameters.keys()
        solver_kwargs = {k: kwargs.pop(k) for k in params}
        return solver_kwargs, kwargs


class AbstractCurvatureFlow(AbstractFlow):
    """Abstract superclass for curvature flows

    Parameters
    ----------
    curvature_fn
        A function `Curve -> ndarray` that returns the signed curvature values at each vertex.

    rescale
        If this is 'length' or 'area', the recurve length/area is rescaled to the original's
        length or area, preventing the usual curvature flow shrinkage.
    """

    class State(AbstractFlow.State):
        initial_area: float
        initial_length: float

    def __init__(
            self,
            curvature_fn: Optional[Callable[[Curve], ndarray]] = None,
            rescale: Optional[Literal['length', 'area']] = None,
    ):
        super().__init__()
        self.curvature_fn = curvature_fn or self.default_curvature_fn
        self.preserve = rescale

    @abstractmethod
    def step(self, curve: Curve, timestep: float, state: Dict[str, Any]) -> Curve: ...

    def _init_state(self, initial: Curve, state: State, **kwargs) -> None:
        super()._init_state(initial=initial, state=state, **kwargs)
        state['initial_area'] = initial.area
        state['initial_length'] = initial.length

    def _postprocess(self, curve: Curve, state: Dict[str, Any]) -> Curve:
        if self.preserve == 'area':
            curve = curve.scale(sqrt(state['initial_area'] / curve.area))
        elif self.preserve == 'length':
            curve = curve.scale(state['initial_length'] / curve.length)

        return curve

    @staticmethod
    def default_curvature_fn(curve: Curve) -> ndarray:
        return curve.exterior_angle_curvatures


class CurveShorteningFlow(AbstractCurvatureFlow):
    r"""Basic curve shortening flow

    At each iteration, vertices coordinates are moved by $\Delta t \kappa_i N_i$, for
    timestep $\Delta t$ and vertex curvatures $\kappa_i$ and normal $N_i$.

    Parameters
    ----------
    resample
        If true, the curve is resampled at each iteration to maintain the same average edge
        length as was present in the initial curve.

    **kwargs
        Remaining kwargs are passed to the `AbstractCurvatureFlow` constructor.
    """
    class State(AbstractCurvatureFlow.State):
        orig_thresh: float

    def __init__(
            self,
            resample: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.resample = resample

    def _init_state(self, initial: Curve, state: State, **kwargs) -> None:
        super()._init_state(initial=initial, state=state, **kwargs)
        state['orig_thresh'] = initial.edge_length.mean()

    def step(self, curve: Curve, timestep: float, state: State) -> Curve:
        curve = curve.translate(
            timestep * self.curvature_fn(curve)[:, newaxis] * curve.normal
        )
        curve = super()._postprocess(curve=curve, state=state)

        if self.resample:
            curve = curve.resample(thresh=state['orig_thresh'])

        return curve


class WillmoreFlow(AbstractCurvatureFlow):
    r"""Willmore Flow

    As explained in [*Robust Fairing via Conformal Curvature Flow.* Keenan Crane, Ulrich Pinkall,
    and Peter Schröder. 2014.](
    https://www.cs.cmu.edu/~kmcrane/Projects/ConformalWillmoreFlow/paper.pdf)

    Parameters
    ----------
    filter_width
    filter_shape
        The `σ` and `k` parameters in Crane §4. These filter the curvature flow direction and
        can be used to prioritize high or low frequency smoothing. See
        `WillmoreFlow.filter_flow_direction`.

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

    Notes
    -----
    Target curvatures can be supplied to `WillmoreFlow.solver` to flow towards desired target
    curvatures.

    """

    class State(AbstractCurvatureFlow.State):
        tgt_curvatures: Optional[ndarray]

    def __init__(
            self,
            constrain: bool = True,
            filter_width: Optional[float] = None,
            filter_shape: Optional[int] = None,
            solve_vertices: bool = True,
            realign: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.constrain = constrain
        self.filter_width = filter_width
        self.filter_shape = filter_shape
        _filter_params_specified = {filter_shape is not None, filter_width is not None}
        if len(_filter_params_specified) != 1:
            raise ValueError(
                "Both `filter_width` and `filter_shape` must be specified to filter curvature flow"
            )
        self._do_filter: bool = _filter_params_specified.pop()

        self.solve_vertices = solve_vertices
        self.realign = realign

    @staticmethod
    def constrain_flow(curve: Curve, dk: ndarray) -> ndarray:
        """Constrain curvature flow as per Crane §5

        Constraints are 1. end points must meet: f(0) = f(L) and 2. tangents must agree
        at endpoints: T(0) = T(L).

        Parameters
        ----------
        curve: Curve
        dk : ndarray
            An `n_vertices` length vector indicating the curvature flow.

        Returns
        -------
        `dk_constrained` the curvature flow direction after applying the constraints.

        """
        mass = curve.dual_edge_length

        def inner_product(f: ndarray, g: ndarray) -> float:
            """The L2 inner product ⟨⟨F G⟩⟩"""
            return (f * mass * g).sum()

        def proj(f: ndarray, g: ndarray) -> ndarray:
            """Projection of f onto g"""
            return inner_product(f, g) / inner_product(g, g) * g

        # Construct orthogonal constraint basis (Crane §4, the `c_i` terms) via Gram–Schmidt
        x, y = curve.pts.T
        c0 = ones(curve.n)
        c1 = x - proj(x, c0)
        c2 = y - proj(y, c1) - proj(y, c0)

        # Subtract flow along the constraint basis
        dk = dk - proj(dk, c0) - proj(dk, c1) - proj(dk, c2)
        return dk

    def filter_flow_direction(self, curve: Curve, dk: ndarray) -> ndarray:
        """Filter curvature flow gradient"""
        sigma, order = self.filter_width, self.filter_shape
        # Square matrix `a` here is the term `id - σ∆^k` in Crane §4
        a = scipy.sparse.eye(curve.n, curve.n) - sigma * curve.laplacian ** order
        dk = dk - scipy.sparse.spsolve(a, dk)  # `v ← v - inv(a)v`
        return dk

    def solver(
            self,
            initial: Curve,
            tgt_curvatures: Optional[ndarray] = None,
            stop_tol: Optional[float] = None,
            **kwargs,
    ) -> Solver:
        """Construct a `Solver` for the flow

        Parameters
        ----------
        initial
            The initial `Curve`.

        tgt_curvatures
            A length `n` vector of target curvatures, if desired.

        **kwargs
            Remaining kwargs passed to the `Solver` constructor.

        Notes
        -----
        If neither `timestep` nor `timestep_fn` are supplied to the solver, sets the solver
        `timestep_fn` to `self.autotimestep_fn` for adaptive timestep selection. When
        `tgt_curvatures` is None, it's probably safe to just use a reasonably large timestep < 1,
        but an adaptive timestep seems to be safer for targeted curvature flow.
        """
        if tgt_curvatures is not None and len(tgt_curvatures) != initial.n:
            raise ValueError(
                f"`len(tgt_curvatures)` (got {len(tgt_curvatures)}) must match "
                f"`initial.n` (got {initial.n})."
            )
        solver = super().solver(initial=initial, **kwargs)
        solver.state['tgt_curvatures'] = tgt_curvatures

        solver.add_curve_loggers(willmore_energy=lambda c: self.energy(c, tgt_curvatures))

        if solver.timestep is solver.timestep_fn is None:
            solver.timestep_fn = self.autotimestep_fn()

        if stop_tol is not None:
            solver.add_stop_fn(self.autostop_fn(stop_tol))

        return solver

    def step(self, curve: Curve, timestep: float, state: State) -> Curve:
        k0 = self.curvature_fn(curve)

        # Calculate curvature gradient, i.e. the derivative of E(k) = ||k||^2
        if (tgt := state.get('tgt_curvatures')) is not None:
            dk = 2 * (tgt - k0)
        else:
            dk = -2 * k0

        if self._do_filter:
            dk = self.filter_flow_direction(curve, dk)

        if self.constrain:
            dk = self.constrain_flow(curve, dk)

        k1 = k0 + timestep * dk
        curve = curve.with_curvatures(
            curvatures=k1,
            solve_vertices=self.solve_vertices,
            realign=self.realign,
        )
        return self._postprocess(curve, state=state)

    def energy(self, c: Curve, tgt_curvatures: Optional[ndarray] = None) -> float:
        r"""Calculate curve energy

        If `tgt_curvature` is None, calculates the Willmore energy

        $$
            E(c) = \sum_i^n \kappa_i^2 l_i$
        $$

        for vertex curvatures $\kappa_i$ and dual edge lengths $l_i$.

        If `tgt_curvatures` is supplied, calculates

        $$
            E(c) = \sum_i^n ( \kappa_i - \hat \kappa_i)^2 l_i
        $$

        for target curvatures $\hat \kappa_i$.

        """
        if tgt_curvatures is None:
            return (self.curvature_fn(c) ** 2 * c.dual_edge_length).sum()
        else:
            return ((self.curvature_fn(c) - tgt_curvatures) ** 2 * c.dual_edge_length).sum()

    @staticmethod
    def autotimestep_fn(
            min_step: Optional[float] = 1e-5,
            max_step: Optional[float] = 0.9,
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
            e = solver.current['willmore_energy']
            if e == 0:
                # Probably doesn't matter what the stepsize is but avoid division by zero
                return max_step
            else:
                return float(clip(1 / sqrt(e), min_step, max_step))

        return timestep_fn

    @staticmethod
    def autostop_fn(tol: float) -> Callable[[Solver], bool]:
        def stop_fn(solver: Solver):
            if solver.previous is None:
                return False

            e1 = solver.current['willmore_energy']
            e0 = solver.previous['willmore_energy']
            dt = solver.current['timestep']
            gradient = abs(e1 - e0) / dt
            if gradient < tol:
                solver.log("Willmore energy gradient {} < tol {}, stopping", gradient, tol)
                return True
            else:
                return False

        return stop_fn


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
    http://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2017/09.04.18.39/doc/Mean%20Curvature%20Flow%20and%20Applications.pdf)
    """
    class State(AbstractCurvatureFlow.State):
        stiffness: scipy.sparse.dia_array

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_state(self, initial: Curve, state: State, **kwargs) -> None:
        super()._init_state(initial=initial, state=state, **kwargs)
        state['stiffness'] = scipy.sparse.diags(-1 / initial.dual_edge_length) @ initial.laplacian

    def step(self, curve: Curve, timestep: float, state: State) -> Curve:
        inv_mass = scipy.sparse.diags(1 / curve.dual_edge_length)  # The $D_t^-1$ matrix
        stiffness = state['stiffness']  # The stiffness $L_0$
        pts = scipy.sparse.linalg.spsolve(inv_mass - timestep * stiffness, inv_mass @ curve.pts)
        curve = Curve(pts)
        curve = super()._postprocess(curve=curve, state=state)
        return curve
