from __future__ import annotations

from typing import Tuple, Union, Any, Sized, Optional, Callable, Literal

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from numpy import ndarray, stack, cos, sin, array, cross, arctan2, concatenate
from scipy.interpolate import CubicSpline


def angle_to_points(theta: ndarray) -> ndarray:
    return stack([cos(theta), sin(theta)], axis=1)


def rotation_matrix(theta: float) -> ndarray:
    """A 2x2 rotation matrix"""
    cos_theta, sin_theta = cos(theta), sin(theta)
    return array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


def reflection_matrix(theta: float) -> ndarray:
    """A 2x2 reflection matrix"""
    cos_2theta, sin_2theta = cos(2 * theta), sin(2 * theta)
    return array([[cos_2theta, sin_2theta], [sin_2theta, -cos_2theta]])


def _align_edges(
        src_pts: ndarray,
        src_edges: ndarray,
        tgt_pts: ndarray,
        tgt_edges: ndarray,
) -> ndarray:
    """Calulate mean change in angle and position between two curves

    Parameters
    ----------
    src_pts, tgt_pts : ndarray (nv, 2)
        Vertex positions

    src_edges, tgt_edges : ndarray (nv, 2)
        Edge vectors, i.e. pts[i + 1] - pts[i]

    Returns
    -------
    aligned_points : ndarray
        The aligned source vertex locations

    """
    # Calculate the mean angle between source and target edge vectors
    # Because these are unnormalized, larger edges are weighted more heavily
    cos_theta = (src_edges * tgt_edges).sum()
    sin_theta = cross(src_edges, tgt_edges).sum()
    theta = arctan2(sin_theta, cos_theta)
    pts = src_pts @ rotation_matrix(-theta)
    # AFTER applying the rotation, check mean change in vertex position
    offset = (tgt_pts - pts).mean(axis=0, keepdims=True)
    return pts + offset


InterpType = Literal['linear', 'cubic', 'pchip']


def _periodic_interp1d(x: ndarray, f: ndarray) -> Callable[[ndarray], ndarray]:
    from scipy.interpolate import interp1d
    if x[0] != 0:
        raise NotImplementedError("x[0] must == 0")
    if len(x) != (len(f) + 1):
        raise NotImplementedError("len(x) must == len(f) + 1")

    f = concatenate([f, f[[0]]])
    fitted = interp1d(x, f, kind='linear', assume_sorted=True, axis=0)

    def _periodic_wrapper(xi: ndarray) -> ndarray:
        return fitted(xi % x[-1])

    return _periodic_wrapper


def _periodic_pchip(x: ndarray, f: ndarray) -> Callable[[ndarray], ndarray]:
    from scipy.interpolate import PchipInterpolator

    if x[0] != 0:
        raise NotImplementedError("x[0] must == 0")
    if len(x) != (len(f) + 1):
        raise NotImplementedError("len(x) must == len(f) + 1")

    # let m = len(x) - 2

    # x is (x0, x1, ..., xm, xL)
    # f is (f0, f1, fm)

    x = concatenate([
        # [x[-3] - x[-1]],
        [x[-2] - x[-1]],  # negative; one before the first element, i.e. (xm -xL)
        x,  # (0, x1, x2, ..., xm, xL)
        [x[-1] + x[1]],  # > L; one past the last element (f[L] = f[0])
        # [x[-1] + x[2]],
    ])

    # n.b. f is one element shorter than x so need an extra pad here
    f = concatenate([
        # f[[-2]],
        f[[-1]],
        f,  # f(0), f(x1), f(x2), ..., f(x_n)
        f[[0]],  # f(L)
        f[[1]],  # f(L + x1)
        # f[[2]],
    ])

    pchip = PchipInterpolator(x, f, extrapolate=False)

    def _periodic_wrapper(xi: ndarray) -> ndarray:
        return pchip(xi % x[-1])

    return _periodic_wrapper


def periodic_interpolator(
        x: ndarray,
        f: ndarray,
        typ: InterpType = 'cubic',
) -> Callable[[ndarray], ndarray]:
    """Construct a periodic interpolator of the function f(x)

    Parameters
    ----------
    x : ndarray
        `(n + 1,)` array of independent variable values.

    f : ndarray
        `(n,)` or `(n, ndim)` array of function values.
        `f(x[-1])` is assumed equal to `f(x[0])`.

    typ : str
        The type of interpolator. One of

        - 'linear' for linear interpolation via `scipy.interpolate.interp1d`
        - `cubic` for cubic spline interpolation via `scipy.interpolate.CubicSpline`
        - 'pchip`
            - for piecewise cubic hermite interpolating polynomial via
            `scipy.interpolate.PchipInterpolator`.

    Returns
    -------
    `Callable`
        A function `f` that returns interpolated values.
    """

    if typ == 'linear':
        return _periodic_interp1d(x, f)
    elif typ == 'cubic':
        return CubicSpline(x, concatenate([f, f[[0]]]), bc_type='periodic')
    elif typ == 'pchip':
        return _periodic_pchip(x, f)
    else:
        raise ValueError(f"Unrecognized interpolator type {typ}")


# A way of typing the extra *args to quiver
# Might be the empty tuple, or a single element tuple
_SingleColorOrNothing = Union[Tuple[()], Tuple[Any]]


def _get_quiver_color_arg(
        n_data: int,
        colorlike_or_array: Union[Sized, ndarray],
) -> Tuple[_SingleColorOrNothing, Optional[Sized]]:
    """
    returns (c, color) such that we can call quiver(_, _, _, _, *c, color=color)

    Returns
    -------
    `c`
    `color`
    """
    # matplotlib.pyplot quiver has two syntaxes for specifying color
    #   quiver(x, y, p, q, c) for unnamed `*args` vector color `c`
    #   quiver(x, y, p, color=single_color) `**kwargs` single color
    if isinstance(colorlike_or_array, str):  # e.g. color='black'
        return (), colorlike_or_array

    try:
        n_maybe_colorlike = len(colorlike_or_array)
    except TypeError as e:
        raise NotImplementedError("Expected `color` to have a length") from e

    if n_data == n_maybe_colorlike == 3:
        # warnings.warn(
        #     f'`color` = {colorlike_or_array} could be either an RGB color or an '
        #     '`n_data` length array of scalars, assuming `n_data`'
        # )
        return (colorlike_or_array,), None
    elif n_data == n_maybe_colorlike:
        # TODO what
        return (colorlike_or_array,), None
    else:
        # i n_maybe_colorlike needs to be 3 here
        return (), colorlike_or_array


def _get_ax(ax: Optional[Axes]) -> Axes:
    if ax:
        return ax

    if len(plt.get_fignums()) == 0:
        # making a new axes so don't feel bad about setting some defaults
        _fig, ax = plt.subplots()
        ax.axis('equal')
        return ax
    else:
        return plt.gca()
