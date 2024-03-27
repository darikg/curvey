"""Sequences of curves"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    Callable,
    Literal,
    Union,
    cast,
    overload,
)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy import arange, array, asanyarray, asarray, ceil, nan, ndarray, sqrt
from numpy.typing import ArrayLike

from .curve import Curve
from .util import _get_ax

NamedMetadata = Union[tuple[str, Sequence], tuple[str, Callable[[Curve], Any]]]
BareMetadata = Union["str", Sequence, Callable[[Curve], Any]]
Metadata = Union[NamedMetadata, BareMetadata]


class Curves:
    """A container of `Curve`s

    The constructor accepts an iterator of `Curve`s:
    ```python
    from curvey import Curve, Curves

    curves = Curves(Curve.circle(n=20, r=r) for r in (1, 2, 3))
    ```

    `Curves` can be iterated over:
    ```python
    for c in curves:
        ...
    ```

    `Curves` can be indexed with:

    - an int: `curves[i]` returns the `Curve` at the `i`ith position

    - a slice or numpy index-like array: `curves[::2]` returns a new `Curves` containing every
        second curve

    - a string: `curves['param']` returns an array of the curve metadata or `Curve` attribute
        associated with each `Curve` in the `Curves`

    - a function: `curves[fn]` is equivalent to `array([fn(c) for c in curves])`

    Use `Curves.subplots` or `Curves.superimposed` to plot every curve in the `Curves` at once.

    Use `Curves.plot` to plot `Curve` metadata values against eachother.

    Parameters
    ----------
    curves
        A iterable of `Curves`. Defaults to an empty sequence if not supplied.
    """

    def __init__(self, curves: Iterable[Curve] | None = None):
        self.curves: list[Curve] = []
        """The `Curve`s contained in this `Curves`"""

        if curves is not None:
            self.curves.extend(curves)

    @property
    def n(self) -> int:
        """Number of curves in the sequence"""
        return len(self.curves)

    def __len__(self) -> int:
        """Number of curves in the sequence"""
        return len(self.curves)

    def __add__(self, other: Curves | list[Curve]) -> Curves:
        """Concatenate two `Curves` to form a new sequence"""
        if isinstance(other, Curves):
            return Curves(self.curves + other.curves)

        # Just hope this raises a reasonable error if it fails
        return Curves(self.curves + other)

    def append(self, curve: Curve):
        """Add a curve to the end of the sequence"""
        self.curves.append(curve)

    def keys(self, mode: Literal["intersection", "union"] = "union") -> set[str]:
        """Unique curve metadata parameter names

        Parameters
        ----------
        mode
            If 'union', return keys that are present on any `Curve` in the `Curves`.
            If 'intersection', return only keys that are present on all `Curve`s in the `Curves`.
        """
        if self.n == 0:
            return set()

        keys = (set(c.data.keys()) for c in self)
        if mode == "intersection":
            return set.intersection(*keys)

        if mode == "union":
            return set.union(*keys)

        raise ValueError(mode)

    def __repr__(self) -> str:
        if keys := self.keys():
            data = ", ".join(k for k in sorted(keys))
            # Gross triple nested curlies here, so it formats like `Curves(n=3, data={foo, bar})`
            return f"Curves(n={self.n}, data={{{data}}})"

        return f"Curves(n={self.n})"

    def __iter__(self) -> Iterator[Curve]:
        """Iterate over curves in the sequence

        The index of the curve is stored in the 'idx' metadata parameter.
        Use `Curves.iter_curves` to supply a custom name for index, if necessary.
        """
        return self.iter_curves()

    def iter_curves(self, idx: str = "idx") -> Iterator[Curve]:
        """Iterate over curves in the sequence

        The index of the curve is stored in the `idx` metadata parameter. This might be useful for
        tracking the original index in a subset.

        ```python
        from curvey import Curves

        orig = Curves(Curve.circle(3) for _ in range(6))
        every_other = Curves(orig.iter_curves('orig_idx'))[::2]
        every_other['orig_idx']
        ```
        """
        for i, c in enumerate(self.curves):
            yield c.with_data(**{idx: i})

    def get_named_data(self, data: BareMetadata | NamedMetadata, **kwargs) -> tuple[str, ndarray]:
        """Get curve metadata (name, values) pairs

        If `data` is just a name, return (name, values).
        If `data` is something that can reasonably interpreted as `values`,
        try to figure out a reasonable name for them.
        """
        if isinstance(data, str):
            return data, self.get_data(data, **kwargs)

        if isinstance(data, tuple) and len(data) == 2:
            name = data[0]
            if not isinstance(name, str):
                msg = f"Expected metadata name to be str, got {name}"
                raise TypeError(msg)
            return name, self.get_data(data[1], **kwargs)

        if callable(data):
            name = data.__name__ if hasattr(data, "__name__") else str(data)
            return name, self.get_data(data, **kwargs)

        name = type(data).__name__
        return name, self.get_data(data, **kwargs)

    def get_data(
        self,
        data: BareMetadata,
        default: Any = nan,
        allow_default: bool = True,
        allow_property: bool = True,
    ) -> ndarray:
        """Concatenate curve metadata into an array of length `n`

        Parameters
        ----------
        data
            One of:

            - Name of the property stored in curve metadata
            - Name of a `Curve` attribute, if `allow_property` is true
            - A function `Curve -> value`
            - An length `n` array or list of values

        allow_property
            If true, `data` may be the name of a Curve attribute, such as 'area' or 'length'

        allow_default
            If true, and the requested data is only available on a subset of curves, return
            `default` for those curves.

        default
            The default value if named parameter `data` is not present in a curve's metadata.
            If not supplied, all curves in the collection must have that metadata parameter,
            otherwise a `KeyError` is raised.

        """
        if isinstance(data, str):
            if data in self.keys("union"):
                if allow_default:
                    return array([c.data.get(data, default) for c in self])

                if data in self.keys("intersection"):
                    msg = f"Metadata '{data}' is only present on some curves"
                    raise KeyError(msg)

                return array([c[data] for c in self])

            if hasattr(Curve, data):
                if allow_property:
                    return array([getattr(c, data) for c in self])

                msg = (
                    f"Metadata '{data}' not found. "
                    "it's a Curve property but `allow_property` is False"
                )
                raise KeyError(msg)

            msg = f"Metadata '{data}' not found"
            raise KeyError(msg)

        if callable(data):
            return array([data(c) for c in self])

        if isinstance(data, Sequence):
            if (n := len(data)) != self.n:
                msg = f"Expected a sequence of length self.n = {self.n}, got {n}"
                raise ValueError(msg)
            return asarray(data)

        msg = f"Unrecognized data type {type(data)}"  # type: ignore [unreachable]
        raise TypeError(msg)

    @overload
    def __getitem__(self, idx: int) -> Curve: ...

    @overload
    def __getitem__(self, idx: slice) -> Curves: ...

    @overload
    def __getitem__(self, idx: str | collections.abc.Callable) -> ndarray: ...

    def __getitem__(
        self,
        idx: str | int | slice | ArrayLike | collections.abc.Callable,
    ) -> Curve | Curves | ndarray:
        """Convenience method to index the sequence

        `Curves[int]` returns the curve stored at that index.
        `Curves[str]` returns a `ndarray` of `n` metadata values.
        `Curves[fn]` for `fn: Callable[[Curve], Any]` returns a `ndarray` of the values
        of that function called on the `n` curves in the sequence.

        Otherwise, `Curves[idx]` returns a new `Curves` for that index,
        obeying list slicing and numpy smart indexing behavior. E.g. `sequence[::3]` returns
        a new curve sequence for every third curve in the original sequence.

        """
        if isinstance(idx, str):
            return self.get_data(idx)

        if callable(idx):
            return self.get_data(idx)

        if isinstance(idx, (int, np.integer)):
            # Recast to int here so indexing with np.int works as expected
            return self.curves[int(idx)]

        # noinspection PyUnresolvedReferences
        idx = arange(self.n)[idx]  # type: ignore [index]
        all_curves = self.curves
        curves = [all_curves[int(i)] for i in idx]
        return Curves(curves=curves)

    def plot(
        self,
        y: Metadata,
        x: Metadata | None = None,
        ax: Axes | None = None,
        label_axes: bool | None = None,
        label: str | bool | None = True,
        **kwargs,
    ):
        """Plot metadata values against each other.

        By default, the independent variable `y` is 'time', if it's present.

        Parameters
        ----------
        y
            The name of the parameter to plot on the y-axis.
            Can also be a function `Curve` -> float, or a `tuple(name, function)`,
            Or a `tuple(name, array)`

        x
            The name of the parameter to plot on the x-axis, or alternative type as described for
            `y`. If not supplied, defaults to `time` if it's present in the curve metadata,
            or index otherwise.

        label_axes
            If true, set x and y labels. Defaults to true if a new axes is created.

        label
            Name to label the plot with, for use in matplotlib legend. Defaults to the name
            of the `y` parameter.

        ax
            The matplotlib axes to plot in. Defaults to the current axes.

        **kwargs
            Remaining kwargs are passed to `matplotlib.pyplot.plot`

        """
        if ax is None:
            ax = plt.gca()
            if label_axes is None:
                label_axes = True

        yname, ydata = self.get_named_data(y)

        if x is None:
            keys = self.keys(mode="union")
            if "time" in keys:
                x = "time"
            elif "idx" in keys:
                x = "idx"
            else:
                x = ("idx", arange(self.n))

        xname, xdata = self.get_named_data(x)

        if label_axes:
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)

        if label and isinstance(label, bool):
            label = yname

        ax.plot(xdata, ydata, label=label, **kwargs)

    def _subplots_builder(
        self,
        n_rows: int | None = 1,
        n_cols: int | None = None,
        share_xy=True,
        figsize: tuple[float, float] = (15, 5),
        idx: str = "idx",
        axs: Sequence[Axes] | None = None,
    ) -> _SubplotsBuilder:
        curves = Curves(list(self.iter_curves(idx=idx)))
        if axs is None:
            return _SubplotsBuilder.from_dims(
                curves=curves,
                nr=n_rows,
                nc=n_cols,
                sharex=share_xy,
                sharey=share_xy,
                figsize=figsize,
            )

        return _SubplotsBuilder.from_axs(curves=curves, axs=axs)

    def subplots(
        self,
        n_rows: int | None = 1,
        n_cols: int | None = None,
        axis: str | None = "scaled",
        show_axes=False,
        plot_fn: Callable[[Curve], None] | None = None,
        subtitle: str | Callable[[Curve], str] | None = None,
        share_xy=True,
        figsize: tuple[float, float] = (15, 5),
        idx: str = "idx",
        axs: Sequence[Axes] | None = None,
        hide_unused: bool = True,
    ):
        """Plot each curve in the sequence in a different subplot

        Parameters
        ----------
        figsize
            The size of the overall superfigure.

        n_rows
            Number of rows.

        n_cols
            Number of columns. By default, `n_cols` = ceil(self.n / n_rows)`. If `n_cols` is specified, and
            `n_rows` * `n_cols` < `self.n`, the curve sequence is automatically subsampled.

        axis
            Argument to `plt.axis`. By default this is 'equal' (i.e., make circles circular).

        show_axes
            Whether to show each subplot axes, i.e. border and x/y ticks, etc.

        plot_fn
            By default this just dispatches to `Curve.plot_edges(directed=False)`.

        subtitle
            A convenience argument to put a title over each subplot. If `subtitle` is a string,
            a title is constructed from the corresponding curve metadata. Otherwise, `subtitle`
            should be a function that accepts a curve and returns a string.

        share_xy
            Whether each subplot should share x/y limits.

        idx
            The index of the curve in this collection is stored in the curve metadata property
            with this name.

        axs
            Array of matplotlib axes to use for subplots. If supplied, `n_rows` and `n_cols`
            are ignored and determined by the shape of this array.

        hide_unused
            If n_axs > n_plots, hide the unused axes.

        Returns
        -------
        axes :
            `(n_rows, n_cols)` array of `matplotlib.axes.Axes` objects

        """
        if not plot_fn:

            def plot_fn(c: Curve):
                c.plot_edges(directed=False)

        if isinstance(subtitle, str):
            _key = subtitle

            def subtitle(c: Curve) -> str:
                return f"{_key} = {c[_key]}"

        b = self._subplots_builder(
            n_rows=n_rows,
            n_cols=n_cols,
            share_xy=share_xy,
            figsize=figsize,
            idx=idx,
            axs=axs,
        )
        if hide_unused and b.n_axs > b.n_plots:
            # Hide unused axes if there's e.g. 3x3 axes but only 8 curves to plot
            for i in range(b.n_plots, b.n_axs):
                b.get_ax(i).axis("off")

        for i in range(b.n_plots):
            b.axis_subplot(i=i, plot_fn=plot_fn, subtitle=subtitle, axis=axis, show_axes=show_axes)

        if b.fig is not None:  # If we created the figure
            b.fig.tight_layout()

        return b.axs

    def superimposed(
        self,
        ax: Axes | None = None,
        plot_fn: Callable[[Curve], Any] | None = None,
        color: Metadata | None = None,
        clim: tuple[float, float] | None = None,
        idx: str = "idx",
    ) -> list[Any]:
        """Plot every curve in the same axes

        Parameters
        ----------
        ax
            Matlplotlib axes to plot in. Default current axes.

        plot_fn
            Function `Curve -> matplotlib_object` to plot the curve.
            By default, this dispatches to `curve.plot`

        color
            The name of a curve metadata parameter to color by. If `plot_fn` is supplied,
            this is ignored.

        clim
            Range to scale color data to.

        idx
            The index of the curve in this collection is stored in this curve's metadata.

        Returns
        -------
        :
            List of objects returned by `plot_fn`.
        """

        ax = _get_ax(ax)

        if not plot_fn:
            if color is not None:
                cmap = plt.get_cmap("viridis")
                _cname, cdata = self.get_named_data(color)
                cnorm = plt.Normalize(*clim) if clim else plt.Normalize()
                cdata = cnorm(cdata)

                def plot_fn(c: Curve) -> Line2D:
                    return c.plot(color=cmap(cdata[c[idx]]), ax=ax)
            else:

                def plot_fn(c: Curve) -> Line2D:
                    return c.plot(ax=ax)

        out = []

        for curve in self.iter_curves(idx=idx):
            out.append(plot_fn(curve))

        return out

    def _animation_frames(self) -> Iterator[int]:
        i, step, n = 0, 1, self.n
        while True:
            yield i
            if (i, step) == (n - 1, 1):
                i, step = n - 2, -1
            elif (i, step) == (0, -1):
                i, step = 1, 1
            else:
                i += step

    def _animate(
        self,
        frames: Iterable[int] | Callable[[], int] | None = None,
        **kwargs,
    ):
        from matplotlib import animation

        kwargs.setdefault("save_count", 2 * self.n + 1)
        frames = frames or self._animation_frames()

        fig, ax = plt.subplots()
        line = self[0].plot(ax=ax)

        for c in self:
            ax.dataLim.update_from_data_xy(c.points)

        ax.axis("equal")
        ax.autoscale_view()

        def update(frame):
            curve = self[frame]
            x, y = curve.closed_points.T
            line.set_xdata(x)
            line.set_ydata(y)
            return line

        # noinspection PyTypeChecker
        return animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=frames,  # type: ignore [arg-type]
            interval=30,
            **kwargs,
        )


@dataclass
class _SubplotsBuilder:
    curves: Curves
    nr: int
    nc: int
    axs: ndarray  # Should be NDArray[Axes]
    fig: Figure | None

    @property
    def n_axs(self) -> int:
        return self.nr * self.nc

    @cached_property
    def plot_idxs(self) -> ndarray:
        if self.curves.n > self.n_axs:
            step = self.curves.n // self.n_axs
            return arange(self.n_axs) * step

        return arange(self.curves.n)

    @property
    def n_plots(self) -> int:
        return min(self.curves.n, self.n_axs)

    @staticmethod
    def from_axs(curves: Curves, axs: Sequence[Axes]) -> _SubplotsBuilder:
        _axs = asanyarray(axs)
        if _axs.ndim == 1:
            nr, nc = 1, len(_axs)
        elif _axs.ndim == 2:
            nr, nc = _axs.shape
        else:
            msg = "Expected axes array to be 1 or 2 dimensional"
            raise ValueError(msg)

        return _SubplotsBuilder(curves=curves, nr=nr, nc=nc, axs=_axs, fig=None)

    @staticmethod
    def get_dims(n_curves: int, nr: int | None, nc: int | None) -> tuple[int, int]:
        if nr is not None and nc is not None:
            return nr, nc

        if nc is None and nr is not None:
            return nr, int(ceil(n_curves / nr))

        if nr is None and nc is not None:
            return int(ceil(n_curves / nc)), nc

        n = int(ceil(sqrt(n_curves)))
        return n, n

    @staticmethod
    def from_dims(
        curves: Curves,
        nr: int | None = 1,
        nc: int | None = None,
        sharex: bool = True,
        sharey: bool = True,
        figsize: tuple[float, float] = (5, 3),
    ) -> _SubplotsBuilder:
        nr_, nc_ = _SubplotsBuilder.get_dims(curves.n, nr, nc)
        fig, axs = plt.subplots(
            nrows=nr_, ncols=nc_, squeeze=False, sharex=sharex, sharey=sharey, figsize=figsize
        )

        return _SubplotsBuilder(curves=curves, nr=nr_, nc=nc_, axs=axs, fig=fig)

    def get_ax(self, i: int) -> Axes:
        return cast(Axes, self._axs_flat[i])

    @cached_property
    def _axs_flat(self) -> ndarray:
        return self.axs.reshape(-1)

    def axis_subplot(
        self,
        i: int,
        plot_fn: Callable[[Curve], Any],
        subtitle: Callable[[Curve], str] | None,
        axis: str | None,
        show_axes: bool,
    ):
        ax = self.get_ax(i)
        plt.sca(ax)  # Set current axes
        curve = self.curves.curves[self.plot_idxs[i]]
        plot_fn(curve)

        if subtitle:
            ax.set_title(subtitle(curve))

        if axis:
            ax.axis(axis)

        if not show_axes:
            # Could just do ax.axis('off') but that prevents x/y labels
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
