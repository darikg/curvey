"""Sequences of curves"""

from __future__ import annotations

from typing import Union, Optional, List, Sequence, Iterator, Tuple, Callable, Any, Set, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy import ndarray, arange, array, ceil, nan, asarray

from .curve import Curve
from .util import _get_ax

NamedMetadata = Union[Tuple[str, Sequence], Tuple[str, Callable[[Curve], Any]]]
BareMetadata = Union['str', Sequence, Callable[[Curve], Any]]
Metadata = Union[NamedMetadata, BareMetadata]


class Curves:
    """A container of `Curve`s

    Use `Curves()` to construct an empty sequence, or see `Curves.single` to initialize with a
    single curve.
    """

    def __init__(self, curves: Optional[List[Curve]] = None):
        """Construct a new `Curves` sequence"""

        self.curves: List[Curve] = curves or []
        """The `Curve`s contained in this `Curves`"""

    @property
    def n(self) -> int:
        """Number of curves in the sequence"""
        return len(self.curves)

    def __len__(self) -> int:
        """Number of curves in the sequence"""
        return len(self.curves)

    def __add__(self, other: Union[Curves, List[Curve]]) -> Curves:
        """Concatenate two `Curves` to form a new sequence"""
        if isinstance(other, Curves):
            return Curves(self.curves + other.curves)
        else:
            # Just hope this raises a reasonable error if it fails
            return Curves(self.curves + other)

    def keys(self, mode: Literal['intersection', 'union'] = 'union') -> Set[str]:
        """Return unique keys of data in the curves"""
        if self.n == 0:
            return set()

        keys = (set(c.data.keys()) for c in self)
        if mode == 'intersection':
            return set.intersection(*keys)
        elif mode == 'union':
            return set.union(*keys)
        else:
            raise ValueError('mode')

    def __repr__(self) -> str:
        if keys := self.keys():
            data = ', '.join(k for k in sorted(keys))
            # Gross triple nested curlies here so it formats like `Curves(n=3, data={foo, bar})`
            return f'Curves(n={self.n}, data={{{data}}})'
        else:
            return f'Curves(n={self.n})'

    def __iter__(self) -> Iterator[Curve]:
        """Iterate over curves in the sequence"""
        return iter(self.curves)

    def iter_curves(self, idx: str = 'idx') -> Iterator[Curve]:
        """Iterate over curves in the sequence

        The index of the curve is stored in the `idx` metadata parameter.
        """
        for i, c in enumerate(self):
            yield c.with_data(**{idx: i})

    def get_named_data(
            self,
            data: Union[BareMetadata, NamedMetadata],
            **kwargs
    ) -> Tuple[str, Sequence]:
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
                raise TypeError(f"Expected metadata name to be str, got {name}")
            return name, self.get_data(data[1], **kwargs)

        if isinstance(data, Callable):
            name = data.__name__ if hasattr(data, '__name__') else str(data)
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
                - A function on `Curve`s
                - An array or list of values

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
            if data in self.keys('union'):
                if allow_default:
                    return array([c.data.get(data, default) for c in self])
                elif data in self.keys('intersection'):
                    raise KeyError(f"Metadata '{data}' is only present on some curves")
                else:
                    return array([c[data] for c in self])
            elif hasattr(Curve, data):
                if allow_property:
                    return array([getattr(c, data) for c in self])
                else:
                    raise KeyError(
                        f"Metadata '{data}' not found; "
                        "it's a Curve property but `allow_property` is False",
                    )
            else:
                raise KeyError(f"Metadata '{data}' not found")
        elif isinstance(data, Callable):
            return array([data(c) for c in self])
        elif isinstance(data, Sequence):
            if (n := len(data)) != self.n:
                raise ValueError(f"Expected a sequence of length self.n = {self.n}, got {n}")
            return asarray(data)
        else:
            raise TypeError(f"Unrecognized data type {type(data).__name}: {data}")

    def __getitem__(
            self,
            idx: Union[str, int, slice, ndarray, Sequence[int]],
    ) -> Union[Curve, Curves, ndarray]:
        """Convenience method to index the sequence

        `CurveSequence[int]` returns the curve stored at that index.
        `CurveSequence[str]` returns a `ndarray` of `n` metadata values.
        `CurveSequence[fn]` for `fn: Callable[[Curve], Any]` returns a `ndarray` of the values
        of that function called on the `n` curves in the sequence.

        Otherwise, `CurveSequence[idx]` returns a new `CurveSequence` for that index,
        obeying list slicing and numpy smart indexing behavior. E.g. `sequence[::3]` returns
        a new curve sequence for every third curve in the original sequence.

        """
        if isinstance(idx, (str, Callable)):
            return self.get_data(idx)
        elif isinstance(idx, (int, np.integer)):
            # Recast to int here so indexing with np.int works as expected
            return self.curves[int(idx)]
        else:
            idx = arange(self.n)[idx]
            curves = [self.curves[i] for i in idx]
            return Curves(curves=curves)

    def plot(
            self,
            y: NamedMetadata,
            x: NamedMetadata = 'time',
            ax: Optional[Axes] = None,
            label_axes: Optional[bool] = None,
            label: Optional[str | bool] = True,
            *args,
            **kwargs,
    ):
        """Plot metadata values against each other.

        By default, the independent variable `y` is 'time', if it's present.

        Parameters
        ----------
        y
            The name of the parameter to plot on the y-axis.
            Can also be a function `Curve` -> float.

        x
            The name of the parameter to plot on the x-axis.
            Can also be a function 'Curve' -> float.

        label_axes
            If true, set x and y labels. Defaults to true if a new axes is created.

        label
            Name to label the plot with, for use in matplotlib legend. Defaults to the name
            of the `y` parameter.

        ax : `matplotlib.axes.Axes`, default current axes
            The axes to plot in.

        Remaining *args and **kwargs are passed to `matplotlib.pyplot.plot`

        """
        if ax is None:
            ax = plt.gca()
            if label_axes is None:
                label_axes = True

        xname, xdata = self.get_named_data(x)
        yname, ydata = self.get_named_data(y)

        if label_axes:
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)

        if label and isinstance(label, bool):
            label = yname

        ax.plot(xdata, ydata, *args, label=label, **kwargs)

    def append(self, curve: Curve):
        """Add a curve to the sequence"""
        self.curves.append(curve)

    def subplots(
            self,
            n_rows: Optional[int] = 1,
            n_cols: Optional[int] = None,
            axis: Optional[str] = 'scaled',
            show_axes=False,
            plot_fn: Optional[Callable[[Curve], None]] = None,
            subtitle: Optional[Union[str, Callable[[Curve], str]]] = None,
            share_xy=True,
            figsize: Tuple[float, float] = (15, 5),
            idx: str = 'idx',
    ):
        """Plot each curve in the sequence in a different subplot

        Parameters
        ----------
        figsize : (float, float)
            The size of the overall superfigure.

        n_rows : int, default 1
            Number of rows.

        n_cols : int, optional
            Number of columns. By default, `n_cols` = ceil(self.n / n_rows)`. If `n_cols` is specified, and
            `n_rows` * `n_cols` < `self.n`, the curve sequence is automatically subsampled.

        axis : str, optional, default 'scaled'
            Argument to `plt.axis`. By default this is 'equal' (i.e., make circles circular).

        show_axes : bool, default False
            Whether to show each subplot axes, i.e. border and x/y ticks, etc.

        plot_fn : function `Curve` -> None
            By default this just dispatches to `Curve.plot_edges(directed=False)`.

        subtitle : str, function `Curve` -> str, optional
            A convenience argument to put a title over each subplot. If `subtitle` is a string,
            a title is constructed from the corresponding curve metadata. Otherwise, `subtitle`
            should be a function that accepts a curve and returns a string.

        share_xy : bool, default True
            Whether each subplot should share x/y limits.

        idx : str
            The index of the curve in this collection is stored in the curve metadata property
            with this name.

        Returns
        -------
        `ndarray[Axes]` (`n_rows`, `n_cols`) array of `matplotlib.axes.Axes` objects

        """
        if not plot_fn:
            def plot_fn(c: Curve):
                c.plot_edges(directed=False)

        if isinstance(key := subtitle, str):
            def subtitle(c: Curve) -> str:
                return f'{key} = {c[key]}'

        plot_idxs = arange(self.n)
        if n_cols is None:
            n_cols = int(ceil(self.n / n_rows))
        else:
            n_plots = n_rows * n_cols
            step = (self.n // n_plots) if (n_plots < self.n) else 1
            plot_idxs = arange(0, self.n, step)
            if len(plot_idxs) > n_plots:
                plot_idxs = plot_idxs[:n_plots]
                plot_idxs[-1] = self.n - 1  # Force the last step

        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            sharex=share_xy,
            sharey=share_xy,
            figsize=figsize,
            squeeze=False,
        )
        axs_flat = axs.reshape(-1)
        all_curves = list(self.iter_curves(idx=idx))  # So we can use the original index

        for plot_idx, ax in zip(plot_idxs, axs_flat):
            fig.sca(ax)  # Set current axis, shouldn't be nec but makes plot_fns more convenient
            curve = all_curves[plot_idx]
            plot_fn(curve)

            if axis:
                ax.axis(axis)

            if not show_axes:
                ax.axis('off')

            if subtitle:
                ax.set_title(subtitle(curve))

        # If (n_rows * n_cols) > self.n, hide the remaining axes
        for ax in axs_flat[len(plot_idxs):]:
            ax.axis('off')

        fig.tight_layout()

        return axs

    def superimposed(
            self,
            ax: Optional[Axes] = None,
            plot_fn: Optional[Callable[[Curve], Any]] = None,
            color: Optional[Metadata] = None,
            clim: Optional[Tuple[float, float]] = None,
            idx: str = 'idx',
    ) -> List[Any]:
        """Plot every curve in the same axes

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, default current axes

        plot_fn : function `Curve` -> `None`, optional
            Function to plot the curve. By default, this dispatches to `curve.plot`

        color : str, optional
            The name of a curve metadata parameter to color by. If `plot_fn` is supplied,
            this is ignored.

        clim
            Range to scale color data to.

        idx : str, default 'idx'
            The index of the curve in this collection is stored in this curve's metadata.

        Returns
        -------
        list
            List of objects returned by `plot_fn`.
        """

        ax = _get_ax(ax)

        if not plot_fn:
            if color is not None:
                cmap = plt.get_cmap('viridis')
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

    def animate(
            self,
            frames: Optional[Union[Sequence[int], Callable[[], int]]] = None,
            **kwargs,
    ):
        from matplotlib import animation
        kwargs.setdefault('save_count', 2 * self.n + 1)
        frames = frames or self._animation_frames()

        fig, ax = plt.subplots()
        line = self[0].plot(ax=ax)

        for c in self:
            ax.dataLim.update_from_data_xy(c.pts)

        ax.axis('equal')
        ax.autoscale_view()

        def update(frame):
            curve = self[frame]
            x, y = curve.explicity_closed_points.T
            line.set_xdata(x)
            line.set_ydata(y)
            return line

        return animation.FuncAnimation(
            fig=fig,
            func=update,  # type: ignore
            frames=frames,
            interval=30,
            **kwargs,
        )
