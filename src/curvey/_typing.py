from __future__ import annotations

from collections.abc import Sequence
from typing import Union

from numpy import ndarray

PointsLike = Union[
    ndarray,
    Sequence[Sequence[float]],
    Sequence[tuple[float, float]],
]

EdgesLike = Union[
    ndarray,
    Sequence[Sequence[int]],
    Sequence[tuple[int, int]],
]

TrisLike = Union[
    ndarray,
    Sequence[Sequence[int]],
    Sequence[tuple[int, int, int]],
]
