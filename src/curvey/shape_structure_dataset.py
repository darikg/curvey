"""Interface to the [2D Shape Structure Dataset](https://2dshapesstructure.github.io/index.html)

Download the shapes zip file [here](https://2dshapesstructure.github.io/data/ShapesJSON.zip).

The dataset groups shapes by class; shapes can be loaded either by an exact name or a class name
and an index.

```python
from curvey.shape_structure_dataset import ShapeStructureDataset

try:
    dataset = ShapeStructureDataset('~/Downloads/ShapesJSON.zip')
except FileNotFoundError:
    print("Couldn't find the ShapesJSON.zip file")
else:
    print(', '.join(dataset.class_names))
    curve = dataset.load_curve('elephant-1')  # or load_curve('elephant', 0)
    curve.plot()
```
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import TypedDict
from zipfile import ZipFile

from numpy import array

from .curve import Curve
from .polygon import Polygon
from .triangulation import Triangulation


class _JsonPoint(TypedDict):
    x: float
    y: float


class _JsonTriangle(TypedDict):
    p1: int
    p2: int
    p3: int


class _JsonShape(TypedDict):
    points: list[_JsonPoint]
    triangles: list[_JsonTriangle]


class ShapeStructureDataset:
    """Interface to the 2D Shape Structure Dataset zip file

    Parameters
    ----------
    dataset
        the path to the `ShapesJSON.zip` zip file. Download it
        [here](https://2dshapesstructure.github.io/data/ShapesJSON.zip)

    Notes
    -----
    Shapes in the dataset are actually a 2d triangular mesh, not a curve. Curves are loaded without
    any processing, assuming vertices are listed in order, so they may have irregularities, such as
    repeated points or self-intersections. Shapes with topological holes like the coffee cup are
    returned as a single curve without a care in the world.
    """

    # regex to match e.g. 'Shapes/Bone-13.json'
    _shape_file_regex = re.compile(r"Shapes/([^.]+)\.json")

    def __init__(self, dataset: str | Path):
        self.dataset = ZipFile(Path(dataset).expanduser())
        self.cache: dict[str, Triangulation] = {}

    def _load_json(self, name: str, idx: int | None = None) -> _JsonShape:
        name = self._canonical_name(name, idx)
        shape_bytes = self.dataset.read(f"Shapes/{name}.json")
        return json.loads(shape_bytes)

    def load_triangulation(
        self,
        name: str,
        idx: int | None = None,
    ) -> Triangulation:
        """The triangulated shape"""
        name = self._canonical_name(name, idx)
        if name in self.cache:
            return self.cache[name]

        data = self._load_json(name)
        pts = array([[d["x"], d["y"]] for d in data["points"]])
        faces = array([[d["p1"], d["p2"], d["p3"]] for d in data["triangles"]])
        tris = self.cache[name] = Triangulation(pts, faces)
        return tris

    def load_polygon(self, name: str, idx: int | None = None) -> Polygon:
        tris = self.load_triangulation(name, idx)
        loops = list(tris.boundary_loops())
        return Polygon(loops[0], loops[1:])

    def load_curve(self, name: str, idx: int | None = None) -> Curve:
        """Construct a `Curve` from the named shape in the dataset

        Can load curves by explicit name, e.g. `dataset.load_curve('Bone-13')`,
        or a class name and an index, e.g. `dataset.load_curve('Bone', 1)`.
        Names are case-insensitive.
        """
        poly = self.load_polygon(name, idx)
        return poly.exterior

    def _canonical_name(self, name: str, idx: int | None = None) -> str:
        if idx is None:
            return name

        return self.names_by_class[name][idx]

    @cached_property
    def all_names(self) -> set[str]:
        """Names of the shapes in the dataset"""
        return set(self._iter_all_names())

    @cached_property
    def names_by_class(self) -> dict[str, tuple[str, ...]]:
        """Shape names grouped by their class

        Returns
        -------
        classes :
            `dict[class_name] -> tuple[object_names, ...]`
        """
        from collections import defaultdict

        classes = defaultdict(list)
        special_classes = ("image", "device", "dino")

        for name in self.all_names:
            for special in special_classes:
                if name.startswith(special):
                    classes[special].append(name)
                    is_special = True
                    break
            else:
                is_special = False

            if is_special:
                continue

            class_name = name.split("-")[0].lower()
            classes[class_name].append(name)
        return {k: tuple(sorted(v)) for k, v in classes.items()}

    @cached_property
    def class_names(self) -> tuple[str, ...]:
        """Names of the unique shape classes in the dataset"""
        return tuple(sorted(self.names_by_class.keys()))

    def _iter_all_names(self) -> Iterator[str]:
        for f in self.dataset.filelist:
            if match := self._shape_file_regex.match(f.filename):
                yield match.group(1)
