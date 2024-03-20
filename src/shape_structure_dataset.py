"""Interface to the 2D Shape Structure Dataset

# Quickstart

Download the dataset [here](https://2dshapesstructure.github.io/data/ShapesJSON.zip).

```python
from curvey.shape_structure_dataset import ShapeStructureDataset

dataset = ShapeStructureDataset('~/Downloads/ShapesJSON.zip')
print(', '.join(dataset.classes))
curve = dataset.load_curve('elephant-1')  # or load_curve('elephant', 0)
curve.plot()
"""


from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path

from typing import TypedDict, List, Union, Dict, Optional, Tuple, Set, Iterator
from zipfile import ZipFile

from numpy import ndarray, array

from .curve import Curve


class _JsonPoint(TypedDict):
    x: float
    y: float


class _JsonTriangle(TypedDict):
    p1: int
    p2: int
    p3: int


class _JsonShape(TypedDict):
    points: List[_JsonPoint]
    triangles: List[_JsonTriangle]


class ShapeStructureDataset:
    """Interface to the 2D Shape Structure Dataset zip file

    https://2dshapesstructure.github.io/

    Parameters
    ----------
    dataset : pathlike, the path to the `ShapesJSON.zip` zip file. Download it from
    `https://2dshapesstructure.github.io/data/ShapesJSON.zip`.
    """

    # regex to match e.g. 'Shapes/Bone-13.json'
    _shape_file_regex = re.compile(r'Shapes/([^.]+)\.json')

    def __init__(self, dataset: Union[str, Path]):
        self.dataset = ZipFile(Path(dataset).expanduser())
        self.cache: Dict[str, ndarray] = {}

    def _load_json(self, name: str, idx: Optional[int] = True) -> _JsonShape:
        name = self._canonical_name(name, idx)
        shape_bytes = self.dataset.read(f'Shapes/{name}.json')
        return json.loads(shape_bytes)

    def load_shape(
            self,
            name: str,
            idx: Optional[int] = None,
            load_triangles: bool = True,
    ) -> Tuple[ndarray, Optional[ndarray]]:
        """Return an (n_verts, 2) array of points and an (n_faces, 3) array of triangles"""
        data = self._load_json(name, idx)
        pts = array([[d['x'], d['y']] for d in data['points']])
        if load_triangles:
            tris = array([[d['p1'], d['p2'], d['p3']] for d in data['triangles']])
        else:
            tris = None
        return pts, tris

    def load_points(self, name: str, idx: Optional[int] = None) -> ndarray:
        """Load the points from the specified dataset.

        Note
        ----
        Some of the shapes in the dataset include repeated points. This method returns point-sets
        as-is, without any further processing.

        The dataset also includes triangulations. Use method `load_shape` to load those as well.
        """
        name = self._canonical_name(name, idx)
        if name in self.cache:
            return self.cache[name]
        pts, _ = self.load_shape(name, load_triangles=False)
        self.cache[name] = pts
        return pts

    def load_curve(self, name: str, idx: Optional[int] = None) -> Curve:
        """Construct a `Curve` from the named shape in the dataset

        Can load curves by explicit name, e.g. `dataset.load_curve('Bone-13')`,
        or a class name and an index, e.g. dataset.load_curve('Bone', 13).
        Names are case-insensitive.

        Note
        ----
        Some of the shapes in the dataset are stored with repeated points. These are dropped
        automatically. To get the original points, use methods `load_points` or `load_shapes`.
        """
        pts = self.load_points(name, idx).copy()  # Make sure the cache can't be mutated
        return Curve(pts).drop_repeated_points()

    def _canonical_name(self, name: str, idx: Optional[int] = None) -> str:
        if idx is None:
            return name
        else:
            return self.names_by_class[name][idx]

    @cached_property
    def all_names(self) -> Set[str]:
        """Names of the shapes in the dataset"""
        return set(self._iter_all_names())

    @cached_property
    def names_by_class(self) -> Dict[str, Tuple[str]]:
        from collections import defaultdict
        classes = defaultdict(list)
        special_classes = ('image', 'device', 'dino')

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

            class_name = name.split('-')[0].lower()
            classes[class_name].append(name)
        return {k: tuple(sorted(v)) for k, v in classes.items()}

    @cached_property
    def classes(self) -> Tuple[str]:
        """Names of the shape classes in the dataset"""
        return tuple(sorted(self.names_by_class.keys()))

    def _iter_all_names(self) -> Iterator[str]:
        for f in self.dataset.filelist:
            if match := self._shape_file_regex.match(f.filename):
                yield match.group(1)
