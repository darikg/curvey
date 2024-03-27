from __future__ import annotations

from pathlib import Path

import pytest

from curvey import Curve
from curvey.shape_structure_dataset import ShapeStructureDataset

_DATASET_FILE = Path("~/Downloads/ShapesJSON.zip").expanduser()
_DATASET_MISSING = not _DATASET_FILE.exists()
_DATASET_MISSING_REASON = "Dataset file could not be found"


def test_missing_zip_file_raises():
    with pytest.raises(FileNotFoundError):
        _ = ShapeStructureDataset("doesnt_exist.zip")


@pytest.fixture()
def dataset():
    return ShapeStructureDataset(_DATASET_FILE)


@pytest.mark.skipif(_DATASET_MISSING, reason=_DATASET_MISSING_REASON)
def test_load_curve(dataset):
    curve = dataset.load_curve("elephant-1")
    assert isinstance(curve, Curve)
