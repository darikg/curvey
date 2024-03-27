from __future__ import annotations

from typing import cast

import pytest

from curvey import Curve, Curves
from curvey.flow import CurveShorteningFlow, SingularityFreeMeanCurvatureFlow, Solver, WillmoreFlow


@pytest.fixture()
def dumbbell():
    return Curve.dumbbell(n=100)


def _assert_flow_becomes_rounder(solver: Solver):
    solver.run()
    current = cast(Curve, solver.current)
    assert current["step"] > 0
    curves = cast(Curves, solver.history)

    for c0, c1 in zip(curves[:-1], curves[1:]):
        assert c1.roundness < c0.roundness


def test_csf(dumbbell):
    solver = CurveShorteningFlow().solver(
        initial=dumbbell,
        timestep=1e-3,
        max_step=5,
    )
    _assert_flow_becomes_rounder(solver)


def test_sf_mcf(dumbbell):
    solver = SingularityFreeMeanCurvatureFlow().solver(
        initial=dumbbell,
        timestep=0.1,
        max_step=5,
    )
    _assert_flow_becomes_rounder(solver)


def test_willmore(dumbbell):
    solver = WillmoreFlow().solver(initial=dumbbell, timestep=0.01, max_step=5)

    _assert_flow_becomes_rounder(solver)
