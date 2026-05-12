from __future__ import annotations

import numpy as np
import pytest

from meteor.rsmap import Map
from meteor.scripts.common import DiffMapSet


@pytest.fixture
def diffmap_set(random_difference_map: Map) -> DiffMapSet:
    base = random_difference_map.copy()
    base["F"] = np.abs(base["F"]) + 1.0  # non-zero amplitudes
    derivative = base.copy()
    derivative["F"] += 1.0  # ensure there is some change
    return DiffMapSet(
        native=base,
        derivative=derivative,
        calculated=base,
    )


@pytest.fixture
def fixed_kparameter() -> float:
    return 0.05


@pytest.fixture
def base_cli_arguments(fixed_kparameter: float) -> list[str]:
    return [
        "fake-derivative.mtz",
        "-da",
        "F",
        "--derivative-uncertainty-column",
        "SIGF",
        "fake-native.mtz",
        "--structure",
        "fake.pdb",
        "-o",
        "fake-output.mtz",
        "-m",
        "fake-output-metadata.csv",
        "--kweight-mode",
        "fixed",
        "--kweight-parameter",
        str(fixed_kparameter),
        "--highres",
        "1.5",
        "--lowres",
        "6.0",
    ]
