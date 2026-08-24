# Copyright (c) 2024 Reciprocal Space Ship
# SPDX-License-Identifier: MIT

from pathlib import Path

import gemmi
import numpy as np
import pytest
from numpy import testing as npt

from meteor import sfcalc
from meteor.rsmap import Map
from meteor.testing import single_carbon_structure

RESOLUTION = 1.0
UNIT_CELL = gemmi.UnitCell(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90)
SPACE_GROUP = gemmi.find_spacegroup_by_name("P1")
CARBON1_POSITION = (5.0, 5.0, 5.0)


@pytest.fixture
def structure() -> gemmi.Structure:
    return single_carbon_structure(CARBON1_POSITION, SPACE_GROUP, UNIT_CELL)


def test_sf_calc(structure: gemmi.Structure) -> None:
    calc_map: Map = sfcalc.gemmi_structure_to_calculated_map(
        structure, high_resolution_limit=RESOLUTION
    )
    assert calc_map.resolution_limits[1] == RESOLUTION
    assert calc_map.spacegroup == SPACE_GROUP
    assert calc_map.cell == UNIT_CELL
    assert not calc_map.has_uncertainties
    assert np.any(calc_map.amplitudes != 0.0)
    assert np.any(calc_map.phases != 0.0)


def test_cif_to_calculated_map(testing_cif_file: Path) -> None:
    calc_map: Map = sfcalc.structure_file_to_calculated_map(
        testing_cif_file, high_resolution_limit=RESOLUTION
    )
    npt.assert_allclose(calc_map.resolution_limits[1], RESOLUTION, atol=1e-5)

    # CRYST1   51.990   62.910   72.030  90.00  90.00  90.00 P 21 21 21
    assert calc_map.spacegroup == gemmi.find_spacegroup_by_name("P 21 21 21")
    assert calc_map.cell == gemmi.UnitCell(
        a=51.990, b=62.910, c=72.030, alpha=90, beta=90, gamma=90
    )

    assert np.any(calc_map.amplitudes != 0.0)
    assert np.any(calc_map.phases != 0.0)


def test_structure_file_to_calculated_map(testing_pdb_file: Path) -> None:
    calc_map: Map = sfcalc.structure_file_to_calculated_map(
        testing_pdb_file, high_resolution_limit=RESOLUTION
    )
    npt.assert_allclose(calc_map.resolution_limits[1], RESOLUTION, atol=1e-5)

    # CRYST1   51.990   62.910   72.030  90.00  90.00  90.00 P 21 21 21
    assert calc_map.spacegroup == gemmi.find_spacegroup_by_name("P 21 21 21")
    assert calc_map.cell == gemmi.UnitCell(
        a=51.990, b=62.910, c=72.030, alpha=90, beta=90, gamma=90
    )

    assert np.any(calc_map.amplitudes != 0.0)
    assert np.any(calc_map.phases != 0.0)


def test_structure_file_to_calculated_map_fails_if_pdb_path_wrong() -> None:
    not_a_valid_path = Path("not-a-valid-path")
    with pytest.raises(OSError, match="could not find file"):
        sfcalc.structure_file_to_calculated_map(not_a_valid_path, high_resolution_limit=RESOLUTION)
