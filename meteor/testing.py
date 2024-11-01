"""used during testing"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np


@dataclass
class MapColumns:
    amplitude: str
    phase: str
    uncertainty: str | None = None


def assert_phases_allclose(array1: np.ndarray, array2: np.ndarray, atol: float = 1e-3) -> None:
    diff = array2 - array1
    diff = (diff + 180) % 360 - 180
    absolute_difference = np.sum(np.abs(diff)) / float(np.prod(diff.shape))
    if not absolute_difference < atol:
        msg = f"per element diff {absolute_difference} > tolerance {atol}"
        raise AssertionError(msg)


def check_test_file_exists(path: Path) -> None:
    if not path.exists():
        msg = f"cannot find {path}, use github LFS to retrieve this file from the parent repo"
        raise OSError(msg)


def single_carbon_structure(
    carbon_position: tuple[float, float, float],
    space_group: gemmi.SpaceGroup,
    unit_cell: gemmi.UnitCell,
) -> gemmi.Structure:
    model = gemmi.Model("single_atom")
    chain = gemmi.Chain("A")

    residue = gemmi.Residue()
    residue.name = "X"
    residue.seqid = gemmi.SeqId("1")

    atom = gemmi.Atom()
    atom.name = "C"
    atom.element = gemmi.Element("C")
    atom.pos = gemmi.Position(*carbon_position)

    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)

    structure = gemmi.Structure()
    structure.add_model(model)
    structure.cell = unit_cell
    structure.spacegroup_hm = space_group.hm

    return structure


def single_carbon_density(
    carbon_position: tuple[float, float, float],
    space_group: gemmi.SpaceGroup,
    unit_cell: gemmi.UnitCell,
    high_resolution_limit: float,
) -> gemmi.Ccp4Map:
    structure = single_carbon_structure(carbon_position, space_group, unit_cell)

    density_map = gemmi.DensityCalculatorX()
    density_map.d_min = high_resolution_limit
    density_map.grid.setup_from(structure)
    density_map.put_model_density_on_grid(structure[0])

    ccp4_map = gemmi.Ccp4Map()
    ccp4_map.grid = density_map.grid
    ccp4_map.update_ccp4_header()

    return ccp4_map
