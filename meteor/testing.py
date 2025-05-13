"""used during testing"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from .rsmap import Map
from .settings import MAP_SAMPLING


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


def map_corrcoeff(map1: Map, map2: Map) -> float:
    map1_np = map1.to_3d_numpy_map(map_sampling=MAP_SAMPLING).flatten()
    map2_np = map2.to_3d_numpy_map(map_sampling=MAP_SAMPLING).flatten()
    rho = np.corrcoef(map1_np, map2_np)
    return float(rho[0, 1])


def check_test_file_exists(path: Path) -> None:
    if not path.exists():
        msg = f"cannot find {path}, use github LFS to retrieve this file from the parent repo"
        raise OSError(msg)


def single_carbon_structure(
    carbon_position: tuple[float, float, float],
    space_group: gemmi.SpaceGroup,
    unit_cell: gemmi.UnitCell,
) -> gemmi.Structure:
    model = gemmi.Model(0)
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


def check_leaf_floats_are_finite(obj: Any, path: str = "root") -> list[str]:
    issues = []

    if isinstance(obj, float):
        if not np.isfinite(obj):
            issues.append(path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            issues.extend(check_leaf_floats_are_finite(value, f"{path}.{key}"))
    elif isinstance(obj, list | tuple | set):
        for i, item in enumerate(obj):
            issues.extend(check_leaf_floats_are_finite(item, f"{path}[{i}]"))
    elif hasattr(obj, "__dict__"):  # Handle objects with attributes
        for key, value in vars(obj).items():
            issues.extend(check_leaf_floats_are_finite(value, f"{path}.{key}"))

    return issues
