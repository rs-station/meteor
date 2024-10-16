from __future__ import annotations

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map
from meteor.testing import MapColumns
from meteor.utils import numpy_array_to_map

RESOLUTION = 1.0
UNIT_CELL = gemmi.UnitCell(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90)
SPACE_GROUP = gemmi.find_spacegroup_by_name("P1")
CARBON1_POSITION = (5.0, 5.0, 5.0)
CARBON2_POSITION = (5.0, 5.2, 5.0)

NP_RNG = np.random.default_rng()


@pytest.fixture
def test_map_columns() -> MapColumns:
    return MapColumns(
        amplitude="F",
        phase="PHI",
        uncertainty="SIGF",
    )


def single_carbon_density(
    carbon_position: tuple[float, float, float],
    space_group: gemmi.SpaceGroup,
    unit_cell: gemmi.UnitCell,
    d_min: float,
) -> gemmi.Ccp4Map:
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

    density_map = gemmi.DensityCalculatorX()
    density_map.d_min = d_min
    density_map.grid.setup_from(structure)
    density_map.put_model_density_on_grid(structure[0])

    ccp4_map = gemmi.Ccp4Map()
    ccp4_map.grid = density_map.grid
    ccp4_map.update_ccp4_header()

    return ccp4_map


def single_atom_map_coefficients(*, noise_sigma: float) -> Map:
    density_map = single_carbon_density(CARBON1_POSITION, SPACE_GROUP, UNIT_CELL, RESOLUTION)
    density_array = np.array(density_map.grid)
    grid_values = density_array + noise_sigma * NP_RNG.normal(size=density_array.shape)
    ccp4_map = numpy_array_to_map(grid_values, spacegroup=SPACE_GROUP, cell=UNIT_CELL)

    map_coefficients = Map.from_ccp4_map(ccp4_map=ccp4_map, high_resolution_limit=RESOLUTION)

    uncertainties = noise_sigma * np.ones_like(map_coefficients.phases)
    uncertainties = rs.DataSeries(uncertainties, index=map_coefficients.index)
    map_coefficients.set_uncertainties(uncertainties)

    return map_coefficients


@pytest.fixture
def ccp4_map() -> gemmi.Ccp4Map:
    return single_carbon_density(CARBON1_POSITION, SPACE_GROUP, UNIT_CELL, RESOLUTION)


@pytest.fixture
def noise_free_map() -> Map:
    return single_atom_map_coefficients(noise_sigma=0.0)


@pytest.fixture
def noisy_map() -> Map:
    return single_atom_map_coefficients(noise_sigma=0.03)


@pytest.fixture
def very_noisy_map() -> Map:
    return single_atom_map_coefficients(noise_sigma=1.0)


@pytest.fixture
def random_difference_map(test_map_columns: MapColumns) -> Map:
    hall = rs.utils.generate_reciprocal_asu(UNIT_CELL, SPACE_GROUP, RESOLUTION, anomalous=False)
    sigma = 1.0

    h, k, l = hall.T  # noqa: E741
    number_of_reflections = len(h)

    ds = rs.DataSet(
        {
            "H": h,
            "K": k,
            "L": l,
            test_map_columns.amplitude: sigma * NP_RNG.normal(size=number_of_reflections),
            test_map_columns.phase: NP_RNG.uniform(-180, 180, size=number_of_reflections),
        },
        spacegroup=SPACE_GROUP,
        cell=UNIT_CELL,
    ).infer_mtz_dtypes()

    ds = ds.set_index(["H", "K", "L"])
    ds[test_map_columns.amplitude] = ds[test_map_columns.amplitude].astype("SFAmplitude")

    uncertainties = sigma * np.ones_like(ds[test_map_columns.amplitude])
    uncertainties = rs.DataSeries(uncertainties, index=ds.index)
    ds[test_map_columns.uncertainty] = uncertainties.astype(rs.StandardDeviationDtype())

    return Map(
        ds,
        amplitude_column=test_map_columns.amplitude,
        phase_column=test_map_columns.phase,
        uncertainty_column=test_map_columns.uncertainty,
    )
