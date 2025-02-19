import gemmi
import numpy as np
import pytest

from meteor import testing as meteortesting
from meteor.rsmap import Map


class AnyOldObject:
    x: float = 0.0
    y: list[float] = [1.0, 2.0]  # noqa: RUF012


def test_map_columns_smoke() -> None:
    meteortesting.MapColumns(amplitude="amp", phase="phase", uncertainty=None)
    meteortesting.MapColumns(amplitude="amp", phase="phase", uncertainty="sig")


def test_phases_allclose() -> None:
    close1 = np.array([0.0, 89.9999, 179.9999, 359.9999, 360.9999])
    close2 = np.array([-0.0001, 90.0, 180.0001, 360.0001, 0.9999])
    far = np.array([0.5, 90.5, 180.0, 360.0, 361.0])

    meteortesting.assert_phases_allclose(close1, close2)

    with pytest.raises(AssertionError):
        meteortesting.assert_phases_allclose(close1, far)


def test_map_corrcoeff(noise_free_map: Map, np_rng: np.random.Generator) -> None:
    assert meteortesting.map_corrcoeff(noise_free_map, noise_free_map) == 1.0

    noisy_map = noise_free_map.copy()
    noisy_map.amplitudes += np_rng.normal(size=len(noise_free_map))
    noisier_map = noise_free_map.copy()
    noisier_map.amplitudes += 10.0 * np_rng.normal(size=len(noise_free_map))

    noisy_cc = meteortesting.map_corrcoeff(noise_free_map, noisy_map)
    noisier_cc = meteortesting.map_corrcoeff(noise_free_map, noisier_map)
    assert 1.0 > noisy_cc > noisier_cc > 0.0


def test_single_carbon_structure_smoke() -> None:
    carbon_position = (4.0, 5.0, 6.0)
    space_group = gemmi.find_spacegroup_by_name("P212121")
    unit_cell = gemmi.UnitCell(a=9.0, b=10.0, c=11.0, alpha=90, beta=90, gamma=90)
    structure = meteortesting.single_carbon_structure(carbon_position, space_group, unit_cell)
    assert isinstance(structure, gemmi.Structure)


def test_single_carbon_density_smoke() -> None:
    carbon_position = (4.0, 5.0, 6.0)
    space_group = gemmi.find_spacegroup_by_name("P212121")
    unit_cell = gemmi.UnitCell(a=9.0, b=10.0, c=11.0, alpha=90, beta=90, gamma=90)
    high_resolution_limit = 1.0
    density = meteortesting.single_carbon_density(
        carbon_position, space_group, unit_cell, high_resolution_limit
    )
    assert isinstance(density, gemmi.Ccp4Map)
    epsilon = 1e-16
    assert np.all(np.array(density.grid) > -epsilon)


def test_check_leaf_floats_are_finite() -> None:
    obj = AnyOldObject()

    issues = meteortesting.check_leaf_floats_are_finite(obj)
    assert len(issues) == 0

    obj.x = np.inf
    issues = meteortesting.check_leaf_floats_are_finite(obj)
    assert len(issues) == 1

    obj.y = [np.inf, np.nan, 1.0]
    issues = meteortesting.check_leaf_floats_are_finite(obj)
    assert len(issues) == 3
