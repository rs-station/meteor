import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor import scale
from meteor.rsmap import Map
from meteor.scale import _compute_anisotropic_scale_factors


@pytest.fixture
def miller_dataseries() -> rs.DataSeries:
    miller_indices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    data = np.array([8.0, 4.0, 2.0, 1.0, 1.0], dtype=np.float32)
    return rs.DataSeries(
        data,
        index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
    )


def test_compute_anisotropic_scale_factors_smoke(miller_dataseries: rs.DataSeries) -> None:
    # test call signature, valid return
    random_params: scale.ScaleParameters = (1.5,) * 7
    scale_factors = _compute_anisotropic_scale_factors(miller_dataseries.index, random_params)
    assert len(scale_factors) == len(miller_dataseries)


def test_compute_scale_factors_identical(miller_dataseries: rs.DataSeries) -> None:
    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=miller_dataseries,
    )
    assert (scale_factors == 1.0).all()

    equal_uncertainties = miller_dataseries.copy()
    equal_uncertainties[:] = np.ones(len(equal_uncertainties))

    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=miller_dataseries,
        reference_uncertainties=equal_uncertainties,
        to_scale_uncertainties=equal_uncertainties,
    )
    assert (scale_factors == 1.0).all()


def test_compute_scale_factors_shuffle_indices(miller_dataseries: rs.DataSeries) -> None:
    shuffled_miller_dataseries = miller_dataseries.sample(frac=1)
    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=shuffled_miller_dataseries,
    )
    assert (scale_factors == 1.0).all()


def test_compute_scale_factors_nans(miller_dataseries: rs.DataSeries) -> None:
    with_nans = miller_dataseries.copy()
    with_nans.iloc[0] = np.nan
    miller_dataseries.iloc[1] = np.nan
    miller_dataseries.iloc[2] = np.nan
    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=with_nans,
    )
    assert (scale_factors == 1.0).all()


def test_compute_scale_factors_scalar(miller_dataseries: rs.DataSeries) -> None:
    multiple = 2.0
    doubled_miller_dataseries = miller_dataseries / multiple

    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=doubled_miller_dataseries,
    )
    np.testing.assert_array_almost_equal(scale_factors, multiple)


def test_compute_scale_factors_anisotropic(miller_dataseries: rs.DataSeries) -> None:
    flat_miller_dataseries = miller_dataseries.copy()
    flat_miller_dataseries[:] = np.ones(len(miller_dataseries))

    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=flat_miller_dataseries,
    )
    np.testing.assert_array_almost_equal(scale_factors, miller_dataseries.values)


@pytest.mark.parametrize("use_uncertainties", [False, True])
def test_scale_maps(random_difference_map: Map, use_uncertainties: bool) -> None:
    multiple = 2.0
    doubled_difference_map: Map = random_difference_map.copy()
    doubled_difference_map.amplitudes /= multiple

    scaled = scale.scale_maps(
        reference_map=random_difference_map,
        map_to_scale=doubled_difference_map,
        weight_using_uncertainties=use_uncertainties,
    )
    np.testing.assert_array_almost_equal(
        scaled.amplitudes,
        random_difference_map.amplitudes,
    )
    np.testing.assert_array_almost_equal(
        scaled.phases,
        random_difference_map.phases,
    )
    np.testing.assert_array_almost_equal(
        scaled.uncertainties / multiple,
        random_difference_map.uncertainties,
    )


@pytest.mark.parametrize("use_uncertainties", [False, True])
def test_scale_maps_nans_in_input(random_difference_map: Map, use_uncertainties: bool) -> None:
    another_difference_map = random_difference_map.copy()
    another_difference_map.loc[0, another_difference_map.amplitude_column_name] = np.nan

    scale.scale_maps(
        reference_map=random_difference_map,
        map_to_scale=another_difference_map,
        weight_using_uncertainties=use_uncertainties,
    )


def test_scale_maps_uncertainty_weighting() -> None:
    x = np.array([1, 2, 3])
    y = np.array([4, 8, 2])
    phi = np.array([0, 0, 0])
    weights = np.array([1, 1, 1e6])

    miller_indices = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    index = pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"])

    reference_map = Map({"F": x, "PHI": phi, "SIGF": weights})
    reference_map.index = index
    map_to_scale = Map({"F": y, "PHI": phi, "SIGF": weights})
    map_to_scale.index = index

    scaled = scale.scale_maps(
        reference_map=reference_map,
        map_to_scale=map_to_scale,
        weight_using_uncertainties=True,
    )

    np.testing.assert_allclose(scaled["F"][(0, 0, 2)], 0.5, atol=1e-4)
    np.testing.assert_allclose(scaled["SIGF"][(0, 0, 2)], 250000.0, rtol=1e-4)
