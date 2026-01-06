import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor import scale
from meteor.rsmap import Map
from meteor.scale import ScaleParameters, compute_scale_factors


def compute_scale_factor_reference_implementation(
    miller_indices: pd.Index,
    scale_parameters: ScaleParameters,
) -> np.ndarray:
    miller_indices_as_array = np.array(list(miller_indices))
    squared_miller_indices = np.square(miller_indices_as_array)

    h_squared = squared_miller_indices[:, 0]
    k_squared = squared_miller_indices[:, 1]
    l_squared = squared_miller_indices[:, 2]

    hk_product = miller_indices_as_array[:, 0] * miller_indices_as_array[:, 1]
    hl_product = miller_indices_as_array[:, 0] * miller_indices_as_array[:, 2]
    kl_product = miller_indices_as_array[:, 1] * miller_indices_as_array[:, 2]

    # Anisotropic scaling term
    exponential_argument = -(
        h_squared * scale_parameters[1]
        + k_squared * scale_parameters[2]
        + l_squared * scale_parameters[3]
        + 2 * hk_product * scale_parameters[4]
        + 2 * hl_product * scale_parameters[5]
        + 2 * kl_product * scale_parameters[6]
    )

    return scale_parameters[0] * np.exp(exponential_argument)


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
    arb_params: scale.ScaleParameters = (1.5,) * 7
    scale_factors = compute_scale_factors(miller_dataseries.index, arb_params)
    assert len(scale_factors) == len(miller_dataseries)


def test_compute_anisotropic_scale_factors(
    miller_dataseries: rs.DataSeries, np_rng: np.random.Generator
) -> None:
    num_random_trials = 5
    for _ in range(num_random_trials):
        random_params: scale.ScaleParameters = tuple(np_rng.random(size=7))
        obtained_output = compute_scale_factors(miller_dataseries.index, random_params)
        expected_output = compute_scale_factor_reference_implementation(
            miller_dataseries.index, random_params
        )
        np.testing.assert_allclose(obtained_output, expected_output)


@pytest.mark.parametrize("use_uncertainties", [False, True])
def test_scale_maps_identical(random_difference_map: Map, use_uncertainties: bool) -> None:
    scaled_map = scale.scale_maps(
        reference_map=random_difference_map,
        map_to_scale=random_difference_map,
        weight_using_uncertainties=use_uncertainties,
    )
    pd.testing.assert_frame_equal(scaled_map, random_difference_map)


@pytest.mark.parametrize("use_uncertainties", [False, True])
@pytest.mark.parametrize("multiple", [0.4, 1.0, 2.5, 13.324])
def test_scale_maps(random_difference_map: Map, use_uncertainties: bool, multiple: float) -> None:
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


@pytest.mark.parametrize("multiple", [0.4, 1.0, 2.5, 13.324])
def test_scale_uncertainties_invariant_global_scale(
    random_difference_map: Map, multiple: float
) -> None:
    doubled_difference_map: Map = random_difference_map.copy()
    doubled_difference_map.uncertainties /= multiple

    scaled = scale.scale_maps(
        reference_map=random_difference_map,
        map_to_scale=doubled_difference_map,
        weight_using_uncertainties=True,
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
        scaled.uncertainties * multiple,
        random_difference_map.uncertainties,
    )


@pytest.mark.parametrize("use_uncertainties", [False, True])
@pytest.mark.parametrize("column", ["F", "PHI", "SIGF"])
def test_scale_maps_nans_in_input(
    random_difference_map: Map, use_uncertainties: bool, column: str
) -> None:
    another_difference_map = random_difference_map.copy()
    another_difference_map.loc[1, column] = np.nan

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
