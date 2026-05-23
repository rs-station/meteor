from collections.abc import Callable

import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor import scale
from meteor.rsmap import Map
from meteor.scale import (
    ParameterLengthMismatchError,
    ScaleMode,
    ScaleParameters,
    ScalingError,
    compute_scale_factors,
)

LSQ_LOSSES_TO_TEST: list[str | Callable] = ["linear", "huber"]


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


@pytest.mark.parametrize("scale_mode", ScaleMode)
def test_compute_anisotropic_scale_factors(
    scale_mode: ScaleMode, miller_dataseries: rs.DataSeries, np_rng: np.random.Generator
) -> None:
    num_random_trials = 5
    for _ in range(num_random_trials):
        random_params = np_rng.normal(size=scale_mode.number_of_parameters)
        obtained_output = compute_scale_factors(
            miller_indices=miller_dataseries.index,
            scale_parameters=tuple(random_params),
            scale_mode=scale_mode,
        )

        params_for_ref = np.zeros(7)
        params_for_ref[: scale_mode.number_of_parameters] = random_params
        if scale_mode == ScaleMode.isotropic:
            params_for_ref[2] = params_for_ref[3] = params_for_ref[1]

        expected_output = compute_scale_factor_reference_implementation(
            miller_dataseries.index, tuple(params_for_ref)
        )
        assert len(obtained_output) == len(miller_dataseries)
        np.testing.assert_allclose(obtained_output, expected_output)


def test_compute_anisotropic_scale_factors_miller_error() -> None:
    scale_mode = ScaleMode.anisotropic
    arbitrary_params = (1.0,) * scale_mode.number_of_parameters
    cut_index = np.ones((5, 2))

    # backslashes needed because this string becomes a regex search
    err_string = r"`miller_indices` should be an \(n, 3\) multi-index of miller HKL indices, got shape: \(5, 2\)"
    with pytest.raises(ValueError, match=err_string):
        _ = compute_scale_factors(
            miller_indices=cut_index,  # type: ignore[arg-type]
            scale_parameters=arbitrary_params,
            scale_mode=scale_mode,
        )


@pytest.mark.parametrize(
    "scale_mode", ["anisotropic", "isotropic", "orthogonal", "scalar", "not-valid"]
)
def test_string_scale_mode(scale_mode: str, miller_dataseries: rs.DataSeries) -> None:
    if scale_mode == "not-valid":
        with pytest.raises(ValueError, match="'not-valid' is not a valid ScaleMode"):
            _ = compute_scale_factors(
                miller_indices=miller_dataseries.index,
                scale_parameters=(1.0,) * 7,
                scale_mode=scale_mode,
            )
    else:
        arbitrary_params = (1.0,) * ScaleMode(scale_mode).number_of_parameters
        _ = compute_scale_factors(
            miller_indices=miller_dataseries.index,
            scale_parameters=arbitrary_params,
            scale_mode=scale_mode,
        )


@pytest.mark.parametrize("scale_mode", ScaleMode)
def test_parameter_length_scale_mode_mismatch(
    scale_mode: ScaleMode, miller_dataseries: rs.DataSeries
) -> None:
    wrong_length_params = (1.0,) * 10
    with pytest.raises(ParameterLengthMismatchError):
        _ = compute_scale_factors(
            miller_indices=miller_dataseries.index,
            scale_parameters=wrong_length_params,
            scale_mode=scale_mode,
        )


@pytest.mark.parametrize("use_uncertainties", [False, True])
@pytest.mark.parametrize("scale_mode", ScaleMode)
@pytest.mark.parametrize("least_squares_loss", LSQ_LOSSES_TO_TEST)
def test_scale_maps_identical(
    random_difference_map: Map,
    use_uncertainties: bool,
    scale_mode: ScaleMode,
    least_squares_loss: str,
) -> None:
    scaled_map = scale.scale_maps(
        reference_map=random_difference_map,
        map_to_scale=random_difference_map,
        weight_using_uncertainties=use_uncertainties,
        scale_mode=scale_mode,
        least_squares_loss=least_squares_loss,
    )
    pd.testing.assert_frame_equal(scaled_map, random_difference_map)


@pytest.mark.parametrize("use_uncertainties", [False, True])
@pytest.mark.parametrize("scale_mode", ScaleMode)
@pytest.mark.parametrize("least_squares_loss", LSQ_LOSSES_TO_TEST)
@pytest.mark.parametrize("multiple", [0.4, 1.0, 2.5, 13.324])
def test_scale_maps(
    random_difference_map: Map,
    use_uncertainties: bool,
    scale_mode: ScaleMode,
    least_squares_loss: str,
    multiple: float,
) -> None:
    multiplied_difference_map: Map = random_difference_map.copy()
    multiplied_difference_map.amplitudes /= multiple

    scaled = scale.scale_maps(
        reference_map=random_difference_map,
        map_to_scale=multiplied_difference_map,
        weight_using_uncertainties=use_uncertainties,
        scale_mode=scale_mode,
        least_squares_loss=least_squares_loss,
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
    multiplied_difference_map: Map = random_difference_map.copy()
    multiplied_difference_map.uncertainties /= multiple

    scaled = scale.scale_maps(
        reference_map=random_difference_map,
        map_to_scale=multiplied_difference_map,
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
@pytest.mark.parametrize("scale_mode", ScaleMode)
@pytest.mark.parametrize("least_squares_loss", LSQ_LOSSES_TO_TEST)
@pytest.mark.parametrize("column", ["F", "PHI", "SIGF"])
def test_scale_maps_nans_in_input(
    random_difference_map: Map,
    use_uncertainties: bool,
    scale_mode: ScaleMode,
    least_squares_loss: str,
    column: str,
) -> None:
    # use positive amplitudes so a single NaN in F doesn't poison the mean used to seed C
    reference_map = random_difference_map.copy()
    reference_map["F"] = np.abs(reference_map["F"]) + 1.0
    another_difference_map = reference_map.copy()
    another_difference_map.loc[1, column] = np.nan

    scale.scale_maps(
        reference_map=reference_map,
        map_to_scale=another_difference_map,
        weight_using_uncertainties=use_uncertainties,
        scale_mode=scale_mode,
        least_squares_loss=least_squares_loss,
    )


def test_scale_maps_uncertainty_weighting() -> None:
    x = np.array([1, 2, 3])
    y = np.array([4, 8, 2])
    phi = np.array([0, 0, 0])
    weights = np.array([1, 1, 1e6])

    miller_indices = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    index = pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"])

    common_columns = {"PHI": phi, "SIGF": weights}
    cell = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)

    reference_map = Map({"F": x} | common_columns, cell=cell, spacegroup=1)
    reference_map.index = index
    map_to_scale = Map({"F": y} | common_columns, cell=cell, spacegroup=1)
    map_to_scale.index = index

    scaled = scale.scale_maps(
        reference_map=reference_map,
        map_to_scale=map_to_scale,
        weight_using_uncertainties=True,
    )

    np.testing.assert_allclose(scaled["F"][(0, 0, 2)], 0.5, atol=1e-4)
    np.testing.assert_allclose(scaled["SIGF"][(0, 0, 2)], 250000.0, rtol=1e-4)


@pytest.mark.parametrize("weight_using_uncertainties", [False, True])
@pytest.mark.parametrize("scale_mode", ScaleMode)
@pytest.mark.parametrize("least_squares_loss", LSQ_LOSSES_TO_TEST)
def test_scale_mismatched_indices(
    weight_using_uncertainties: bool, scale_mode: ScaleMode, least_squares_loss: str, noisy_map: Map
) -> None:
    missing_indices = noisy_map.copy()
    missing_indices.drop(missing_indices.index[:512], inplace=True)

    _ = scale.scale_maps(
        reference_map=missing_indices,
        map_to_scale=noisy_map,
        weight_using_uncertainties=weight_using_uncertainties,
        scale_mode=scale_mode,
        least_squares_loss=least_squares_loss,
    )

    _ = scale.scale_maps(
        reference_map=noisy_map,
        map_to_scale=missing_indices,
        weight_using_uncertainties=weight_using_uncertainties,
        scale_mode=scale_mode,
        least_squares_loss=least_squares_loss,
    )


@pytest.mark.parametrize("scale_mode", ScaleMode)
def test_scale_maps_large_mismatch_protein_cell(scale_mode: ScaleMode) -> None:
    # Regression test for issue #149: when the reference and map_to_scale amplitudes differ
    # by a large overall factor and the cell is big enough that Miller indices reach typical
    # protein-crystal magnitudes, the anisotropic optimization used to take a Newton step
    # that drove the B parameters to values where exp(-h^T B h) overflowed to +inf.
    cell = gemmi.UnitCell(a=80.0, b=80.0, c=100.0, alpha=90, beta=90, gamma=120)
    spacegroup = gemmi.find_spacegroup_by_name("P 31 2 1")
    resolution = 2.0
    scale_mismatch = 16.0

    # local RNG so we don't perturb the session-scoped fixture state for other tests
    rng = np.random.default_rng(seed=149)
    hkl = rs.utils.generate_reciprocal_asu(cell, spacegroup, resolution, anomalous=False)
    n = hkl.shape[0]
    amplitudes = (np.abs(rng.normal(size=n)) * 200.0).astype("float32")
    phases = rng.uniform(-180, 180, size=n).astype("float32")
    uncertainties = np.ones(n, dtype="float32")

    ds = (
        rs.DataSet(
            {
                "H": hkl[:, 0],
                "K": hkl[:, 1],
                "L": hkl[:, 2],
                "F": amplitudes,
                "PHI": phases,
                "SIGF": uncertainties,
            },
            spacegroup=spacegroup,
            cell=cell,
        )
        .infer_mtz_dtypes()
        .set_index(["H", "K", "L"])
    )

    reference_map = Map(
        ds,
        amplitude_column="F",
        phase_column="PHI",
        uncertainty_column="SIGF",
        cell=cell,
        spacegroup=spacegroup,
    )

    mismatched = ds.copy()
    mismatched["F"] = (mismatched["F"].astype(float) / scale_mismatch).astype("float32")
    map_to_scale = Map(
        mismatched,
        amplitude_column="F",
        phase_column="PHI",
        uncertainty_column="SIGF",
        cell=cell,
        spacegroup=spacegroup,
    )

    scaled = scale.scale_maps(
        reference_map=reference_map,
        map_to_scale=map_to_scale,
        scale_mode=scale_mode,
    )
    np.testing.assert_allclose(scaled.amplitudes, reference_map.amplitudes, rtol=1e-3)


@pytest.mark.parametrize("scale_mode", ScaleMode)
def test_compute_scale_factors_clips_overflow(
    scale_mode: ScaleMode, miller_dataseries: rs.DataSeries
) -> None:
    # Pre-clip insurance: pathological B parameters used to overflow `exp(-h^T B h)`
    # to +inf and poison the residual vector. The clip in compute_scale_factors keeps
    # the output finite for every mode that uses the exponent.
    huge_b = 1e6
    params = (1.0,) + (huge_b,) * (scale_mode.number_of_parameters - 1)
    out = compute_scale_factors(
        miller_indices=miller_dataseries.index,
        scale_parameters=params,
        scale_mode=scale_mode,
    )
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_compute_scale_factors_rejects_non_finite_parameters(
    miller_dataseries: rs.DataSeries, bad_value: float
) -> None:
    params = (1.0, bad_value, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ScalingError, match=r"`scale_parameters` must all be finite"):
        _ = compute_scale_factors(
            miller_indices=miller_dataseries.index,
            scale_parameters=params,
            scale_mode=ScaleMode.anisotropic,
        )


def test_compute_scale_factors_accepts_ndarray_miller_indices(
    miller_dataseries: rs.DataSeries,
) -> None:
    # The hot path inside `scale_maps` passes a precomputed (n, 3) ndarray instead of
    # a pd.Index. Verify the array form gives the same answer as the Index form.
    params = (1.5, 0.01, 0.02, 0.03, 0.0, 0.0, 0.0)
    miller_arr = np.asarray(list(miller_dataseries.index))

    from_index = compute_scale_factors(
        miller_indices=miller_dataseries.index,
        scale_parameters=params,
        scale_mode=ScaleMode.anisotropic,
    )
    from_array = compute_scale_factors(
        miller_indices=miller_arr,
        scale_parameters=params,
        scale_mode=ScaleMode.anisotropic,
    )
    np.testing.assert_array_equal(from_index, from_array)


def test_scale_maps_raises_when_no_finite_common_reflections(random_difference_map: Map) -> None:
    # If every common reflection is NaN we cannot fit anything; should raise a clear
    # ScalingError up front instead of letting scipy fail mysteriously.
    all_nan = random_difference_map.copy()
    all_nan.amplitudes *= np.nan

    with pytest.raises(ScalingError, match=r"No finite common reflections"):
        scale.scale_maps(
            reference_map=random_difference_map,
            map_to_scale=all_nan,
        )


def test_scale_maps_recovers_scale_with_partial_nan_inputs(random_difference_map: Map) -> None:
    # Regression test for #162: dropping reflections via NaN must not change the
    # residual vector length seen by scipy across iterations. Inject NaNs into a
    # subset of `map_to_scale` and confirm the recovered scale matches a clean fit.
    multiple = 3.0
    reference = random_difference_map.copy()
    reference.amplitudes = np.abs(reference.amplitudes) + 1.0

    scaled_clean = reference.copy()
    scaled_clean.amplitudes = scaled_clean.amplitudes / multiple

    scaled_with_nans = scaled_clean.copy()
    nan_rows = np.zeros(len(scaled_with_nans), dtype=bool)
    nan_rows[::7] = True  # ~14% of reflections marked missing
    scaled_with_nans.loc[nan_rows, scaled_with_nans.amplitudes.name] = np.nan

    recovered = scale.scale_maps(
        reference_map=reference,
        map_to_scale=scaled_with_nans,
        scale_mode=ScaleMode.scalar,
        least_squares_loss="linear",
    )

    # the finite entries should be recovered to within numerical tolerance
    finite = np.isfinite(np.asarray(recovered.amplitudes, dtype=np.float64))
    np.testing.assert_allclose(
        np.asarray(recovered.amplitudes, dtype=np.float64)[finite],
        np.asarray(reference.amplitudes, dtype=np.float64)[finite],
        rtol=1e-4,
    )


def test_scale_maps_raises_on_non_finite_initial_c(random_difference_map: Map) -> None:
    zeroed = random_difference_map.copy()
    zeroed.amplitudes *= 0.0
    with pytest.raises(ScalingError, match=r"`initial_c` is .*: either not finite or negative"):
        scale.scale_maps(
            reference_map=random_difference_map,
            map_to_scale=zeroed,
        )


def test_scale_maps_raises_on_negative_initial_c(random_difference_map: Map) -> None:
    # Negate `map_to_scale` amplitudes so `initial_c` is negative.
    negated = random_difference_map.copy()
    negated.amplitudes = -np.abs(negated.amplitudes) - 1.0
    reference = random_difference_map.copy()
    reference.amplitudes = np.abs(reference.amplitudes) + 1.0

    with pytest.raises(ScalingError, match=r"`initial_c` is .*: either not finite or negative"):
        scale.scale_maps(
            reference_map=reference,
            map_to_scale=negated,
        )


@pytest.mark.parametrize("multiple", [0.6, 1.0, 2.3])
def test_scalar_scale_reciprocal_vs_real_space(random_difference_map: Map, multiple: float) -> None:
    map_sampling = 3

    real_space = random_difference_map.to_3d_numpy_map(map_sampling=map_sampling)

    # #133 right now there is an issue where the round trip to and from a numpy map rescales the
    # map values - do one round trip first and use THAT rescaled value as a starting point
    m1 = Map.from_3d_numpy_map(
        real_space,
        spacegroup=random_difference_map.spacegroup,
        cell=random_difference_map.cell,
        high_resolution_limit=random_difference_map.resolution_limits[1],
    )

    different_real_space = multiple * real_space.copy()
    m2 = Map.from_3d_numpy_map(
        different_real_space,
        spacegroup=random_difference_map.spacegroup,
        cell=random_difference_map.cell,
        high_resolution_limit=random_difference_map.resolution_limits[1],
    )

    # confirm reciprocal space scalar scaling recovers `multiple`
    rescaled_m2 = scale.scale_maps(
        reference_map=m1,
        map_to_scale=m2,
        weight_using_uncertainties=False,
        scale_mode=ScaleMode.scalar,
        least_squares_loss="linear",
    )

    m1.canonicalize_amplitudes()
    rescaled_m2.canonicalize_amplitudes()

    np.testing.assert_allclose(m1.amplitudes, rescaled_m2.amplitudes, rtol=0.01, atol=0.01)
    np.testing.assert_allclose(m1.phases, rescaled_m2.phases, rtol=0.01, atol=0.01)
