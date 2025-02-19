from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from meteor import tv
from meteor.metadata import TvScanMetadata
from meteor.rsmap import Map
from meteor.testing import map_corrcoeff
from meteor.validate import map_negentropy

DEFAULT_WEIGHTS_TO_SCAN = np.logspace(-2, 0, 25)


@pytest.mark.parametrize(
    "weights_to_scan",
    [
        None,
        [-1.0, 0.0, 1.0],
    ],
)
@pytest.mark.parametrize("full_output", [False, True])
def test_tv_denoise_map_smoke(
    weights_to_scan: None | Sequence[float],
    full_output: bool,
    random_difference_map: Map,
) -> None:
    output = tv.tv_denoise_difference_map(
        random_difference_map,
        weights_to_scan=weights_to_scan,
        full_output=full_output,
    )  # type: ignore[call-overload]
    if full_output:
        assert len(output) == 2
        assert isinstance(output[0], Map)
        assert isinstance(output[1], TvScanMetadata)
    else:
        assert isinstance(output, Map)


def test_tv_denoise_zero_weight(random_difference_map: Map) -> None:
    weight = 0.0
    output = tv.tv_denoise_difference_map(
        random_difference_map,
        weights_to_scan=[weight],
        full_output=False,
    )
    random_difference_map.canonicalize_amplitudes()
    output.canonicalize_amplitudes()
    pd.testing.assert_frame_equal(random_difference_map, output, atol=0.1, rtol=0.1)


def test_tv_denoise_nan_input(random_difference_map: Map) -> None:
    weight = 0.0
    random_difference_map.iloc[0] = np.nan
    _ = tv.tv_denoise_difference_map(
        random_difference_map,
        weights_to_scan=[weight],
        full_output=False,
    )


@pytest.mark.parametrize("weights_to_scan", [None, DEFAULT_WEIGHTS_TO_SCAN])
def test_tv_denoise_map(
    weights_to_scan: None | Sequence[float],
    noise_free_map: Map,
    noisy_map: Map,
) -> None:
    def cc_to_noise_free(test_map: Map) -> float:
        return map_corrcoeff(test_map, noise_free_map)

    # Normally, the `tv_denoise_difference_map` function only returns the best result -- since we
    # know the ground truth, work around this to test all possible results.

    best_cc: float = 0.0
    best_weight: float = 0.0

    for trial_weight in DEFAULT_WEIGHTS_TO_SCAN:
        denoised_map, result = tv.tv_denoise_difference_map(
            noisy_map,
            weights_to_scan=[
                trial_weight,
            ],
            full_output=True,
        )
        cc = cc_to_noise_free(denoised_map)
        if cc > best_cc:
            best_cc = cc
            best_weight = trial_weight

    # now run the denoising algorithm and make sure we get a result that's close
    # to the one that minimizes the RMS error to the ground truth
    denoised_map, result = tv.tv_denoise_difference_map(
        noisy_map,
        weights_to_scan=weights_to_scan,
        full_output=True,
    )

    assert cc_to_noise_free(denoised_map) > cc_to_noise_free(noisy_map)
    np.testing.assert_allclose(
        result.optimal_parameter_value, best_weight, rtol=0.5, err_msg="opt weight"
    )


def test_final_map_has_reported_negentropy(noisy_map: Map) -> None:
    # regression test: previously the written map dropped a few indices to be consistent w/input
    # this caused the negentropy values to be off

    # simulate missing reflections that will be filled
    noisy_map.drop(noisy_map.index[:512], inplace=True)

    weight = 0.01
    output_map, metadata = tv.tv_denoise_difference_map(
        noisy_map,
        weights_to_scan=[weight],
        full_output=True,
    )
    actual_negentropy = map_negentropy(output_map)
    assert len(metadata.parameter_scan_results) == 1

    # it seems converting to real space and back can cause a small (few %) discrepency
    assert np.isclose(actual_negentropy, metadata.optimal_negentropy, atol=0.05)
    assert np.isclose(
        actual_negentropy, metadata.parameter_scan_results[0].objective_value, atol=0.05
    )
