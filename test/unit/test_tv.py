from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meteor import tv
from meteor.rsmap import Map
from meteor.testing import map_corrcoeff
from meteor.validate import map_negentropy

DEFAULT_WEIGHTS_TO_SCAN = np.logspace(-2, 0, 25)


def test_tv_denoise_result(tv_denoise_result_source_data: dict) -> None:
    tdr_obj = tv.TvDenoiseResult(**tv_denoise_result_source_data)
    assert tv_denoise_result_source_data == asdict(tdr_obj)

    json = tdr_obj.json()
    roundtrip = tv.TvDenoiseResult.from_json(json)
    assert tv_denoise_result_source_data == asdict(roundtrip)


def test_tv_denoise_result_to_file(tv_denoise_result_source_data: dict, tmp_path: Path) -> None:
    tdr_obj = tv.TvDenoiseResult(**tv_denoise_result_source_data)
    filepath = tmp_path / "tmp.json"
    tdr_obj.to_json_file(filepath)
    roundtrip = tv.TvDenoiseResult.from_json_file(filepath)
    assert tv_denoise_result_source_data == asdict(roundtrip)


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
        assert isinstance(output[1], tv.TvDenoiseResult)
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
        result.optimal_tv_weight, best_weight, rtol=0.5, err_msg="opt weight"
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
    assert len(metadata.negentropy_at_weights) == 1

    # it seems converting to real space and back can cause a small (few %) discrepency
    assert np.isclose(actual_negentropy, metadata.optimal_negentropy, atol=0.05)
    assert np.isclose(actual_negentropy, metadata.negentropy_at_weights[0], atol=0.05)
