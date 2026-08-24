# Copyright (c) 2024 Reciprocal Space Ship
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.diffmaps import compute_difference_map
from meteor.iterative import (
    IterativeTvDenoiser,
    _assert_are_dataseries,
)
from meteor.metadata import TvIterationMetadata, TvScanMetadata
from meteor.rsmap import Map
from meteor.testing import map_corrcoeff


@pytest.fixture
def testing_denoiser() -> IterativeTvDenoiser:
    return IterativeTvDenoiser(
        tv_weights_to_scan=[0.1],
        convergence_tolerance=0.01,
        max_iterations=100,
    )


def test_assert_are_dataseries() -> None:
    ds = rs.DataSeries([1, -1, 1], index=[1, 2, 3])
    _assert_are_dataseries(ds)
    _assert_are_dataseries(ds, ds)

    with pytest.raises(TypeError):
        _assert_are_dataseries(1)  # type: ignore[arg-type]


def test_init(testing_denoiser: IterativeTvDenoiser) -> None:
    assert isinstance(testing_denoiser, IterativeTvDenoiser)


def test_tv_denoise_complex_difference_sf(
    testing_denoiser: IterativeTvDenoiser,
    random_difference_map: Map,
) -> None:
    # use a huge TV weight, make sure random noise goes down
    testing_denoiser.tv_weights_to_scan = [100.0]
    noise = random_difference_map.to_structurefactor()

    denoised_sfs, metadata = testing_denoiser._tv_denoise_complex_difference_sf(
        noise, cell=random_difference_map.cell, spacegroup=random_difference_map.spacegroup
    )

    assert isinstance(denoised_sfs, rs.DataSeries)
    assert isinstance(metadata, TvScanMetadata)

    # weak check, but makes sure something happened
    assert np.sum(np.abs(denoised_sfs)) < np.sum(np.abs(noise))


def test_iteratively_denoise_sf_amplitudes_smoke(
    testing_denoiser: IterativeTvDenoiser, random_difference_map: Map
) -> None:
    # tests for correctness below

    denoised_sfs, metadata = testing_denoiser._iteratively_denoise_sf_amplitudes(
        initial_derivative=random_difference_map.to_structurefactor(),
        native=random_difference_map.to_structurefactor() + 1.0,
        cell=random_difference_map.cell,
        spacegroup=random_difference_map.spacegroup,
    )

    assert isinstance(denoised_sfs, rs.DataSeries)
    assert np.issubdtype(denoised_sfs.dtype, np.complexfloating)

    assert isinstance(metadata, list)
    assert len(metadata) > 1
    assert isinstance(metadata[0], TvIterationMetadata)


def test_iterative_tv_halts_on_decreasing_negentropy(
    testing_denoiser: IterativeTvDenoiser,
    random_difference_map: Map,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_negentropies = [1.0, 2.0, 1.5]
    call_count = {"n": 0}

    def mock_tv_step(
        complex_difference_sf: rs.DataSeries, *, cell: object, spacegroup: object
    ) -> tuple[rs.DataSeries, TvScanMetadata]:
        idx = call_count["n"]
        call_count["n"] += 1
        if idx >= len(scripted_negentropies):
            msg = (
                f"algorithm did not halt: called {idx + 1} times, "
                f"only {len(scripted_negentropies)} negentropies scripted"
            )
            raise AssertionError(msg)
        meta = TvScanMetadata(
            initial_negentropy=0.0,
            optimal_parameter_value=0.1,
            optimal_negentropy=scripted_negentropies[idx],
            map_sampling=5.0,
            parameter_scan_results=[],
        )
        # return the input unchanged so the rest of the loop's arithmetic stays valid
        return complex_difference_sf.copy(), meta

    monkeypatch.setattr(testing_denoiser, "_tv_denoise_complex_difference_sf", mock_tv_step)
    # neither convergence nor max_iterations should be the reason we stop
    testing_denoiser.convergence_tolerance = -1.0
    testing_denoiser.max_iterations = 1000

    sfs = random_difference_map.to_structurefactor()
    _, metadata = testing_denoiser._iteratively_denoise_sf_amplitudes(
        initial_derivative=sfs,
        native=sfs + 1.0,
        cell=random_difference_map.cell,
        spacegroup=random_difference_map.spacegroup,
    )

    # the third call is the one with decreasing negentropy and should trigger break
    # before the iteration body runs, so only two iterations get recorded in metadata
    assert call_count["n"] == 3
    assert len(metadata) == 2
    assert [m.negentropy_after_tv for m in metadata] == [1.0, 2.0]


def test_iterative_tv_denoiser_different_indices(
    noise_free_map: Map, very_noisy_map: Map, testing_denoiser: IterativeTvDenoiser
) -> None:
    # regression test to make sure we can accept maps with different indices
    labels = pd.MultiIndex.from_arrays(
        [
            (1, 2),
        ]
        * 3,
        names=("H", "K", "L"),
    )

    n = len(very_noisy_map)
    very_noisy_map.drop(labels, inplace=True)
    assert len(very_noisy_map) == n - 2

    denoised_map, metadata = testing_denoiser(derivative=very_noisy_map, native=noise_free_map)
    assert isinstance(metadata, list)
    assert isinstance(denoised_map, Map)


def test_iterative_tv_denoise_retains_scale(
    noise_free_map: Map, very_noisy_map: Map, testing_denoiser: IterativeTvDenoiser
) -> None:
    vanilla_difference_map = compute_difference_map(very_noisy_map, noise_free_map)
    denoised_map, _ = testing_denoiser(derivative=very_noisy_map, native=noise_free_map)
    mssq_vanilla = np.mean(np.square(vanilla_difference_map.amplitudes))
    mssq_denoised = np.mean(np.square(denoised_map.amplitudes))

    # the itTV method should natively retain the sum{squares} scale to 20% or better
    np.testing.assert_allclose(mssq_denoised, mssq_vanilla, rtol=0.2)


def test_iterative_tv_denoiser(
    noise_free_map: Map, noisy_map: Map, testing_denoiser: IterativeTvDenoiser
) -> None:
    # the test case is the denoising of a difference: between a noisy map and its noise-free origin
    # such a diffmap is ideally totally flat, so should have very low TV

    denoised_map, metadata = testing_denoiser(derivative=noisy_map, native=noise_free_map)

    # make sure metadata exists
    assert isinstance(metadata, list)

    # test correctness by comparing denoised dataset to noise-free
    noisy_cc = map_corrcoeff(noisy_map, noise_free_map)
    denoised_cc = map_corrcoeff(denoised_map, noise_free_map)

    # insist on improvement
    assert denoised_cc > noisy_cc

    # insist that the negentropy and phase change decrease (or stay approx same) at every iteration
    metadata_as_df = pd.DataFrame([row.model_dump() for row in metadata])
    for expected_col in ["iteration", "tv_weight", "negentropy_after_tv", "average_phase_change"]:
        assert expected_col in metadata_as_df.columns

    negentropy_change = metadata_as_df["negentropy_after_tv"].diff().to_numpy()
    assert (negentropy_change[1:-1] >= -0.05).all()

    phase_change_change = metadata_as_df["average_phase_change"].diff().to_numpy()
    assert (phase_change_change[1:-1] <= 0.1).all()
