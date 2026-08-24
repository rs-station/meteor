# Copyright (c) 2024 Reciprocal Space Ship
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map
from meteor.scripts.common import (
    PHASE_COLUMN_NAME,
    DiffmapArgParser,
    DiffMapSet,
    WeightMode,
    kweight_diffmap_according_to_mode,
)
from meteor.utils import ResolutionCutOverlapError

NONCANONICAL_ASU_TEST_SPACEGROUP: int = 154
NONCANONICAL_ASU_TEST_CELL: tuple[float, float, float, float, float, float] = (
    83.5,
    83.5,
    90.3,
    90.0,
    90.0,
    120.0,
)


def dataset_in_canonical_asu() -> rs.DataSet:
    index = pd.MultiIndex.from_arrays(
        [[1, 2, 3, 4, 5], [1, 1, 2, 2, 3], [1, 2, 3, 4, 5]], names=("H", "K", "L")
    )
    data = {
        "F": np.array([2.0, 3.0, 1.0, 4.0, 5.0]),
        "SIGF": np.array([0.5, 0.5, 1.0, 0.2, 0.3]),
    }
    dataset = rs.DataSet(
        data,
        index=index,
        cell=NONCANONICAL_ASU_TEST_CELL,
        spacegroup=NONCANONICAL_ASU_TEST_SPACEGROUP,
    ).infer_mtz_dtypes()
    return dataset.hkl_to_asu()


def mocked_read_mtz_in_noncanonical_asu(dummy_filename: str) -> rs.DataSet:
    # the same reflections as `dataset_in_canonical_asu`, but indexed as their Friedel mates,
    # which live outside the canonical ASU -- this is what an MTZ reindexed to resolve an
    # indexing ambiguity can look like
    assert isinstance(dummy_filename, str), "read_mtz takes a string only"

    canonical_dataset = dataset_in_canonical_asu()
    friedel_mate_dataset = canonical_dataset.copy()
    friedel_mate_dataset.index = pd.MultiIndex.from_arrays(
        [-canonical_dataset.index.get_level_values(level) for level in ("H", "K", "L")],
        names=("H", "K", "L"),
    )
    return friedel_mate_dataset


def mocked_read_mtz(dummy_filename: str) -> rs.DataSet:
    # if read_mtz gets a Path, it freaks out; requires str
    assert isinstance(dummy_filename, str), "read_mtz takes a string only"

    index = pd.MultiIndex.from_arrays(
        [[1, 1, 5, 6], [1, 2, 5, 6], [1, 3, 5, 6]], names=("H", "K", "L")
    )
    data = {
        "F": np.array([2.0, 3.0, 1.0, np.nan]),
        "SIGF": np.array([0.5, 0.5, 1.0, np.nan]),
    }
    cell = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    return rs.DataSet(data, index=index, cell=cell, spacegroup=1).infer_mtz_dtypes()


def test_diffmap_set_smoke(diffmap_set: DiffMapSet) -> None:
    assert isinstance(diffmap_set, DiffMapSet)


@pytest.mark.parametrize("use_uncertainties", [False, True])
def test_diffmap_set_scale(random_difference_map: Map, use_uncertainties: bool) -> None:
    diffmap_set = DiffMapSet(
        native=random_difference_map.copy(),
        derivative=random_difference_map.copy(),
        # `Map * float` yields a `Map` at runtime, but mypy cannot see through the
        # inherited pandas `__mul__`, so the scalar-multiply result is annotated away
        calculated=random_difference_map.copy() * 2.0,  # type: ignore[arg-type]
    )

    # upon scale, both native and derivative should also become 2x bigger
    native_amps_before = diffmap_set.native["F"].to_numpy()
    derivative_amps_before = diffmap_set.native["F"].to_numpy()

    diffmap_set.scale(weight_using_uncertainties=use_uncertainties)

    assert np.all(native_amps_before * 2 == diffmap_set.native["F"].to_numpy())
    assert np.all(derivative_amps_before * 2 == diffmap_set.derivative["F"].to_numpy())


def test_diffmap_argparser_parse_args(
    base_cli_arguments: list[str], fixed_kparameter: float
) -> None:
    parser = DiffmapArgParser()
    args = parser.parse_args(base_cli_arguments)

    assert args.derivative_mtz == Path("fake-derivative.mtz")
    assert args.derivative_amplitude_column == "F"
    assert args.derivative_uncertainty_column == "SIGF"
    assert args.native_mtz == Path("fake-native.mtz")
    assert args.native_amplitude_column == "infer"
    assert args.native_uncertainty_column == "infer"
    assert args.structure == Path("fake.pdb")
    assert args.mtzout == Path("fake-output.mtz")
    assert args.metadataout == Path("fake-output-metadata.csv")
    assert args.kweight_mode == WeightMode.fixed
    assert args.kweight_parameter == fixed_kparameter
    assert args.highres == 1.5
    assert args.lowres == 6.0


def test_diffmap_argparser_check_output_filepaths(
    base_cli_arguments: list[str], tmp_path: Path
) -> None:
    parser = DiffmapArgParser()
    args = parser.parse_args(base_cli_arguments)

    # this should pass; no files on disk
    parser.check_output_filepaths(args)

    existing = tmp_path / "exists.foo"
    existing.open("a").close()

    args.mtzout = existing
    with pytest.raises(IOError, match="file: "):
        parser.check_output_filepaths(args)

    args.mtzout = Path("fine-output-filename.mtz")
    parser.check_output_filepaths(args)

    args.metadataout = existing
    with pytest.raises(IOError, match="file: "):
        parser.check_output_filepaths(args)


@mock.patch("meteor.scripts.common.rs.read_mtz", mocked_read_mtz)
@pytest.mark.parametrize("highres", [0.1, 2.0, 100.0, None])
@pytest.mark.parametrize("lowres", [0.1, 30.0, 100.0, None])
def test_contruct_map_rescuts(
    highres: float | None,
    lowres: float | None,
) -> None:
    # map phases have an extra index
    calculated_map_phases = rs.DataSeries([60.0, 181.0, -91.0, 0.0])
    index = pd.MultiIndex.from_arrays(
        [[1, 1, 5, 6], [0, 2, 5, 6], [0, 3, 5, 7]], names=("H", "K", "L")
    )
    calculated_map_phases.index = index

    amplitude_column_requested = "F"
    uncertainty_column_requested = "SIGF"

    # 1. the rescuts overlap, guarenteed no data left
    if highres and lowres and highres >= lowres:
        with pytest.raises(ResolutionCutOverlapError):
            _ = DiffmapArgParser._construct_map(
                name="fake-name",
                mtz_file=Path("function-is-mocked.mtz"),
                calculated_map_phases=calculated_map_phases,
                amplitude_column=amplitude_column_requested,
                uncertainty_column=uncertainty_column_requested,
                high_resolution_limit=highres,
                low_resolution_limit=lowres,
            )

    # 2. rescuts remove all the data
    elif (highres and (highres > 10.0)) or (lowres and (lowres < 10.0)):
        with pytest.raises(RuntimeError, match="resolution cut removed all reflections"):
            _ = DiffmapArgParser._construct_map(
                name="fake-name",
                mtz_file=Path("function-is-mocked.mtz"),
                calculated_map_phases=calculated_map_phases,
                amplitude_column=amplitude_column_requested,
                uncertainty_column=uncertainty_column_requested,
                high_resolution_limit=highres,
                low_resolution_limit=lowres,
            )

    # 3. we should have some reflections left
    else:
        constructed_map = DiffmapArgParser._construct_map(
            name="fake-name",
            mtz_file=Path("function-is-mocked.mtz"),
            calculated_map_phases=calculated_map_phases,
            amplitude_column=amplitude_column_requested,
            uncertainty_column=uncertainty_column_requested,
            high_resolution_limit=highres,
            low_resolution_limit=lowres,
        )
        assert len(constructed_map) > 0
        assert len(constructed_map) <= len(index)
        assert constructed_map.has_uncertainties


@mock.patch("meteor.scripts.common.rs.read_mtz", mocked_read_mtz)
@pytest.mark.parametrize("amplitude_column_requested", ["infer", "F", "doesnt-exist"])
@pytest.mark.parametrize("uncertainty_column_requested", ["infer", "SIGF", "doesnt-exist"])
def test_contruct_map_column_lookup(
    amplitude_column_requested: str, uncertainty_column_requested: str
) -> None:
    calculated_map_phases = rs.DataSeries([60.0, 181.0, -91.0, 0.0])
    index = pd.MultiIndex.from_arrays(
        [[1, 1, 5, 6], [0, 2, 5, 6], [0, 3, 5, 7]], names=("H", "K", "L")
    )
    calculated_map_phases.index = index
    highres = 1.0
    lowres = 10.0

    # if the user requests a column not present in the MTZ to load, throw an exception
    if (amplitude_column_requested == "doesnt-exist") or (
        uncertainty_column_requested == "doesnt-exist"
    ):
        with pytest.raises(KeyError, match="requested"):
            _ = DiffmapArgParser._construct_map(
                name="fake-name",
                mtz_file=Path("function-is-mocked.mtz"),
                calculated_map_phases=calculated_map_phases,
                amplitude_column=amplitude_column_requested,
                uncertainty_column=uncertainty_column_requested,
                high_resolution_limit=highres,
                low_resolution_limit=lowres,
            )
    else:
        constructed_map = DiffmapArgParser._construct_map(
            name="fake-name",
            mtz_file=Path("function-is-mocked.mtz"),
            calculated_map_phases=calculated_map_phases,
            amplitude_column=amplitude_column_requested,
            uncertainty_column=uncertainty_column_requested,
            high_resolution_limit=highres,
            low_resolution_limit=lowres,
        )
        assert len(constructed_map) > 0
        assert len(constructed_map) <= len(index)
        assert constructed_map.has_uncertainties


@mock.patch("meteor.scripts.common.rs.read_mtz", mocked_read_mtz_in_noncanonical_asu)
def test_construct_map_moves_noncanonical_asu_to_canonical_asu() -> None:
    # regression test: reflections are matched across the native/derivative/calculated maps by
    # Miller index. an MTZ indexed in a non-canonical ASU used to be read in as-is, so its indices
    # never lined up, and the difference map silently collapsed to the few reflections the two
    # conventions share -- surfacing much later as an opaque "Golden minimization failed" from the
    # TV weight search
    canonical_dataset = dataset_in_canonical_asu()
    calculated_map_phases = rs.DataSeries(
        np.linspace(-90.0, 90.0, len(canonical_dataset)),
        index=canonical_dataset.index,
        name=PHASE_COLUMN_NAME,
    )

    # the input is really out of the canonical ASU, otherwise this test proves nothing
    mtz_as_read = mocked_read_mtz_in_noncanonical_asu("function-is-mocked.mtz")
    assert not np.any(rs.utils.in_asu(mtz_as_read.get_hkls(), mtz_as_read.spacegroup))

    constructed_map = DiffmapArgParser._construct_map(
        name="fake-name",
        mtz_file=Path("function-is-mocked.mtz"),
        calculated_map_phases=calculated_map_phases,
        amplitude_column="F",
        uncertainty_column="SIGF",
    )

    # every reflection should survive, in the canonical ASU, with its calculated phase attached
    assert len(constructed_map) == len(canonical_dataset)
    assert np.all(rs.utils.in_asu(constructed_map.get_hkls(), constructed_map.spacegroup))
    assert constructed_map.index.sort_values().equals(canonical_dataset.index.sort_values())
    assert not constructed_map.phases.isna().to_numpy().any()

    sorted_map = constructed_map.sort_index()
    np.testing.assert_allclose(
        sorted_map.phases.to_numpy(),
        calculated_map_phases.sort_index().to_numpy(),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        sorted_map.amplitudes.to_numpy(),
        canonical_dataset.sort_index()["F"].to_numpy(),
        rtol=1e-5,
    )
@mock.patch("meteor.scripts.common.rs.read_mtz")
def test_construct_map_rejects_duplicate_indices_after_asu_mapping(
    read_mtz_mock: mock.Mock,
) -> None:
    canonical_dataset = dataset_in_canonical_asu()
    noncanonical_dataset = mocked_read_mtz_in_noncanonical_asu("function-is-mocked.mtz")
    mtz_with_symmetry_equivalent_indices = rs.concat(
        [canonical_dataset, noncanonical_dataset.iloc[[0]]]
    )
    assert mtz_with_symmetry_equivalent_indices.index.is_unique

    read_mtz_mock.return_value = mtz_with_symmetry_equivalent_indices
    calculated_map_phases = rs.DataSeries(
        np.zeros(len(canonical_dataset)),
        index=canonical_dataset.index,
        name=PHASE_COLUMN_NAME,
    )

    with pytest.raises(
        ValueError,
        match=r"function-is-mocked\.mtz contains duplicate Miller indices",
    ):
        DiffmapArgParser._construct_map(
            name="fake-name",
            mtz_file=Path("function-is-mocked.mtz"),
            calculated_map_phases=calculated_map_phases,
            amplitude_column="F",
            uncertainty_column="SIGF",
        )



def test_load_difference_maps(random_difference_map: Map, base_cli_arguments: list[str]) -> None:
    parser = DiffmapArgParser()
    args = parser.parse_args(base_cli_arguments)

    def return_a_map(*args: Any, **kwargs: Any) -> Map:
        return random_difference_map

    mocked_fxn_1 = "meteor.scripts.common.structure_file_to_calculated_map"
    mocked_fxn_2 = "meteor.scripts.common.DiffmapArgParser._construct_map"

    with mock.patch(mocked_fxn_1, return_a_map), mock.patch(mocked_fxn_2, return_a_map):
        mapset = DiffmapArgParser.load_difference_maps(args)
        assert isinstance(mapset.native, Map)
        assert isinstance(mapset.derivative, Map)
        assert isinstance(mapset.calculated, Map)


@pytest.mark.parametrize("mode", list(WeightMode))
def test_kweight_diffmap_according_to_mode(
    mode: WeightMode, diffmap_set: DiffMapSet, fixed_kparameter: float
) -> None:
    # ensure the two maps aren't exactly the same to prevent numerical issues
    diffmap_set.derivative.loc[0, diffmap_set.derivative._amplitude_column] += 1.0

    diffmap, _ = kweight_diffmap_according_to_mode(
        mapset=diffmap_set, kweight_mode=mode, kweight_parameter=fixed_kparameter
    )
    assert len(diffmap) > 0
    assert isinstance(diffmap, Map)

    if mode == WeightMode.fixed:
        with pytest.raises(TypeError):
            _ = kweight_diffmap_according_to_mode(
                mapset=diffmap_set, kweight_mode=mode, kweight_parameter=None
            )
