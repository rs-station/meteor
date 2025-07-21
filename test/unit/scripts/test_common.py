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
    DiffmapArgParser,
    DiffMapSet,
    WeightMode,
    kweight_diffmap_according_to_mode,
)
from meteor.utils import ResolutionCutOverlapError


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
        calculated=random_difference_map.copy() * 2.0,
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
