from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from meteor.metadata import DiffmapMetadata
from meteor.rsmap import Map
from meteor.scripts import diffmap
from meteor.scripts.common import DiffMapSet, WeightMode
from meteor.scripts.diffmap import (
    TvDiffmapArgParser,
)

TV_WEIGHT = 0.1


@pytest.fixture
def tv_cli_arguments(base_cli_arguments: list[str]) -> list[str]:
    new_cli_arguments = [
        "-tv",
        "fixed",
        "--tv-weight",
        str(TV_WEIGHT),
    ]
    return [*base_cli_arguments, *new_cli_arguments]


@pytest.fixture
def parsed_tv_cli_args(tv_cli_arguments: list[str]) -> argparse.Namespace:
    parser = TvDiffmapArgParser()
    return parser.parse_args(tv_cli_arguments)


def test_tv_diffmap_parser(parsed_tv_cli_args: argparse.Namespace) -> None:
    assert parsed_tv_cli_args.tv_denoise_mode == WeightMode.fixed
    assert parsed_tv_cli_args.tv_weight == TV_WEIGHT


@pytest.mark.parametrize("kweight_mode", list(WeightMode))
@pytest.mark.parametrize("tv_denoise_mode", list(WeightMode))
def test_compute_meteor_difference_map(
    diffmap_set: DiffMapSet,
    fixed_kparameter: float,
    kweight_mode: WeightMode,
    tv_denoise_mode: WeightMode,
) -> None:
    final_map, final_metadata = diffmap.compute_meteor_difference_map(
        diffmap_set=diffmap_set,
        kweight_mode=kweight_mode,
        tv_denoise_mode=tv_denoise_mode,
        kweight_parameter=fixed_kparameter,
        tv_weight=TV_WEIGHT,
    )
    assert isinstance(final_map, Map)
    assert isinstance(final_metadata, DiffmapMetadata)
    assert len(final_map > 0)


def test_main(diffmap_set: DiffMapSet, tmp_path: Path, fixed_kparameter: float) -> None:
    def mock_load_difference_maps(self: Any, args: argparse.Namespace) -> DiffMapSet:
        return diffmap_set

    output_mtz_path = tmp_path / "out.mtz"
    output_metadata_path = tmp_path / "metadata.csv"

    cli_arguments = [
        "fake-derivative.mtz",
        "fake-native.mtz",
        "--structure",
        "fake.pdb",
        "-o",
        str(output_mtz_path),
        "-m",
        str(output_metadata_path),
        "--kweight-mode",
        "fixed",
        "--kweight-parameter",
        str(fixed_kparameter),
        "-tv",
        "fixed",
        "-l",
        str(TV_WEIGHT),
    ]

    fxn_to_mock = "meteor.scripts.diffmap.TvDiffmapArgParser.load_difference_maps"
    with mock.patch(fxn_to_mock, mock_load_difference_maps):
        diffmap.main(cli_arguments)

    assert output_mtz_path.exists()
    assert output_metadata_path.exists()
