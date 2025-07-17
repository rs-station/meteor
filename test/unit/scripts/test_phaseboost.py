from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from meteor.metadata import EvaluatedPoint, TvScanMetadata
from meteor.rsmap import Map
from meteor.scripts import phaseboost
from meteor.scripts.common import DiffMapSet, WeightMode
from meteor.scripts.phaseboost import IterativeDiffmapMetadata, IterativeTvArgParser

TV_WEIGHTS_TO_SCAN = [0.01, 0.05]


def mock_tv_denoise_difference_map(
    diffmap: Map, *, full_output: bool, weights_to_scan: Sequence[float] | np.ndarray | None = None
) -> tuple[Map, TvScanMetadata]:
    fake_metadata = TvScanMetadata(
        initial_negentropy=0.001,
        optimal_parameter_value=0.1,
        optimal_negentropy=1.0,
        map_sampling=3,
        parameter_scan_results=[EvaluatedPoint(parameter_value=0.1, objective_value=0.1)],
    )
    return diffmap, fake_metadata


@pytest.fixture
def tv_cli_arguments(base_cli_arguments: list[str]) -> list[str]:
    new_cli_arguments = [
        "--tv-weights-to-scan",
        *[str(weight) for weight in TV_WEIGHTS_TO_SCAN],
        "--convergence-tolerance",
        "0.1",
        "--max-iterations",
        "3",
    ]
    return [*base_cli_arguments, *new_cli_arguments]


@pytest.fixture
def parsed_tv_cli_args(tv_cli_arguments: list[str]) -> argparse.Namespace:
    parser = IterativeTvArgParser()
    return parser.parse_args(tv_cli_arguments)


def test_tv_diffmap_parser(parsed_tv_cli_args: argparse.Namespace) -> None:
    assert parsed_tv_cli_args.tv_weights_to_scan == TV_WEIGHTS_TO_SCAN


@pytest.mark.parametrize("kweight_mode", list(WeightMode))
def test_compute_meteor_iteratively_phased_difference_map(
    diffmap_set: DiffMapSet,
    fixed_kparameter: float,
    kweight_mode: WeightMode,
) -> None:
    patch = mock.patch(
        "meteor.scripts.phaseboost.tv_denoise_difference_map",
        mock_tv_denoise_difference_map,
    )

    with patch:
        final_map, combined_metadata = phaseboost.compute_meteor_phaseboost_map(
            diffmap_set=diffmap_set,
            kweight_mode=kweight_mode,
            kweight_parameter=fixed_kparameter,
        )

    assert isinstance(final_map, Map)
    assert isinstance(combined_metadata, IterativeDiffmapMetadata)
    assert len(final_map > 0)


def test_main(diffmap_set: DiffMapSet, tmp_path: Path, fixed_kparameter: float) -> None:
    def mock_load_maps(self: Any, args: argparse.Namespace) -> DiffMapSet:
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
        "-x",
        *[str(weight) for weight in TV_WEIGHTS_TO_SCAN],
        "--convergence-tolerance",
        "0.1",
        "--max-iterations",
        "3",
    ]

    patch1 = mock.patch(
        "meteor.scripts.phaseboost.IterativeTvArgParser.load_difference_maps",
        mock_load_maps,
    )
    patch2 = mock.patch(
        "meteor.scripts.phaseboost.tv_denoise_difference_map",
        mock_tv_denoise_difference_map,
    )

    with patch1, patch2:
        phaseboost.main(cli_arguments)

    assert output_mtz_path.exists()
    assert output_metadata_path.exists()
