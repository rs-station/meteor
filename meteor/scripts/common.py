"""shared code for the CLI"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import numpy as np
import reciprocalspaceship as rs
import structlog

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.metadata import KparameterScanMetadata
from meteor.mtzio import find_observed_amplitude_column, find_observed_uncertainty_column
from meteor.rsmap import Map
from meteor.scale import scale_maps
from meteor.settings import (
    COMPUTED_MAP_RESOLUTION_LIMIT,
    KWEIGHT_PARAMETER_DEFAULT,
)
from meteor.sfcalc import structure_file_to_calculated_map
from meteor.utils import cut_resolution

log = structlog.get_logger()

INFER_COLUMN_NAME: str = "infer"
PHASE_COLUMN_NAME: str = "PHI"
DEFAULT_OUTPUT_MTZ: Path = Path("meteor_difference_map.mtz")
DEFAULT_OUTPUT_METADATA_FILE: Path = Path("meteor_metadata.json")
NEGATIVE_NEGENTROPY_WARNING_MESSAGE: str = """\
Negentropy estimate is negative, but by definition negentropy is a positive value. This indicates
the map is very close to Gaussian and the true negentropy is close to zero. Your map may have too
little non-Gaussian signal for `meteor` to optimize."""


class InvalidWeightModeError(ValueError): ...


class WeightMode(StrEnum):
    optimize = auto()
    fixed = auto()
    none = auto()


@dataclass
class DiffMapSet:
    native: Map
    derivative: Map
    calculated: Map

    def scale(self, *, weight_using_uncertainties: bool = True) -> None:
        self.native = scale_maps(
            reference_map=self.calculated,
            map_to_scale=self.native,
            weight_using_uncertainties=weight_using_uncertainties,
        )
        log.info(
            "scaling: native --> calculated",
            weight_using_uncertainties=weight_using_uncertainties,
        )

        self.derivative = scale_maps(
            reference_map=self.calculated,
            map_to_scale=self.derivative,
            weight_using_uncertainties=weight_using_uncertainties,
        )
        log.info(
            "scaling: derivative --> calculated",
            weight_using_uncertainties=weight_using_uncertainties,
        )


class DiffmapArgParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        required_group = self.add_argument_group("required")
        required_group.add_argument(
            "derivative_mtz",
            type=Path,
            help="Path to MTZ containing the `derivative` data; positional arg (order matters).",
        )
        required_group.add_argument(
            "native_mtz",
            type=Path,
            help="Path to MTZ containing the `native` data; positional arg (order matters)",
        )
        required_group.add_argument(
            "-s",
            "--structure",
            type=Path,
            required=True,
            help="Specify CIF or PDB file path, for phases (usually a native model)",
        )

        labels_group = self.add_argument_group("mtz column labels (input)")
        labels_group.add_argument(
            "-da",
            "--derivative-amplitude-column",
            type=str,
            default=INFER_COLUMN_NAME,
            help="specify the MTZ column for the amplitudes; will try to guess if not provided",
        )
        labels_group.add_argument(
            "-du",
            "--derivative-uncertainty-column",
            type=str,
            default=INFER_COLUMN_NAME,
            help="specify the MTZ column for the uncertainties; will try to guess if not provided",
        )
        labels_group.add_argument(
            "-na",
            "--native-amplitude-column",
            type=str,
            default=INFER_COLUMN_NAME,
            help="specify the MTZ column for the amplitudes; will try to guess if not provided",
        )
        labels_group.add_argument(
            "-nu",
            "--native-uncertainty-column",
            type=str,
            default=INFER_COLUMN_NAME,
            help="specify the MTZ column for the uncertainties; will try to guess if not provided",
        )

        input_data_group = self.add_argument_group("input data specifics")
        input_data_group.add_argument(
            "--highres",
            type=float,
            default=0.0,
            help="Cut at high resolution. Default: no resolution cut beyond the data.",
        )
        input_data_group.add_argument(
            "--lowres",
            type=float,
            default=np.inf,
            help="Cut at low resolution. Default: no resolution cut beyond the data.",
        )

        output_group = self.add_argument_group("output")
        output_group.add_argument(
            "-o",
            "--mtzout",
            type=Path,
            default=DEFAULT_OUTPUT_MTZ,
            help=f"Specify output MTZ file path. Default: {DEFAULT_OUTPUT_MTZ}.",
        )
        output_group.add_argument(
            "-m",
            "--metadataout",
            type=Path,
            default=DEFAULT_OUTPUT_METADATA_FILE,
            help=f"Specify output metadata file path. Default: {DEFAULT_OUTPUT_METADATA_FILE}.",
        )

        kweight_group = self.add_argument_group("k weighting settings")
        kweight_group.add_argument(
            "-k",
            "--kweight-mode",
            type=WeightMode,
            default=WeightMode.optimize,
            choices=list(WeightMode),
            help="How to pick the k-parameter. Optimize means max negentropy. Default: `optimize`.",
        )
        kweight_group.add_argument(
            "-w",
            "--kweight-parameter",
            type=float,
            default=KWEIGHT_PARAMETER_DEFAULT,
            help=(
                f"If `--kweight-mode {WeightMode.fixed}`, set the kweight-parameter to this value. "
                f"Default: {KWEIGHT_PARAMETER_DEFAULT}."
            ),
        )

    @staticmethod
    def check_output_filepaths(args: argparse.Namespace) -> None:
        for filename in [args.mtzout, args.metadataout]:
            if filename.exists():
                msg = f"file: {filename} already exists, refusing to overwrite"
                raise OSError(msg)

    @staticmethod
    def _construct_map(  # noqa: PLR0913
        *,
        name: str,
        mtz_file: Path,
        calculated_map_phases: rs.DataSeries,
        amplitude_column: str,
        uncertainty_column: str,
        high_resolution_limit: float | None = None,
        low_resolution_limit: float | None = None,
    ) -> Map:
        log.info(
            "Reading structure factors...",
            file=str(mtz_file),
            map=name,
        )

        mtz = rs.read_mtz(str(mtz_file))

        if PHASE_COLUMN_NAME in mtz.columns:
            log.warning(
                "phase column already in MTZ; overwriting with computed data",
                file=str(mtz_file),
                column=PHASE_COLUMN_NAME,
            )
        mtz[PHASE_COLUMN_NAME] = calculated_map_phases

        found_amplitude_column = (
            find_observed_amplitude_column(mtz.columns)
            if amplitude_column is INFER_COLUMN_NAME
            else amplitude_column
        )
        if found_amplitude_column not in mtz.columns:
            msg = f"requested {found_amplitude_column} not in {mtz_file} columns, found: {mtz.columns}"
            raise KeyError(msg)
        log.info("  amplitudes", sought=amplitude_column, found=found_amplitude_column)

        found_uncertainty_column = (
            find_observed_uncertainty_column(mtz.columns)
            if uncertainty_column is INFER_COLUMN_NAME
            else uncertainty_column
        )
        if found_uncertainty_column not in mtz.columns:
            msg = f"requested {found_uncertainty_column} not in {mtz_file} columns, found: {mtz.columns}"
            raise KeyError(msg)
        log.info("  uncertainties", sought=uncertainty_column, found=found_uncertainty_column)

        mtz.dropna(axis="index", how="any", subset=found_amplitude_column, inplace=True)
        mtz = cut_resolution(
            mtz,
            low_resolution_limit=low_resolution_limit,
            high_resolution_limit=high_resolution_limit,
        )

        if len(mtz) == 0:
            msg = f"resolution cut removed all reflections for map: {name}"
            raise RuntimeError(msg)

        return Map(
            mtz,
            amplitude_column=found_amplitude_column,
            phase_column=PHASE_COLUMN_NAME,
            uncertainty_column=found_uncertainty_column,
        )

    @staticmethod
    def load_difference_maps(args: argparse.Namespace) -> DiffMapSet:
        # note: method accepts `args`, in case the passed arguments are mutable

        log.info("Loading PDB & computing FC/PHIC", file=str(args.structure))
        calculated_map = structure_file_to_calculated_map(
            args.structure, high_resolution_limit=COMPUTED_MAP_RESOLUTION_LIMIT
        )

        derivative_map = DiffmapArgParser._construct_map(
            name="derivative",
            mtz_file=args.derivative_mtz,
            calculated_map_phases=calculated_map.phases,
            amplitude_column=args.derivative_amplitude_column,
            uncertainty_column=args.derivative_uncertainty_column,
            high_resolution_limit=args.highres,
            low_resolution_limit=args.lowres,
        )

        native_map = DiffmapArgParser._construct_map(
            name="native",
            mtz_file=args.native_mtz,
            calculated_map_phases=calculated_map.phases,
            amplitude_column=args.native_amplitude_column,
            uncertainty_column=args.native_uncertainty_column,
            high_resolution_limit=args.highres,
            low_resolution_limit=args.lowres,
        )

        return DiffMapSet(
            native=native_map,
            derivative=derivative_map,
            calculated=calculated_map,
        )


def kweight_diffmap_according_to_mode(
    *, mapset: DiffMapSet, kweight_mode: WeightMode, kweight_parameter: float | None = None
) -> tuple[Map, KparameterScanMetadata | None]:
    """
    Make and k-weight a difference map using a specified `WeightMode`.

    Three modes are possible to pick the k-parameter:
      * `WeightMode.optimize`, max-negentropy value will and picked, this may take some time
      * `WeightMode.fixed`, `kweight_parameter` is used
      * `WeightMode.none`, then no k-weighting is done (note this is NOT equivalent to
         kweight_parameter=0.0)

    Parameters
    ----------
    mapset: DiffMapSet
        The set of `derivative`, `native`, `computed` maps to use to compute the diffmap.

    kweight_mode: WeightMode
        How to set the k-parameter: {optimize, fixed, none}. See above. If `fixed`, then
        `kweight_parameter` is required.

    kweight_parameter: float | None
        If kweight_mode == WeightMode.fixed, then this must be a float that specifies the
        k-parameter to use.

    Returns
    -------
    diffmap: meteor.rsmap.Map
        The difference map, k-weighted if requested.

    kparameter_metadata: KparameterScanMetadata | None
        The information about the k_weight optimization used.
        Only really interesting if WeightMode.optimize.
    """
    log.info("Computing difference map.")

    if kweight_mode == WeightMode.optimize:
        diffmap, kparameter_metadata = max_negentropy_kweighted_difference_map(
            mapset.derivative, mapset.native
        )
        log.info("  using negentropy max.", kparameter=kparameter_metadata.optimal_parameter_value)
        if np.isnan(kparameter_metadata.optimal_parameter_value):
            msg = "determined `k-parameter` is NaN, something went wrong..."
            raise RuntimeError(msg)

    elif kweight_mode == WeightMode.fixed:
        if not isinstance(kweight_parameter, float):
            msg = f"`kweight_parameter` is type `{type(kweight_parameter)}`, must be `float`"
            raise TypeError(msg)

        diffmap = compute_kweighted_difference_map(
            mapset.derivative, mapset.native, k_parameter=kweight_parameter
        )
        kparameter_metadata = None
        log.info("  using fixed", kparameter=kweight_parameter)

    elif kweight_mode == WeightMode.none:
        diffmap = compute_difference_map(mapset.derivative, mapset.native)
        kparameter_metadata = None
        log.info(" requested no k-weighting")

    else:
        raise InvalidWeightModeError(kweight_mode)

    return diffmap, kparameter_metadata
