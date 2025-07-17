"""source code for meteor.diffmap"""

from __future__ import annotations

from typing import Any

import structlog

from meteor.metadata import DiffmapMetadata, EvaluatedPoint, TvScanMetadata
from meteor.rsmap import Map
from meteor.settings import MAP_SAMPLING, TV_WEIGHT_DEFAULT
from meteor.tv import tv_denoise_difference_map
from meteor.validate import map_negentropy

from .common import (
    NEGATIVE_NEGENTROPY_WARNING_MESSAGE,
    DiffmapArgParser,
    DiffMapSet,
    InvalidWeightModeError,
    WeightMode,
    kweight_diffmap_according_to_mode,
)

log = structlog.get_logger()


class TvDiffmapArgParser(DiffmapArgParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        tv_group = self.add_argument_group("TV denoising settings")
        tv_group.add_argument(
            "-tv",
            "--tv-denoise-mode",
            type=WeightMode,
            default=WeightMode.optimize,
            choices=list(WeightMode),
            help=(
                "How to find a TV regularization weight (lambda). Optimize means pick maximum "
                "negentropy. Default: `optimize`."
            ),
        )
        tv_group.add_argument(
            "-l",
            "--tv-weight",
            type=float,
            default=TV_WEIGHT_DEFAULT,
            help=(
                f"If `--tv-denoise-mode {WeightMode.fixed}`, set the TV weight parameter to this "
                f"value. Default: {TV_WEIGHT_DEFAULT}."
            ),
        )


def denoise_diffmap_according_to_mode(
    *,
    diffmap: Map,
    tv_denoise_mode: WeightMode,
    tv_weight: float | None = None,
) -> tuple[Map, TvScanMetadata]:
    """
    Denoise a difference map `diffmap` using a specified `WeightMode`.

    Three modes are possible:
      * `WeightMode.optimize`, max-negentropy value will and picked, this may take some time
      * `WeightMode.fixed`, `tv_weight` is used
      * `WeightMode.none`, then no TV denoising is done (equivalent to weight = 0.0)

    Parameters
    ----------
    diffmap: meteor.rsmap.Map
        The map to denoise.

    tv_denoise_mode: WeightMode
        How to set the TV weight parameter: {optimize, fixed, none}. See above. If `fixed`, the
        `tv_weight` parameter is required.

    tv_weight: float | None
        If tv_denoise_mode == WeightMode.fixed, then this must be a float that specifies the weight
        to use.

    Returns
    -------
    final_map: meteor.rsmap.Map
        The difference map, denoised if requested

    metadata: meteor.metadata.TvScanMetadata
        Information regarding the denoising process.
    """
    if tv_denoise_mode == WeightMode.optimize:
        log.info("Searching for max-negentropy TV denoising weight", method="golden-section search")
        log.info("This may take some time...")

        final_map, metadata = tv_denoise_difference_map(diffmap, full_output=True)

        log.info(
            "Optimal TV weight found",
            weight=f"{metadata.optimal_parameter_value:.2e}",
            initial_negentropy=f"{metadata.initial_negentropy:.2e}",
            final_negentropy=f"{metadata.optimal_negentropy:.2e}",
        )

    elif tv_denoise_mode == WeightMode.fixed:
        if not isinstance(tv_weight, float):
            msg = f"`tv_weight` is type `{type(tv_weight)}`, must be `float`"
            raise TypeError(msg)

        log.info("TV denoising with fixed weight", weight=tv_weight)
        final_map, metadata = tv_denoise_difference_map(
            diffmap, full_output=True, weights_to_scan=[tv_weight]
        )

        log.info(
            "Map TV-denoised with fixed weight",
            weight=tv_weight,
            initial_negentropy=f"{metadata.initial_negentropy:.2e}",
            final_negentropy=f"{metadata.optimal_negentropy:.2e}",
        )

    elif tv_denoise_mode == WeightMode.none:
        final_map = diffmap
        final_negentropy = map_negentropy(final_map)
        metadata = TvScanMetadata(
            initial_negentropy=final_negentropy,
            optimal_negentropy=final_negentropy,
            optimal_parameter_value=0.0,
            map_sampling=MAP_SAMPLING,
            parameter_scan_results=[
                EvaluatedPoint(parameter_value=0.0, objective_value=final_negentropy)
            ],
        )

        log.info("Requested no TV denoising")

    else:
        raise InvalidWeightModeError(tv_denoise_mode)

    return final_map, metadata


def compute_meteor_difference_map(
    diffmap_set: DiffMapSet,
    *,
    kweight_mode: WeightMode = WeightMode.optimize,
    tv_denoise_mode: WeightMode = WeightMode.optimize,
    kweight_parameter: float | None = None,
    tv_weight: float | None = None,
) -> tuple[Map, DiffmapMetadata]:
    diffmap_set.scale()

    diffmap, kparameter_metadata = kweight_diffmap_according_to_mode(
        kweight_mode=kweight_mode,
        kweight_parameter=kweight_parameter,
        mapset=diffmap_set,
    )
    final_map, tv_metadata = denoise_diffmap_according_to_mode(
        tv_denoise_mode=tv_denoise_mode, tv_weight=tv_weight, diffmap=diffmap
    )
    aggregate_metadata = DiffmapMetadata(
        k_parameter_optimization=kparameter_metadata, tv_weight_optmization=tv_metadata
    )

    if aggregate_metadata.tv_weight_optmization.optimal_negentropy <= 0.0:
        log.warning(
            NEGATIVE_NEGENTROPY_WARNING_MESSAGE,
            final_negentropy=aggregate_metadata.tv_weight_optmization.optimal_negentropy,
        )

    return final_map, aggregate_metadata


def main(command_line_arguments: list[str] | None = None) -> None:
    parser = TvDiffmapArgParser(
        description=(
            "Compute an isomorphous difference map, optionally applying k-weighting and/or "
            "TV-denoising if desired. \n\n In the terminology adopted, this script computes a "
            "`derivative` minus a `native` map, using a constant phase approximation. Phases, "
            "typically from a model of the `native` data, are computed from a CIF/PDB model you "
            "must provide."
        )
    )
    args = parser.parse_args(command_line_arguments)
    parser.check_output_filepaths(args)
    mapset = parser.load_difference_maps(args)

    final_map, aggregate_metadata = compute_meteor_difference_map(
        diffmap_set=mapset,
        kweight_mode=args.kweight_mode,
        tv_denoise_mode=args.tv_denoise_mode,
        kweight_parameter=args.kweight_parameter,
        tv_weight=args.tv_weight,
    )

    log.info("Writing output.", file=str(args.mtzout))
    final_map.write_mtz(args.mtzout)

    log.info("Writing metadata.", file=str(args.metadataout))
    with args.metadataout.open("w") as f:
        f.write(aggregate_metadata.model_dump_json(round_trip=True, indent=4))


if __name__ == "__main__":
    main()
