from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.rsmap import Map
from meteor.settings import MAP_SAMPLING, TV_WEIGHT_DEFAULT
from meteor.tv import TvDenoiseResult, tv_denoise_difference_map
from meteor.validate import negentropy

from .common import DiffmapArgParser, DiffMapSet, InvalidWeightModeError, WeightMode

log = structlog.get_logger()


class TvDiffmapArgParser(DiffmapArgParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument(
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
        self.add_argument(
            "-l",
            "--tv-weight",
            type=float,
            default=TV_WEIGHT_DEFAULT,
            help=(
                f"If `--tv-denoise-mode {WeightMode.fixed}`, set the TV weight parameter to this "
                f"value. Default: {TV_WEIGHT_DEFAULT}."
            ),
        )


def kweight_diffmap_according_to_mode(
    *, mapset: DiffMapSet, kweight_mode: WeightMode, kweight_parameter: float | None = None
) -> tuple[Map, float | None]:
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

    kweight_parameter: float | None
        The `kweight_parameter` used. Only really interesting if WeightMode.optimize.
    """
    log.info("Computing difference map.")

    if kweight_mode == WeightMode.optimize:
        diffmap, kweight_parameter = max_negentropy_kweighted_difference_map(
            mapset.derivative, mapset.native
        )
        log.info("  using negentropy optimized", kparameter=kweight_parameter)
        if kweight_parameter is np.nan:
            msg = "determined `k-parameter` is NaN, something went wrong..."
            raise RuntimeError(msg)

    elif kweight_mode == WeightMode.fixed:
        if not isinstance(kweight_parameter, float):
            msg = f"`kweight_parameter` is type `{type(kweight_parameter)}`, must be `float`"
            raise TypeError(msg)

        diffmap = compute_kweighted_difference_map(
            mapset.derivative, mapset.native, k_parameter=kweight_parameter
        )

        log.info("  using fixed", kparameter=kweight_parameter)

    elif kweight_mode == WeightMode.none:
        diffmap = compute_difference_map(mapset.derivative, mapset.native)
        kweight_parameter = None
        log.info(" requested no k-weighting")

    else:
        raise InvalidWeightModeError(kweight_mode)

    return diffmap, kweight_parameter


def denoise_diffmap_according_to_mode(
    *,
    diffmap: Map,
    tv_denoise_mode: WeightMode,
    tv_weight: float | None = None,
) -> tuple[Map, TvDenoiseResult]:
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

    metadata: meteor.tv.TvDenoiseResult
        Information regarding the denoising process.
    """
    if tv_denoise_mode == WeightMode.optimize:
        log.info(
            "Searching for max-negentropy TV denoising weight", method="golden ration optimization"
        )
        log.info("This may take some time...")

        final_map, metadata = tv_denoise_difference_map(diffmap, full_output=True)

        log.info(
            "Optimal TV weight found",
            weight=metadata.optimal_weight,
            initial_negentropy=f"{metadata.initial_negentropy:.2e}",
            final_negetropy=f"{metadata.optimal_negentropy:.2e}",
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
            final_negetropy=f"{metadata.optimal_negentropy:.2e}",
        )

    elif tv_denoise_mode == WeightMode.none:
        final_map = diffmap

        realspace_map = final_map.to_ccp4_map(map_sampling=MAP_SAMPLING)
        map_negetropy = negentropy(np.array(realspace_map.grid))
        metadata = TvDenoiseResult(
            initial_negentropy=map_negetropy,
            optimal_negentropy=map_negetropy,
            optimal_weight=0.0,
            map_sampling_used_for_tv=MAP_SAMPLING,
            weights_scanned=[0.0],
            negentropy_at_weights=[map_negetropy],
        )

        log.info("Requested no TV denoising")

    else:
        raise InvalidWeightModeError(tv_denoise_mode)

    return final_map, metadata


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

    diffmap, kparameter_used = kweight_diffmap_according_to_mode(
        kweight_mode=args.kweight_mode, kweight_parameter=args.kweight_parameter, mapset=mapset
    )
    final_map, metadata = denoise_diffmap_according_to_mode(
        tv_denoise_mode=args.tv_denoise_mode, tv_weight=args.tv_weight, diffmap=diffmap
    )

    log.info("Writing output.", file=str(args.mtzout))
    final_map.write_mtz(args.mtzout)

    log.info("Writing metadata.", file=str(args.metadataout))
    metadata.k_parameter_used = kparameter_used
    metadata.to_json_file(args.metadataout)


if __name__ == "__main__":
    main()