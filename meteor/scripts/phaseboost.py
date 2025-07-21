"""source code for meteor.phaseboost"""

from __future__ import annotations

from typing import Any

import structlog

from meteor.iterative import IterativeTvDenoiser
from meteor.metadata import IterativeDiffmapMetadata
from meteor.rsmap import Map
from meteor.settings import (
    DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
    ITERATIVE_TV_CONVERGENCE_TOLERANCE,
    ITERATIVE_TV_MAX_ITERATIONS,
)
from meteor.tv import tv_denoise_difference_map

from .common import (
    NEGATIVE_NEGENTROPY_WARNING_MESSAGE,
    DiffmapArgParser,
    DiffMapSet,
    WeightMode,
    kweight_diffmap_according_to_mode,
)

log = structlog.get_logger()


class IterativeTvArgParser(DiffmapArgParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        it_tv_group = self.add_argument_group("iterative TV settings")
        it_tv_group.add_argument(
            "-x",
            "--tv-weights-to-scan",
            nargs="+",
            type=float,
            default=DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
            help=(
                "Choose what TV weights to evaluate at every iteration. Can be a single float."
                f"Default: {DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION}."
            ),
        )
        it_tv_group.add_argument(
            "--convergence-tolerance",
            type=float,
            default=ITERATIVE_TV_CONVERGENCE_TOLERANCE,
            help=(
                "If the average phase change drops below this value at each iteration, stop."
                f"Default: {ITERATIVE_TV_CONVERGENCE_TOLERANCE}."
            ),
        )
        it_tv_group.add_argument(
            "--max-iterations",
            type=float,
            default=ITERATIVE_TV_MAX_ITERATIONS,
            help=(
                "If the number of iterations exceeds this value, stop."
                f"Default: {ITERATIVE_TV_MAX_ITERATIONS}."
            ),
        )


def compute_meteor_phaseboost_map(  # noqa: PLR0913
    diffmap_set: DiffMapSet,
    *,
    kweight_mode: WeightMode = WeightMode.optimize,
    kweight_parameter: float | None = None,
    tv_weights_to_scan: list[float] = DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
    convergence_tolerance: float = ITERATIVE_TV_CONVERGENCE_TOLERANCE,
    max_iterations: int = ITERATIVE_TV_MAX_ITERATIONS,
) -> tuple[Map, IterativeDiffmapMetadata]:
    log.info("Launching iterative TV phase retrieval", tv_weights_to_scan=tv_weights_to_scan)
    log.info("This will take time, typically minutes...")

    diffmap_set.scale()

    denoiser = IterativeTvDenoiser(
        tv_weights_to_scan=tv_weights_to_scan,
        convergence_tolerance=convergence_tolerance,
        max_iterations=max_iterations,
        verbose=True,
    )
    diffmap_set.derivative, it_tv_metadata = denoiser(
        derivative=diffmap_set.derivative, native=diffmap_set.native
    )
    log.info("Convergence.")

    diffmap, kparameter_metadata = kweight_diffmap_according_to_mode(
        kweight_mode=kweight_mode, kweight_parameter=kweight_parameter, mapset=diffmap_set
    )

    log.info("Final real-space TV denoising pass...", method="golden-section search")
    log.info("This may take some time (up to a few minutes)...")
    final_map, final_tv_metadata = tv_denoise_difference_map(diffmap, full_output=True)

    log.info(
        "Optimal TV weight found",
        weight=f"{final_tv_metadata.optimal_parameter_value:.2e}",
        final_negentropy=round(final_tv_metadata.optimal_negentropy, 4),
    )

    combined_metadata = IterativeDiffmapMetadata(
        kparameter_metadata=kparameter_metadata,
        iterative_tv_iterations=it_tv_metadata,
        final_tv_pass=final_tv_metadata,
    )
    if combined_metadata.final_tv_pass.optimal_negentropy <= 0.0:
        log.warning(
            NEGATIVE_NEGENTROPY_WARNING_MESSAGE,
            final_negentropy=combined_metadata.final_tv_pass.optimal_negentropy,
        )

    return final_map, combined_metadata


def main(command_line_arguments: list[str] | None = None) -> None:
    parser = IterativeTvArgParser(
        description=(
            "Compute a difference map, where the phases of the derivative structure are estimated "
            "using the assumption that the resulting map should have a low total variation. Phases "
            "are estimated using a crystallographic analog of the Gerchberg-Saxton algorithm, with "
            "TV denoising as the real-space constraint.\n\n K-weighting can optionally be used to "
            "weight the algorithm input. \n\n In the terminology adopted, this script computes a "
            "`derivative` minus a `native` map, modifying the derivative phases. Native phases,"
            "typically from a model of the `native` data, are computed from a CIF/PDB model you "
            "must provide."
        )
    )
    args = parser.parse_args(command_line_arguments)
    parser.check_output_filepaths(args)
    mapset = parser.load_difference_maps(args)

    final_map, combined_metadata = compute_meteor_phaseboost_map(
        diffmap_set=mapset,
        kweight_mode=args.kweight_mode,
        kweight_parameter=args.kweight_parameter,
        tv_weights_to_scan=args.tv_weights_to_scan,
        convergence_tolerance=args.convergence_tolerance,
        max_iterations=args.max_iterations,
    )

    log.info("Writing output.", file=str(args.mtzout))
    final_map.write_mtz(args.mtzout)

    log.info("Writing metadata.", file=str(args.metadataout))
    with args.metadataout.open("w") as f:
        f.write(combined_metadata.model_dump_json(round_trip=True, indent=4))


if __name__ == "__main__":
    main()
