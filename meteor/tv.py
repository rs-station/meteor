"""total variation denoising of maps"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import structlog
from skimage.restoration import denoise_tv_chambolle

from .metadata import TvScanMetadata
from .rsmap import Map
from .settings import (
    BRACKET_FOR_GOLDEN_OPTIMIZATION,
    MAP_SAMPLING,
    TV_MAX_NUM_ITER,
    TV_MAX_WEIGHT_EXPECTED,
    TV_STOP_TOLERANCE,
)
from .validate import ScalarMaximizer, negentropy

log = structlog.get_logger()


def _tv_denoise_array(*, map_as_array: np.ndarray, weight: float) -> np.ndarray:
    """Closure convienence function to generate more readable code."""
    if weight <= 0.0:
        return map_as_array
    return denoise_tv_chambolle(  # type: ignore[no-untyped-call]
        map_as_array,
        weight=weight,
        eps=TV_STOP_TOLERANCE,
        max_num_iter=TV_MAX_NUM_ITER,
    )


@overload
def tv_denoise_difference_map(
    difference_map: Map,
    *,
    full_output: Literal[False],
    weights_to_scan: Sequence[float] | np.ndarray | None = None,
) -> Map: ...


@overload
def tv_denoise_difference_map(
    difference_map: Map,
    *,
    full_output: Literal[True],
    weights_to_scan: Sequence[float] | np.ndarray | None = None,
) -> tuple[Map, TvScanMetadata]: ...


def tv_denoise_difference_map(
    difference_map: Map,
    *,
    full_output: bool = False,
    weights_to_scan: Sequence[float] | np.ndarray | None = None,
) -> Map | tuple[Map, TvScanMetadata]:
    """Single-pass TV denoising of a difference map.

    Automatically selects the optimal level of regularization (the TV weight, aka lambda) by
    maximizing the negentropy of the denoised map. Two modes can be used to dictate which
    candidate values of weights are assessed:

      1. By default (`weights_to_scan=None`), the golden-section search algorithm selects
         a weights value according to the bounds and convergence criteria set in meteor.settings.
      2. Alternatively, an explicit list of weights values to assess can be provided using
        `weights_to_scan`.

    Parameters
    ----------
    difference_map : Map
        The input dataset containing the difference map coefficients (amplitude and phase)
        that will be used to compute the difference map.

    full_output : bool, optional
        If `True`, the function returns both the denoised map coefficients and a `TvScanMetadata`
         object containing the optimal weight and the associated negentropy. If `False`, only
         the denoised map coefficients are returned. Default is `False`.

    weights_to_scan : Sequence[float] | None, optional
        A sequence of weight values to explicitly scan for determining the optimal value. If
        `None`, the function uses the golden-section search method to determine the optimal
        weight. Default is `None`.

    Returns
    -------
    Map | tuple[Map, TvScanMetadata]
        If `full_output` is `False`, returns a `Map`, the denoised map coefficients.
        If `full_output` is `True`, returns a tuple containing:
        - `Map`: The denoised map coefficients.
        - `TvScanMetadata`: An object w/ the optimal weight and the corresponding negentropy.

    Raises
    ------
    AssertionError
        If the golden-section search fails to find an optimal weight.

    Notes
    -----
    - The function is designed to maximize the negentropy of the denoised map, which is a
      measure of the map's "randomness."
      Higher negentropy generally corresponds to a more informative and less noisy map.
    - The golden-section search is a robust method for optimizing unimodal functions,
      particularly suited for scenarios where an explicit list of candidate values is not provided.

    Example
    -------
    >>> coefficients = Map.read_mtz("./path/to/difference_map.mtz", ...)  # load dataset
    >>> denoised_map, result = tv_denoise_difference_map(coefficients, full_output=True)
    >>> print(f"Optimal: {result.optimal_tv_weight}, Negentropy: {result.optimal_negentropy}")
    """
    realspace_map_array = difference_map.to_3d_numpy_map(map_sampling=MAP_SAMPLING)

    def negentropy_objective(tv_weight: float) -> float:
        denoised_map = _tv_denoise_array(map_as_array=realspace_map_array, weight=tv_weight)
        return negentropy(denoised_map)

    maximizer = ScalarMaximizer(objective=negentropy_objective)
    if weights_to_scan is not None:
        maximizer.optimize_over_explicit_values(arguments_to_scan=weights_to_scan)
    else:
        maximizer.optimize_with_golden_algorithm(bracket=BRACKET_FOR_GOLDEN_OPTIMIZATION)

    if maximizer.argument_optimum > TV_MAX_WEIGHT_EXPECTED:
        log.warning(
            "TV regularization weight much larger than expected, something probably went wrong",
            weight=f"{maximizer.argument_optimum:.2f}",
            limit=TV_MAX_WEIGHT_EXPECTED,
        )

    # denoise using the optimized parameters and convert to an rs.DataSet
    final_realspace_map_as_array = _tv_denoise_array(
        map_as_array=realspace_map_array,
        weight=maximizer.argument_optimum,
    )
    final_map = Map.from_3d_numpy_map(
        final_realspace_map_as_array,
        spacegroup=difference_map.spacegroup,
        cell=difference_map.cell,
        high_resolution_limit=difference_map.resolution_limits[1],
    )

    # propogate uncertainties
    if difference_map.has_uncertainties:
        final_map.set_uncertainties(difference_map.uncertainties)

    if full_output:
        initial_negentropy = negentropy(realspace_map_array)
        tv_result = TvScanMetadata(
            initial_negentropy=float(initial_negentropy),
            optimal_parameter_value=float(maximizer.argument_optimum),
            optimal_negentropy=float(maximizer.objective_maximum),
            map_sampling=MAP_SAMPLING,
            parameter_scan_results=maximizer.parameter_scan_results,
        )
        return final_map, tv_result

    return final_map
