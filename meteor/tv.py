from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence, overload

import numpy as np
from skimage.restoration import denoise_tv_chambolle

from .rsmap import Map
from .settings import (
    MAP_SAMPLING,
    TV_LAMBDA_RANGE,
    TV_MAX_NUM_ITER,
    TV_STOP_TOLERANCE,
)
from .utils import (
    numpy_array_to_map,
)
from .validate import ScalarMaximizer, negentropy


@dataclass
class TvDenoiseResult:
    optimal_lambda: float
    optimal_negentropy: float
    map_sampling_used_for_tv: float
    lambdas_scanned: set[float] = field(default_factory=set)


def _tv_denoise_array(*, map_as_array: np.ndarray, weight: float) -> np.ndarray:
    """Closure convienence function to generate more readable code."""
    return denoise_tv_chambolle(
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
    lambda_values_to_scan: Sequence[float] | np.ndarray | None = None,
) -> Map: ...


@overload
def tv_denoise_difference_map(
    difference_map: Map,
    *,
    full_output: Literal[True],
    lambda_values_to_scan: Sequence[float] | np.ndarray | None = None,
) -> tuple[Map, TvDenoiseResult]: ...


def tv_denoise_difference_map(
    difference_map: Map,
    *,
    full_output: bool = False,
    lambda_values_to_scan: Sequence[float] | np.ndarray | None = None,
) -> Map | tuple[Map, TvDenoiseResult]:
    """Single-pass TV denoising of a difference map.

    Automatically selects the optimal level of regularization (the TV lambda parameter) by
    maximizing the negentropy of the denoised map. Two modes can be used to dictate which
    candidate values of lambda are assessed:

      1. By default (`lambda_values_to_scan=None`), the golden-section search algorithm selects
         a lambda value according to the bounds and convergence criteria set in meteor.settings.
      2. Alternatively, an explicit list of lambda values to assess can be provided using
        `lambda_values_to_scan`.

    Parameters
    ----------
    difference_map : Map
        The input dataset containing the difference map coefficients (amplitude and phase)
        that will be used to compute the difference map.
    full_output : bool, optional
        If `True`, the function returns both the denoised map coefficients and a `TvDenoiseResult`
         object containing the optimal lambda and the associated negentropy. If `False`, only
         the denoised map coefficients are returned. Default is `False`.
    lambda_values_to_scan : Sequence[float] | None, optional
        A sequence of lambda values to explicitly scan for determining the optimal value. If
        `None`, the function uses the golden-section search method to determine the optimal
        lambda. Default is `None`.

    Returns
    -------
    Map | tuple[Map, TvDenoiseResult]
        If `full_output` is `False`, returns a `Map`, the denoised map coefficients.
        If `full_output` is `True`, returns a tuple containing:
        - `Map`: The denoised map coefficients.
        - `TvDenoiseResult`: An object w/ the optimal lambda and the corresponding negentropy.

    Raises
    ------
    AssertionError
        If the golden-section search fails to find an optimal lambda.

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
    >>> print(f"Optimal Lambda: {result.optimal_lambda}, Negentropy: {result.optimal_negentropy}")
    """
    realspace_map = difference_map.to_ccp4_map(map_sampling=MAP_SAMPLING)
    realspace_map_array = np.array(realspace_map.grid)

    def negentropy_objective(tv_lambda: float):
        denoised_map = _tv_denoise_array(map_as_array=realspace_map_array, weight=tv_lambda)
        return negentropy(denoised_map)

    maximizer = ScalarMaximizer(objective=negentropy_objective)
    if lambda_values_to_scan is not None:
        maximizer.optimize_over_explicit_values(arguments_to_scan=lambda_values_to_scan)
    else:
        maximizer.optimize_with_golden_algorithm(bracket=TV_LAMBDA_RANGE)

    # denoise using the optimized parameters and convert to an rs.DataSet
    final_realspace_map_as_array = _tv_denoise_array(
        map_as_array=realspace_map_array,
        weight=maximizer.argument_optimum,
    )
    final_realspace_map_as_ccp4 = numpy_array_to_map(
        final_realspace_map_as_array,
        spacegroup=difference_map.spacegroup,
        cell=difference_map.cell,
    )

    final_map = Map.from_ccp4_map(
        ccp4_map=final_realspace_map_as_ccp4,
        high_resolution_limit=difference_map.resolution_limits[1],
    )

    # sometimes `compute_coefficients_from_map` adds reflections -- systematic absences or
    # reflections just beyond the resolution limt; remove those
    extra_indices = final_map.index.difference(difference_map.index)
    final_map.drop(extra_indices, axis=0, inplace=True)
    sym_diff = difference_map.index.symmetric_difference(final_map.index)
    if len(sym_diff) > 0:
        msg = "something went wrong, input and output coefficients do not have identical indices"
        raise IndexError(msg)

    if full_output:
        tv_result = TvDenoiseResult(
            optimal_lambda=maximizer.argument_optimum,
            optimal_negentropy=maximizer.objective_maximum,
            map_sampling_used_for_tv=MAP_SAMPLING,
            lambdas_scanned=maximizer.values_evaluated,
        )
        return final_map, tv_result

    return final_map
