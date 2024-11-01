"""total variation denoising of maps"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, overload

import numpy as np
from skimage.restoration import denoise_tv_chambolle

from .rsmap import Map
from .settings import (
    BRACKET_FOR_GOLDEN_OPTIMIZATION,
    MAP_SAMPLING,
    TV_MAX_NUM_ITER,
    TV_STOP_TOLERANCE,
)
from .validate import ScalarMaximizer, negentropy


@dataclass
class TvDenoiseResult:
    # constants for JSON format
    _scan_name = "scan"
    _weight_name = "weight"
    _negentropy_name = "negentropy"

    initial_negentropy: float
    optimal_tv_weight: float
    optimal_negentropy: float
    map_sampling_used_for_tv: float
    tv_weights_scanned: list[float]
    negentropy_at_weights: list[float]
    k_parameter_used: float | None = None

    def json(self) -> dict:
        json_payload = asdict(self)
        json_payload.pop("tv_weights_scanned")
        json_payload.pop("negentropy_at_weights")
        json_payload[self._scan_name] = [
            {
                self._weight_name: float(self.tv_weights_scanned[idx]),
                self._negentropy_name: float(self.negentropy_at_weights[idx]),
            }
            for idx in range(len(self.tv_weights_scanned))
        ]
        return json_payload

    def to_json_file(self, filename: Path) -> None:
        with filename.open("w") as f:
            json.dump(self.json(), f, indent=4)

    @classmethod
    def from_json(cls, json_payload: dict) -> TvDenoiseResult:
        try:
            data = json_payload.pop(cls._scan_name)
            json_payload["tv_weights_scanned"] = [float(point[cls._weight_name]) for point in data]
            json_payload["negentropy_at_weights"] = [
                float(point[cls._negentropy_name]) for point in data
            ]
            return cls(**json_payload)

        except Exception as exptn:
            msg = "could not load json payload; mis-formatted"
            raise ValueError(msg) from exptn

    @classmethod
    def from_json_file(cls, filename: Path) -> TvDenoiseResult:
        with filename.open("r") as f:
            json_payload = json.load(f)
        return cls.from_json(json_payload)


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
) -> tuple[Map, TvDenoiseResult]: ...


def tv_denoise_difference_map(
    difference_map: Map,
    *,
    full_output: bool = False,
    weights_to_scan: Sequence[float] | np.ndarray | None = None,
) -> Map | tuple[Map, TvDenoiseResult]:
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
        If `True`, the function returns both the denoised map coefficients and a `TvDenoiseResult`
         object containing the optimal weight and the associated negentropy. If `False`, only
         the denoised map coefficients are returned. Default is `False`.
    weights_to_scan : Sequence[float] | None, optional
        A sequence of weight values to explicitly scan for determining the optimal value. If
        `None`, the function uses the golden-section search method to determine the optimal
        weight. Default is `None`.

    Returns
    -------
    Map | tuple[Map, TvDenoiseResult]
        If `full_output` is `False`, returns a `Map`, the denoised map coefficients.
        If `full_output` is `True`, returns a tuple containing:
        - `Map`: The denoised map coefficients.
        - `TvDenoiseResult`: An object w/ the optimal weight and the corresponding negentropy.

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
    realspace_map = difference_map.to_ccp4_map(map_sampling=MAP_SAMPLING)
    realspace_map_array = np.array(realspace_map.grid)

    def negentropy_objective(tv_weight: float) -> float:
        denoised_map = _tv_denoise_array(map_as_array=realspace_map_array, weight=tv_weight)
        return negentropy(denoised_map)

    maximizer = ScalarMaximizer(objective=negentropy_objective)
    if weights_to_scan is not None:
        maximizer.optimize_over_explicit_values(arguments_to_scan=weights_to_scan)
    else:
        maximizer.optimize_with_golden_algorithm(bracket=BRACKET_FOR_GOLDEN_OPTIMIZATION)

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

    # sometimes `from_ccp4_map` adds reflections -- systematic absences or
    # reflections just beyond the resolution limt; remove those
    extra_indices = final_map.index.difference(difference_map.index)
    final_map.drop(extra_indices, inplace=True)
    sym_diff = difference_map.index.symmetric_difference(final_map.index)
    if len(sym_diff) > 0:
        msg = "something went wrong, input and output coefficients do not have identical indices"
        raise IndexError(msg)

    if full_output:
        initial_negentropy = negentropy(realspace_map_array)
        tv_result = TvDenoiseResult(
            initial_negentropy=float(initial_negentropy),
            optimal_tv_weight=float(maximizer.argument_optimum),
            optimal_negentropy=float(maximizer.objective_maximum),
            map_sampling_used_for_tv=MAP_SAMPLING,
            tv_weights_scanned=maximizer.values_evaluated,
            negentropy_at_weights=maximizer.objective_at_values,
        )
        return final_map, tv_result

    return final_map
