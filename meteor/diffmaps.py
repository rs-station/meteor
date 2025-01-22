"""library for computing difference density maps"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import reciprocalspaceship as rs

from .rsmap import Map, assert_is_map
from .settings import (
    DEFAULT_KPARAMS_TO_SCAN,
    MAP_SAMPLING,
)
from .utils import assert_isomorphous, filter_common_indices
from .validate import ScalarMaximizer, map_negentropy


@dataclass
class KWeightScreenResult:
    # constants for JSON format
    _scan_name = "scan"
    _weight_name = "weight"
    _negentropy_name = "negentropy"

    optimal_k_weight: float
    optimal_negentropy: float
    map_sampling_used: float
    k_weights_scanned: list[float]
    negentropy_at_weights: list[float]

    def json(self) -> dict:
        json_payload = asdict(self)
        json_payload.pop("k_weights_scanned")
        json_payload.pop("negentropy_at_weights")
        json_payload[self._scan_name] = [
            {
                self._weight_name: float(self.k_weights_scanned[idx]),
                self._negentropy_name: float(self.negentropy_at_weights[idx]),
            }
            for idx in range(len(self.k_weights_scanned))
        ]
        return json_payload

    def to_json_file(self, filename: Path) -> None:
        with filename.open("w") as f:
            json.dump(self.json(), f, indent=4)

    @classmethod
    def from_json(cls, json_payload: dict) -> KWeightScreenResult:
        try:
            data = json_payload.pop(cls._scan_name)
            json_payload["k_weights_scanned"] = [
                float(point[cls._weight_name]) for point in data
            ]
            json_payload["negentropy_at_weights"] = [
                float(point[cls._negentropy_name]) for point in data
            ]
            return cls(**json_payload)

        except Exception as exptn:
            msg = "could not load json payload; mis-formatted"
            raise ValueError(msg) from exptn

    @classmethod
    def from_json_file(cls, filename: Path) -> KWeightScreenResult:
        with filename.open("r") as f:
            json_payload = json.load(f)
        return cls.from_json(json_payload)


def compute_difference_map(
    derivative: Map, native: Map, *, check_isomorphous: bool = True
) -> Map:
    """
    Computes amplitude and phase differences between native and derivative structure factor sets.

    It converts the amplitude and phase pairs from both the native and derivative structure factor
    sets into complex numbers, computes the difference, and then converts the result back
    into amplitudes and phases.

    If uncertainty columns are provided for both native and derivative data, it also propagates the
    uncertainty of the difference in amplitudes.

    Parameters
    ----------
    derivative: Map
        the derivative amplitudes, phases, uncertainties

    native: Map
        the native amplitudes, phases, uncertainties

    check_isomorphous: bool
        perform a check to ensure the two datasets are isomorphous; recommended. Default: True.

    Returns
    -------
    diffmap: Map
        map corresponding to the complex difference (derivative - native)
    """
    assert_is_map(derivative, require_uncertainties=False)
    assert_is_map(native, require_uncertainties=False)
    if check_isomorphous:
        assert_isomorphous(derivative=derivative, native=native)

    derivative, native = filter_common_indices(derivative, native)

    delta_complex = derivative.to_structurefactor() - native.to_structurefactor()
    delta = Map.from_structurefactor(delta_complex)

    delta.cell = native.cell
    delta.spacegroup = native.spacegroup

    if derivative.has_uncertainties and native.has_uncertainties:
        prop_uncertainties = np.sqrt(
            derivative.uncertainties**2 + native.uncertainties**2
        )
        delta.set_uncertainties(prop_uncertainties)

    return delta


def compute_kweights(difference_map: Map, *, k_parameter: float) -> rs.DataSeries:
    """
    Compute weights for each structure factor based on DeltaF and its uncertainty.

    Parameters
    ----------
    difference_map: Map
        A map of structure factor differences (DeltaF).

    k_parameter: float
        A scaling factor applied to the squared `df` values in the weight calculation.

    Returns
    -------
    weights: rs.DataSeries
        A series of computed weights, where higher uncertainties and larger differences lead to
        lower weights.
    """
    assert_is_map(difference_map, require_uncertainties=True)

    inverse_weights = (
        1
        + (difference_map.uncertainties**2 / (difference_map.uncertainties**2).mean())
        + k_parameter
        * (difference_map.amplitudes**2 / (difference_map.amplitudes**2).mean())
    )
    return 1.0 / inverse_weights


def compute_kweighted_difference_map(
    derivative: Map, native: Map, *, k_parameter: float, check_isomorphous: bool = True
) -> Map:
    """
    Compute k-weighted derivative - native structure factor map.

    This function first computes the standard difference map using `compute_difference_map`.
    Then, it applies k-weighting to the amplitude differences based on the provided `k_parameter`.

    Assumes amplitudes have already been scaled prior to invoking this function.

    Parameters
    ----------
    derivative: Map
        the derivative amplitudes, phases, uncertainties

    native: Map
        the native amplitudes, phases, uncertainties

    check_isomorphous: bool
        perform a check to ensure the two datasets are isomorphous; recommended. Default: True.

    Returns
    -------
    diffmap: Map
        the k-weighted difference map
    """
    # require uncertainties at the beginning
    assert_is_map(derivative, require_uncertainties=True)
    assert_is_map(native, require_uncertainties=True)
    if check_isomorphous:
        assert_isomorphous(derivative=derivative, native=native)

    difference_map = compute_difference_map(
        derivative, native, check_isomorphous=check_isomorphous
    )
    weights = compute_kweights(difference_map, k_parameter=k_parameter)

    difference_map.amplitudes *= weights
    difference_map.uncertainties *= weights

    return difference_map


def max_negentropy_kweighted_difference_map(
    derivative: Map,
    native: Map,
    *,
    full_output: bool = True,
    k_parameter_values_to_scan: np.ndarray | Sequence[float] = DEFAULT_KPARAMS_TO_SCAN,
    check_isomorphous: bool = True,
) -> rs.DataSet:
    """
    Compute k-weighted differences between native and derivative amplitudes and phases.

    Determines an "optimal" k_parameter, between 0.0 and 1.0, that maximizes the resulting
    difference map negentropy. Assumes that scaling has already been applied to the amplitudes
    before calling this function.

    Parameters
    ----------
    derivative: Map
        the derivative amplitudes, phases, uncertainties

    native: Map
        the native amplitudes, phases, uncertainties

    k_parameter_values_to_scan : np.ndarray | Sequence[float]
        The values to scan to optimize the k-weighting parameter, by default is 0.00, 0.01 ... 1.00

    check_isomorphous: bool
        perform a check to ensure the two datasets are isomorphous; recommended. Default: True.

    Returns
    -------
    kweighted_dataset: rs.DataSet
        dataset with added columns

    opt_k_parameter: float
        optimized k-weighting parameter
    """
    assert_is_map(derivative, require_uncertainties=True)
    assert_is_map(native, require_uncertainties=True)
    if check_isomorphous:
        assert_isomorphous(derivative=derivative, native=native)

    def negentropy_objective(k_parameter: float) -> float:
        kweighted_map = compute_kweighted_difference_map(
            derivative,
            native,
            k_parameter=k_parameter,
            check_isomorphous=check_isomorphous,
        )
        return map_negentropy(kweighted_map)

    maximizer = ScalarMaximizer(objective=negentropy_objective)
    maximizer.optimize_over_explicit_values(
        arguments_to_scan=k_parameter_values_to_scan
    )
    opt_k_parameter = float(maximizer.argument_optimum)

    kweighted_dataset = compute_kweighted_difference_map(
        derivative,
        native,
        k_parameter=opt_k_parameter,
        check_isomorphous=check_isomorphous,
    )

    if full_output:
        kweight_screen_result = KWeightScreenResult(
            optimal_k_weight=float(maximizer.argument_optimum),
            optimal_negentropy=float(maximizer.objective_maximum),
            map_sampling_used=MAP_SAMPLING,
            k_weights_scanned=maximizer.values_evaluated,
            negentropy_at_weights=maximizer.objective_at_values,
        )
        return kweighted_dataset, opt_k_parameter, kweight_screen_result

    return kweighted_dataset, opt_k_parameter
