# Copyright (c) 2024 Reciprocal Space Ship
# SPDX-License-Identifier: MIT

"""library for computing difference density maps"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import reciprocalspaceship as rs

from .metadata import KparameterScanMetadata
from .rsmap import Map, assert_is_map
from .settings import DEFAULT_KPARAMS_TO_SCAN
from .utils import assert_isomorphous, filter_common_indices
from .validate import ScalarMaximizer, map_negentropy


def compute_difference_map(derivative: Map, native: Map) -> Map:
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

    Returns
    -------
    diffmap: Map
        map corresponding to the complex difference (derivative - native)
    """
    assert_is_map(derivative, require_uncertainties=False)
    assert_is_map(native, require_uncertainties=False)
    assert_isomorphous(derivative=derivative, native=native, warning_only=True)

    derivative, native = filter_common_indices(derivative, native)

    delta_complex = derivative.to_structurefactor() - native.to_structurefactor()
    delta = Map.from_structurefactor(
        delta_complex,
        cell=native.cell,
        spacegroup=native.spacegroup,
    )

    if derivative.has_uncertainties and native.has_uncertainties:
        prop_uncertainties = np.sqrt(derivative.uncertainties**2 + native.uncertainties**2)
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
        lower weights. The zero-frequency (0, 0, 0) coefficient receives unit weight
        and is excluded from both moment estimates and weight normalization.
    """
    assert_is_map(difference_map, require_uncertainties=True)

    nonzero_frequency = ~difference_map.index.isin([(0, 0, 0)])
    if not np.any(nonzero_frequency):
        msg = "dataset is either empty or only consists of F000"
        raise ValueError(msg)

    measured_difference_map = difference_map.loc[nonzero_frequency]
    mean_squared_amplitude = (measured_difference_map.amplitudes**2).mean()

    # compute k-weights
    amplitude_penalty = (
        k_parameter * (measured_difference_map.amplitudes**2 / mean_squared_amplitude)
        if mean_squared_amplitude != 0.0
        else 0.0
    )
    kweights = 1.0 / (
        1
        + (
            measured_difference_map.uncertainties**2
            / (measured_difference_map.uncertainties**2).mean()
        )
        + amplitude_penalty
    )
    normalized_weights = kweights / np.mean(kweights)

    return normalized_weights.reindex(difference_map.index, fill_value=1.0)


def compute_kweighted_difference_map(derivative: Map, native: Map, *, k_parameter: float) -> Map:
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

    Returns
    -------
    diffmap: Map
        the k-weighted difference map
    """
    # require uncertainties at the beginning
    assert_is_map(derivative, require_uncertainties=True)
    assert_is_map(native, require_uncertainties=True)

    difference_map = compute_difference_map(derivative, native)
    weights = compute_kweights(difference_map, k_parameter=k_parameter)

    difference_map.amplitudes *= weights
    difference_map.uncertainties *= weights

    return difference_map


def max_negentropy_kweighted_difference_map(
    derivative: Map,
    native: Map,
    *,
    k_parameter_values_to_scan: np.ndarray | Sequence[float] = DEFAULT_KPARAMS_TO_SCAN,
) -> tuple[rs.DataSet, KparameterScanMetadata]:
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

    Returns
    -------
    kweighted_dataset: rs.DataSet
        dataset with added columns

    kparameter_metadata: KparameterScanMetadata
        metadata detailing k-weight scan
    """
    assert_is_map(derivative, require_uncertainties=True)
    assert_is_map(native, require_uncertainties=True)

    def negentropy_objective(k_parameter: float) -> float:
        kweighted_map = compute_kweighted_difference_map(
            derivative,
            native,
            k_parameter=k_parameter,
        )
        return map_negentropy(kweighted_map)

    maximizer = ScalarMaximizer(objective=negentropy_objective)
    maximizer.optimize_over_explicit_values(arguments_to_scan=k_parameter_values_to_scan)
    opt_k_parameter = float(maximizer.argument_optimum)

    kweighted_dataset = compute_kweighted_difference_map(
        derivative,
        native,
        k_parameter=opt_k_parameter,
    )

    unweighted_diffmap = compute_difference_map(derivative, native)

    kparameter_metadata = KparameterScanMetadata(
        initial_negentropy=map_negentropy(unweighted_diffmap),
        optimal_parameter_value=float(maximizer.argument_optimum),
        optimal_negentropy=float(maximizer.objective_maximum),
        parameter_scan_results=maximizer.parameter_scan_results,
    )

    return kweighted_dataset, kparameter_metadata
