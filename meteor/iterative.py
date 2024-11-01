"""iterative TV-based phase retrieval"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import structlog

from .rsmap import Map
from .settings import (
    DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
    ITERATIVE_TV_CONVERGENCE_TOLERANCE,
    ITERATIVE_TV_MAX_ITERATIONS,
)
from .tv import TvDenoiseResult, tv_denoise_difference_map
from .utils import (
    average_phase_diff_in_degrees,
    complex_array_to_rs_dataseries,
    filter_common_indices,
)

log = structlog.get_logger()


def _project_derivative_on_experimental_set(
    *,
    native: np.ndarray,
    derivative_amplitudes: np.ndarray,
    difference: np.ndarray,
) -> np.ndarray:
    """
    Project the `derivative` structure factor onto the set of experimentally observed amplitudes.

    Specifically, we change the amplitude of the complex-valued `derivative` to ensure that both

        difference = derivative - native

    and that the modulus |derivative| is equal to the specified (user-input) `derivative_amplitudes`

    Parameters
    ----------
    native: np.ndarray
        The experimentally observed native amplitudes and computed phases, as a complex array.

    derivative_amplitudes: np.ndarray
        An array of the experimentally observed derivative amplitudes. Typically real-valued, but
        a complex-valued array with arbitrary phase can be passed (phases discarded).

    difference: np.ndarray
        The estimated complex structure factor difference, derivative-minus-native.

    Returns
    -------
    projected_derivative: np.ndarray
        The complex-valued derivative structure factors, with experimental amplitude and phase
        adjusted to ensure that difference = derivative - native.
    """
    projected_derivative = difference + native
    projected_derivative *= np.abs(derivative_amplitudes) / np.abs(projected_derivative)
    return projected_derivative


def _complex_derivative_from_iterative_tv(  # noqa: PLR0913
    *,
    native: np.ndarray,
    initial_derivative: np.ndarray,
    tv_denoise_function: Callable[[np.ndarray], tuple[np.ndarray, TvDenoiseResult]],
    convergence_tolerance: float = ITERATIVE_TV_CONVERGENCE_TOLERANCE,
    max_iterations: int = ITERATIVE_TV_MAX_ITERATIONS,
    verbose: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Estimate the derivative phases using the iterative TV algorithm.

    This function contains the algorithm logic.

    Parameters
    ----------
    native: np.ndarray
        The complex native structure factors, usually experimental amplitudes and calculated phases

    initial_complex_derivative : np.ndarray
        The complex derivative structure factors, usually with experimental amplitudes and esimated
        phases (often calculated from the native structure)

    tv_denoise_function: Callable[[np.ndarray], tuple[np.ndarray, TvDenoiseResult]]
        A function capable of applying the TV denoising operation to *Fourier space* objects. This
        function should therefore map one complex np.ndarray to a denoised complex np.ndarray and
        the TvDenoiseResult for that TV run.

    convergance_tolerance: float
        If the change in the estimated derivative SFs drops below this value (phase, per-component)
        then return. Default 1e-4.

    max_iterations: int
        If this number of iterations is reached, stop early. Default 1000.

    verbose: bool
        Log or not.

    Returns
    -------
    estimated_complex_derivative: np.ndarray
        The derivative SFs, with the same amplitudes but phases altered to minimize the TV.

    metadata: pd.DataFrame
        Information about the algorithm run as a function of iteration. For each step, includes:
        the tv_weight used, the negentropy (after the TV step), and the average phase change in
        degrees.
    """
    derivative = np.copy(initial_derivative)
    difference = initial_derivative - native

    converged: bool = False
    num_iterations: int = 0
    metadata: list[dict[str, float]] = []

    while not converged:
        difference_tvd, tv_metadata = tv_denoise_function(difference)
        updated_derivative = _project_derivative_on_experimental_set(
            native=native,
            derivative_amplitudes=np.abs(derivative),
            difference=difference_tvd,
        )

        phase_change = average_phase_diff_in_degrees(derivative, updated_derivative)
        derivative = updated_derivative
        difference = derivative - native

        converged = phase_change < convergence_tolerance
        num_iterations += 1

        metadata.append(
            {
                "iteration": num_iterations,
                "tv_weight": tv_metadata.optimal_tv_weight,
                "negentropy_after_tv": tv_metadata.optimal_negentropy,
                "average_phase_change": phase_change,
            },
        )
        if verbose:
            log.info(
                f"  iteration {num_iterations:04d}",  # noqa: G004
                phase_change=round(phase_change, 4),
                negentropy=round(tv_metadata.optimal_negentropy, 4),
                tv_weight=tv_metadata.optimal_tv_weight,
            )

        if num_iterations > max_iterations:
            break

    return derivative, pd.DataFrame(metadata)


def iterative_tv_phase_retrieval(  # noqa: PLR0913
    initial_derivative: Map,
    native: Map,
    *,
    convergence_tolerance: float = ITERATIVE_TV_CONVERGENCE_TOLERANCE,
    max_iterations: int = ITERATIVE_TV_MAX_ITERATIONS,
    tv_weights_to_scan: list[float] = DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
    verbose: bool = False,
) -> tuple[Map, pd.DataFrame]:
    """
    Here is a brief pseudocode sketch of the alogrithm. Structure factors F below are complex unless
    explicitly annotated |*|.

        Input: |F|, |Fh|, phi_c
        Note: F = |F| * exp{ phi_c } is the native/dark data,
             |Fh| represents the derivative/triggered/light data

        Initialize:
         - D_F = ( |Fh| - |F| ) * exp{ phi_c }

        while not converged:
            D_rho = FT{ D_F }                       Fourier transform
            D_rho' = TV{ D_rho }                    TV denoise: apply real space prior
            D_F' = FT-1{ D_rho' }                   back Fourier transform
            Fh' = (D_F' + F) * [|Fh| / |D_F' + F|]  Fourier space projection onto experimental set
            D_F = Fh' - F

    Where the TV weight parameter is determined using golden section optimization. The algorithm
    iterates until the changes in the derivative phase drop below a specified threshold.

    Parameters
    ----------
    initial_derivative: Map
        the derivative amplitudes, and initial guess for the phases

    native: Map
        the native amplitudes, phases

    convergance_tolerance: float
        If the change in the estimated derivative SFs drops below this value (phase, per-component)
        then return. Default 1e-4.

    max_iterations: int
        If this number of iterations is reached, stop early. Default 1000.

    tv_weights_to_scan : list[float], optional
        A list of TV regularization weights (λ values) to be scanned for optimal results,
        by default [0.001, 0.01, 0.1, 1.0].

    verbose: bool
        Log or not.

    Returns
    -------
    output_map: Map
        The estimated derivative phases, along with the input amplitudes and input computed phases.

    metadata: pd.DataFrame
        Information about the algorithm run as a function of iteration. For each step, includes:
        the tv_weight used, the negentropy (after the TV step), and the average phase change in
        degrees.
    """
    initial_derivative, native = filter_common_indices(initial_derivative, native)

    # clean TV denoising interface that is crystallographically intelligent
    # maintains state for the HKL index, spacegroup, and cell information
    def tv_denoise_closure(difference: np.ndarray) -> tuple[np.ndarray, TvDenoiseResult]:
        diffmap = Map.from_structurefactor(difference, index=native.index)
        diffmap.cell = native.cell
        diffmap.spacegroup = native.spacegroup

        denoised_map, tv_metadata = tv_denoise_difference_map(
            diffmap,
            weights_to_scan=tv_weights_to_scan,
            full_output=True,
        )

        return denoised_map.complex_amplitudes, tv_metadata

    # estimate the derivative phases using the iterative TV algorithm
    if verbose:
        log.info(
            "convergence criteria:",
            phase_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
        )
    it_tv_complex_derivative, metadata = _complex_derivative_from_iterative_tv(
        native=native.complex_amplitudes,
        initial_derivative=initial_derivative.complex_amplitudes,
        tv_denoise_function=tv_denoise_closure,
        convergence_tolerance=convergence_tolerance,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    _, derivative_phases = complex_array_to_rs_dataseries(
        it_tv_complex_derivative,
        index=initial_derivative.index,
    )

    # combine the determined derivative phases with the input to generate a complete output
    output_dataset = initial_derivative.copy()
    output_dataset.phases = derivative_phases

    return output_dataset, metadata
