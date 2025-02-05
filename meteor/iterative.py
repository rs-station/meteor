"""iterative TV-based phase retrieval"""

from __future__ import annotations

import numpy as np
import pandas as pd
import reciprocalspaceship as rs
import structlog

from .rsmap import Map
from .settings import (
    DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
    ITERATIVE_TV_CONVERGENCE_TOLERANCE,
    ITERATIVE_TV_MAX_ITERATIONS,
)
from .tv import TvDenoiseResult, tv_denoise_difference_map
from .utils import (
    CellType,
    SpacegroupType,
    assert_isomorphous,
    average_phase_diff_in_degrees,
    filter_common_indices,
)

log = structlog.get_logger()


def _assert_are_dataseries(*args: list[rs.DataSeries]) -> None:
    for arg in args:
        if not isinstance(arg, rs.DataSeries):
            msg = f"`{arg!s}` must be type rs.DataSeries, got {type(arg)}"
            raise TypeError(msg)


class IterativeTvDenoiser:
    """
    An implementation of `meteor`'s iterative TV phase update algorithm.

    The big idea is to iteratively TV denoise a map, then project it back onto the set of:
     - experimentally determined structure factor amplitudes
     - fixed (likely computed) phases for the `native` dataset

    The only thing left to change are the `derivative` phases, which are latent.

    Here is a brief pseudocode sketch of the alogrithm. Structure factors F below are complex
    unless explicitly annotated |*|.

        Input: |F|, |Fh|, phi_c
        Note: F = |F| * exp{ phi_c } is the native/dark data,
            |Fh| represents the derivative/triggered/light data

        Initialize:
        - D_F = ( |Fh| - |F| ) * exp{ phi_c }

        while not converged:
            D_rho = FT{ D_F }                       Fourier transform
            D_rho' = TV{ D_rho }                    TV denoise: apply real space prior
            D_F' = FT-1{ D_rho' }                   back Fourier transform
            Fh' = (D_F' + F) * [|Fh| / |D_F' + F|]  Fourier space projection to experimental set
            D_F = Fh' - F

    Where the TV weight parameter is determined at each step by optimizing over a fixed range of
    values. Golden-section search could also be used, but tests have revealed a limited set of about
    three order-of-magnitude values to scan leads to faster convergence in most cases. The algorithm
    iterates until the changes in the derivative phase drop below a specified threshold.
    """

    def __init__(
        self,
        *,
        convergence_tolerance: float = ITERATIVE_TV_CONVERGENCE_TOLERANCE,
        max_iterations: int = ITERATIVE_TV_MAX_ITERATIONS,
        tv_weights_to_scan: list[float] = DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
        verbose: bool = False,
    ) -> None:
        """
        Initialize an iterative TV denoiser.

        Parameters
        ----------
        convergance_tolerance: float
            If the change in the estimated derivative SFs drops below this value (phase,
            per-component) then return. Default 1e-4.

        max_iterations: int
            If this number of iterations is reached, stop early. Default 1000.

        tv_weights_to_scan : list[float], optional
            A list of TV regularization weights (λ values) to be scanned for optimal results,
            by default [0.001, 0.01, 0.1, 1.0].

        verbose: bool
            Log or not.
        """
        self.tv_weights_to_scan = tv_weights_to_scan
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose

        if verbose:
            log.info(
                "convergence criteria:",
                phase_tolerance=convergence_tolerance,
                max_iterations=max_iterations,
            )

    def _tv_denoise_complex_difference_sf(
        self,
        complex_difference_sf: rs.DataSeries,
        *,
        cell: CellType,
        spacegroup: SpacegroupType,
    ) -> tuple[rs.DataSeries, TvDenoiseResult]:
        """Apply a single iteration of TV denoising to set of complex SFs, return complex SFs"""
        diffmap = Map.from_structurefactor(
            complex_difference_sf,
            index=complex_difference_sf.index,
            cell=cell,
            spacegroup=spacegroup,
        )

        denoised_map, tv_metadata = tv_denoise_difference_map(
            diffmap,
            weights_to_scan=self.tv_weights_to_scan,
            full_output=True,
        )

        return denoised_map.to_structurefactor(), tv_metadata

    def _iteratively_denoise_sf_amplitudes(
        self,
        *,
        initial_derivative: rs.DataSeries,
        native: rs.DataSeries,
        cell: CellType,
        spacegroup: SpacegroupType,
    ) -> tuple[rs.DataSeries, pd.DataFrame]:
        """
        Estimate the derivative phases using the iterative TV algorithm.

        This function contains the algorithm logic.

        Parameters
        ----------
        native: rs.DataSeries
            The complex native structure factors, usually experimental amplitudes and calculated phases

        initial_complex_derivative : rs.DataSeries
            The complex derivative structure factors, usually with experimental amplitudes and esimated
            phases (often calculated from the native structure)

        cell: gemmi.Cell | Sequence[float] | np.ndarray
            Unit cell, should match both the `native` and `derivative` datasets (usual: use native)

        spacegroup: gemmi.SpaceGroup | str | int
            The spacegroup; both the `native` and `derivative` datasets

        Returns
        -------
        estimated_complex_derivative: rs.DataSeries
            The derivative SFs, with the same amplitudes but phases altered to minimize the TV.

        metadata: pd.DataFrame
            Information about the algorithm run as a function of iteration. For each step, includes:
            the tv_weight used, the negentropy (after the TV step), and the average phase change in
            degrees.
        """
        derivative = initial_derivative.copy()
        _assert_are_dataseries(native, derivative)

        converged: bool = False
        num_iterations: int = 0
        metadata: list[dict[str, float]] = []

        # do differences with rs.DataSeries, handles missing indices
        difference: rs.DataSeries = initial_derivative - native

        while not converged:
            denoised_difference_sfs, tv_metadata = self._tv_denoise_complex_difference_sf(
                difference, cell=cell, spacegroup=spacegroup
            )

            # project onto the native amplitudes to obtain an "updated_derivative"
            #   Fh' = (D_F' + F) * [|Fh| / |D_F' + F|]
            denoised_difference_sfs, native = filter_common_indices(denoised_difference_sfs, native)
            updated_derivative: rs.DataSeries = denoised_difference_sfs + native

            updated_derivative, derivative = filter_common_indices(updated_derivative, derivative)
            updated_derivative *= np.abs(derivative) / np.abs(updated_derivative)

            # compute phase change, THEN set: derivative <- updated_derivative
            phase_change = average_phase_diff_in_degrees(derivative, updated_derivative)
            derivative = updated_derivative

            difference = derivative - native

            converged = phase_change < self.convergence_tolerance
            num_iterations += 1

            metadata.append(
                {
                    "iteration": num_iterations,
                    "tv_weight": tv_metadata.optimal_tv_weight,
                    "negentropy_after_tv": tv_metadata.optimal_negentropy,
                    "average_phase_change": phase_change,
                },
            )
            if self.verbose:
                log.info(
                    f"  iteration {num_iterations:04d}",  # noqa: G004
                    phase_change=round(phase_change, 4),
                    negentropy=round(tv_metadata.optimal_negentropy, 4),
                    tv_weight=tv_metadata.optimal_tv_weight,
                )

            if num_iterations > self.max_iterations:
                break

        return derivative, pd.DataFrame(metadata)

    def __call__(
        self,
        *,
        derivative: Map,
        native: Map,
        check_isomorphous: bool = True,
    ) -> tuple[Map, pd.DataFrame]:
        """
        Denoise by estimating new, low-TV phases for the `derivative` dataset.

        Parameters
        ----------
        derivative: Map
            the derivative amplitudes, and initial guess for the phases

        native: Map
            the native amplitudes, phases

        check_isomorphous: bool
            perform a check to ensure the two datasets are isomorphous; recommended. Default: True.

        Returns
        -------
        updated_derivative: Map
            The estimated derivative phases, along with the input amplitudes and input phases.

        metadata: pd.DataFrame
            Information about the algorithm run as a function of iteration. For each step, includes:
            the tv_weight used, the negentropy (after the TV step), and the average phase change in
            degrees.
        """
        if check_isomorphous:
            assert_isomorphous(derivative=derivative, native=native)

        it_tv_complex_derivative, metadata = self._iteratively_denoise_sf_amplitudes(
            native=native.to_structurefactor(),
            initial_derivative=derivative.to_structurefactor(),
            cell=native.cell,
            spacegroup=native.spacegroup,
        )

        updated_derivative_map = Map.from_structurefactor(
            it_tv_complex_derivative,
            cell=derivative.cell,
            spacegroup=derivative.spacegroup,
        )

        if derivative.has_uncertainties:
            updated_derivative_map.set_uncertainties(
                derivative.uncertainties,
                column_name=derivative.uncertainties_column_name,
            )

        return updated_derivative_map, metadata
