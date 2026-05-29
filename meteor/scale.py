"""(anisotropic) scaling of crystallographic datasets"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import numpy as np
import pandas as pd
import scipy.optimize as opt
import structlog

from .rsmap import Map
from .utils import filter_common_indices

ScaleParameters = tuple[float, ...]
""" 7x float tuple to hold anisotropic scaling parameters """

log = structlog.get_logger()

DIMENSION_OF_MILLER_INDEX: int = 3
MAX_SCALE_FACTOR: float = 1e300
NON_FINITE_RESIDUAL_PENALTY: float = 1e30
MIN_FRACTION_COMMON_INDICES: float = 0.5


class ParameterLengthMismatchError(ValueError): ...


class ScalingError(RuntimeError): ...


class ScaleMode(StrEnum):
    anisotropic = "anisotropic"
    orthogonal = "orthogonal"
    isotropic = "isotropic"
    scalar = "scalar"

    @property
    def number_of_parameters(self) -> int:
        if self is ScaleMode.anisotropic:
            return 7
        if self is ScaleMode.orthogonal:
            return 4
        if self is ScaleMode.isotropic:
            return 2
        if self is ScaleMode.scalar:
            return 1
        raise NotImplementedError


def _cast_to_miller_array(miller_indices: pd.Index | np.ndarray) -> np.ndarray:
    """Coerce a pd.MultiIndex or a precomputed (n, 3) array to an (n, 3) ndarray."""
    if isinstance(miller_indices, np.ndarray):
        vector_h = miller_indices
    else:
        vector_h = np.array(list(miller_indices))
    if vector_h.ndim != 2 or vector_h.shape[1] != DIMENSION_OF_MILLER_INDEX:  # noqa: PLR2004
        msg = "`miller_indices` should be an (n, 3) multi-index of miller HKL indices, "
        msg += f"got shape: {vector_h.shape}"
        raise ValueError(msg)
    return vector_h


def compute_scale_factors(
    *,
    miller_indices: pd.Index | np.ndarray,
    scale_parameters: ScaleParameters,
    scale_mode: str | ScaleMode,
) -> np.ndarray:
    """Evaluate the anisotropic scale factor `C * exp(-h^T B h)` for each Miller index.

    `miller_indices` may be a `pd.MultiIndex` or a precomputed `(n, 3)` ndarray.
    The hot path (called once per least-squares iteration) prefers the ndarray
    form so we don't re-build it on every call.
    """
    if isinstance(scale_mode, str):
        scale_mode = ScaleMode(scale_mode)

    vector_h = _cast_to_miller_array(miller_indices)

    sp_as_array = np.asarray(scale_parameters, dtype=np.float64)
    if sp_as_array.shape != (scale_mode.number_of_parameters,):
        msg = f"`scale_parameters` should be length {scale_mode.number_of_parameters} "
        msg += f"for mode={scale_mode}, got length: {len(scale_parameters)}"
        raise ParameterLengthMismatchError(msg)

    if not np.all(np.isfinite(sp_as_array)):
        msg = f"`scale_parameters` must all be finite, got: {tuple(scale_parameters)}"
        raise ScalingError(msg)

    # this code is part of a few tight loops; code below is fast and clear
    matrix_B = np.zeros((3, 3), dtype=np.float64)  # noqa: N806 (variable capitalization)

    if scale_mode == ScaleMode.anisotropic:
        matrix_B[0, 0] = sp_as_array[1]
        matrix_B[1, 1] = sp_as_array[2]
        matrix_B[2, 2] = sp_as_array[3]
        matrix_B[0, 1] = matrix_B[1, 0] = sp_as_array[4]
        matrix_B[0, 2] = matrix_B[2, 0] = sp_as_array[5]
        matrix_B[1, 2] = matrix_B[2, 1] = sp_as_array[6]

    elif scale_mode == ScaleMode.orthogonal:
        matrix_B[0, 0] = sp_as_array[1]
        matrix_B[1, 1] = sp_as_array[2]
        matrix_B[2, 2] = sp_as_array[3]

    elif scale_mode == ScaleMode.isotropic:
        matrix_B[0, 0] = sp_as_array[1]
        matrix_B[1, 1] = sp_as_array[1]
        matrix_B[2, 2] = sp_as_array[1]

    # NOTE: early return -- we don't need to compute the einsum for scale_mode "scalar"
    elif scale_mode == ScaleMode.scalar:
        return sp_as_array[0] * np.ones(vector_h.shape[0], dtype=np.float64)

    # the einsum implements sum_i{ h^T . B . h }
    exponential_argument = -np.einsum("ni,ij,nj->n", vector_h, matrix_B, vector_h)

    scale_factors = sp_as_array[0] * np.exp(exponential_argument)
    return np.clip(scale_factors, -MAX_SCALE_FACTOR, MAX_SCALE_FACTOR)


def scale_maps(
    *,
    reference_map: Map,
    map_to_scale: Map,
    scale_mode: ScaleMode = ScaleMode.anisotropic,
    weight_using_uncertainties: bool = True,
    least_squares_loss: str | Callable[[np.ndarray], np.ndarray] = "huber",
) -> Map:
    """
    Scale a dataset to align it with a reference dataset using anisotropic scaling.

    This function scales the dataset (`map_to_scale`) by comparing it to a reference dataset
    (`reference_map`) based on a specified column. The scaling applies an anisotropic model of
    the form:

        C * exp{ -(h**2 B11 + k**2 B22 + l**2 B33 +
                    2hk B12 + 2hl  B13 +  2kl B23) }

    The parameters Bxy are fit using least squares, optionally with uncertainty (inverse variance)
    weighting. Any of `scipy`'s loss functions can be employed; the Huber loss is the default.

    NB! All intensity, amplitude, and standard deviation columns in `map_to_scale` will be
    modified (scaled). To access the scale parameters directly, use
    `meteor.scale.compute_scale_factors`.

    Parameters
    ----------
    reference_map : Map
        The reference dataset map.
    map_to_scale : Map
        The map dataset to be scaled.
    scale_mode : ScaleMode (StrEnum, default: `anisotropic`)
        Should be one of:
          - 'anisotropic' (fit all Bxy, as above)
          - 'orthogonal' (off-diagonal Bxy are zero)
          - 'isotropic' (off-diagonal Bxy are zero and dialgonal Bxy are identical)
          - 'scalar' (only fit the scale constant C)
    weight_using_uncertainties : bool, optional (default: True)
        Whether or not to weight the scaling by uncertainty values. If True, uncertainty values are
        extracted from the `uncertainty_column` in both datasets, and robust (Huber) inverse
        variance weighting is used in the LSQ procedure.
    least_squares_loss: str, optional (default: "huber")
        This value is passed directly to the `loss` argument in scipy.optimize.least_squares. Refer
        to the documentation for `scipy.optimize.least_squares` [2]. The default value ("huber")
        should be a good choice for just about any situation. If you want to more directly replicate
        SCALEIT's behavior, use "linear" instead.

    Returns
    -------
    scaled_map: Map
        A copy of `map_to_scale`, with the amplitudes and uncertainties scaled anisotropically to
        best match `reference_map`.

    See Also
    --------
    compute_scale_factors : function to compute the scale factors directly

    Citations:
    ----------
    [1] SCALEIT https://www.ccp4.ac.uk/html/scaleit.html
    [2] scipy.optimize.least_squares
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    """
    # we want to compute the scaling factors (scalars) on the common set of indices,
    # but then apply the scaling operation to the entire set of `map_to_scale.amplitudes``,
    # even if the corresponding amplitudes don't appear in the `reference_map` Map
    unmodified_map_to_scale = map_to_scale.copy()
    reference_map, map_to_scale = filter_common_indices(reference_map, map_to_scale)

    ref_amps = np.asarray(reference_map.amplitudes, dtype=np.float64)
    to_amps = np.asarray(map_to_scale.amplitudes, dtype=np.float64)

    use_ref_sigmas = reference_map.has_uncertainties and weight_using_uncertainties
    use_to_sigmas = map_to_scale.has_uncertainties and weight_using_uncertainties
    ref_sigmas = (
        np.asarray(reference_map.uncertainties, dtype=np.float64) if use_ref_sigmas else None
    )
    to_sigmas = np.asarray(map_to_scale.uncertainties, dtype=np.float64) if use_to_sigmas else None

    valid = np.isfinite(ref_amps) & np.isfinite(to_amps)
    if ref_sigmas is not None:
        valid &= np.isfinite(ref_sigmas) & (ref_sigmas > 0.0)
    if to_sigmas is not None:
        valid &= np.isfinite(to_sigmas) & (to_sigmas > 0.0)

    n_valid = int(valid.sum())
    if n_valid == 0:
        msg = (
            "No finite common reflections to fit. "
            "Check input maps for missing values or invalid uncertainties."
        )
        raise ScalingError(msg)

    fraction_common_indices = float(n_valid / len(valid))
    if fraction_common_indices < MIN_FRACTION_COMMON_INDICES:
        log.warning(
            "very small number of common indices between datasets to be scaled together",
            fraction_common_indices=fraction_common_indices,
        )

    ref_amps = ref_amps[valid]
    to_amps = to_amps[valid]
    ref_variance: np.ndarray | float = ref_sigmas[valid] ** 2 if ref_sigmas is not None else 1.0
    to_variance: np.ndarray | float = to_sigmas[valid] ** 2 if to_sigmas is not None else 1.0
    sqrt_inverse_variance = 1.0 / np.sqrt(ref_variance + to_variance)

    miller_array = _cast_to_miller_array(reference_map.index)[valid]

    def compute_residuals(scale_parameters: ScaleParameters) -> np.ndarray:
        scale_factors = compute_scale_factors(
            miller_indices=miller_array,
            scale_parameters=scale_parameters,
            scale_mode=scale_mode,
        )
        residuals = sqrt_inverse_variance * (scale_factors * to_amps - ref_amps)

        return np.nan_to_num(
            residuals,
            nan=NON_FINITE_RESIDUAL_PENALTY,
            posinf=NON_FINITE_RESIDUAL_PENALTY,
            neginf=-NON_FINITE_RESIDUAL_PENALTY,
        )

    initial_c = float(ref_amps.mean() / to_amps.mean())
    if not np.isfinite(initial_c) or initial_c < 0.0:
        msg = (
            f"`initial_c` is {initial_c}: either not finite or negative. "
            "Check input for errors and outliers"
        )
        raise ScalingError(msg)

    initial_scaling_parameters: ScaleParameters = (initial_c,) + (0.0,) * (
        scale_mode.number_of_parameters - 1
    )
    optimization_result = opt.least_squares(
        compute_residuals,
        initial_scaling_parameters,
        loss=least_squares_loss,
    )
    optimized_parameters: ScaleParameters = tuple(optimization_result.x)

    optimized_scale_factors = compute_scale_factors(
        miller_indices=unmodified_map_to_scale.index,
        scale_parameters=optimized_parameters,
        scale_mode=scale_mode,
    )

    scaled_map = unmodified_map_to_scale.copy()
    scaled_map.amplitudes *= optimized_scale_factors
    if scaled_map.has_uncertainties:
        scaled_map.uncertainties *= optimized_scale_factors

    return scaled_map
