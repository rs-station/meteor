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


class ParameterLengthMismatchError(ValueError): ...


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


def compute_scale_factors(
    *, miller_indices: pd.Index, scale_parameters: ScaleParameters, scale_mode: str | ScaleMode
) -> np.ndarray:
    if isinstance(scale_mode, str):
        scale_mode = ScaleMode(scale_mode)

    vector_h = np.array(list(miller_indices))
    if vector_h.shape[1] != DIMENSION_OF_MILLER_INDEX:
        msg = "`miller_indices` should be an (n, 3) multi-index of miller HKL indices, "
        msg += f"got shape: {vector_h.shape}"
        raise ValueError(msg)

    sp_as_array = np.array(scale_parameters)
    if sp_as_array.shape != (scale_mode.number_of_parameters,):
        msg = f"`scale_parameters` should be length {scale_mode.number_of_parameters} "
        msg += f"for mode={scale_mode}, got length: {len(scale_parameters)}"
        raise ParameterLengthMismatchError(msg)

    # this code is part of a few tight loops; code below is fast and clear
    matrix_B = np.zeros((3, 3), dtype=sp_as_array.dtype)  # noqa: N806 (variable capitalization)

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

    else:
        msg = f"mode {scale_mode} not valid"
        raise ValueError(msg)

    # the einsum implements sum_i{ h^T . B . h }
    exponential_argument = -np.einsum("ni,ij,nj->n", vector_h, matrix_B, vector_h)

    return sp_as_array[0] * np.exp(exponential_argument)


def scale_maps(
    *,
    reference_map: Map,
    map_to_scale: Map,
    scale_mode: ScaleMode = ScaleMode.anisotropic,
    weight_using_uncertainties: bool = True,
    least_squares_loss: str | Callable = "huber",
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

    one = np.array(1.0)  # a constant if there are no uncertainties to use
    ref_variance: np.ndarray = (
        np.square(reference_map.uncertainties)
        if (reference_map.has_uncertainties and weight_using_uncertainties)
        else one
    )
    to_scale_variance: np.ndarray = (
        np.square(map_to_scale.uncertainties)
        if (map_to_scale.has_uncertainties and weight_using_uncertainties)
        else one
    )
    sqrt_inverse_variance = 1.0 / np.sqrt(ref_variance + to_scale_variance)

    def compute_residuals(scale_parameters: ScaleParameters) -> np.ndarray:
        scale_factors = compute_scale_factors(
            miller_indices=reference_map.index,
            scale_parameters=scale_parameters,
            scale_mode=scale_mode,
        )

        difference_after_scaling = (
            scale_factors * map_to_scale.amplitudes - reference_map.amplitudes
        )
        residuals = np.array(sqrt_inverse_variance * difference_after_scaling, dtype=np.float64)

        # filter NaNs in input -- are simply missing values
        residuals = residuals[np.isfinite(residuals)]

        if not isinstance(residuals, np.ndarray):
            msg = "scipy optimizers' behavior is unstable unless `np.ndarray`s are used"
            raise TypeError(msg)

        return residuals

    initial_scaling_parameters: ScaleParameters = (1.0,) + (0.0,) * (
        scale_mode.number_of_parameters - 1
    )
    optimization_result = opt.least_squares(
        compute_residuals,
        initial_scaling_parameters,
        loss=least_squares_loss,
    )
    optimized_parameters: ScaleParameters = optimization_result.x

    optimized_scale_factors = compute_scale_factors(
        miller_indices=unmodified_map_to_scale.index,
        scale_parameters=optimized_parameters,
        scale_mode=scale_mode,
    )
    if optimized_scale_factors[0] <= 0.0:
        log.warning("the scale constant `C` (in `C*exp{-B}`) is negative - likely wrong")

    if len(optimized_scale_factors) != len(unmodified_map_to_scale.index):
        msg1 = "length mismatch: `optimized_scale_factors` - something went wrong"
        msg2 = f"({len(optimized_scale_factors)}) vs `values_to_scale` ({len(unmodified_map_to_scale.index)})"
        raise RuntimeError(msg1, msg2)

    scaled_map = unmodified_map_to_scale.copy()
    scaled_map.amplitudes *= optimized_scale_factors
    if scaled_map.has_uncertainties:
        scaled_map.uncertainties *= optimized_scale_factors

    return scaled_map
