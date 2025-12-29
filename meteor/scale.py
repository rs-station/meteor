"""(anisotropic) scaling of crystallographic datasets"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy.optimize as opt
import structlog

from .rsmap import Map
from .utils import filter_common_indices

ScaleParameters = tuple[float, float, float, float, float, float, float]
""" 7x float tuple to hold anisotropic scaling parameters """

log = structlog.get_logger()


def _compute_anisotropic_scale_factors(
    miller_indices: pd.Index,
    anisotropic_scale_parameters: ScaleParameters,
) -> np.ndarray:
    miller_indices_as_array = np.array(list(miller_indices))
    squared_miller_indices = np.square(miller_indices_as_array)

    h_squared = squared_miller_indices[:, 0]
    k_squared = squared_miller_indices[:, 1]
    l_squared = squared_miller_indices[:, 2]

    hk_product = miller_indices_as_array[:, 0] * miller_indices_as_array[:, 1]
    hl_product = miller_indices_as_array[:, 0] * miller_indices_as_array[:, 2]
    kl_product = miller_indices_as_array[:, 1] * miller_indices_as_array[:, 2]

    # Anisotropic scaling term
    exponential_argument = -(
        h_squared * anisotropic_scale_parameters[1]
        + k_squared * anisotropic_scale_parameters[2]
        + l_squared * anisotropic_scale_parameters[3]
        + 2 * hk_product * anisotropic_scale_parameters[4]
        + 2 * hl_product * anisotropic_scale_parameters[5]
        + 2 * kl_product * anisotropic_scale_parameters[6]
    )

    return anisotropic_scale_parameters[0] * np.exp(exponential_argument)


def scale_maps(
    *,
    reference_map: Map,
    map_to_scale: Map,
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

    def compute_residuals(scaling_parameters: ScaleParameters) -> np.ndarray:
        scale_factors = _compute_anisotropic_scale_factors(
            reference_map.index,
            scaling_parameters,
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

    initial_scaling_parameters: ScaleParameters = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    optimization_result = opt.least_squares(
        compute_residuals,
        initial_scaling_parameters,
        loss=least_squares_loss,
    )
    optimized_parameters: ScaleParameters = optimization_result.x

    optimized_scale_factors = _compute_anisotropic_scale_factors(
        unmodified_map_to_scale.index,
        optimized_parameters,
    )

    if len(optimized_scale_factors) != len(unmodified_map_to_scale.index):
        msg1 = "length mismatch: `optimized_scale_factors` - something went wrong"
        msg2 = f"({len(optimized_scale_factors)}) vs `values_to_scale` ({len(unmodified_map_to_scale.index)})"
        raise RuntimeError(msg1, msg2)

    scaled_map = map_to_scale.copy()
    scaled_map.amplitudes *= optimized_scale_factors
    if scaled_map.has_uncertainties:
        scaled_map.uncertainties *= optimized_scale_factors

    return scaled_map
