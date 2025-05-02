
from skimage.filters import gaussian
import numpy as np
import structlog
from collections.abc import Sequence
from typing import Literal, overload


# from .metadata import GaussianScanMetadata 
from .rsmap import Map
from .metadata import GaussianScanMetadata
from .settings import (
    GAUSSIAN_BRACKET_FOR_GOLDEN_OPTIMIZATION,
    MAP_SAMPLING,
    GAUSSIAN_SIGMA_MAX,
    GAUSSIAN_SIGMA_MIN,
)

from .validate import ScalarMaximizer, negentropy

log = structlog.get_logger()

def _low_pass_denoise_array(*, map_as_array: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0.0: # No filtering if weight is zero or negative
        return map_as_array 
    return gaussian(  # type: ignore[no-untyped-call]
        map_as_array,
        sigma=weight,
        preserve_range=True
    )

@overload
def low_pass_denoise_difference_map(
    difference_map: Map,
    *,
    full_output: Literal[False],
    weights_to_scan: Sequence[float] | np.ndarray | None = None,
) -> Map: ...


def low_pass_denoise_difference_map(
    difference_map: Map,
    *,
    full_output: bool = False,
    weights_to_scan: Sequence[float] | np.ndarray | None = None,
) -> Map:

    realspace_map_array = difference_map.to_3d_numpy_map(map_sampling=MAP_SAMPLING)
    # Compute voxel size (assuming difference_map.cell is a tuple (a, b, c) in Å)
 
    cell_lengths = np.array([difference_map.cell.a, difference_map.cell.b, difference_map.cell.c])
    print("Difference map cell paratemers are: ")
    print(difference_map.cell.parameters)


    voxel_size = cell_lengths / MAP_SAMPLING  # if MAP_SAMPLING is a scalar
    print("Voxel size")
    print(voxel_size)
    avg_voxel_size = np.mean(voxel_size)


    def negentropy_objective(lp_weight: float) -> float:
        denoised_map = _low_pass_denoise_array(map_as_array=realspace_map_array, weight=lp_weight)
        return negentropy(denoised_map)

    maximizer = ScalarMaximizer(objective=negentropy_objective)
    if weights_to_scan is not None:
        maximizer.optimize_over_explicit_values(arguments_to_scan=weights_to_scan)
    else:
        #  this needs to change for the lp Gaussian 
        print("BRACKET_FOR_GOLDEN_OPTIMIZATION = ",GAUSSIAN_BRACKET_FOR_GOLDEN_OPTIMIZATION)

        ## normalised the brackets with the voxels size
        GAUSSIAN_NORM_BRACKET_FOR_GOLDEN_OPTIMIZATION = GAUSSIAN_BRACKET_FOR_GOLDEN_OPTIMIZATION / avg_voxel_size
        print("After normalisaiton NORM_BRACKET_FOR_GOLDEN_OPTIMIZATION = ",GAUSSIAN_NORM_BRACKET_FOR_GOLDEN_OPTIMIZATION)

        log.info("Optimizing Gaussian smoothing sigma using golden-section search.")
        maximizer.optimize_with_golden_algorithm(bracket=GAUSSIAN_NORM_BRACKET_FOR_GOLDEN_OPTIMIZATION)

    # if maximizer.argument_optimum > GAUSSIAN_SIGMA_MAX:
    #     log.warning(
    #         "Gaussian smoothing sigma much larger than expected, something probably went wrong",
    #         sigma=f"{maximizer.argument_optimum:.2f}",
    #         limit=GAUSSIAN_SIGMA_MAX,
    #     )
    # # Compute voxel size (assuming difference_map.cell is a tuple (a, b, c) in Å)
    # voxel_size = np.array(difference_map.cell.parameters) / MAP_SAMPLING
    # avg_voxel_size = np.mean(voxel_size)  # Average voxel size for isotropic filtering
    # # Normalize sigma
    # normalised_sigma = maximizer.argument_optimum / avg_voxel_size
    normalised_sigma = maximizer.argument_optimum
    log.info(
        "Applying Gaussian smoothing with normalized sigma",
        original_sigma=maximizer.argument_optimum,
        normalised_sigma=normalised_sigma,
        voxel_size=avg_voxel_size,
    )

    # # denoise using the optimized parameters and convert to an rs.DataSet
    # final_realspace_map_as_array = _low_pass_denoise_array(
    #     map_as_array=realspace_map_array,
    #     weight=maximizer.argument_optimum,
    # )
    # Apply Gaussian filtering with normalized sigma
    final_realspace_map_as_array = _low_pass_denoise_array(
        map_as_array=realspace_map_array,
        weight=normalised_sigma,
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

# need to fix the Metadata for lp filtering
    if full_output:
        initial_negentropy = negentropy(realspace_map_array)
        lp_result = GaussianScanMetadata(
            initial_negentropy=float(initial_negentropy),
            optimal_parameter_value=float(maximizer.argument_optimum),
            optimal_negentropy=float(maximizer.objective_maximum),
            negentropy_gain=float(maximizer.objective_maximum - initial_negentropy),
            negentropy_gain_ratio=float((maximizer.objective_maximum - initial_negentropy) / initial_negentropy),
            map_sampling=MAP_SAMPLING,
            normalised_sigma=float(normalised_sigma),
            parameter_scan_results=maximizer.parameter_scan_results,
        )
        return final_map, lp_result

    return final_ma



