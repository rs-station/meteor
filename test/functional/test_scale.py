from pathlib import Path

import reciprocalspaceship as rs
from numpy import testing as npt

from meteor.rsmap import Map
from meteor.scale import scale_maps


def test_scaling_regression(testing_mtz_file: Path) -> None:
    ds = rs.read_mtz(str(testing_mtz_file))

    on = Map(ds, amplitude_column="F_on", phase_column="PHI_k", uncertainty_column="SIGF_on")
    off = Map(ds, amplitude_column="F_off", phase_column="PHI_k", uncertainty_column="SIGF_off")
    calculated = Map(ds, amplitude_column="FC_nochrom", phase_column="PHI_k")

    scaled_on_truth = Map(
        ds,
        amplitude_column="F_on_scaled",
        phase_column="PHI_k",
        uncertainty_column="SIGF_on_scaled",
    )
    scaled_off_truth = Map(
        ds,
        amplitude_column="F_off_scaled",
        phase_column="PHI_k",
        uncertainty_column="SIGF_off_scaled",
    )

    scaled_on = scale_maps(
        map_to_scale=on,
        reference_map=calculated,
        weight_using_uncertainties=False,
        least_squares_loss="linear",
    )
    scaled_off = scale_maps(
        map_to_scale=off,
        reference_map=calculated,
        weight_using_uncertainties=False,
        least_squares_loss="linear",
    )

    npt.assert_allclose(scaled_on.amplitudes, scaled_on_truth.amplitudes, atol=1e-3)
    npt.assert_allclose(scaled_off.amplitudes, scaled_off_truth.amplitudes, atol=1e-3)
