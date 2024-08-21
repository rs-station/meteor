import numpy as np
import gemmi as gm
import reciprocalspaceship as rs
from typing import overload, Literal


def resolution_limits(dataset: rs.DataSet) -> tuple[float, float]:
    dHKL = dataset.compute_dHKL()["dHKL"]
    return dHKL.max(), dHKL.min()


def cut_resolution(
    dataset: rs.DataSet,
    *,
    dmax_limit: float | None = None,
    dmin_limit: float | None = None,
) -> rs.DataSet:
    dHKL = dataset.compute_dHKL()["dHKL"]
    if dmax_limit:
        dataset = dataset.loc[(dHKL <= dmax_limit)]
    if dmin_limit:
        dataset = dataset.loc[(dHKL >= dmin_limit)]
    return dataset


@overload
def canonicalize_amplitudes(
    dataset: rs.DataSet,
    *,
    amplitude_label: str,
    phase_label: str,
    inplace: Literal[False],
) -> rs.DataSet: ...


@overload
def canonicalize_amplitudes(
    dataset: rs.DataSet,
    *,
    amplitude_label: str,
    phase_label: str,
    inplace: Literal[True],
) -> None: ...


def canonicalize_amplitudes(
    dataset: rs.DataSet,
    *,
    amplitude_label: str,
    phase_label: str,
    inplace: bool = False,
) -> rs.DataSet | None:
    dataset.canonicalize_phases(inplace=inplace)
    if not inplace:
        dataset = dataset.copy(deep=True)

    negative_amplitude_indices = dataset[amplitude_label] < 0.0
    dataset[amplitude_label] = np.abs(dataset[amplitude_label])
    dataset.loc[negative_amplitude_indices, phase_label] += 180.0

    if not inplace:
        return dataset
    else:
        return None


def compute_map_from_coefficients(
    *,
    map_coefficients: rs.DataSet,
    amplitude_label: str,
    phase_label: str,
    map_sampling: int,
) -> gm.Ccp4Map:
    map_coefficients_gemmi_format = map_coefficients.to_gemmi()
    ccp4_map = gm.Ccp4Map()
    ccp4_map.grid = map_coefficients_gemmi_format.transform_f_phi_to_map(
        amplitude_label, phase_label, sample_rate=map_sampling
    )
    ccp4_map.update_ccp4_header()

    return ccp4_map


def compute_coefficients_from_map(
    *,
    map: np.ndarray | gm.Ccp4Map,
    high_resolution_limit: float,
    amplitude_label: str,
    phase_label: str,
) -> rs.DataSet:
    if isinstance(map, np.ndarray):
        return _compute_coefficients_from_numpy_array(
            map_array=map,
            high_resolution_limit=high_resolution_limit,
            amplitude_label=amplitude_label,
            phase_label=phase_label,
        )
    elif isinstance(map, gm.Ccp4Map):
        return _compute_coefficients_from_ccp4_map(
            ccp4_map=map,
            high_resolution_limit=high_resolution_limit,
            amplitude_label=amplitude_label,
            phase_label=phase_label,
        )
    else:
        raise TypeError(f"invalid type {type(map)} for `map`")


def _compute_coefficients_from_numpy_array(
    *,
    map_array: np.ndarray,
    high_resolution_limit: float,
    amplitude_label: str,
    phase_label: str,
) -> rs.DataSet: ...


def _compute_coefficients_from_ccp4_map(
    *,
    ccp4_map: gm.Ccp4Map,
    high_resolution_limit: float,
    amplitude_label: str,
    phase_label: str,
) -> rs.DataSet:
    # to ensure we include the final shell of reflections, add a small buffer to the resolution
    high_resolution_buffer = 0.05

    gemmi_structure_factors = gm.transform_map_to_f_phi(ccp4_map.grid, half_l=False)
    data = gemmi_structure_factors.prepare_asu_data(
        dmin=high_resolution_limit - high_resolution_buffer, with_sys_abs=True
    )

    mtz = gm.Mtz(with_base=True)
    mtz.spacegroup = gemmi_structure_factors.spacegroup
    mtz.set_cell_for_all(gemmi_structure_factors.unit_cell)
    mtz.add_dataset("FromMap")
    mtz.add_column(amplitude_label, "F")
    mtz.add_column(phase_label, "P")
    mtz.set_data(data)
    mtz.switch_to_asu_hkl()

    return rs.DataSet.from_gemmi(mtz)