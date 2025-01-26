"""crystallographic helper functions"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import gemmi
import numpy as np
import reciprocalspaceship as rs
from reciprocalspaceship.decorators import cellify, spacegroupify
from reciprocalspaceship.utils import canonicalize_phases

CellType = Sequence[float] | np.ndarray | gemmi.UnitCell
SpacegroupType = str | int | gemmi.SpaceGroup


class ShapeMismatchError(Exception): ...


class NotIsomorphousError(RuntimeError): ...


class ResolutionCutOverlapError(ValueError): ...


def assert_isomorphous(*, derivative: rs.DataSet, native: rs.DataSet) -> None:
    if not native.is_isomorphous(derivative):
        msg = "`derivative` and `native` datasets are not similar enough; "
        msg += f"they have cell/spacegroup: {derivative.cell}/{native.cell} and "
        msg += f"{derivative.spacegroup}/{native.spacegroup} respectively"
        raise NotIsomorphousError(msg)


def filter_common_indices(df1: rs.DataSet, df2: rs.DataSet) -> tuple[rs.DataSet, rs.DataSet]:
    common_indices = df1.index.intersection(df2.index)
    df1_common = df1.loc[common_indices].copy()
    df2_common = df2.loc[common_indices].copy()
    if len(df1_common) == 0 or len(df2_common) == 0:
        msg = "cannot find any HKL incdices in common between `df1` and `df2`"
        raise IndexError(msg)
    return df1_common, df2_common


def cut_resolution(
    dataset: rs.DataSet,
    *,
    low_resolution_limit: float | None = None,
    high_resolution_limit: float | None = None,
) -> rs.DataSet:
    if (
        high_resolution_limit
        and low_resolution_limit
        and (high_resolution_limit >= low_resolution_limit)
    ):
        msg = f"requested highres >= lowres: {high_resolution_limit} >= {low_resolution_limit}"
        raise ResolutionCutOverlapError(msg)

    d_hkl = dataset.compute_dHKL()
    if isinstance(d_hkl, rs.DataSet):
        d_hkl = d_hkl["dHKL"]

    if low_resolution_limit:
        dataset = dataset.loc[(d_hkl <= low_resolution_limit)]
    if high_resolution_limit:
        dataset = dataset.loc[(d_hkl >= high_resolution_limit)]
    return dataset


@overload
def canonicalize_amplitudes(
    dataset: rs.DataSet,
    *,
    amplitude_column: str,
    phase_column: str,
    inplace: Literal[False],
) -> rs.DataSet: ...


@overload
def canonicalize_amplitudes(
    dataset: rs.DataSet,
    *,
    amplitude_column: str,
    phase_column: str,
    inplace: Literal[True],
) -> None: ...


def canonicalize_amplitudes(
    dataset: rs.DataSet,
    *,
    amplitude_column: str,
    phase_column: str,
    inplace: bool = False,
) -> rs.DataSet | None:
    if not inplace:
        dataset = dataset.copy()

    negative_amplitude_indices = dataset[amplitude_column] < 0.0
    dataset[amplitude_column] = np.abs(dataset[amplitude_column])
    dataset.loc[negative_amplitude_indices, phase_column] += 180.0

    dataset[phase_column] = canonicalize_phases(dataset[phase_column])

    if not inplace:
        return dataset
    return None


def average_phase_diff_in_degrees(
    array1: np.ndarray | rs.DataSeries, array2: np.ndarray | rs.DataSeries
) -> float:
    if isinstance(array1, rs.DataSeries) and isinstance(array2, rs.DataSeries):
        array1, array2 = filter_common_indices(array1, array2)

    if array1.shape != array2.shape:
        msg = f"inputs not same shape: {array1.shape} vs {array2.shape}"
        raise ShapeMismatchError(msg)

    phase1 = np.rad2deg(np.angle(array1))
    phase2 = np.rad2deg(np.angle(array2))

    diff = phase2 - phase1
    diff = (diff + 180) % 360 - 180

    return float(np.sum(np.abs(diff)) / float(np.prod(array1.shape)))


@cellify("cell")
@spacegroupify("spacegroup")
def numpy_array_to_map(
    array: np.ndarray,
    *,
    spacegroup: SpacegroupType,
    cell: CellType,
) -> gemmi.Ccp4Map:
    ccp4_map = gemmi.Ccp4Map()
    ccp4_map.grid = gemmi.FloatGrid(array.astype(np.float32))

    if isinstance(cell, gemmi.UnitCell):
        ccp4_map.grid.unit_cell = cell
    else:
        ccp4_map.grid.unit_cell.set(*cell)

    if not isinstance(spacegroup, gemmi.SpaceGroup):
        spacegroup = gemmi.SpaceGroup(spacegroup)
    ccp4_map.grid.spacegroup = spacegroup

    return ccp4_map
