from __future__ import annotations

from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs

from .settings import GEMMI_HIGH_RESOLUTION_BUFFER
from .utils import (
    canonicalize_amplitudes,
    complex_array_to_rs_dataseries,
)


class MissingUncertaintiesError(AttributeError): ...


class MapMutabilityError(RuntimeError): ...


def _assert_is_map(obj: Any, *, require_uncertainties: bool) -> None:
    if not isinstance(obj, Map):
        msg = f"expected {obj} to be a rsmap.Map, got {type(obj)}"
        raise TypeError(msg)
    if require_uncertainties and (not obj.has_uncertainties):
        msg = f"{obj} Map missing required uncertainty column"
        raise MissingUncertaintiesError(msg)


class Map(rs.DataSet):
    """
    A high-level interface for a crystallographic map of any kind.

    Specifically, this class is based on a reciprocalspaceship `DataSet` (ie. a
    crystographically-aware pandas `DataFrame`), but is restricted to three and only three
    special columns corresponding to:

        - (real) `amplitudes`
        - `phases`
        - `uncertainties`, ie the standard deviation for a Gaussian around the amplitude mean

    In addition, the class maintains an `index` of Miller indices, as well as the crystallographic
    metadata supported by `rs.DataSet`, most notably a `cell` and `spacegroup`.

    These structured data enable this class to perform some routine map-based caluclations, such as

        - computing a real-space map, or computing map coefficients from a map
        - converting between complex Cartesian and polar (amplitude/phase) structure factors
        - reading and writing mtz and ccp4 map files

    all in a way that automatically facilitates the bookkeeping tasks normally associated with
    these relatively simple operations.
    """

    def __init__(
        self,
        data: dict | pd.DataFrame | rs.DataSet,
        *,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
        uncertainty_column: str | None = "SIGF",
        **kwargs,
    ) -> None:
        super().__init__(data=data, **kwargs)

        if not hasattr(self, "index"):
            msg = "required index not found, please either pass a `DataFrame` or `DataSet` with an "
            msg += "index of Miller indices, or pass `index=...` to `Map(...)` at instantiation"
            raise IndexError(msg)

        columns_to_keep = [amplitude_column, phase_column]
        for column in columns_to_keep:
            if column not in self.columns:
                msg = "amplitude and phase columns must be in input `data`... "
                msg += f"looking for {amplitude_column} and {phase_column}, got {self.columns}"
                raise KeyError(msg)

        self._amplitude_column = amplitude_column
        self._phase_column = phase_column
        self._uncertainty_column = uncertainty_column
        if uncertainty_column and (uncertainty_column in self.columns):
            columns_to_keep.append(uncertainty_column)

        # this feels dangerous, but I cannot find a better way | author: @tjlane
        excess_columns = set(self.columns) - set(columns_to_keep)
        for column in excess_columns:
            del self[column]

        # ensure types correct
        self.amplitudes = self._verify_amplitude_type(self.amplitudes, fix=True)
        self.phases = self._verify_phase_type(self.phases, fix=True)
        if self.has_uncertainties:
            self.uncertainties = self._verify_uncertainty_type(self.uncertainties, fix=True)

    @property
    def _constructor(self):
        return Map

    @property
    def _constructor_sliced(self):
        return rs.DataSeries

    def _verify_type(
        self,
        name: str,
        allowed_types: list[type],
        dataseries: rs.DataSeries,
        *,
        fix: bool,
        cast_fix_to: type,
    ) -> rs.DataSeries:
        if dataseries.dtype not in allowed_types:
            if fix:
                return dataseries.astype(cast_fix_to)
            msg = f"dtype for passed {name} not allowed, got: {dataseries.dtype} allow {allowed_types}"
            raise AssertionError(msg)
        return dataseries

    def _verify_amplitude_type(
        self,
        dataseries: rs.DataSeries,
        *,
        fix: bool = True,
    ) -> rs.DataSeries:
        name = "amplitude"
        amplitude_dtypes = [
            rs.StructureFactorAmplitudeDtype(),
            rs.FriedelStructureFactorAmplitudeDtype(),
            rs.NormalizedStructureFactorAmplitudeDtype(),
            rs.AnomalousDifferenceDtype(),
        ]
        return self._verify_type(
            name,
            amplitude_dtypes,
            dataseries,
            fix=fix,
            cast_fix_to=rs.StructureFactorAmplitudeDtype(),
        )

    def _verify_phase_type(self, dataseries: rs.DataSeries, *, fix: bool = True) -> rs.DataSeries:
        name = "phase"
        phase_dtypes = [rs.PhaseDtype()]
        return self._verify_type(
            name, phase_dtypes, dataseries, fix=fix, cast_fix_to=rs.PhaseDtype()
        )

    def _verify_uncertainty_type(
        self,
        dataseries: rs.DataSeries,
        *,
        fix: bool = True,
    ) -> rs.DataSeries:
        name = "uncertainties"
        uncertainty_dtypes = [
            rs.StandardDeviationDtype(),
            rs.StandardDeviationFriedelIDtype(),
            rs.StandardDeviationFriedelSFDtype(),
        ]
        return self._verify_type(
            name, uncertainty_dtypes, dataseries, fix=fix, cast_fix_to=rs.StandardDeviationDtype()
        )

    def __setitem__(self, key: str, value) -> None:
        if key not in self.columns:
            msg = "column assignment not allowed for Map objects"
            raise MapMutabilityError(msg)
        super().__setitem__(key, value)

    def insert(self, loc: int, column: str, value: Any, *, allow_duplicates: bool = False) -> None:
        allowed_columns = ["H", "K", "L", "dHKL"]
        if column in allowed_columns:
            super().insert(loc, column, value, allow_duplicates=allow_duplicates)
        else:
            msg = "general column assignment not allowed for Map objects"
            msg += f"special columns allowed: {allowed_columns}; "
            msg += "see also Map.set_uncertainties(...)"
            raise MapMutabilityError(msg)

    def drop(self, labels=None, *, axis=0, columns=None, inplace=False, **kwargs):
        if (axis == 1) or (columns is not None):
            msg = "columns are fixed for Map objects"
            raise MapMutabilityError(msg)
        return super().drop(labels=labels, axis=axis, columns=columns, inplace=inplace, **kwargs)

    def get_hkls(self):
        # TODO: audit this
        return self.index.to_frame().to_numpy(dtype=np.int32)

    def compute_dHKL(self) -> rs.DataSeries:  # noqa: N802, inhereted from reciprocalspaceship
        # TODO: audit this
        d_hkl = self.cell.calculate_d_array(self.get_hkls())
        return rs.DataSeries(d_hkl, dtype="R", index=self.index)

    @property
    def resolution_limits(self) -> tuple[float, float]:
        d_hkl = self.compute_dHKL()
        return np.max(d_hkl), np.min(d_hkl)

    @property
    def amplitudes(self) -> rs.DataSeries:
        return self[self._amplitude_column]

    @amplitudes.setter
    def amplitudes(self, values: rs.DataSeries) -> None:
        values = self._verify_amplitude_type(values)
        self[self._amplitude_column] = values

    @property
    def phases(self) -> rs.DataSeries:
        return self[self._phase_column]

    @phases.setter
    def phases(self, values: rs.DataSeries) -> None:
        values = self._verify_phase_type(values)
        self[self._phase_column] = values

    @property
    def has_uncertainties(self) -> bool:
        if self._uncertainty_column is None:
            return False
        return self._uncertainty_column in self.columns

    @property
    def uncertainties(self) -> rs.DataSeries:
        if self.has_uncertainties:
            return self[self._uncertainty_column]
        msg = "uncertainties not set for Map object"
        raise AttributeError(msg)

    @uncertainties.setter
    def uncertainties(self, values: rs.DataSeries) -> None:
        if self.has_uncertainties:
            values = self._verify_uncertainty_type(values)
            self[self._uncertainty_column] = values  # type: ignore[index]
        else:
            msg = "uncertainties unset, and Pandas forbids assignment via attributes; "
            msg += "to initialize, use Map.set_uncertainties(...)"
            raise AttributeError(msg)

    def set_uncertainties(self, values: rs.DataSeries, column_name: str = "SIGF") -> None:
        values = self._verify_uncertainty_type(values)

        if self.has_uncertainties:
            self.uncertainties = values
        else:
            # otherwise, create a new column
            self._uncertainty_column = column_name
            position = len(self.columns)
            if position != 2:  # noqa: PLR2004, should be 2: just amplitudes & phases
                msg = "Misconfigured columns"
                raise RuntimeError(msg)
            super().insert(position, self._uncertainty_column, values, allow_duplicates=False)

    @property
    def complex_amplitudes(self) -> np.ndarray:
        return self.amplitudes.to_numpy() * np.exp(1j * np.deg2rad(self.phases.to_numpy()))

    def canonicalize_amplitudes(self):
        canonicalize_amplitudes(
            self,
            amplitude_label=self._amplitude_column,
            phase_label=self._phase_column,
            inplace=True,
        )

    def to_structurefactor(self) -> rs.DataSeries:
        return super().to_structurefactor(self._amplitude_column, self._phase_column)

    # dev note: `rs.DataSet.from_structurefactor` exists, but it operates on a column that's already
    # part of the dataset; having such a (redundant) column is forbidden by `Map` - @tjlane
    @classmethod
    def from_structurefactor(
        cls,
        complex_structurefactor: np.ndarray | rs.DataSeries,
        *,
        index: pd.Index,
        cell: Any = None,
        spacegroup: Any = None,
    ) -> Map:
        # recprocalspaceship has a `from_structurefactor` method, but it is occasionally
        # mangling indices for me when the input is a numpy array, as of 16 OCT 24 - @tjlane
        amplitudes, phases = complex_array_to_rs_dataseries(complex_structurefactor, index=index)
        dataset = rs.DataSet(
            rs.concat([amplitudes.rename("F"), phases.rename("PHI")], axis=1),
            index=index,
            cell=cell,
            spacegroup=spacegroup,
        )
        return cls(dataset)

    def to_gemmi(self) -> rs.DataSet:
        # the parent DataSet.to_gemmi() modifies columns, so we need to cast to DataSet - @tjlane
        # TODO: can you more intelligently cast to parent?
        return rs.DataSet(self).to_gemmi()

    @classmethod
    def from_gemmi(
        cls,
        gemmi_mtz: gemmi.Mtz,
        *,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
        uncertainty_column: str | None = "SIGF",
    ) -> Map:
        return cls(
            rs.DataSet(gemmi_mtz),
            amplitude_column=amplitude_column,
            phase_column=phase_column,
            uncertainty_column=uncertainty_column,
        )

    def to_ccp4_map(self, *, map_sampling: int) -> gemmi.Ccp4Map:
        map_coefficients_gemmi_format = self.to_gemmi()
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = map_coefficients_gemmi_format.transform_f_phi_to_map(
            self._amplitude_column,
            self._phase_column,
            sample_rate=map_sampling,
        )
        ccp4_map.update_ccp4_header()
        return ccp4_map

    @classmethod
    def from_ccp4_map(
        cls,
        ccp4_map: gemmi.Ccp4Map,
        *,
        high_resolution_limit: float,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
    ) -> Map:
        # to ensure we include the final shell of reflections, add a small buffer to the resolution
        gemmi_structure_factors = gemmi.transform_map_to_f_phi(ccp4_map.grid, half_l=False)
        data = gemmi_structure_factors.prepare_asu_data(
            dmin=high_resolution_limit - GEMMI_HIGH_RESOLUTION_BUFFER,
            with_sys_abs=True,
        )

        mtz = gemmi.Mtz(with_base=True)
        mtz.spacegroup = gemmi_structure_factors.spacegroup
        mtz.set_cell_for_all(gemmi_structure_factors.unit_cell)
        mtz.add_dataset("FromMap")

        mtz.add_column(amplitude_column, "F")
        mtz.add_column(phase_column, "P")

        mtz.set_data(data)
        mtz.switch_to_asu_hkl()
        dataset = super().from_gemmi(mtz)

        return cls(dataset, amplitude_column=amplitude_column, phase_column=phase_column)

    def write_mtz(self, file_path: str | Path) -> None:
        # TODO: can you more intelligently cast to parent?
        # dev note: also modifies the columns; requires cast to DataSet - @tjlane
        rs.DataSet(self).write_mtz(str(file_path))

    @classmethod
    def read_mtz_file(
        cls,
        file_path: str | Path,
        *,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
        uncertainty_column: str | None = "SIGF",
    ) -> Map:
        gemmi_mtz = gemmi.read_mtz_file(str(file_path))
        return cls.from_gemmi(
            gemmi_mtz,
            amplitude_column=amplitude_column,
            phase_column=phase_column,
            uncertainty_column=uncertainty_column,
        )
