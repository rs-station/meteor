from __future__ import annotations

from typing import TYPE_CHECKING, Type, Any

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs

from .utils import (
    canonicalize_amplitudes,
    complex_array_to_rs_dataseries,
    rs_dataseries_to_complex_array,
)

if TYPE_CHECKING:
    from pathlib import Path


GEMMI_HIGH_RESOLUTION_BUFFER = 1e-6

CellType = Type[tuple[float, float, float, float, float, float] | gemmi.UnitCell]
SpaceGroupType = Type[int | str | gemmi.SpaceGroup]


class MissingUncertaintiesError(AttributeError): ...


def _assert_is_map(obj: Any, require_uncertainties: bool) -> None:
    if not isinstance(obj, Map):
        msg = f"expected {obj} to be a rsmap.Map, got {type(obj)}"
        raise TypeError(msg)
    if require_uncertainties:
        if not obj.has_uncertainties:
            msg = f"{obj} Map missing required uncertainty column"
            raise MissingUncertaintiesError(msg)



# TODO: docstring for this class
# TODO: audit __init__ in light of https://github.com/rs-station/reciprocalspaceship/blob/main/reciprocalspaceship/dataset.py
class Map(rs.DataSet):

    # TODO: these can get out of sync with the column names if the columns are renamed
    # and what are the consequences of "class" variables like this?
    _amplitude_column = "F"
    _phase_column = "PHI"
    _uncertainty_column = "SIGF"

    def __init__(
        self,
        dataset: rs.DataSet,
        *,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
        uncertainty_column: str | None = None,
        **kwargs,
    ) -> None:
        # add to dataset
        labels = [amplitude_column, phase_column]
        if uncertainty_column:
            labels.append(uncertainty_column)
        sub_dataset = dataset[labels].copy()
        super().__init__(sub_dataset, **kwargs)

        # rename columns
        self[amplitude_column].rename(self._amplitude_column, inplace=True)
        self[phase_column].rename(self._phase_column, inplace=True)
        if uncertainty_column:
            self[uncertainty_column].rename(self._uncertainty_column, inplace=True)

        # ensure types correct
        self._assert_amplitude_type(self[self._amplitude_column])
        self._assert_phase_type(self[self._phase_column])
        if self.has_uncertainties:
            self._assert_uncertainty_type(self[self._uncertainty_column])

        # touchups
        canonicalize_amplitudes(
            self,
            amplitude_label=self._amplitude_column,
            phase_label=self._phase_column,
            inplace=True,
        )

    @property
    def _constructor(self):
        return Map

    @property
    def _constructor_sliced(self):
        return rs.DataSeries

    def _assert_amplitude_type(self, dataseries: rs.DataSeries) -> None:
        amplitude_dtypes = [
            rs.StructureFactorAmplitudeDtype(),
            rs.FriedelStructureFactorAmplitudeDtype(),
            rs.NormalizedStructureFactorAmplitudeDtype(),
            rs.AnomalousDifferenceDtype(),
        ]
        if dataseries.dtype not in amplitude_dtypes:
            msg = f"amplitude dtype not allowed, got: {dataseries.dtype} allow {amplitude_dtypes}"
            raise AssertionError(msg)

    def _assert_phase_type(self, dataseries: rs.DataSeries) -> None:
        allowed_phase_dtypes = [rs.PhaseDtype()]
        if dataseries.dtype not in allowed_phase_dtypes:
            msg = f"phase dtype not allowed, got: {dataseries.dtype} allow {allowed_phase_dtypes}"
            raise AssertionError(msg)

    def _assert_uncertainty_type(self, dataseries: rs.DataSeries) -> None:
        uncertainty_dtypes = [
            rs.StandardDeviationDtype(),
            rs.StandardDeviationFriedelIDtype(),
            rs.StandardDeviationFriedelSFDtype(),
        ]
        if dataseries.dtype not in uncertainty_dtypes:
            msg = f"phase dtype not allowed, got: {dataseries.dtype} allow {uncertainty_dtypes}"
            raise AssertionError(msg)

    def __setitem__(self, key: str, value) -> None:
        if key not in self.columns:
            msg = "column assignment not allowed for Map objects"
            raise KeyError(msg)
        super().__setitem__(key, value)

    def insert(self, *args, **kwargs) -> None:  # noqa: ARG002
        msg = "column assignment not allowed for Map objects"
        raise NotImplementedError(msg)

    @property
    def amplitudes(self) -> rs.DataSeries:
        return self[self._amplitude_column]

    @amplitudes.setter
    def amplitudes(self, values: rs.DataSeries) -> None:
        self._assert_amplitude_type(values)
        self[self._amplitude_column] = values

    @property
    def phases(self) -> rs.DataSeries:
        return self[self._phase_column]

    @phases.setter
    def phases(self, values: rs.DataSeries) -> None:
        self._assert_phase_type(values)
        self[self._phase_column] = values

    @property
    def has_uncertainties(self) -> bool:
        return self._uncertainty_column in self.columns

    @property
    def uncertainties(self) -> rs.DataSeries:
        if self.has_uncertainties:
            return self[self._uncertainty_column]
        msg = "uncertainties not set for Map object"
        raise AttributeError(msg)

    @uncertainties.setter
    def uncertainties(self, values: rs.DataSeries) -> None:
        self._assert_uncertainty_type(values)
        if self.has_uncertainties:
            self[self._uncertainty_column] = values
        else:
            position = len(self.columns)
            if position != 2:  # noqa: PLR2004, should be 2: just amplitudes & phases
                msg = "Misconfigured columns"
                raise RuntimeError(msg)
            super().insert(position, self._uncertainty_column, values, allow_duplicates=False)

    @property
    def complex_amplitudes(self) -> np.ndarray:
        return rs_dataseries_to_complex_array(amplitudes=self.amplitudes, phases=self.phases)

    def to_gemmi(self) -> rs.DataSet:
        # the parent DataSet.to_gemmi() modifies columns, so we need to cast to DataSet
        return rs.DataSet(self).to_gemmi()

    def to_structurefactor(self) -> rs.DataSeries:
        return super().to_structurefactor(self._amplitude_column, self._phase_column)

    def to_ccp4_map(self, *, map_sampling: int) -> gemmi.Ccp4Map:
        map_coefficients_gemmi_format = self.to_gemmi()
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = map_coefficients_gemmi_format.transform_f_phi_to_map(
            self._amplitude_column, self._phase_column, sample_rate=map_sampling
        )
        ccp4_map.update_ccp4_header()
        return ccp4_map

    @classmethod
    def from_structurefactor(
        cls,
        complex_structurefactor: np.ndarray | rs.DataSeries,
        *,
        index: pd.Index | None = None,
        cell: CellType | None = None,
        spacegroup: SpaceGroupType | None = None,
    ) -> Map:
        if isinstance(complex_structurefactor, np.ndarray):
            if not isinstance(index, pd.Index):
                msg = "if `complex_structurefactor` is a numpy array, `index` must be provided"
                raise TypeError(msg)
            amplitudes, phases = complex_array_to_rs_dataseries(
                complex_structure_factors=complex_structurefactor, index=index
            )

        elif isinstance(complex_structurefactor, rs.DataSeries):
            amplitudes, phases = super().from_structurefactor(complex_structurefactor)

        else:
            msg = f"`complex_structurefactor` invalid type: {type(complex_structurefactor)}"
            raise TypeError(msg)

        dataset = rs.concat(
            [amplitudes.rename(cls._amplitude_column), phases.rename(cls._phase_column)], axis=1
        )
        dataset.cell = cell
        dataset.spacegroup = spacegroup

        return cls(dataset, amplitude_column=cls._amplitude_column, phase_column=cls._phase_column)

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
            dmin=high_resolution_limit - GEMMI_HIGH_RESOLUTION_BUFFER, with_sys_abs=True
        )

        mtz = gemmi.Mtz(with_base=True)
        mtz.spacegroup = gemmi_structure_factors.spacegroup
        mtz.set_cell_for_all(gemmi_structure_factors.unit_cell)
        mtz.add_dataset("FromMap")

        mtz.add_column(amplitude_column, "F")
        mtz.add_column(phase_column, "P")

        mtz.set_data(data)
        mtz.switch_to_asu_hkl()
        dataset = rs.DataSet.from_gemmi(mtz)

        return cls(dataset, amplitude_column=amplitude_column, phase_column=phase_column)

    @classmethod
    def from_mtz_file(
        cls,
        file_path: Path,
        *,
        amplitude_column: str,
        phase_column: str,
        uncertainty_column: str | None = None,
    ) -> Map:
        dataset = super().from_mtz_file(file_path)
        return cls(
            dataset,
            amplitude_column=amplitude_column,
            phase_column=phase_column,
            uncertainty_column=uncertainty_column,
        )

    @classmethod
    def from_dict(cls, data, *, index=None, cell=None, spacegroup=None) -> Map:
        dataset = rs.DataSet(data=data, index=index, cell=cell, spacegroup=spacegroup)

        for required_column in [cls._amplitude_column, cls._phase_column]:
            if required_column not in dataset.columns:
                msg = f"cannot find required key {required_column} in input dict"
                raise KeyError(msg)

        dataset[cls._amplitude_column] = dataset[cls._amplitude_column].astype(
            rs.StructureFactorAmplitudeDtype()
        )
        dataset[cls._phase_column] = dataset[cls._phase_column].astype(rs.PhaseDtype())

        if cls._uncertainty_column in dataset.columns:
            dataset[cls._uncertainty_column] = dataset[cls._uncertainty_column].astype(
                rs.StandardDeviationDtype()
            )
            return cls(dataset, uncertainty_column=cls._uncertainty_column)

        return cls(dataset)

    def from_records(self, *args, **kwargs) -> None:
        raise NotImplementedError
