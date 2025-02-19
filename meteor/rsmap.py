"""Map class definition and related functions"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, overload

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
from reciprocalspaceship.decorators import cellify, spacegroupify

from .settings import GEMMI_HIGH_RESOLUTION_BUFFER
from .utils import (
    CellType,
    ShapeMismatchError,
    SpacegroupType,
    canonicalize_amplitudes,
    numpy_array_to_map,
)

NUMBER_OF_DIMENSIONS_IN_UNIVERSE: Final[int] = 3


class MissingUncertaintiesError(AttributeError): ...


class MapMutabilityError(RuntimeError): ...


def assert_is_map(obj: Any, *, require_uncertainties: bool) -> None:
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

    # these columns are always allowed
    _allowed_columns: ClassVar[list[str]] = ["H", "K", "L"]

    # in addition, __init__ specifies 3 columns special that can be named dynamically to support:
    # amplitudes, phases, uncertainties; all other columns are forbidden

    @cellify
    @spacegroupify
    def __init__(
        self,
        data: dict | pd.DataFrame | rs.DataSet,
        *,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
        uncertainty_column: str | None = "SIGF",
        **kwargs: Any,
    ) -> None:
        super().__init__(data=data, **kwargs)

        self._amplitude_column = amplitude_column
        self._phase_column = phase_column
        self._uncertainty_column = uncertainty_column

        for column in [self._amplitude_column, self._phase_column]:
            if column not in self.columns:
                msg = "amplitude and phase columns must be in input `data`... "
                msg += f"looking for `{column}`, found `{self.columns}`"
                raise KeyError(msg)

        columns_to_keep = [*self._allowed_columns, amplitude_column, phase_column]
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
    def _constructor(self) -> Callable[[Any], Map]:
        def constructor_fxn(*args: Any, **kwargs: Any) -> Map:
            return Map(
                *args,
                amplitude_column=self._amplitude_column,
                phase_column=self._phase_column,
                uncertainty_column=self._uncertainty_column,
                **kwargs,
            )

        return constructor_fxn

    @property
    def _constructor_sliced(self) -> Callable[[Any], rs.DataSeries]:
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

    def __setitem__(self, key: str, value: Any) -> None:
        allowed = list(self.columns) + self._allowed_columns
        if key not in allowed:
            msg = "column assignment not allowed for Map objects"
            raise MapMutabilityError(msg)
        super().__setitem__(key, value)

    def insert(self, loc: int, column: str, value: Any, *, allow_duplicates: bool = False) -> None:
        if column in self._allowed_columns:
            super().insert(loc, column, value, allow_duplicates=allow_duplicates)
        else:
            msg = "general column assignment not allowed for Map objects"
            msg += f"special columns allowed: {self._allowed_columns}; "
            msg += "see also Map.set_uncertainties(...)"
            raise MapMutabilityError(msg)

    @overload
    def drop(self, labels: Any, *, inplace: Literal[True]) -> None: ...

    @overload
    def drop(self, labels: Any, *, inplace: Literal[False]) -> Map: ...

    def drop(self, labels: Any, *, inplace: bool = False) -> None | Map:
        return super().drop(labels=labels, axis="index", columns=None, inplace=inplace)

    def get_hkls(self) -> np.ndarray:
        # overwrite rs implt'n, return w/o modifying self -> same behavior, under testing - @tjlane
        # this is a rather horrible thing to do and we should fix it
        # best is to push changes upstream
        hkl_names = ["H", "K", "L"]
        if self.index.names == hkl_names:
            hkls = self.index.to_frame().to_numpy(dtype=np.int32)
        elif all(col in self.columns for col in hkl_names):
            # we need to pull out each column as a separate DataSeries so that we don't try to
            # create a new Map object without F, PHI
            hkls = np.vstack([self[col].to_numpy(dtype=np.int32) for col in hkl_names]).T
        else:
            msg = f"cannot find `H`, `K`, and `L` columns in index or columns {self.columns}"
            raise ValueError(msg)

        if hkls.shape[-1] != NUMBER_OF_DIMENSIONS_IN_UNIVERSE:
            msg = f"something went wrong, HKL array has a funny shape: {hkls.shape}"
            raise RuntimeError(msg)

        return hkls

    def compute_dHKL(self) -> rs.DataSeries:  # noqa: N802, caps from reciprocalspaceship
        # rs adds a "dHKL" column to the DataFrame
        # that could be enabled by adding "dHKL" to _allowed_columns - @tjlane
        if not hasattr(self, "cell"):
            msg = "no `cell` attribute set, cannot compute resolution (d-values)"
            raise AttributeError(msg)
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
    def amplitude_column_name(self) -> str:
        return self._amplitude_column

    @property
    def phases(self) -> rs.DataSeries:
        return self[self._phase_column]

    @phases.setter
    def phases(self, values: rs.DataSeries) -> None:
        values = self._verify_phase_type(values)
        self[self._phase_column] = values

    @property
    def phase_column_name(self) -> str:
        return self._phase_column

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
            number_of_columns = len(self.columns)
            number_of_columns_with_just_amplitudes_and_phases = 2
            if number_of_columns != number_of_columns_with_just_amplitudes_and_phases:
                msg = "Misconfigured columns"
                raise RuntimeError(msg)
            super().insert(
                number_of_columns, self._uncertainty_column, values, allow_duplicates=False
            )

    @property
    def uncertainties_column_name(self) -> str:
        if self.has_uncertainties:
            if not isinstance(self._uncertainty_column, str):
                msg = "misconfigured uncertainty column"
                raise RuntimeError(msg)
            return self._uncertainty_column
        msg = "uncertainties not set for Map object"
        raise AttributeError(msg)

    def canonicalize_amplitudes(self) -> None:
        canonicalize_amplitudes(
            self,
            amplitude_column=self._amplitude_column,
            phase_column=self._phase_column,
            inplace=True,
        )

    def to_structurefactor(self) -> rs.DataSeries:
        """Return a DataSeries of complex structure factor amplitudes"""
        return super().to_structurefactor(self._amplitude_column, self._phase_column)

    @overload
    @classmethod
    def from_structurefactor(
        cls,
        complex_structurefactor: rs.DataSeries,
        *,
        index: None = None,
        cell: CellType | None = None,
        spacegroup: SpacegroupType | None = None,
    ) -> Map: ...

    @overload
    @classmethod
    def from_structurefactor(
        cls,
        complex_structurefactor: np.ndarray,
        *,
        index: pd.Index,
        cell: CellType | None = None,
        spacegroup: SpacegroupType | None = None,
    ) -> Map: ...

    @classmethod
    @cellify("cell")
    @spacegroupify("spacegroup")
    def from_structurefactor(
        cls,
        complex_structurefactor: np.ndarray | rs.DataSeries,
        *,
        index: pd.Index | None = None,
        cell: CellType | None = None,
        spacegroup: SpacegroupType | None = None,
    ) -> Map:
        # 1. `rs.DataSet.from_structurefactor` exists, but it operates on a column that's already
        #    part of the dataset; having such a (redundant) column is forbidden by `Map`
        # 2. recprocalspaceship has a `from_structurefactor` method, but it is occasionally
        #    mangling indices for me when the input is a numpy array, as of 16 OCT 24
        #
        # hopefully we can resolve these and reuse code! - @tjlane

        if index is None:
            if isinstance(complex_structurefactor, rs.DataSeries) and hasattr(
                complex_structurefactor, "index"
            ):
                index = complex_structurefactor.index

            else:
                msg = "if `complex_structurefactor` is not a `DataSeries` with an `index` attribute"
                msg += ", and index must be provided"
                raise ValueError(msg)

        elif index.shape != complex_structurefactor.shape:
            msg = f"`complex_structurefactor` {complex_structurefactor.shape} does not have same "
            msg += f"shape as `index` {index.shape}"
            raise ShapeMismatchError(msg)

        amplitudes = rs.DataSeries(
            np.abs(complex_structurefactor),
            index=index,
            dtype=rs.StructureFactorAmplitudeDtype(),
            name="F",
        )

        phases = rs.DataSeries(
            np.angle(complex_structurefactor, deg=True),
            index=index,
            dtype=rs.PhaseDtype(),
            name="PHI",
        )

        dataset = rs.DataSet(
            rs.concat([amplitudes, phases], axis=1),
            index=index,
            cell=cell,
            spacegroup=spacegroup,
        )

        return cls(dataset)

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

    def to_3d_numpy_map(self, *, map_sampling: int) -> np.ndarray:
        realspace_map = self.to_ccp4_map(map_sampling=map_sampling)
        return np.array(realspace_map.grid)

    @classmethod
    @cellify("cell")
    def from_3d_numpy_map(
        cls, map_grid: np.ndarray, *, spacegroup: Any, cell: CellType, high_resolution_limit: float
    ) -> Map:
        """
        Create a `Map` from a 3d grid of voxel values stored in a numpy array.

        Parameters
        ----------
        map_grid: np.ndarray
            The array, laid out in Gemmi format
        spacegroup: Any
            Specifies which spacegroup, can be an int, gemmi.SpaceGroup, ...
        cell
            Specifies cell, can be a tuple, gemmi.Cell, ...
        high_resolution_limit: float
            The resolution of the map, irregardless of the sampling; we need this to infer the map
            sampling

        Returns
        -------
        map: Map
            The map coefficients

        See Also
        --------
        For information about Gemmi data layout: https://gemmi.readthedocs.io/en/latest/grid.html
        """
        if len(map_grid.shape) != NUMBER_OF_DIMENSIONS_IN_UNIVERSE:
            msg = "`map_grid` should be a 3D array representing a realspace map"
            raise ValueError(msg)
        ccp4 = numpy_array_to_map(
            map_grid,
            spacegroup=spacegroup,
            cell=cell,
        )
        return cls.from_ccp4_map(
            ccp4_map=ccp4,
            high_resolution_limit=high_resolution_limit,
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
        path_cast_to_str = str(file_path)
        super().write_mtz(path_cast_to_str)

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
