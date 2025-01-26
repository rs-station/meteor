"""MTZ input/output helper functions"""

from __future__ import annotations

import re

from .settings import (
    COMPUTED_AMPLITUDE_COLUMNS,
    COMPUTED_PHASE_COLUMNS,
    OBSERVED_AMPLITUDE_COLUMNS,
    OBSERVED_INTENSITY_COLUMNS,
    OBSERVED_UNCERTAINTY_COLUMNS,
)


class AmbiguousMtzColumnError(ValueError): ...


def _infer_mtz_column(columns_to_search: list[str], columns_to_look_for: list[str]) -> str:
    # the next line consumes ["FOO", "BAR", "BAZ"] and produces regex strings like "^(FOO|BAR|BAZ)$"
    regex = re.compile(f"^({'|'.join(columns_to_look_for)})$")
    matches = [
        regex.match(column) for column in columns_to_search if regex.match(column) is not None
    ]

    if len(matches) == 0:
        msg = "cannot infer MTZ column name; "
        msg += f"cannot find any of {columns_to_look_for} in {columns_to_search}"
        raise AmbiguousMtzColumnError(msg)
    if len(matches) > 1:
        msg = "cannot infer MTZ column name; "
        msg += f">1 instance of {columns_to_look_for} in {columns_to_search}"
        raise AmbiguousMtzColumnError(msg)

    [match] = matches
    if match is None:
        msg = "`None` not filtered during regex matching"
        raise RuntimeError(msg)

    return match.group(0)


def find_observed_intensity_column(mtz_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_columns, OBSERVED_INTENSITY_COLUMNS)


def find_observed_amplitude_column(mtz_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_columns, OBSERVED_AMPLITUDE_COLUMNS)


def find_observed_uncertainty_column(mtz_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_columns, OBSERVED_UNCERTAINTY_COLUMNS)


def find_computed_amplitude_column(mtz_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_columns, COMPUTED_AMPLITUDE_COLUMNS)


def find_computed_phase_column(mtz_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_columns, COMPUTED_PHASE_COLUMNS)
