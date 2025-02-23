from __future__ import annotations

from collections.abc import Callable

import pytest

from meteor import mtzio

FIND_column_FUNC_TYPE = Callable[[list[str]], str]


OBSERVED_INTENSITY_CASES = [
    ([], "raise"),
    (["F"], "raise"),
    (["I", "F"], "I"),
    (["IMEAN", "F"], "IMEAN"),
    (["I-obs", "F"], "I-obs"),
    (["I", "IMEAN"], "raise"),
]


OBSERVED_AMPLITUDE_CASES = [
    ([], "raise"),
    (["IMEAN"], "raise"),
    (["I", "F"], "F"),
    (["IMEAN", "FP"], "FP"),
    (["IMEAN", "FW-F"], "FW-F"),
    (["I-obs", "FPH0"], "FPH0"),
    (["I-obs", "FPH5"], "FPH5"),
    (["I-obs", "FPH51"], "raise"),
    (["FP", "FPH1"], "raise"),
]


OBSERVED_UNCERTAINTY_CASES = [
    ([], "raise"),
    (["F"], "raise"),
    (["SIGF", "F"], "SIGF"),
    (["FW-SIGF", "F"], "FW-SIGF"),
    (["SIGFP", "F"], "SIGFP"),
    (["I-obs", "SIGFPH0"], "SIGFPH0"),
    (["I-obs", "SIGFPH1"], "SIGFPH1"),
    (["I-obs", "SIGFPH10"], "raise"),
    (["SIGFPH1", "SIGFPH2"], "raise"),
]


COMPUTED_AMPLITUDE_CASES = [
    ([], "raise"),
    (["F"], "raise"),
    (["I", "F"], "raise"),
    (["FC", "F"], "FC"),
    (["FC", "FC"], "raise"),
]


COMPUTED_PHASE_CASES = [
    ([], "raise"),
    (["F"], "raise"),
    (["PHIC", "F"], "PHIC"),
    (["PHIC", "PHIC"], "raise"),
]


def test_infer_mtz_column() -> None:
    to_search = ["FOO", "BAR", "BAZ"]
    assert mtzio._infer_mtz_column(to_search, ["FOO"]) == "FOO"
    assert mtzio._infer_mtz_column(to_search, ["BAR"]) == "BAR"
    with pytest.raises(mtzio.AmbiguousMtzColumnError):
        _ = mtzio._infer_mtz_column(to_search, [])
    with pytest.raises(mtzio.AmbiguousMtzColumnError):
        _ = mtzio._infer_mtz_column(to_search, ["FOO", "BAR"])


def validate_find_column_result(
    function: FIND_column_FUNC_TYPE, columns: list[str], expected_result: str
) -> None:
    if expected_result == "raise":
        with pytest.raises(mtzio.AmbiguousMtzColumnError):
            _ = function(columns)
    else:
        assert function(columns) == expected_result


@pytest.mark.parametrize(("columns", "expected_result"), OBSERVED_INTENSITY_CASES)
def test_find_observed_intensity_column(columns: list[str], expected_result: str) -> None:
    validate_find_column_result(mtzio.find_observed_intensity_column, columns, expected_result)


@pytest.mark.parametrize(("columns", "expected_result"), OBSERVED_AMPLITUDE_CASES)
def test_find_observed_amplitude_column(columns: list[str], expected_result: str) -> None:
    validate_find_column_result(mtzio.find_observed_amplitude_column, columns, expected_result)


@pytest.mark.parametrize(("columns", "expected_result"), OBSERVED_UNCERTAINTY_CASES)
def test_find_observed_uncertainty_column(columns: list[str], expected_result: str) -> None:
    validate_find_column_result(mtzio.find_observed_uncertainty_column, columns, expected_result)


@pytest.mark.parametrize(("columns", "expected_result"), COMPUTED_AMPLITUDE_CASES)
def test_find_computed_amplitude_column(columns: list[str], expected_result: str) -> None:
    validate_find_column_result(mtzio.find_computed_amplitude_column, columns, expected_result)


@pytest.mark.parametrize(("columns", "expected_result"), COMPUTED_PHASE_CASES)
def test_find_computed_phase_column(columns: list[str], expected_result: str) -> None:
    validate_find_column_result(mtzio.find_computed_phase_column, columns, expected_result)
