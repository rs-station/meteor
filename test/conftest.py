# Copyright (c) 2024 Reciprocal Space Ship
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from meteor.testing import check_test_file_exists


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent.resolve() / "data"


@pytest.fixture(scope="session")
def testing_cif_file(data_dir: Path) -> Path:
    return data_dir / "8a6g-chromophore-removed.cif"


@pytest.fixture(scope="session")
def testing_pdb_file(data_dir: Path) -> Path:
    return data_dir / "8a6g-chromophore-removed.pdb"


@pytest.fixture(scope="session")
def testing_mtz_file(data_dir: Path) -> Path:
    path = data_dir / "scaled-test-data.mtz"
    check_test_file_exists(path)
    return path


@pytest.fixture(scope="session")
def np_rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)
