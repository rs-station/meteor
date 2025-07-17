from pathlib import Path

import numpy as np
import reciprocalspaceship as rs

from meteor import settings
from meteor.metadata import IterativeDiffmapMetadata
from meteor.rsmap import Map
from meteor.scripts import phaseboost
from meteor.testing import check_leaf_floats_are_finite
from meteor.utils import filter_common_indices

# faster tests
settings.MAP_SAMPLING = 1
settings.TV_MAX_NUM_ITER = 10


def test_script_produces_consistent_results(
    testing_pdb_file: Path,
    testing_mtz_file: Path,
    tmp_path: Path,
) -> None:
    output_mtz = tmp_path / "test-output.mtz"
    output_metadata = tmp_path / "test-output-metadata.csv"

    cli_args = [
        str(testing_mtz_file),  # derivative
        "--derivative-amplitude-column",
        "F_on",
        "--derivative-uncertainty-column",
        "SIGF_on",
        str(testing_mtz_file),  # native
        "--native-amplitude-column",
        "F_off",
        "--native-uncertainty-column",
        "SIGF_off",
        "--structure",
        str(testing_pdb_file),
        "-o",
        str(output_mtz),
        "-m",
        str(output_metadata),
        "-x",
        "0.01",
        "--max-iterations",
        "5",
    ]

    phaseboost.main(cli_args)

    with output_metadata.open("r") as f:
        metadata = IterativeDiffmapMetadata.model_validate_json(f.read())

    result_map = Map.read_mtz_file(output_mtz)

    # 1. make sure the negentropy increased during iterative TV
    negentropy_at_first_iter = metadata.iterative_tv_iterations[0].negentropy_after_tv
    negentropy_at_last_iter = metadata.iterative_tv_iterations[-1].negentropy_after_tv
    assert negentropy_at_last_iter > negentropy_at_first_iter

    # 2. make sure negentropy increased in the final TV pass
    assert metadata.final_tv_pass.optimal_negentropy >= metadata.final_tv_pass.initial_negentropy

    # 3. make sure computed DF are close to those stored on disk
    reference_dataset = rs.read_mtz(str(testing_mtz_file))
    reference_amplitudes = reference_dataset["F_itTV"]

    result_amplitudes, reference_amplitudes = filter_common_indices(
        result_map.amplitudes, reference_amplitudes
    )
    rho = np.corrcoef(result_amplitudes.to_numpy(), reference_amplitudes.to_numpy())[0, 1]
    assert rho > 0.95

    # 4. regression, make sure no NaNs creep into metadata
    issues = check_leaf_floats_are_finite(metadata)
    if len(issues) != 0:
        msg = f"non-finite values in metadata {issues}"
        raise ValueError(msg)
