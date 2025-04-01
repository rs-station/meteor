"""runtime settings for `meteor`"""

from __future__ import annotations

import numpy as np

# map computation
COMPUTED_MAP_RESOLUTION_LIMIT: float = 1.0
GEMMI_HIGH_RESOLUTION_BUFFER: float = 1e-6
MAP_SAMPLING: int = 3


# known map labels, see:
# https://www.ccp4.ac.uk/html/mtzformat.html
# https://www.globalphasing.com/buster/wiki/index.cgi?MTZcolumns

OBSERVED_INTENSITY_COLUMNS: list[str] = [
    "I",  # generic
    "IMEAN",  # CCP4
    "I-obs",  # phenix
    "IOBS",  # phenix
]
OBSERVED_AMPLITUDE_COLUMNS: list[str] = [
    "F",  # generic
    "FP",  # CCP4 & GLPh native
    "FW-F",  # reciprocalspaceship French-Wilson default
    r"FPH\d",  # CCP4 derivative
    "F-obs",  # phenix
    "FOBS",  # phenix
    "F-obs-filtered",  # phenix
]
OBSERVED_UNCERTAINTY_COLUMNS: list[str] = [
    "SIGF",  # generic
    "SIGFP",  # CCP4 & GLPh native
    "FW-SIGF",  # reciprocalspaceship French-Wilson default
    r"SIGFPH\d",  # CCP4 derivative
    "SIGF-obs",  # phenix
    "SIGFOBS",  # phenix
    "SIGF-obs-filtered",  # phenix
]
COMPUTED_AMPLITUDE_COLUMNS: list[str] = ["FC"]
COMPUTED_PHASE_COLUMNS: list[str] = ["PHIC"]


# k-weighting
KWEIGHT_PARAMETER_DEFAULT: float = 0.05
DEFAULT_KPARAMS_TO_SCAN = np.linspace(0.0, 1.0, 101)
K_PARAMETER_NAME: str = "k_parameter"


# tv denoising
TV_WEIGHT_DEFAULT: float = 0.01
BRACKET_FOR_GOLDEN_OPTIMIZATION: tuple[float, float] = (0.0, 0.01)  # the braket can expand
TV_MAX_WEIGHT_EXPECTED = 0.1  # this value sets the threshold for a warning to the user
TV_STOP_TOLERANCE: float = 0.00000005  # inner loop; not for iterative-tv phase retrieval
TV_MAX_NUM_ITER: int = 50  # inner loop; not for iterative-tv phase retrieval
TV_WEIGHT_PARAMETER_NAME: str = "tv_weight"

# iterative tv
ITERATIVE_TV_CONVERGENCE_TOLERANCE: float = 0.001
ITERATIVE_TV_MAX_ITERATIONS: int = 100
DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION: list[float] = [0.001, 0.01, 0.1]
