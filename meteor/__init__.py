""".. include:: ../README.md"""

__all__ = [
    "__version__",
    "compute_meteor_difference_map",
    "compute_meteor_phaseboost_map",
    "version",
]

from meteor.scripts.diffmap import compute_meteor_difference_map
from meteor.scripts.phaseboost import compute_meteor_phaseboost_map

from ._version import __version__, version
