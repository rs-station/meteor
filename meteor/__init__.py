""".. include:: ../README.md"""

__all__ = ["__version__", "version"]

from ._version import __version__, version
from meteor.scripts.compute_difference_map import compute_meteor_difference_map
from meteor.scripts.compute_iterative_tv_map import compute_iterative_difference_map
