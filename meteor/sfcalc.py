"""computing structure factors from models"""

from pathlib import Path

import gemmi

from .rsmap import Map


def gemmi_structure_to_calculated_map(
    structure: gemmi.Structure,
    *,
    high_resolution_limit: float,
    map_sampling: float = 3.0,
) -> Map:
    """Compute a calculated map from a gemmi structure.

    Parameters
    ----------
    structure : gemmi.Structure
        The input gemmi structure.
    high_resolution_limit : float
        The high resolution limit for the map.
    map_sampling : float, optional
        The map sampling rate, by default 3.0. It does not affect the size of the generated
        structure but can affect the quality of the generated map. We advise lower map_sampling
        when generating for very high resolutions. Otherwise leave as is.

    Returns
    -------
    Map
        The calculated map.
    """
    density_map = gemmi.DensityCalculatorX()
    density_map.d_min = high_resolution_limit

    # factor 1/2 for consistency between density generation and density/structure factor conversion
    # voxel_spacing ≈ d_min/(2*density_map.rate)) = d_min/map_sampling
    density_map.rate = map_sampling / 2

    density_map.grid.setup_from(structure)
    for i, _ in enumerate(structure):
        density_map.put_model_density_on_grid(structure[i])

    ccp4_map = gemmi.Ccp4Map()
    ccp4_map.grid = density_map.grid
    ccp4_map.update_ccp4_header()

    return Map.from_ccp4_map(ccp4_map, high_resolution_limit=high_resolution_limit)


def structure_file_to_calculated_map(
    cif_or_pdb_file: Path, *, high_resolution_limit: float, map_sampling: float = 3.0
) -> Map:
    if not cif_or_pdb_file.exists():
        msg = f"could not find file: {cif_or_pdb_file}"
        raise OSError(msg)
    structure = gemmi.read_structure(str(cif_or_pdb_file))
    return gemmi_structure_to_calculated_map(
        structure, high_resolution_limit=high_resolution_limit, map_sampling=map_sampling
    )
