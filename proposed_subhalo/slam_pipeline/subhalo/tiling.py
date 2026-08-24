"""
Subhalo Detection: Image-Plane Tiling
=====================================

The existing ``detect.subhalo_grid_search`` runs an ``af.SearchGridSearch``, which
evaluates every grid cell **inside a single process / single job**. For large
subhalo scans on a cluster it is faster and more robust to split the image plane
into a 2D grid of tiles and farm each tile out to an **independent job** (one
HTCondor job, one Nautilus fit, one tile).

This module mirrors the model-building logic of
``slam_pipeline.subhalo.detect.subhalo_grid_search`` exactly, but replaces the
grid search with a single ``af.Nautilus`` fit whose subhalo (y, x) centre is
confined to ONE tile via uniform priors.

Coordinate convention (PyAutoLens):
    centre_0 = image-plane **y** coordinate  (vertical, row)
    centre_1 = image-plane **x** coordinate  (horizontal, col)

Layout / row-major order used here:
    for iy in range(n):          # y (centre_0), top -> bottom
        for ix in range(n):      # x (centre_1), left -> right
            index = iy * n + ix
"""

from typing import Dict, List

import numpy as np

import autofit as af
import autolens as al


def make_tiles(
    grid_dimension_arcsec: float = 3.0,
    number_of_tiles: int = 5,
) -> List[Dict[str, float]]:
    """
    Split the square search region ``[-grid_dimension_arcsec, +grid_dimension_arcsec]``
    (in arcseconds) into a ``number_of_tiles x number_of_tiles`` grid of tiles.

    Parameters
    ----------
    grid_dimension_arcsec
        Half-side length of the square search region centred on (0, 0)".
    number_of_tiles
        Number of tiles per side. Total jobs = number_of_tiles**2.

    Returns
    -------
    list of dict, one per tile, each with:
        index        : 0 .. (n*n - 1)            (row-major, y outer, x inner)
        iy, ix       : integer tile coordinates
        y0, y1       : centre_0 (y) prior bounds -> lower/upper (arcsec)
        x0, x1       : centre_1 (x) prior bounds -> lower/upper (arcsec)
    """
    n = int(number_of_tiles)
    if n < 1:
        raise ValueError("number_of_tiles must be >= 1")

    edges = np.linspace(-float(grid_dimension_arcsec), float(grid_dimension_arcsec), n + 1)

    tiles: List[Dict[str, float]] = []
    for iy in range(n):  # y (centre_0)
        for ix in range(n):  # x (centre_1)
            tiles.append(
                {
                    "index": iy * n + ix,
                    "iy": iy,
                    "ix": ix,
                    "y0": float(edges[iy]),
                    "y1": float(edges[iy + 1]),
                    "x0": float(edges[ix]),
                    "x1": float(edges[ix + 1]),
                }
            )
    return tiles


def _lens_model_from(
    mass_result,
    subhalo_no_subhalo_settings_dict: dict,
) -> af.Model:
    """
    Build the lens model component for the subhalo search, exactly as
    ``detect.subhalo_grid_search`` does.

    This is factored out so the no-subhalo baseline and every tile use the
    identical lens model construction (the only thing that differs between
    them is whether a subhalo galaxy is added).
    """
    if subhalo_no_subhalo_settings_dict.get("chaining", False):
        lens = mass_result.model_centred.galaxies.lens
        if subhalo_no_subhalo_settings_dict.get("reset_multipoles", False):
            # NOTE: the upstream detect.py has a latent typo where it sets
            # ``lens.multipole.m = ...`` for multipole_1/_3/_4. We set the
            # attribute on EACH multipole model itself, which is correct.
            if hasattr(lens, "multipole_1"):
                lens.multipole_1 = af.Model(al.mp.PowerLawMultipole)
                lens.multipole_1.m = 1
            if hasattr(lens, "multipole_3"):
                lens.multipole_3 = af.Model(al.mp.PowerLawMultipole)
                lens.multipole_3.m = 3
            if hasattr(lens, "multipole_4"):
                lens.multipole_4 = af.Model(al.mp.PowerLawMultipole)
                lens.multipole_4.m = 4
        if subhalo_no_subhalo_settings_dict.get("reset_shear", False):
            lens.shear = af.Model(al.mp.ExternalShear)
    else:
        lens = mass_result.model.galaxies.lens

    return lens


def analysis_from(
    dataset,
    source_pix_result_1,
    mass_result,
) -> al.AnalysisImaging:
    """
    Build the ``AnalysisImaging`` for a (no-)subhalo fit, mirroring
    ``detect.subhalo_grid_search``: adapt images from source_pix[1], positions
    likelihood from the mass result (factor 3.0, threshold 0.2").
    """
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    return al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            mass_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
    )


def subhalo_tile(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    subhalo_no_subhalo_result: af.Result,
    subhalo_no_subhalo_settings_dict: dict,
    tile: Dict[str, float],
    subhalo_mass: af.Model,
    subhalo_mass_limits: list = [1e6, 1e11],
    n_live: int = 200,
    n_batch: int = 20,
) -> af.Result:
    """
    Fit a DM subhalo whose (y, x) centre is confined to a single image-plane
    tile, reusing the model-building logic of ``detect.subhalo_grid_search``.

    The returned result's ``samples.log_evidence`` is compared against the
    no-subhalo baseline (same model, no subhalo) to form one cell of the
    delta-log-evidence map.
    """

    analysis = analysis_from(
        dataset=dataset,
        source_pix_result_1=source_pix_result_1,
        mass_result=mass_result,
    )

    # --- Subhalo galaxy with an MCR-NFW mass profile --------------------
    subhalo = af.Model(al.Galaxy, mass=subhalo_mass)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=subhalo_mass_limits[0], upper_limit=subhalo_mass_limits[1]
    )
    # Confine the subhalo centre to THIS tile only (the whole point of tiling).
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=tile["y0"], upper_limit=tile["y1"]
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=tile["x0"], upper_limit=tile["x1"]
    )

    subhalo.redshift = subhalo_no_subhalo_result.instance.galaxies.lens.redshift
    subhalo.mass.redshift_object = (
        subhalo_no_subhalo_result.instance.galaxies.lens.redshift
    )
    subhalo.mass.redshift_source = (
        subhalo_no_subhalo_result.instance.galaxies.source.redshift
    )

    # --- Lens + source model, mirrored from detect.subhalo_grid_search --
    lens = _lens_model_from(
        mass_result=mass_result,
        subhalo_no_subhalo_settings_dict=subhalo_no_subhalo_settings_dict,
    )
    source = al.util.chaining.source_from(result=mass_result)

    model = af.Collection(
        galaxies=af.Collection(lens=lens, subhalo=subhalo, source=source),
    )

    search = af.Nautilus(
        name=f"subhalo_tile_{tile['index']:03d}",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)
