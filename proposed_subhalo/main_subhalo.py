"""
Subhalo Detection Runner (one job per image-plane tile)
=======================================================

This is the efficient "load-from-output" mode. It does **not** re-run the
source -> light -> mass SLaM chain. Your submitted ``main_runner_*.py`` jobs
(and ``prepare_subhalo_baseline.py``) already wrote those results to
``output/``; this script loads them back via the Aggregator and runs only the
single subhalo tile it is responsible for.

Launch 16 independent jobs, one per tile, e.g.:

    python main_subhalo.py COSJ100024+021749 0 F444W
    python main_subhalo.py COSJ100024+021749 1 F444W
    ...
    python main_subhalo.py COSJ100024+021749 15 F444W

Arguments
---------
argv[1]  dataset name   (e.g. "COSJ100024+021749")
argv[2]  tile index     0 .. (NUMBER_OF_TILES**2 - 1)
argv[3]  filter         (optional, default "F444W")

Each job writes its own result under a tile-specific ``unique_tag`` and a small
summary JSON under ``output/subhalo_tiles/<dataset>/tile_###.json``. After all
16 finish, run ``combine_tiles.py <dataset>`` to assemble the delta-log-evidence
map.
"""

import json
import os
import sys
import time
from os import path

from autoconf import jax_wrapper  # noqa: F401  (sets JAX env before other imports)

import autofit as af
import autolens as al
from autoconf import conf
import numpy as np

from slam_pipeline.subhalo import tiling, detect
from slam_pipeline.subhalo.loaders import (
    load_result,
    load_search_output,
    log_evidence_of,
)


# ---------------------------------------------------------------------------
# Paths (identical layout to the main_runner_*.py scripts)
# ---------------------------------------------------------------------------
cosma_path = path.join(path.sep, "users", "wing-yan.chan", "PyAutoLens-Project")
workspace_path = cosma_path
output_path = path.join(cosma_path, "output")
config_path = path.join(cosma_path, "config")
conf.instance.push(new_path=config_path, output_path=output_path)
data_path = path.join(workspace_path, "data", "cowls")

use_jax = True


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
dataset_name = str(sys.argv[1])
tile_index = int(sys.argv[2])
filt = str(sys.argv[3]) if len(sys.argv) > 3 else "F444W"


# ---------------------------------------------------------------------------
# Subhalo scan configuration  <-- TUNE HERE
# ---------------------------------------------------------------------------
GRID_DIMENSION_ARCSEC = 3.0        # search region is +- this many arcsec
NUMBER_OF_TILES = 4                # tiles per side -> 4x4 = 16 jobs
SUBHALO_MASS_LIMITS = [1e6, 1e11]  # M_200 prior range (Msun)
N_LIVE = 200                       # nautilus live points per tile
N_BATCH = 20

# path_prefixes where the already-finished chain lives.
# main_runner_J24.py -> source: "model_setup_free_source", mass: "mass_models".
# main_runner_J18.py -> source: "model_setup_free_source_weight_3" (or _4), mass: "mass_models".
# Override via argv[4]/argv[5] if needed.
SOURCE_PREFIX = str(sys.argv[4]) if len(sys.argv) > 4 else "model_setup_free_source"
MASS_PREFIX = str(sys.argv[5]) if len(sys.argv) > 5 else "mass_models"
# The base smooth mass model for subhalo detection. The main runners fit BOTH
# "mass_EPL" (PowerLaw) and, after it, "mass_multipole" (PowerLaw + m=3,4
# multipoles) under MASS_PREFIX. Subhalo detection must build on the final, most
# complete smooth model so residuals are not just smooth-mass inadequacy, so we
# load "mass_multipole" (the last search the chain runs).
MASS_NAME = str(sys.argv[6]) if len(sys.argv) > 6 else "mass_multipole"

# Pixel scale + PSF per filter. CHECK these against your data reduction.
FILTER_PIXEL_SCALES = {
    "F444W": 0.063,
    "F277W": 0.0315,
    "F150W": 0.0315,
    "F115W": 0.0315,
}
PSF_FILENAME = "psf95pc.fits"


# Build the tile we are responsible for.
tiles = tiling.make_tiles(
    grid_dimension_arcsec=GRID_DIMENSION_ARCSEC,
    number_of_tiles=NUMBER_OF_TILES,
)
tile = tiles[tile_index]
print(
    f"RUNNING Dataset: {dataset_name}, Filter: {filt}, Tile {tile_index} "
    f"(iy={tile['iy']}, ix={tile['ix']}; "
    f"y [{tile['y0']:.2f},{tile['y1']:.2f}], x [{tile['x0']:.2f},{tile['x1']:.2f}])"
)


# ---------------------------------------------------------------------------
# Dataset: load and mask (identical to main_runner_*.py)
# ---------------------------------------------------------------------------
ps = FILTER_PIXEL_SCALES[filt]
dataset_path = path.join(data_path, dataset_name, filt)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, PSF_FILENAME),
    pixel_scales=ps,
    check_noise_map=False,
)

with open(path.join(data_path, dataset_name, "info.json"), "r") as file:
    info_file = json.load(file)

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(data_path, dataset_name, "positions.json"))
)
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=path.join(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=dataset.pixel_scales,
    invert=True,
)
dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
mask_radius = info_file["mask_radius"]
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.2, 0.4],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
dataset = dataset.apply_sparse_operator_cpu()


# ---------------------------------------------------------------------------
# Load the COMPLETED chain results from output/ (the chain is NOT re-run).
# ---------------------------------------------------------------------------
# source_pix[1] -- full Result (needed for positions_likelihood_from + adapt images)
source_analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=use_jax,
    positions_likelihood_list=[al.PositionsLH(threshold=0.2, positions=positions)],
)
source_pix_result_1 = load_result(
    output_path=output_path,
    path_prefix=SOURCE_PREFIX,
    unique_tag=dataset_name,
    name="source_pix[1]",
    analysis=source_analysis,
)

# mass_multipole -- full Result (the base lens model for subhalo detection)
mass_analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=use_jax,
    positions_likelihood_list=[
        source_pix_result_1.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)
mass_result = load_result(
    output_path=output_path,
    path_prefix=MASS_PREFIX,
    unique_tag=dataset_name,
    name=MASS_NAME,
    analysis=mass_analysis,
)
print(f"Loaded source_pix[1] and {MASS_NAME} results from output/.")


# ---------------------------------------------------------------------------
# No-subhalo baseline: ONE shared result for all 16 tiles.
#
# It is written under a tile-independent unique_tag by
# prepare_subhalo_baseline.py. If that preparation job is still running when a
# tile starts, wait for it (rather than 16 jobs racing to fit the same
# baseline). If it never appears, fall back to fitting it here.
# ---------------------------------------------------------------------------
baseline_unique_tag = f"{dataset_name}_subhalo_baseline"
BASELINE_WAIT_TIMEOUT_S = 6 * 3600  # 6 h; raise if your baseline takes longer
BASELINE_POLL_S = 60


def _baseline_completed() -> bool:
    try:
        so = load_search_output(
            output_path=output_path,
            path_prefix="subhalo",
            unique_tag=baseline_unique_tag,
            name="subhalo_base",
        )
        return bool(so.is_complete)
    except FileNotFoundError:
        return False


if not _baseline_completed():
    print(
        f"No-subhalo baseline ({baseline_unique_tag}/subhalo_base) not found. "
        f"Waiting up to {BASELINE_WAIT_TIMEOUT_S/3600:.1f} h for "
        f"prepare_subhalo_baseline.py to finish..."
    )
    waited = 0
    while waited < BASELINE_WAIT_TIMEOUT_S and not _baseline_completed():
        time.sleep(BASELINE_POLL_S)
        waited += BASELINE_POLL_S
        if waited % 600 == 0:
            print(f"  ...still waiting for baseline ({waited/60:.0f} min elapsed)")

if not _baseline_completed():
    # Fallback: fit the baseline in this job. The unique_tag is shared, so if
    # multiple tiles reach here the filesystem-level resume ensures only one
    # samples; the rest load the completed result.
    print("Baseline still absent after wait; fitting it in this job as a fallback.")
    settings_baseline = af.SettingsSearch(
        path_prefix="subhalo",
        unique_tag=baseline_unique_tag,
        info=None,
        session=None,
        number_of_cores=16,
    )
    detect.subhalo_no_subhalo(
        settings_search=settings_baseline,
        dataset=dataset,
        source_pix_result_1=source_pix_result_1,
        mass_result=mass_result,
        n_batch=N_BATCH,
        use_chained_model=False,
        reset_multipoles=True,
        reset_shear=True,
    )

# Load the (now complete) baseline as a full Result.
baseline_analysis = tiling.analysis_from(
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    mass_result=mass_result,
)
subhalo_no_subhalo_result = load_result(
    output_path=output_path,
    path_prefix="subhalo",
    unique_tag=baseline_unique_tag,
    name="subhalo_base",
    analysis=baseline_analysis,
)
subhalo_no_subhalo_settings_dict = {
    "chaining": False,
    "reset_multipoles": True,
    "reset_shear": True,
}
baseline_log_evidence = float(subhalo_no_subhalo_result.samples.log_evidence)
print(f"Loaded no-subhalo baseline; log_evidence = {baseline_log_evidence:.4f}")


# ---------------------------------------------------------------------------
# SUBHALO TILE FIT: the only Nautilus run this job performs.
# ---------------------------------------------------------------------------
tile_unique_tag = f"{dataset_name}_subhalo_tile_{tile_index:03d}"

settings_search = af.SettingsSearch(
    path_prefix="subhalo",
    unique_tag=tile_unique_tag,
    info=None,
    session=None,
    number_of_cores=16,
)

subhalo_mass = af.Model(al.mp.NFWMCRLudlowSph)

tile_result = tiling.subhalo_tile(
    settings_search=settings_search,
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    mass_result=mass_result,
    subhalo_no_subhalo_result=subhalo_no_subhalo_result,
    subhalo_no_subhalo_settings_dict=subhalo_no_subhalo_settings_dict,
    tile=tile,
    subhalo_mass=subhalo_mass,
    subhalo_mass_limits=SUBHALO_MASS_LIMITS,
    n_live=N_LIVE,
    n_batch=N_BATCH,
)


# ---------------------------------------------------------------------------
# Summary: write one JSON per tile so combine_tiles.py can build the map.
# ---------------------------------------------------------------------------
summary_dir = path.join(output_path, "subhalo_tiles", dataset_name, filt)
os.makedirs(summary_dir, exist_ok=True)

tile_log_evidence = float(tile_result.samples.log_evidence)
subhalo_inst = tile_result.instance.galaxies.subhalo.mass

summary = {
    "dataset": dataset_name,
    "filter": filt,
    "tile_index": tile_index,
    "number_of_tiles": NUMBER_OF_TILES,
    "grid_dimension_arcsec": GRID_DIMENSION_ARCSEC,
    "tile": tile,
    "log_evidence_no_subhalo": baseline_log_evidence,
    "log_evidence_with_subhalo": tile_log_evidence,
    "delta_log_evidence": tile_log_evidence - baseline_log_evidence,
    "best_fit_mass_at_200": float(subhalo_inst.mass_at_200),
    "best_fit_centre_y": float(subhalo_inst.centre[0]),
    "best_fit_centre_x": float(subhalo_inst.centre[1]),
}

with open(path.join(summary_dir, f"tile_{tile_index:03d}.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSUMMARY", json.dumps(summary, indent=2))
print(f"\nWrote {path.join(summary_dir, f'tile_{tile_index:03d}.json')}")
