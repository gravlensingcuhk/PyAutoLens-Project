"""
Subhalo Detection Runner (final refine stage)
=============================================

This is the last stage of the tiled subhalo pipeline, run as ONE job **after**
all tile jobs (``main_subhalo.py``) have finished. It mirrors the third search
of the reference ``detect.py`` grid-search pipeline (``subhalo_refine``): it

  1. reads the per-tile summaries to find the highest-evidence tile,
  2. loads that tile's completed Result,
  3. refits the whole image plane with a DM subhalo whose (y, x) centre is a
     Gaussian **centred on that tile's best-fit subhalo position** (no longer
     confined to a tile), and
  4. compares the refined log-evidence to the no-subhalo baseline to give the
     final detection significance.

Launch it once, after the tiles + ``combine_tiles.py``:

    python subhalo_refine.py COSJ100024+021749 F444W

Arguments
---------
argv[1]  dataset name        (e.g. "COSJ100024+021749")
argv[2]  filter              (optional, default "F444W")
argv[3]  source path_prefix  (optional, default "model_setup_free_source")
argv[4]  mass   path_prefix  (optional, default "mass_models")
argv[5]  mass   search name  (optional, default "mass_multipole")
"""

import json
import os
import sys
from os import path

from autoconf import jax_wrapper  # noqa: F401  (sets JAX env before other imports)

import autofit as af
import autolens as al
from autoconf import conf
import numpy as np

from slam_pipeline.subhalo import tiling
from slam_pipeline.subhalo.loaders import load_result


# ---------------------------------------------------------------------------
# Paths (identical layout to the main_runner_*.py / main_subhalo.py scripts)
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
filt = str(sys.argv[2]) if len(sys.argv) > 2 else "F444W"
SOURCE_PREFIX = str(sys.argv[3]) if len(sys.argv) > 3 else "model_setup_free_source"
MASS_PREFIX = str(sys.argv[4]) if len(sys.argv) > 4 else "mass_models"
MASS_NAME = str(sys.argv[5]) if len(sys.argv) > 5 else "mass_multipole"


# ---------------------------------------------------------------------------
# Refine configuration  <-- keep in sync with main_subhalo.py where relevant
# ---------------------------------------------------------------------------
SUBHALO_MASS_LIMITS = [1e6, 1e11]  # M_200 prior range (Msun)
CENTRE_SIGMA_SCALE = 1.0           # 'a' for model_centred_absolute (Gaussian centre-prior width)
N_LIVE = 600                       # nautilus live points (larger than tiles: final refined fit)
N_BATCH = 16

FILTER_PIXEL_SCALES = {
    "F444W": 0.063,
    "F277W": 0.0315,
    "F150W": 0.0315,
    "F115W": 0.0315,
}
PSF_FILENAME = "psf95pc.fits"


# ---------------------------------------------------------------------------
# Dataset: load and mask (identical to main_subhalo.py)
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
# Load the completed chain results from output/ (chain is NOT re-run).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Load the shared no-subhalo baseline (produced by prepare_subhalo_baseline.py).
# ---------------------------------------------------------------------------
baseline_unique_tag = f"{dataset_name}_subhalo_baseline"
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
baseline_log_evidence = float(subhalo_no_subhalo_result.samples.log_evidence)
print(f"Loaded no-subhalo baseline; log_evidence = {baseline_log_evidence:.4f}")


# ---------------------------------------------------------------------------
# Find the highest-evidence tile from the per-tile summary JSONs.
# ---------------------------------------------------------------------------
summary_dir = path.join(output_path, "subhalo_tiles", dataset_name, filt)
tile_files = sorted(f for f in os.listdir(summary_dir) if f.startswith("tile_") and f.endswith(".json"))
if not tile_files:
    raise FileNotFoundError(
        f"No tile summaries in {summary_dir}. Run the tile jobs (main_subhalo.py) first."
    )

records = [json.load(open(path.join(summary_dir, f))) for f in tile_files]
winner = max(records, key=lambda r: r["delta_log_evidence"])
winner_index = int(winner["tile_index"])
print(
    f"Highest-evidence tile: index {winner_index} "
    f"(delta_logE = {winner['delta_log_evidence']:.3f}, "
    f"centre = ({winner['best_fit_centre_y']:.3f}, {winner['best_fit_centre_x']:.3f}) arcsec, "
    f"M_200 = {winner['best_fit_mass_at_200']:.3e} Msun)"
)


# ---------------------------------------------------------------------------
# Load the winning tile's full Result (needed for model_centred_absolute).
# ---------------------------------------------------------------------------
winning_tile_result = load_result(
    output_path=output_path,
    path_prefix="subhalo",
    unique_tag=f"{dataset_name}_subhalo_tile_{winner_index:03d}",
    name=f"subhalo_tile_{winner_index:03d}",
    analysis=tiling.analysis_from(
        dataset=dataset,
        source_pix_result_1=source_pix_result_1,
        mass_result=mass_result,
    ),
)


# ---------------------------------------------------------------------------
# Refine: full-frame fit with the subhalo centre a Gaussian at the winner.
# ---------------------------------------------------------------------------
settings_search = af.SettingsSearch(
    path_prefix="subhalo",
    unique_tag=f"{dataset_name}_subhalo_refine",
    info=None,
    session=None,
    number_of_cores=16,
)

refine_result = tiling.subhalo_refine(
    settings_search=settings_search,
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    mass_result=mass_result,
    subhalo_no_subhalo_result=subhalo_no_subhalo_result,
    winning_tile_result=winning_tile_result,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    subhalo_mass_limits=SUBHALO_MASS_LIMITS,
    centre_sigma_scale=CENTRE_SIGMA_SCALE,
    n_live=N_LIVE,
    n_batch=N_BATCH,
)


# ---------------------------------------------------------------------------
# Summary: final detection significance vs the no-subhalo baseline.
# ---------------------------------------------------------------------------
refine_log_evidence = float(refine_result.samples.log_evidence)
subhalo_inst = refine_result.instance.galaxies.subhalo.mass

summary = {
    "dataset": dataset_name,
    "filter": filt,
    "winning_tile_index": winner_index,
    "log_evidence_no_subhalo": baseline_log_evidence,
    "log_evidence_with_subhalo_refined": refine_log_evidence,
    "delta_log_evidence_refined": refine_log_evidence - baseline_log_evidence,
    "best_fit_mass_at_200": float(subhalo_inst.mass_at_200),
    "best_fit_centre_y": float(subhalo_inst.centre[0]),
    "best_fit_centre_x": float(subhalo_inst.centre[1]),
}

with open(path.join(summary_dir, "refine.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSUMMARY", json.dumps(summary, indent=2))
print(f"\nWrote {path.join(summary_dir, 'refine.json')}")
