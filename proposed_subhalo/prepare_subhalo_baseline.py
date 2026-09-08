"""
Prepare the no-subhalo baseline (run ONCE, before the 25 tile jobs)
===================================================================

The tiled subhalo scan compares every tile's log-evidence against a SINGLE
no-subhalo baseline. Running that baseline once (instead of once per tile)
saves ~25 days of compute and gives every tile the *same* reference evidence,
which is what makes the delta-log-evidence map clean and comparable.

This script:

  1. loads the masked dataset exactly as the main runners do;
  2. loads the already-completed ``source_pix[1]`` and ``mass_multipole`` results
     from ``output/`` via the Aggregator (the chain is NOT re-run);
  3. runs ONE ``af.Nautilus`` fit of the mass model *without* a subhalo,
     writing it to a tile-independent ``unique_tag`` so every tile job loads
     the same baseline.

Launch it on its own, e.g.:

    uv run python prepare_subhalo_baseline.py COSJ100024+021749 F444W

Arguments
---------
argv[1]  dataset name        (e.g. "COSJ100024+021749")
argv[2]  filter              (optional, default "F444W")
argv[3]  source path_prefix  (optional, default "model_setup_free_source")
argv[4]  mass   path_prefix  (optional, default "mass_models")
argv[5]  mass   search name  (optional, default "mass_multipole")

The path_prefix defaults match ``main_runner_J24.py``. Override them if you ran
under a different prefix (e.g. J18 uses ``model_setup_free_source_weight_3``).
Pass argv[5]="mass_EPL" to base detection on the pre-multipole PowerLaw instead.
"""

import json
import sys
from os import path

from autoconf import jax_wrapper  # noqa: F401  (sets JAX env before other imports)

import autofit as af
import autolens as al
from autoconf import conf
import numpy as np

from slam_pipeline.subhalo import detect
from slam_pipeline.subhalo.loaders import load_search_output, log_evidence_of


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
filt = str(sys.argv[2]) if len(sys.argv) > 2 else "F444W"
source_prefix = str(sys.argv[3]) if len(sys.argv) > 3 else "model_setup_free_source"
mass_prefix = str(sys.argv[4]) if len(sys.argv) > 4 else "mass_models"
# Final smooth mass model the chain produces (mass_EPL is fit first, then
# mass_multipole with m=3,4 multipoles). The baseline must match the mass model
# the tiles load in main_subhalo.py, so it too defaults to "mass_multipole".
mass_name = str(sys.argv[5]) if len(sys.argv) > 5 else "mass_multipole"

# Pixel scale per filter. CHECK against your data reduction if you change filters.
FILTER_PIXEL_SCALES = {
    "F444W": 0.063,
    "F277W": 0.0315,
    "F150W": 0.0315,
    "F115W": 0.0315,
}
PSF_FILENAME = "psf95pc.fits"

ps = FILTER_PIXEL_SCALES[filt]


# ---------------------------------------------------------------------------
# Dataset: load and mask (identical to main_runner_*.py)
# ---------------------------------------------------------------------------
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

print(
    f"BASELINE  Dataset: {dataset_name}, Filter: {filt}, Mask: {mask_radius}\n"
    f"          source_prefix={source_prefix}, mass_prefix={mass_prefix}"
)


# ---------------------------------------------------------------------------
# Load the COMPLETED source_pix[1] and mass_multipole results via the Aggregator.
# The chain is NOT re-run.  We load full Result objects (not just SearchOutput)
# because detect.subhalo_no_subhalo calls positions_likelihood_from(...) and
# uses model_centred.
# ---------------------------------------------------------------------------
from slam_pipeline.subhalo.loaders import load_result  # noqa: E402

# source_pix[1] needs an analysis to reconstruct its Result (positions etc.).
# Build the same analysis object the pipeline used, minus the adapt images
# (those are derived from the result itself inside detect.subhalo_no_subhalo).
source_analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=use_jax,
    positions_likelihood_list=[al.PositionsLH(threshold=0.2, positions=positions)],
)

source_pix_result_1 = load_result(
    output_path=output_path,
    path_prefix=source_prefix,
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
    path_prefix=mass_prefix,
    unique_tag=dataset_name,
    name=mass_name,
    analysis=mass_analysis,
)


# ---------------------------------------------------------------------------
# Run the no-subhalo baseline ONCE, under a tile-independent unique_tag so all
# 25 tile jobs load the exact same result.
#
# If this baseline was already produced (e.g. a re-run), Nautilus sees the
# .completed marker and returns instantly without re-sampling.
# ---------------------------------------------------------------------------
N_BATCH = 20

baseline_unique_tag = f"{dataset_name}_subhalo_baseline"

settings_search = af.SettingsSearch(
    path_prefix="subhalo",
    unique_tag=baseline_unique_tag,
    info=None,
    session=None,
    number_of_cores=16,
)

subhalo_no_subhalo_result, subhalo_no_subhalo_settings_dict = detect.subhalo_no_subhalo(
    settings_search=settings_search,
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    mass_result=mass_result,
    n_batch=N_BATCH,
    use_chained_model=False,
    reset_multipoles=True,
    reset_shear=True,
)

baseline_log_evidence = float(subhalo_no_subhalo_result.samples.log_evidence)

print("\n=== SUBHALO BASELINE COMPLETE ===")
print(f"  dataset        : {dataset_name}")
print(f"  filter         : {filt}")
print(f"  unique_tag     : {baseline_unique_tag}")
print(f"  log_evidence   : {baseline_log_evidence:.4f}")
print(
    "Tile jobs will load this result from:\n"
    f"  {path.join(output_path, 'subhalo', baseline_unique_tag, 'subhalo_base')}"
)
