"""
Re-make the fit pngs of every archived stage with the caustic-square max zoom (2026-11-14).

`slam_pipeline/plot_patch.py` changes the "Source Plane (Max Zoom)" panel of fit.png for both
parametric and pixelized sources, so this rebuilds every stage's figure: source_lp[1],
source_pix[1/2], light[1], mass_EPL, mass_multipole, mass_light_dark[1], for every zip found
in the given output dirs. Pixelized fits are rebuilt with the zip's own adapt images and mesh
grids (`image/adapt_images.fits`, `image/adapt_image_plane_mesh_grids.fits`), masked to the
dataset. Output: `figures/fit_png_caustic_zoom/<output_dir>/<lens>_<stage>_<id8>.png`
(+`_log10`). A stage whose full subplot fails gets a 1x3 source-plane-panels fallback.

Run from the project root:  uv run python figures/remake_fit_pngs.py
"""

import json
import os
import sys
import tempfile
import traceback
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as afits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import slam_pipeline  # applies the caustic-zoom patch
import autolens as al
from autolens.imaging.plot import fit_imaging_plots as fip
from scan_runs import (DIRS, LENS_BAND, LENS_SHORT, STAGE_ORDER, classify_zip,
                       extras_for, figure_name)

DATA_ROOT = os.path.join("data", "cowls_sam")
OUT_ROOT = os.path.join("figures", "fit_png_caustic_zoom")

# (output_dir, lens, band, noise_tag) — the dataset each archived chain was fit on.
RUNS = [
    ("output_sam_v4", "COSJ100018+022138", "F150W", None),
    ("output_sam_v4", "COSJ100024+021749", "F444W", None),
    ("output_sam_v4", "COSJ095921+020638", "F150W", None),
    ("output_sam_v5", "COSJ100018+022138", "F150W", "psffloor"),
    ("output_sam_v5", "COSJ100024+021749", "F444W", "psffloor"),
    ("output_sam_v5", "COSJ095921+020638", "F150W", "psffloor"),
    ("output_sam_v6", "COSJ100018+022138", "F150W", "psfcoh"),
    ("output_sam_v6", "COSJ100024+021749", "F444W", "psfcoh"),
    ("output_sam_v6", "COSJ095921+020638", "F150W", "psfcoh"),
    ("output_sam_v4_f277", "COSJ100018+022138", "F277W", None),
    ("output_test_as", "COSJ100018+022138", "F150W", None),
    ("output_test_as_sp1", "COSJ100018+022138", "F150W", None),
    ("output_test_asp", "COSJ100018+022138", "F150W", None),
]

STAGES = [(prefix, stage, stage_short) for prefix, stage, stage_short in STAGE_ORDER]


def dataset_for(lens, band, noise_tag):
    noise_file = f"noise_map_{noise_tag}.fits" if noise_tag else "noise_map.fits"
    dataset_path = os.path.join(DATA_ROOT, lens, band)
    pixel_scale = 0.03 if band in ("F115W", "F150W") else 0.06
    dataset = al.Imaging.from_fits(
        data_path=os.path.join(dataset_path, "data.fits"),
        noise_map_path=os.path.join(dataset_path, noise_file),
        psf_path=os.path.join(dataset_path, "psf.fits"),
        pixel_scales=pixel_scale,
    )
    info = json.load(open(os.path.join(DATA_ROOT, lens, "info.json")))
    mask = al.Mask2D.circular(shape_native=dataset.shape_native, pixel_scales=pixel_scale,
                              radius=info["mask_radius"])
    return dataset.apply_mask(mask=mask)


def adapt_images_from_zip(tmp, tracer, dataset):
    """The zip's adapt images/mesh grids, keyed by the source galaxy, masked to the dataset."""
    img_path = os.path.join(tmp, "image", "adapt_images.fits")
    grid_path = os.path.join(tmp, "image", "adapt_image_plane_mesh_grids.fits")
    if not os.path.exists(img_path):
        return None
    ny, nx = dataset.shape_native

    def to_masked(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (ny, nx):  # centre-crop a full stamp to the trimmed shape
            cy, cx = (arr.shape[0] - ny) // 2, (arr.shape[1] - nx) // 2
            arr = arr[cy:cy + ny, cx:cx + nx]
        return al.Array2D.no_mask(values=arr, pixel_scales=dataset.pixel_scales).apply_mask(
            mask=dataset.mask)

    source = tracer.galaxies[-1]
    img_hdul = afits.open(img_path)
    kwargs = {}
    src_hdu = "('GALAXIES', 'SOURCE')"
    if src_hdu in img_hdul:
        kwargs["galaxy_image_dict"] = {source: to_masked(img_hdul[src_hdu].data)}
    if os.path.exists(grid_path):
        grid_hdul = afits.open(grid_path)
        kwargs["galaxy_image_plane_mesh_grid_dict"] = {
            source: al.Grid2DIrregular(values=np.asarray(grid_hdul[1].data, dtype=np.float64))}
    return al.AdaptImages(**kwargs) if kwargs else None


def rebuild(tmp, dataset, out_png, out_png_log):
    tracer = al.from_json(os.path.join(tmp, "files", "tracer.json"))
    adapt = adapt_images_from_zip(tmp, tracer, dataset)
    fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt)
    try:
        fip.subplot_fit(fit, output_path=tmp, output_format="png")
        fip.subplot_fit_log10(fit, output_path=tmp, output_format="png")
        os.replace(os.path.join(tmp, "fit.png"), out_png)
        os.replace(os.path.join(tmp, "fit_log10.png"), out_png_log)
        return "full"
    except Exception:
        # Fallback: just the three source-plane panels (max/mid/no zoom).
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, zoom, scale, title in [
            (axes[0], True, 1.0, "Source Plane (Max Zoom)"),
            (axes[1], True, 2.0, "Source Plane (Mid Zoom)"),
            (axes[2], False, 1.0, "Source Plane (No Zoom)"),
        ]:
            fip._plot_source_plane(fit, ax, len(fit.tracer.planes) - 1,
                                   zoom_to_brightest=zoom, zoom_extent_scale=scale,
                                   title=title)
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        return "panels-only"


for output_dir, lens, band, noise_tag in RUNS:
    dataset = None
    for prefix, stage, stage_short in STAGES:
        stage_dir = os.path.join(output_dir, prefix, lens, stage)
        zips = sorted(f for f in os.listdir(stage_dir) if f.endswith(".zip")) \
            if os.path.isdir(stage_dir) else []
        if not zips:
            continue
        if dataset is None:
            dataset = dataset_for(lens, band, noise_tag)
        out_dir = os.path.join(OUT_ROOT, output_dir)
        os.makedirs(out_dir, exist_ok=True)
        for z in zips:
            zip_path = os.path.join(stage_dir, z)
            cls = classify_zip(zip_path)
            if cls.get("error"):
                print(f"FAIL {output_dir} {lens} {stage} {z[:8]}: unreadable: {cls['error']}")
                continue
            name = figure_name(lens, stage_short, DIRS[output_dir]["noise"],
                               DIRS[output_dir].get("band") or LENS_BAND[LENS_SHORT[lens]],
                               cls, z[:8], extra=extras_for(output_dir, lens, zip_path))
            out_png = os.path.join(out_dir, name)
            out_png_log = out_png.replace(".png", "_log10.png")
            if os.path.exists(out_png):
                continue
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    zipfile.ZipFile(os.path.join(stage_dir, z)).extractall(tmp)
                    kind = rebuild(tmp, dataset, out_png, out_png_log)
                print(f"{output_dir} {lens} {stage} {tag}: {kind}")
            except Exception as exc:
                print(f"FAIL {output_dir} {lens} {stage} {tag}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)
