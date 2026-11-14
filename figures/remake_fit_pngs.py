"""
Re-make the parametric-stage fit pngs with the caustic-square max zoom (2026-11-14).

`slam_pipeline/plot_patch.py` changes the "Source Plane (Max Zoom)" panel of fit.png for
parametric sources only (the pixelized stages' panels use the inversion path, which is
unchanged), so only the source_lp[1] (and light[1], not rebuilt here) figures differ.
This script rebuilds fit.png and fit_log10.png from each archived source_lp zip into
`figures/fit_png_caustic_zoom/<output_dir>/`, one file per (output_dir, lens).

Run from the project root:  uv run python figures/remake_fit_pngs.py
"""

import json
import os
import sys
import tempfile
import zipfile

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slam_pipeline  # applies the caustic-zoom patch
import autolens as al
from autolens.imaging.plot import fit_imaging_plots as fip

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
    ("output_sam_v4_f277", "COSJ100018+022138", "F277W", None),
    ("output_test_as", "COSJ100018+022138", "F150W", None),
    ("output_test_as_sp1", "COSJ100018+022138", "F150W", None),
    ("output_test_asp", "COSJ100018+022138", "F150W", None),
]

for output_dir, lens, band, noise_tag in RUNS:
    stage_dir = os.path.join(output_dir, "model_setup_free_source", lens, "source_lp[1]")
    zips = sorted(f for f in os.listdir(stage_dir) if f.endswith(".zip")) if os.path.isdir(stage_dir) else []
    if not zips:
        print(f"SKIP {output_dir} {lens}: no source_lp zips")
        continue

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
    dataset = dataset.apply_mask(mask=mask)

    out_dir = os.path.join(OUT_ROOT, output_dir)
    os.makedirs(out_dir, exist_ok=True)

    for z in zips:
        tag = z.replace(".zip", "")[:8]
        with tempfile.TemporaryDirectory() as tmp:
            zipfile.ZipFile(os.path.join(stage_dir, z)).extractall(tmp)
            tracer = al.from_json(os.path.join(tmp, "files", "tracer.json"))
            fit = al.FitImaging(dataset=dataset, tracer=tracer)
            fip.subplot_fit(fit, output_path=tmp, output_format="png")
            fip.subplot_fit_log10(fit, output_path=tmp, output_format="png")
            os.replace(os.path.join(tmp, "fit.png"),
                       os.path.join(out_dir, f"{lens}_source_lp_{tag}.png"))
            os.replace(os.path.join(tmp, "fit_log10.png"),
                       os.path.join(out_dir, f"{lens}_source_lp_{tag}_log10.png"))
        print(f"wrote {out_dir}/{lens}_source_lp_{tag}.png (+log10)")
