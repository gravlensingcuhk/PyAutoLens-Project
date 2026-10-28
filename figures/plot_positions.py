"""
Two-panel positions overlay for a COWLS lens, as HST_lenses_data_reduction's
`scripts/make_positions.py` shows it: the deflector's radial-median profile subtracted on
the LEFT, the band as observed on the RIGHT, both asinh-stretched (a = 0.1) between the 5th
and 99.5th percentiles, jet colourmap, positions as white crosses with their index.

Usage:
    uv run python figures/plot_positions.py <lens> <filter> <pixel_scale> <positions.json> <out.png> [title] [source_lp_result]

`source_lp_result` (optional) is a finished source_lp search's output folder or .zip; its
`image/model_galaxy_images.fits` lens-galaxy image (the first galaxy HDU) is subtracted from
the data for a third, lens-subtracted panel.

The data are read with autolens (`Array2D.from_fits`, unflipped, see the FITS-orientation
note in CLAUDE.md), so the arcsec frame is the one the runners fit in and positions.json is
written in.
"""
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch, ManualInterval

from autonerves import jax_wrapper  # noqa: F401  (first autolens import)
import autolens as al
from autogalaxy.gui.display_util import radial_median_subtract


def stretched(values, vmin_percent=5.0, vmax_percent=99.5, asinh_a=0.1):
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    vmin, vmax = np.percentile(finite, [vmin_percent, vmax_percent])
    if not vmax > vmin:
        vmax = vmin + (abs(vmin) or 1.0)
    return np.nan_to_num(AsinhStretch(a=asinh_a)(ManualInterval(vmin=vmin, vmax=vmax)(vals)))


def lens_model_image_from(result, pixel_scale):
    """The source_lp lens-galaxy model image, from the result folder or its zip."""
    result = Path(result)
    if result.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(result) as z:
            z.extract("image/model_galaxy_images.fits", tmp)
        result = tmp
    return np.asarray(
        al.Array2D.from_fits(
            str(result / "image/model_galaxy_images.fits"), hdu=1, pixel_scales=float(pixel_scale)
        ).native
    )


def main(lens, filt, pixel_scale, positions_path, out, title=None, source_lp_result=None):
    data = al.Array2D.from_fits(
        f"data/cowls/{lens}/{filt}/data.fits", pixel_scales=float(pixel_scale)
    )
    base = np.asarray(data.native)
    positions = np.asarray(al.from_json(positions_path))

    n_y, n_x = base.shape
    hw = int(n_x / 2) * float(pixel_scale)
    ext = [-hw, hw, -hw, hw]
    panels = [
        ("radial-subtracted", radial_median_subtract(base)),
        ("as-observed", base),
    ]
    if source_lp_result is not None:
        panels.append(("source_lp lens-subtracted", base - lens_model_image_from(source_lp_result, pixel_scale)))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    for ax, (name, values) in zip(axes, panels):
        im = ax.imshow(stretched(values), cmap="jet", extent=ext)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.scatter(positions[:, 1], positions[:, 0], marker="+", c="white", s=180, linewidths=1.5)
        for i, (y, x) in enumerate(positions):
            ax.annotate(str(i), (x, y), color="white", fontsize=10, xytext=(4, 4), textcoords="offset points")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_title(name)
        ax.set_xlabel("arcsec")
        ax.set_ylabel("arcsec")
    fig.suptitle(title or f"{lens}  {filt}  {positions_path}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
