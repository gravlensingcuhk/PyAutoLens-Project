"""
Add a PSF-mismatch floor to the noise maps (2026-11-07).

__Why__

The completed v4 fits' chi-squared maps are dominated by the bright lens core:
J24's F444W lens peaks at S/N ~ 200, while the reduction's own `psf_err.fits`
(the per-pixel scatter of the injected STPSF realisations) reaches ~3.6% of the
PSF peak (J24) and ~10% (J18 F150W). A few-percent error on the convolved model
at a 200-sigma core is tens of sigma of unmodelable chi-squared in the central
pixels, which drags the mass centre and slope around. The runners currently read
`psf.fits` but never `psf_err.fits`.

__What__

For each lens/band, the extra variance from the PSF kernel scatter, propagated
through the convolution the model performs:

    var_psf(x) = sum_k  psf_err(k)^2 * data(x - k)^2        (a convolution)
    noise_psffloor = sqrt(noise_map^2 + var_psf)

Using data^2 as the model-brightness proxy is the standard approximation (the
true model is unknown before the fit); it puts the extra uncertainty exactly
where bright pixels are and leaves sky pixels untouched. `noise_map.fits` is
kept untouched; this writes `noise_map_psffloor.fits`, which the runners read
only with PYAUTOLENS_NOISE=psffloor, and a quick-look PNG per band.

Run from the project root:  uv run python data_prep/add_psf_noise_floor.py
"""
import os

import numpy as np
from astropy.io import fits
from scipy.signal import convolve2d

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = os.path.join("data", "cowls_sam")
LENSES = {
    "COSJ100018+022138": ["F115W", "F150W", "F277W", "F444W"],
    "COSJ100024+021749": ["F115W", "F150W", "F277W", "F444W"],
    "COSJ095921+020638": ["F115W", "F150W", "F277W", "F444W"],
}

for lens, bands in LENSES.items():
    for band in bands:
        dataset_path = os.path.join(DATA_ROOT, lens, band)
        out_path = os.path.join(dataset_path, "noise_map_psffloor.fits")
        png_path = os.path.join(dataset_path, "noise_map_psffloor.png")

        data = fits.getdata(os.path.join(dataset_path, "data.fits")).astype(float)
        noise = fits.getdata(os.path.join(dataset_path, "noise_map.fits")).astype(float)
        psf = fits.getdata(os.path.join(dataset_path, "psf.fits")).astype(float)
        psf_err = fits.getdata(os.path.join(dataset_path, "psf_err.fits")).astype(float)

        var_psf = convolve2d(data**2, psf_err**2, mode="same", boundary="fill", fillvalue=0.0)
        noise_floor = np.sqrt(noise**2 + var_psf)

        header = fits.getheader(os.path.join(dataset_path, "noise_map.fits"))
        fits.writeto(out_path, noise_floor, header=header, overwrite=True)

        peak = np.unravel_index(np.argmax(data), data.shape)
        arc = data > np.percentile(data, 99)
        print(f"{lens} {band}:")
        print(f"  psf_err peak / psf peak = {psf_err.max() / psf.max():.3%}")
        print(f"  core pixel:  data={data[peak]:.3f}  noise={noise[peak]:.4f}  "
              f"psf floor={np.sqrt(var_psf[peak]):.4f} ({np.sqrt(var_psf[peak]) / noise[peak]:.1f}x noise)")
        print(f"  bright (>99th pct): median floor/noise = "
              f"{np.median(np.sqrt(var_psf[arc]) / noise[arc]):.2f}, "
              f"max = {np.max(np.sqrt(var_psf[arc]) / noise[arc]):.2f}")
        print(f"  sky (50th pct of data): median floor/noise = "
              f"{np.median(np.sqrt(var_psf[data < np.percentile(data, 50)]) / noise[data < np.percentile(data, 50)]):.3f}")

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, arr, title, cmap in [
            (axes[0], np.arcsinh(data / noise.max()), "data (arcsinh of S/N)", "viridis"),
            (axes[1], noise, "noise_map", "viridis"),
            (axes[2], np.sqrt(var_psf), "psf floor term", "inferno"),
            (axes[3], noise_floor / noise, "noise_psffloor / noise", "magma"),
        ]:
            im = ax.imshow(arr, origin="lower", cmap=cmap)
            ax.set_title(f"{lens} {band}\n{title}", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(png_path, dpi=100)
        plt.close(fig)
        print(f"  wrote {out_path} and {png_path}")
