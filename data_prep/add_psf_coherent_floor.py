"""
Add a coherence-aware PSF-mismatch floor to the noise maps (2026-11-14, "v6").

__Why__

`add_psf_noise_floor.py` (v5) propagates `psf_err.fits` through the convolution
assuming the kernel-pixel errors are mutually independent. The coherence check
(`psf_coherence_check.py`, 2026-11-14) decomposed the reduction's per-star
residuals and found 37-51% of the PSF error power is a COHERENT pattern shared
by all stars. On extended bright structure the independent-scatter formula
under-propagates it: at J0959's F150W arc pixels the true PSF error is ~2.2x
the shot noise, not the 0.21x the v5 floor adds.

__What__

Per filter, the per-star residuals (rebuilt from `star_check.npz` exactly as
`psf_star_check.py --error-maps` does) are decomposed into

    coherent   = |mean_i res_i|           (shared signed pattern)
    incoherent = sqrt(psf_err^2 - coh^2)  (phase scatter + star-to-star; the
                                           shipped psf_err carries both)

and propagated as

    var_psf(x) = sum_k incoherent(k)^2 * data(x-k)^2   (random part, quadrature)
               + ( sum_k coherent(k)   * data(x-k) )^2  (systematic, correlated)

    noise_psfcoh = sqrt(noise_map^2 + var_psf)

Runners read `noise_map_psfcoh.fits` with PYAUTOLENS_NOISE=psfcoh (v6 chains,
output_sam_v6). `noise_map.fits` and `noise_map_psffloor.fits` are untouched.

Run from the project root:  uv run python data_prep/add_psf_coherent_floor.py
"""

import glob
import os

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.signal import convolve2d

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REDUCTION = os.path.expanduser("~/Code/current_projects/JWST_lenses_data_reduction")
DATA_ROOT = os.path.join("data", "cowls_sam")
LENSES = {
    "COSJ100018+022138": ["F115W", "F150W", "F277W", "F444W"],
    "COSJ100024+021749": ["F115W", "F150W", "F277W", "F444W"],
    "COSJ095921+020638": ["F115W", "F150W", "F277W", "F444W"],
}
CAL_MIN_SNR = 50.0
EXTENDED_RATIO = 2.0


def core_fit_ab(r, t, e, rad, r_fit):
    """Weighted least squares real = a * twin + b within r_fit px (psf_star_check.core_fit)."""
    m = (rad <= r_fit) & np.isfinite(r) & (e > 0)
    w = 1.0 / e[m] ** 2
    A = np.column_stack([t[m], np.ones(m.sum())])
    coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], r[m] * np.sqrt(w), rcond=None)[0]
    return float(coef[0]), float(coef[1])


def coherent_map_for_filter(filt):
    """The shared signed PSF residual pattern of a filter, from the reduction's star checks."""
    res = []
    for f in sorted(glob.glob(os.path.join(REDUCTION, "data", "psf", "*", "*", filt, "star_check.npz"))):
        z = np.load(f)
        for i in range(len(z["real"])):
            if (z["extended"][i] or not np.isfinite(z["ratio"][i])
                    or z["ratio"][i] < 1 / EXTENDED_RATIO or z["peak_snr"][i] < CAL_MIN_SNR):
                continue
            r, t, e = (z[k][i].astype(float) for k in ("real", "twin", "err"))
            C = r.shape[0]
            yy, xx = np.mgrid[:C, :C]
            rad = np.hypot(xx - C // 2, yy - C // 2)
            a, b = core_fit_ab(r, t, e, rad, float(z["r_fit"]))
            res.append(((r - b) / a - t) / t.sum())
    res = np.array(res)
    coh = np.abs(np.ma.mean(sigma_clip(res, sigma=3.0, axis=0), axis=0).filled(0.0))
    return coh, len(res)


def to_kernel(m, k):
    """Centre-crop or zero-pad an error map to the lens's PSF kernel size."""
    c0 = m.shape[0] // 2
    h = k // 2
    if m.shape[0] >= k:
        return m[c0 - h:c0 + h + 1, c0 - h:c0 + h + 1]
    out = np.zeros((k, k))
    s = m.shape[0] // 2
    out[h - s:h + s + 1, h - s:h + s + 1] = m
    return out


# The coherent pattern is pooled per filter (same stars for every lens, as the reduction does).
coherent_maps = {filt: coherent_map_for_filter(filt) for filt in
                 ["F115W", "F150W", "F277W", "F444W"]}
for filt, (coh, n) in coherent_maps.items():
    print(f"{filt}: coherent pattern from {n} stars, err_frac {float(np.sqrt((coh ** 2).sum())):.4f}")

for lens, bands in LENSES.items():
    for band in bands:
        dataset_path = os.path.join(DATA_ROOT, lens, band)
        out_path = os.path.join(dataset_path, "noise_map_psfcoh.fits")
        png_path = os.path.join(dataset_path, "noise_map_psfcoh.png")

        data = fits.getdata(os.path.join(dataset_path, "data.fits")).astype(float)
        noise = fits.getdata(os.path.join(dataset_path, "noise_map.fits")).astype(float)
        psf_err = fits.getdata(os.path.join(dataset_path, "psf_err.fits")).astype(float)
        k = fits.getdata(os.path.join(dataset_path, "psf.fits")).shape[0]

        coh, _ = coherent_maps[band]
        coh_k = to_kernel(coh, k)
        inc_k = np.sqrt(np.clip(psf_err ** 2 - to_kernel(coh, psf_err.shape[0]) ** 2, 0.0, None))

        var_psf = (convolve2d(data ** 2, inc_k ** 2, mode="same", boundary="fill", fillvalue=0.0)
                   + np.abs(convolve2d(data, coh_k, mode="same", boundary="fill", fillvalue=0.0)) ** 2)
        noise_floor = np.sqrt(noise ** 2 + var_psf)

        header = fits.getheader(os.path.join(dataset_path, "noise_map.fits"))
        fits.writeto(out_path, noise_floor, header=header, overwrite=True)

        peak = np.unravel_index(np.argmax(data), data.shape)
        yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
        r = np.hypot(yy - data.shape[0] / 2, xx - data.shape[1] / 2) * (0.03 if band in ("F115W", "F150W") else 0.06)
        arc = (data / noise > 8) & (r > 0.4) & (r < 2.0)
        print(f"{lens} {band}: core floor/noise {np.sqrt(var_psf[peak]) / noise[peak]:.2f}, "
              f"arc ({arc.sum()} px) median floor/noise {np.median(np.sqrt(var_psf[arc]) / noise[arc]):.2f}, "
              f"max {np.max(np.sqrt(var_psf[arc]) / noise[arc]):.2f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, arr, title in [
            (axes[0], np.sqrt(var_psf), "psf floor (coherent-aware)"),
            (axes[1], noise_floor / noise, "noise_psfcoh / noise"),
            (axes[2], noise_floor / np.sqrt(noise ** 2 + convolve2d(
                data ** 2, psf_err ** 2, mode="same", boundary="fill", fillvalue=0.0)),
             "psfcoh / psffloor (v6 / v5)"),
        ]:
            im = ax.imshow(arr, origin="lower", cmap="magma")
            ax.set_title(f"{lens} {band}\n{title}", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(png_path, dpi=100)
        plt.close(fig)
        print(f"  wrote {out_path} and {png_path}")
