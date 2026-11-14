"""
Check the coherent-mode assumption behind the PSF noise floor (2026-11-14).

__Why__

`add_psf_noise_floor.py` propagates `psf_err.fits` through the convolution as
if the kernel-pixel errors were mutually independent:

    var_psf(x) = sum_k psf_err(k)^2 * data(x-k)^2

If the true PSF error is a COHERENT mode (the same signed residual pattern in
every star — model deficiency, defocus, jitter), the error at x is the
correlated sum `(data * err)(x)` and the independent-scatter formula
underestimates it by up to ~sqrt(N_eff) on smooth bright regions.

__What__

The reduction ships `psf_err.fits` as the UNSIGNED rms over stars of
(real - twin) (psf_star_check.py --error-maps), which mixes both parts. But
the per-star signed residuals survive in
`JWST_lenses_data_reduction/data/psf/**/star_check.npz`. This script rebuilds
the residuals exactly as `error_maps()` does and decomposes them per filter:

    coherent   = mean_i  res_i            (the systematic pattern)
    incoherent = std_i   res_i            (star-to-star scatter about it)

with the star noise variance subtracted from the incoherent part as in
error_maps(). It then propagates both through the convolution on each lens's
data and reports, at the lens-core pixel and at bright arc pixels:

    sigma_inc  = sqrt( sum_k incoherent(k)^2 * data(x-k)^2 )   (current recipe)
    err_coh    = | sum_k coherent(k)   * data(x-k) |           (deterministic)

If err_coh << sigma_inc the current floor is conservative; if err_coh is
comparable or larger, the floor should add a coherent term.

Run from the project root:  uv run python data_prep/psf_coherence_check.py
"""

import glob
import os

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.signal import convolve2d

REDUCTION = os.path.expanduser("~/Code/current_projects/JWST_lenses_data_reduction")
DATA_ROOT = os.path.join("data", "cowls_sam")
CAL_MIN_SNR = 50.0          # psf_star_check.CAL_MIN_SNR
EXTENDED_RATIO = 2.0        # psf_star_check.EXTENDED_RATIO (1/x for narrow)
R_FIT_FRACTION = 0.5        # core_fit radius = r_fit (stored per star in the npz)

LENSES = ["COSJ100018+022138", "COSJ100024+021749", "COSJ095921+020638"]


def core_fit_ab(r, t, e, rad, r_fit):
    """Weighted least squares real = a * twin + b within r_fit px (psf_star_check.core_fit)."""
    m = (rad <= r_fit) & np.isfinite(r) & (e > 0)
    w = 1.0 / e[m] ** 2
    A = np.column_stack([t[m], np.ones(m.sum())])
    coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], r[m] * np.sqrt(w), rcond=None)[0]
    return float(coef[0]), float(coef[1])


def residuals_for_filter(filt):
    """Per-star signed residuals in unit-sum kernel units, as error_maps() builds them."""
    res, var = [], []
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
            tot = t.sum()
            res.append(((r - b) / a - t) / tot)
            var.append((e / (a * tot)) ** 2)
    return np.array(res), np.array(var)


for filt in ["F115W", "F150W", "F277W", "F444W"]:
    res, var = residuals_for_filter(filt)
    if len(res) < 2:
        print(f"{filt}: {len(res)} stars -- too few, skipping")
        continue
    # error_maps() total: rms of (res^2 - var) over stars, clipped.
    sq = sigma_clip(res ** 2 - var, sigma=3.0, axis=0)
    em_total = np.sqrt(np.clip(np.ma.mean(sq, axis=0).filled(0.0), 0.0, None))
    # coherent part: mean signed residual (sigma-clipped, same mask).
    coh = np.abs(np.ma.mean(sigma_clip(res, sigma=3.0, axis=0), axis=0).filled(0.0))
    # incoherent part: scatter about the mean, noise-subtracted.
    inc = np.sqrt(np.clip(np.ma.mean(sq, axis=0).filled(0.0) - coh ** 2, 0.0, None))
    f = lambda m: float(np.sqrt(np.sum(m ** 2)))
    frac_coh = f(coh) / f(em_total)
    print(f"\n{filt}: {len(res)} stars | err_frac total {f(em_total):.4f}, "
          f"coherent {f(coh):.4f} ({frac_coh:.0%} of total), incoherent {f(inc):.4f}")

    for lens in LENSES:
        dp = os.path.join(DATA_ROOT, lens, filt)
        if not os.path.exists(os.path.join(dp, "data.fits")):
            continue
        data = fits.getdata(os.path.join(dp, "data.fits")).astype(float)
        # centre-crop or zero-pad the error maps to the lens's kernel size
        k = fits.getdata(os.path.join(dp, "psf.fits")).shape[0]
        def to_kernel(m):
            c0 = m.shape[0] // 2
            h = k // 2
            if m.shape[0] >= k:
                return m[c0 - h:c0 + h + 1, c0 - h:c0 + h + 1]
            out = np.zeros((k, k))
            s = m.shape[0] // 2
            out[h - s:h + s + 1, h - s:h + s + 1] = m
            return out
        coh_k = to_kernel(coh)
        em_k = to_kernel(em_total)
        sig_inc = np.sqrt(convolve2d(data ** 2, em_k ** 2, mode="same", boundary="fill", fillvalue=0.0))
        err_coh = np.abs(convolve2d(data, coh_k, mode="same", boundary="fill", fillvalue=0.0))
        peak = np.unravel_index(np.argmax(data), data.shape)
        bright = data > np.percentile(data, 99)
        print(f"  {lens}: core sigma_inc {sig_inc[peak]:.4f} vs err_coh {err_coh[peak]:.4f} "
              f"(ratio {err_coh[peak] / max(sig_inc[peak], 1e-12):.2f}) | "
              f"bright-pixel median ratio {np.median(err_coh[bright] / np.maximum(sig_inc[bright], 1e-12)):.2f}, "
              f"max {np.max(err_coh[bright] / np.maximum(sig_inc[bright], 1e-12)):.2f}")
