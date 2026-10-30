"""
Rebuild the COSJ100024+021749 F444W noise map (2026-10-29).

__Why__

The noise map shipped with the F444W cutout is `max(0.02, sqrt(k * data))` with
k ~ 3.4e-4 (an above-floor fit gives nm^2 = 3.4e-4 * data with no intercept): a Poisson term
floored at 0.02 e/s, and 99.8% of the pixels sit on the floor. The clipped scatter of the empty
sky around the lens is 0.0068 (MAD and std agree), so the floor is 2.9x the sky noise. In the
`updated_output*/` fits the reduced chi-squared was 0.1 beyond 1", 0.33 on the arc and 2.6 in
the lens core: the core drove the fit and the arc was barely weighted.

__What__

    sigma^2 = sigma_sky^2 + k * max(data, 0)

with sigma_sky measured here (sigma-clipped, outside the extra-galaxies mask, 2.6"-6" from the
centre) and k kept from the shipped map (the only Poisson calibration available: the cutout has
no exposure or gain information, and the PRIMER WHT map in `primer/F444W/` is not in inverse
variance of these units). Neighbouring sky pixels are correlated at 0.66 (drizzle); that is
not modelled, as it was not in the shipped map either.

The shipped map is kept as `noise_map_cowls.fits`; the runner reads `noise_map.fits`.

Run from the project root:  uv run python data_prep/rebuild_noise_map_J24.py
"""
import os
import shutil

import numpy as np
from astropy.io import fits

DATASET = os.path.join("data", "cowls", "COSJ100024+021749", "F444W")
PIXEL_SCALE = 0.06
SKY_ANNULUS = (2.6, 6.0)  # arcsec
FLOOR_OF_SHIPPED_MAP = 0.02

data = fits.getdata(os.path.join(DATASET, "data.fits")).astype(float)
shipped_path = os.path.join(DATASET, "noise_map_cowls.fits")
if not os.path.exists(shipped_path):
    shutil.copy(os.path.join(DATASET, "noise_map.fits"), shipped_path)
shipped = fits.getdata(shipped_path).astype(float)
extra = fits.getdata(os.path.join(DATASET, "mask_extra_galaxies.fits"))

# Poisson coefficient of the shipped map from its above-floor pixels: nm^2 = k * data.
above = shipped > FLOOR_OF_SHIPPED_MAP * 1.005
k = np.sum(shipped[above] ** 2 * data[above]) / np.sum(data[above] ** 2)

# Sky scatter, sigma-clipped, outside the extra galaxies.
n = data.shape[0]
c = (n - 1) / 2
yy, xx = np.mgrid[0:n, 0:n]
r = np.hypot(yy - c, xx - c) * PIXEL_SCALE
sky = data[(r >= SKY_ANNULUS[0]) & (r < SKY_ANNULUS[1]) & (extra == 0)]
for _ in range(5):
    med = np.median(sky)
    s = 1.4826 * np.median(np.abs(sky - med))
    sky = sky[np.abs(sky - med) < 3.5 * s]
sigma_sky = sky.std()

noise = np.sqrt(sigma_sky**2 + k * np.clip(data, 0, None))
fits.writeto(os.path.join(DATASET, "noise_map.fits"), noise.astype(np.float32), overwrite=True)

print(f"k (Poisson) = {k:.3e}  (1/k = {1/k:.0f}),  sigma_sky = {sigma_sky:.5f}  from {sky.size} px")
for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.6), (1.6, 2.2), (2.6, 6.0)]:
    m = (r >= lo) & (r < hi)
    print(f"  r {lo}-{hi}: data {np.median(data[m]):.3f}  shipped {np.median(shipped[m]):.4f}  rebuilt {np.median(noise[m]):.4f}")
