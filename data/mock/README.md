# data/mock/

Mock lens datasets, one per COWLS lens, simulated on 2026-11-14 by
`data_prep/simulate_mock_lenses.py`. The ground truth is **fully parametric**:
Sersic lens light, SIE + external shear, and a parametric source (Sersics,
plus a true point source for J0959's AGN). The sources are Sersics *fitted to
each lens's best pixelized source reconstruction*, so the mocks keep the real
arc/ring morphology that a source_lp parametric source cannot reproduce (the
first version of these mocks, straight source_lp Sersic sources, broke J24's
arc into two blobs and made J0959's ring almost invisible).

| Lens | Band | mass + shear from | source from | theta_E | SIE q | shear |
| --- | --- | --- | --- | --- | --- | --- |
| `COSJ100018+022138` | F150W | source_pix[2] `ce03e2ba` (v4, delaunay+constsplit) | 2 Sersics | 0.369" | 0.72 | 0.071 |
| `COSJ100024+021749` | F444W | source_pix[2] `70a89aec` (v6, delaunay+aspower) | 3 Sersics | 0.667" | 0.70 | 0.100 |
| `COSJ095921+020638` | F150W | source_lp `9017b6f2` (v6) | 2 Sersics + point AGN | 0.719" | 0.48 | 0.063 |

J0959 is special: the pixelized fit's mass (r_E 0.759) puts the reconstructed
AGN position *outside* its caustic (only 2 images), while the source_lp mass
(r_E 0.719) quadruples the source_lp AGN position and lands two images within
0.02" of the real marks. So J0959's truth takes mass + shear + the AGN point
position from the v6 source_lp, and the host Sersics from the pixelized
reconstruction (source_pix[2] `ff53a53a`, v4 rtu-img). The AGN flux is measured
from 3x3 apertures on the real lens-subtracted data at the model image
positions (the source_lp point-basis flux would make the knots ~10x too
bright). The AGN is added as flux x |magnification| in the image pixel
containing each solved image (`al.PointSolver` + finite-difference Jacobian) —
a delta evaluated as a light profile would mis-account flux at 0.03"/px.

## How each mock is built

- **lens light**: the source_pix zip's MGE bases (bulge + point nucleus), each
  converted to a single flux-conserving Sersic.
- **mass + shear**: the zip's max-likelihood values (source_lp zip for J0959, above).
- **source**: Sersics fitted to the pixelized source reconstruction (flux-weighted
  linear residuals), then intensity-rescaled so the *lensed* flux on the
  inversion's bright pixels matches the inversion's lensed image (source-plane
  flux conservation does not conserve lensed flux across the caustic).
- **forward model**: full parametric tracer on a 4x4 sub-sampled grid, the AGN
  added per-image, then `al.SimulatorImaging` with the real `psf.fits` kernel.
- **noise**: `sigma^2 = (I + b)/t`, Poisson on `(I + b) * t` counts, fixed seed.
  `t`, `b` are effective values fitted to the real `noise_map.fits` (the FITS
  headers carry no EXPTIME); median noise matches the real map to ~1%.

## Layout

Same as `data/cowls_sam/`: `<lens>/<band>/{data,noise_map,psf,psf_err,noise_map_psffloor,mask_extra_galaxies}.fits`
plus `info.json`, `positions.json`, `primary_band.json`, and additionally:

- `<lens>/model_parameters.json` — the ground truth as a flat, readable parameter list
  (Sersic lens light + source components with q/PA/r_eff/n, SIE, shear, AGN images
  and magnifications, fit quality).
- `<lens>/tracer.json` — the parametric truth as a reloadable `al.Tracer`
  (the AGN is a point source, so it is not in the tracer; see model_parameters.json).
- `<lens>/source_plane_model.fits` — the parametric source truth on a 0.0025"/px grid.
- `<lens>/source_plane_reconstruction.fits` — the pixelized reconstruction the source fits.
- `<lens>/mock.json` — simulator parameters and provenance.
- `<lens>/<band>/preview.png`, `zoom_comparison.png` — quick-looks against the real stamp.

## Caveats

- `positions.json` / `info.json` are the real lens's; marks land on or near the mock's
  bright features (J0959's AGN images match two marks to 0.02" by construction).
- `mask_extra_galaxies.fits` is all-clear: no contaminants are simulated.
- `noise_map_psffloor.fits` is for layout parity (`PYAUTOLENS_NOISE=psffloor`); the mock
  PSF is exact by construction. No `noise_map_psfcoh.fits`.
- Only the primary band per lens is simulated.
- The J24 and J0959 (pixelized) source_pix fits had the shear at the narrow prior edge;
  J0959's mock uses the source_lp mass instead (above).
