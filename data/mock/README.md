# data/mock/

Mock lens datasets, one per COWLS lens, simulated on 2026-11-14 by
`data_prep/simulate_mock_lenses.py`. Each mock's truth comes from **one completed
source_pix[2] zip**, so the mass model and source are mutually consistent:

| Lens | Band | source_pix[2] zip | mesh+reg | theta_E | SIE q | shear |
| --- | --- | --- | --- | --- | --- | --- |
| `COSJ100018+022138` | F150W | `output_sam_v4/.../ce03e2ba7a348bfae8e799470da902b1.zip` | delaunay+constsplit | 0.369" | 0.72 | 0.071 @ 47 deg |
| `COSJ100024+021749` | F444W | `output_sam_v6/.../70a89aec12a107e3f127174e95d6e347.zip` | delaunay+aspower (psfcoh) | 0.667" | 0.70 | 0.100 @ 58 deg |
| `COSJ095921+020638` | F150W | `output_sam_v4/.../ff53a53af2fd522c9bdc23ae57706a1c.zip` | rtu-img 60x60 | 0.759" | 0.46 | 0.105 @ 54 deg |

Why the pixelized source: the first version of these mocks used the source_lp
parametric source (Sersic-converted MGE), and it could not reproduce the real
arc morphology — J24's arc became two disconnected blobs and J0959's ring was
almost invisible. The pixelized reconstructions reproduce the observed
arcs/ring by construction. Note the J24 and J0959 truths have the shear at the
narrow (+/-0.1) prior edge of the fits they came from.

Components of the truth:

- **lens light**: the zip's MGE bases (bulge + point nucleus), each converted to a
  single flux-conserving Sersic (spherical for the nucleus) by least-squares on
  flux-weighted log-residuals + exact intensity rescale.
- **mass + shear**: the zip's max-likelihood Isothermal + ExternalShear, unchanged.
- **source**: the zip's max-likelihood pixelized source reconstruction. The mock
  image uses the inversion's own *unconvolved* lensed image
  (`inversion.mapped_reconstructed_data`, validated to reproduce the fit's
  convolved model to machine precision) embedded in the full frame; a
  forward-traced interpolation of the mesh was tried and discarded (it painted
  the mesh's background across the frame). The source-plane map itself is saved
  as `source_plane_reconstruction.fits` (0.0025"/px, origin in
  `model_parameters.json`).

## Layout

Same as `data/cowls_sam/`: `<lens>/<band>/{data,noise_map,psf,psf_err,noise_map_psffloor,mask_extra_galaxies}.fits`
plus `info.json`, `positions.json`, `primary_band.json`, and additionally:

- `<lens>/model_parameters.json` — the ground truth as a flat, readable parameter list
  (Sersic lens light with q/PA/r_eff/n, SIE, shear, source-map provenance and flux).
- `<lens>/tracer.json` — Sersic lens + mass + shear as a reloadable `al.Tracer`
  (the source galaxy is empty; its truth is the map).
- `<lens>/source_plane_reconstruction.fits` — the source truth map.
- `<lens>/mock.json` — simulator parameters and provenance.
- `<lens>/<band>/preview.png`, `zoom_comparison.png` — quick-looks against the real stamp.

## Simulation

- Forward model: lensed source from the inversion's unconvolved lensed image; Sersic
  lens light evaluated on a 4x4 sub-sampled grid; summed and passed through
  `al.SimulatorImaging` with the real `psf.fits` kernel (so `psf.fits` is copied verbatim).
- Noise: `sigma^2 = (I + b)/t` with Poisson realisation on `(I + b) * t` counts, fixed seed
  per lens. `t` and `b` are **effective** values fitted to the real `noise_map.fits`
  (binned-median linear fit of noise^2 vs data; the FITS headers carry no EXPTIME).
  Median noise matches the real map to ~1%; the 1/f correction is not reproduced.

## Caveats

- `positions.json` / `info.json` are the real lens's; marks land on S/N 2-61 pixels.
- `mask_extra_galaxies.fits` is all-clear: no contaminants are simulated.
- `noise_map_psffloor.fits` is for layout parity (`PYAUTOLENS_NOISE=psffloor`); the mock PSF
  is exact by construction. No `noise_map_psfcoh.fits` (the coherent floor was fitted to
  real residuals).
- Only the primary band per lens is simulated.
- The truth mass/shear are the source_pix[2] fits' values, which differ slightly from the
  source_lp ones (J24 shear at the 0.1 prior edge, J0959 theta_E 0.76 vs 0.72 in source_lp).
