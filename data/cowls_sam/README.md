# data/cowls_sam/

The re-reduced COWLS stamps of Sam's JWST reduction (`JWST_lenses_data_reduction`,
`data/stamps/cowls_spectacular/prime/<lens>/<filter>/`), copied on 2026-11-02 from that
repository at commit 8d873c8 (the J18 contaminant masks of all four bands and the J18 F150W mask
were uncommitted edits of 2026-11-02 21:14 at the time). The runners read this directory since
2026-11-02; `data/cowls/` holds the earlier COWLS-shipped products (J24 F444W noise map rebuilt
2026-10-29) and is kept for the archived runs.

| File here | Source file | Notes |
| --- | --- | --- |
| `<filter>/data.fits` | `stamp12_sci.fits` | 12" x 12" stamp, MJy/sr; 400 px at 0.03"/px (F115W, F150W), 200 px at 0.06"/px (F277W, F444W) |
| `<filter>/noise_map.fits` | `stamp12_noise.fits` | 1 sigma per pixel, includes the measured 1/f correction (`NOISECOR` header) |
| `<filter>/psf.fits` | `psf.fits` | STPSF injected into the lens's own exposures (`PSFMETH = inject_stpsf`), 37 x 37 (SW) / 41 x 41 (LW), sums to 1 |
| `<filter>/psf_err.fits` | `psf_err.fits` | per-pixel PSF uncertainty, not used by the runners |
| `<filter>/mask_extra_galaxies.fits` | `stamp12_mask.fits` | 1 = exclude (contaminant), 0 = use; the runners load it with `invert=True` and noise-scale the 1s |
| `positions.json` | `<primary filter>/stamp12_positions.json` | hand-marked multiple images, PyAutoLens (y, x) arcsec from the stamp centre; identical across bands |
| `info.json` | `data/cowls/<lens>/info.json` | `mask_radius`, redshifts, unchanged |
| `primary_band.json` | `primary_band.json` | the reduction's band choice and per-band arc S/N |

Orientation and conventions: the reduction's README ("Orientation", "Image positions"). The
stamps, PSFs, masks and positions are mutually consistent as `al.Imaging.from_fits` loads them
here (checked 2026-11-02: the marked positions fall on S/N 10-17 arc pixels in J18 F150W and
43-147 in J24 F444W). The old `data/cowls/` stamps are on the same centre to within a pixel
(the 419 vs 400 px cut), but the positions were re-marked in the reduction's GUI and differ from
the old marks by up to 0.1", so the old `positions.json` is not reused.
