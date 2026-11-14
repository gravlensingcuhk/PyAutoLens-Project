"""
Simulate a mock lens for each COWLS lens from its best reconstruction (2026-11-14).

__What__

The truth is taken from one completed source_pix[2] zip per lens, so mass model
and source are mutually consistent:

- lens light: the zip's MGE bases (bulge + point nucleus), each converted to a
  single flux-conserving Sersic (spherical for the point basis);
- mass + shear: the zip's max-likelihood Isothermal + ExternalShear, unchanged;
- source: the zip's max-likelihood *pixelized* source reconstruction, sampled
  on its mesh and interpolated (linear, zero outside the convex hull). A
  parametric source cannot reproduce the real arc/ring morphology -- the
  source_lp Sersic mocks (first version of this script) showed J24's arc as
  two disconnected blobs and J0959's ring almost invisible, both inherited
  from the parametric source, not the conversion.

The image is forward-ray-traced on a 4x4 sub-sampled grid (lens equation with
the zip's mass + shear, source brightness interpolated at the traced
positions), the Sersic lens light added, then convolved with the real
`psf.fits` kernel and given Poisson noise via `al.SimulatorImaging`, producing
`data/mock/<lens>/<band>/` with the same layout as `data/cowls_sam/`.

The ground truth is stored per lens as `model_parameters.json` (readable:
Sersic lens light, SIE, shear, source-map provenance), `tracer.json` (Sersic
lens + mass + shear, reloadable; the source galaxy is empty -- its truth is
the map) and `source_plane_reconstruction.fits` (the source map on a regular
0.0025"/px grid).

__Noise parameters__

Effective `exposure_time` (t) and `background_sky_level` (b) fitted to the real
noise map: the simulator's noise model is sigma^2 = (I + b) / t, so a
binned-median linear fit of real noise_map^2 against real data I gives 1/t
(slope) and b/t (intercept). The FITS headers carry no EXPTIME; the real map's
1/f correction is not reproduced.

__Which zips__

- J18   source_pix[2]  output_sam_v4  ce03e2ba  (delaunay+constsplit; clean compact
  source, full ring; the RTU/AdaptSplit reconstructions fragment)
- J24   source_pix[2]  output_sam_v6  70a89aec  (delaunay+aspower, psfcoh)
- J0959 source_pix[2]  output_sam_v4  ff53a53a  (rtu-img 60x60, ptsrc; clear ring)

Run from the project root:  uv run python data_prep/simulate_mock_lenses.py
"""

import json
import os
import shutil
import tempfile
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io import fits as afits
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import least_squares
from scipy.signal import convolve2d

import autolens as al

REAL_ROOT = os.path.join("data", "cowls_sam")
MOCK_ROOT = os.path.join("data", "mock")

LENSES = {
    "COSJ100018+022138": {
        "band": "F150W",
        "pixel_scale": 0.03,
        "seed": 101,
        "source_pix_zip": "output_sam_v4/model_setup_free_source/COSJ100018+022138/"
                          "source_pix[2]/ce03e2ba7a348bfae8e799470da902b1.zip",
        "noise_file": "noise_map.fits",
    },
    "COSJ100024+021749": {
        "band": "F444W",
        "pixel_scale": 0.06,
        "seed": 102,
        "source_pix_zip": "output_sam_v6/model_setup_free_source/COSJ100024+021749/"
                          "source_pix[2]/70a89aec12a107e3f127174e95d6e347.zip",
        "noise_file": "noise_map_psfcoh.fits",
    },
    "COSJ095921+020638": {
        "band": "F150W",
        "pixel_scale": 0.03,
        "seed": 103,
        "source_pix_zip": "output_sam_v4/model_setup_free_source/COSJ095921+020638/"
                          "source_pix[2]/ff53a53af2fd522c9bdc23ae57706a1c.zip",
        "noise_file": "noise_map.fits",
        # The pixelized fit's mass (r_E 0.759) puts the reconstructed AGN position
        # OUTSIDE its caustic (only 2 images). The source_lp mass (r_E 0.719)
        # quadruples the source_lp AGN position and lands two images within
        # 0.02" of the real marks, so J0959's truth takes mass + shear + the AGN
        # point (position, flux) from the v6 source_lp; the host Sersics still
        # come from the pixelized reconstruction.
        "mass_agn_zip": "output_sam_v6/model_setup_free_source/COSJ095921+020638/"
                        "source_lp[1]/9017b6f2b5a2f87d0b6d06627324b265.zip",
    },
}

LABELS = {("lens", "bulge"): "lens_bulge", ("lens", "point"): "lens_nucleus"}

FIT_GRID_SCALE = 0.005  # arcsec/px for the MGE -> Sersic fit
FIT_GRID_HALF = 2.0     # arcsec half-width
SUB = 4                 # sub-sampling of the forward ray-traced image grid
SRC_GRID_SCALE = 0.0025  # arcsec/px of the saved source-plane map

# Number of parametric source components fitted to each pixelized reconstruction.
N_SOURCE_COMPONENTS = {
    "COSJ100018+022138": 2,   # compact clumpy core + broad smooth disk
    "COSJ100024+021749": 3,   # three clumps along the banana
    "COSJ095921+020638": 3,   # AGN spike + host core + extended halo
}

# Lenses whose source has a genuine point-source AGN. After the Sersic fit, the
# most compact component is removed from the smooth profiles and treated as a
# true point source: its multiple images are solved with `al.PointSolver`, its
# magnifications from a finite-difference Jacobian of the lens equation, and
# its flux x |mu| is added to the image pixel containing each image (a delta
# evaluated as a light profile mis-accounts flux when the pixel scale is much
# larger than the profile width).
POINT_AGN = {"COSJ095921+020638"}


def fit_source_parametric(mesh, recon, n_comp):
    """Fit `n_comp` elliptical Sersics to the source-plane reconstruction.

    The map is interpolated onto a regular grid; the fit uses sqrt-flux-weighted
    linear residuals over the significant pixels (> 0.3% of peak), initialised
    from the brightest separated peaks. Total flux is conserved over the >1%-peak
    region. Returns (profiles, map_info, quality).
    """
    interp = LinearNDInterpolator(mesh, recon, fill_value=0.0)
    pad = 0.05
    y0, y1 = mesh[:, 0].min() - pad, mesh[:, 0].max() + pad
    x0, x1 = mesh[:, 1].min() - pad, mesh[:, 1].max() + pad
    nys = int((y1 - y0) / SRC_GRID_SCALE) + 1
    nxs = int((x1 - x0) / SRC_GRID_SCALE) + 1
    sy, sx = np.meshgrid(np.linspace(y0, y1, nys), np.linspace(x0, x1, nxs), indexing="ij")
    src_map = np.clip(interp(sy, sx), 0.0, None)

    peak = float(src_map.max())
    use = src_map > 3e-3 * peak
    yy, xx = sy[use], sx[use]
    vals = src_map[use]
    fit_grid = al.Grid2DIrregular(values=np.column_stack([yy, xx]))
    weight = np.sqrt(vals / peak)

    # Initial centres: brightest peaks separated by >= 0.08".
    from scipy.ndimage import maximum_filter, label as ndi_label, center_of_mass
    peaks_mask = (src_map == maximum_filter(src_map, size=int(0.08 / SRC_GRID_SCALE))) \
        & (src_map > 0.05 * peak)
    lab, _ = ndi_label(peaks_mask)
    peaks = sorted(
        (center_of_mass(src_map, lab, i) for i in range(1, lab.max() + 1)),
        key=lambda c: -src_map[int(round(c[0])), int(round(c[1]))],
    )
    centres = [(y0 + c[0] * SRC_GRID_SCALE, x0 + c[1] * SRC_GRID_SCALE) for c in peaks]
    cy_flux = float(np.average(yy, weights=vals))
    cx_flux = float(np.average(xx, weights=vals))
    while len(centres) < n_comp:
        centres.append((cy_flux, cx_flux))
    centres = centres[:n_comp]

    def unpack(p):
        profs = []
        for i in range(n_comp):
            cy, cx, e1, e2, log_i, log_r, n = p[7 * i:7 * i + 7]
            profs.append(al.lp.Sersic(centre=(cy, cx), ell_comps=(e1, e2),
                                      intensity=10 ** log_i,
                                      effective_radius=10 ** log_r, sersic_index=n))
        return profs

    def residual(p):
        model = np.sum([np.asarray(pr.image_2d_from(grid=fit_grid))
                        for pr in unpack(p)], axis=0)
        return (model - vals) * weight

    p0, lo, hi = [], [], []
    for i, (cy, cx) in enumerate(centres):
        r_init = [0.02, 0.08, 0.25][i] if i < 3 else 0.2
        p0 += [cy, cx, 0.0, 0.0, np.log10(max(src_map.max() * 0.3, 1e-6)),
               np.log10(r_init), 1.5]
        lo += [cy - 0.4, cx - 0.4, -0.95, -0.95, -8, np.log10(0.004), 0.5]
        hi += [cy + 0.4, cx + 0.4, 0.95, 0.95, 8, np.log10(2.0), 6.0]
    res = least_squares(residual, p0, bounds=(lo, hi), max_nfev=500)
    profs = unpack(res.x)

    # Conserve flux over the >1%-peak region.
    region = src_map > 1e-2 * peak
    full_grid = al.Grid2DIrregular(
        values=np.column_stack([sy[region].ravel(), sx[region].ravel()]))
    model_flux = float(np.sum([np.asarray(pr.image_2d_from(grid=full_grid)).sum()
                               for pr in profs]))
    scale = float(src_map[region].sum() / model_flux)
    for pr in profs:
        pr.intensity *= scale

    rms = float(np.sqrt(np.mean((res.fun / weight) ** 2)))
    quality = {"rms_surface_brightness": rms,
               "rms_over_peak": rms / peak,
               "flux_conserved_over": ">1% of peak region",
               "n_components": n_comp}
    map_info = {"src_map": src_map, "y0": y0, "x0": x0, "sy": sy, "sx": sx}
    return profs, map_info, quality


def noise_params_from(data, noise, pixel_scale):
    """Effective (exposure_time, background_sky_level) matching sigma^2 = (I + b) / t."""
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    radius = np.hypot(yy - ny / 2, xx - nx / 2) * pixel_scale
    use = (radius < 4.2) & np.isfinite(data) & np.isfinite(noise) & (noise > 0)
    I, s2 = data[use].ravel(), noise[use].ravel() ** 2
    bins = np.quantile(I, np.linspace(0, 1, 26))
    binned = [(I[(I >= lo) & (I < hi)], s2[(I >= lo) & (I < hi)])
              for lo, hi in zip(bins[:-1], bins[1:])]
    Ib = np.array([np.median(a) for a, _ in binned if len(a) > 20])
    Sb = np.array([np.median(s) for _, s in binned if len(s) > 20])
    (inv_t, c), *_ = np.linalg.lstsq(np.vstack([Ib, np.ones_like(Ib)]).T, Sb, rcond=None)
    t = 1.0 / inv_t
    b = c * t
    return t, b


def ell_to_q_pa(e1, e2):
    e = np.hypot(e1, e2)
    q = (1.0 - e) / (1.0 + e) if e > 0 else 1.0
    return float(q), float(0.5 * np.degrees(np.arctan2(e2, e1)))


def fit_coords():
    n = int(FIT_GRID_HALF * 2 / FIT_GRID_SCALE) + 1
    ax = np.linspace(-FIT_GRID_HALF, FIT_GRID_HALF, n)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    return np.column_stack([yy.ravel(), xx.ravel()])


def fit_sersic(target, coords, spherical):
    """Least-squares-fit a single Sersic to a basis image (flux-weighted log
    residuals), then rescale intensity to conserve total flux."""
    target = np.asarray(target)
    yy, xx = coords[:, 0], coords[:, 1]
    peak = float(target.max())
    floor = peak * 1e-5
    use = target > floor
    cy0 = float(np.average(yy[use], weights=target[use]))
    cx0 = float(np.average(xx[use], weights=target[use]))
    fit_grid = al.Grid2DIrregular(values=np.column_stack([yy[use], xx[use]]))
    weight = np.sqrt(target[use] / peak)
    log_t = np.log10(target[use] + floor)

    def residual(p):
        if spherical:
            cy, cx, log_i, log_r, n = p
            prof = al.lp.SersicSph(centre=(cy, cx), intensity=10 ** log_i,
                                   effective_radius=10 ** log_r, sersic_index=n)
        else:
            cy, cx, e1, e2, log_i, log_r, n = p
            prof = al.lp.Sersic(centre=(cy, cx), ell_comps=(e1, e2),
                                intensity=10 ** log_i, effective_radius=10 ** log_r,
                                sersic_index=n)
        model = np.asarray(prof.image_2d_from(grid=fit_grid))
        return (np.log10(model + floor) - log_t) * weight

    if spherical:
        p0 = [cy0, cx0, np.log10(peak), np.log10(0.1), 2.0]
        bounds = ([cy0 - 0.05, cx0 - 0.05, -8, -4, 0.5],
                  [cy0 + 0.05, cx0 + 0.05, 6, 0.7, 8.0])
    else:
        p0 = [cy0, cx0, 0.0, 0.0, np.log10(peak), np.log10(0.1), 2.0]
        bounds = ([cy0 - 0.05, cx0 - 0.05, -0.99, -0.99, -8, -4, 0.5],
                  [cy0 + 0.05, cx0 + 0.05, 0.99, 0.99, 6, 0.7, 8.0])
    res = least_squares(residual, p0, bounds=bounds, max_nfev=200)
    p = res.x
    if spherical:
        prof = al.lp.SersicSph(centre=(p[0], p[1]), intensity=10 ** p[2],
                               effective_radius=10 ** p[3], sersic_index=p[4])
    else:
        prof = al.lp.Sersic(centre=(p[0], p[1]), ell_comps=(p[2], p[3]),
                            intensity=10 ** p[4], effective_radius=10 ** p[5],
                            sersic_index=p[6])
    model_full = np.asarray(prof.image_2d_from(grid=al.Grid2DIrregular(values=coords)))
    prof.intensity *= float(target.sum() / model_full.sum())
    quality = {"log_rms": float(np.sqrt(np.mean((res.fun / weight) ** 2))),
               "flux_conserved": True}
    return prof, quality


def sersic_params_json(prof):
    e1, e2 = getattr(prof, "ell_comps", (0.0, 0.0))
    q, pa = ell_to_q_pa(e1, e2)
    d = {"profile": type(prof).__name__,
         "centre": [float(prof.centre[0]), float(prof.centre[1])],
         "ell_comps": [float(e1), float(e2)],
         "axis_ratio": q, "angle_deg": pa,
         "intensity": float(prof.intensity)}
    if hasattr(prof, "effective_radius"):
        d["effective_radius"] = float(prof.effective_radius)
    if hasattr(prof, "sersic_index"):
        d["sersic_index"] = float(prof.sersic_index)
    if hasattr(prof, "sigma"):
        d["sigma"] = float(prof.sigma)
        d["note"] = "delta-like point source (AGN); PSF-shaped after convolution"
    return d


def agn_images_from(mass_tracer, beta):
    """(position, |magnification|) for each image of a point source at beta.

    Images from `al.PointSolver`; magnifications from a finite-difference
    Jacobian of the lens equation (this autolens version has no tracer
    magnification method).
    """
    solver = al.PointSolver.for_limits_and_scale(
        y_min=-2.5, y_max=2.5, x_min=-2.5, x_max=2.5,
        scale=0.05, pixel_scale_precision=0.001)
    pos = np.asarray(solver.solve(tracer=mass_tracer,
                                  source_plane_coordinate=tuple(beta)))
    pos = pos[np.isfinite(pos).all(axis=1)]
    eps = 1e-4
    images = []
    for p in pos:
        jac = np.zeros((2, 2))
        for ax in range(2):
            dp = np.zeros(2)
            dp[ax] = eps
            gp = np.asarray(mass_tracer.deflections_yx_2d_from(
                grid=al.Grid2DIrregular(values=np.array([p + dp]))))[0]
            gm = np.asarray(mass_tracer.deflections_yx_2d_from(
                grid=al.Grid2DIrregular(values=np.array([p - dp]))))[0]
            jac[:, ax] = (gp - gm) / (2 * eps)
        mu = 1.0 / abs(np.linalg.det(np.eye(2) - jac))
        images.append((np.asarray(p), float(mu)))
    return images


def truth_from_zip(cfg):
    """Load the source_pix zip and rebuild its max-likelihood fit.

    Returns (tracer, lensed_slim, slim_coords, mesh, recon):
    - tracer: the zip's max-likelihood tracer;
    - lensed_slim: the *unconvolved* lensed source image on the masked slim grid
      (the inversion's `mapped_reconstructed_data`; convolving it with the PSF
      reproduces the fit's per-galaxy model image to machine precision);
    - slim_coords: (y, x) of those slim pixels, for embedding in the full frame;
    - mesh, recon: the source-plane mesh and reconstruction (for the truth map).
    """
    lens, band, ps = cfg["_lens"], cfg["band"], cfg["pixel_scale"]
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(cfg["source_pix_zip"]).extractall(tmp)
        tracer = al.from_json(os.path.join(tmp, "files", "tracer.json"))
        dp = os.path.join(REAL_ROOT, lens, band)
        dataset = al.Imaging.from_fits(
            data_path=os.path.join(dp, "data.fits"),
            noise_map_path=os.path.join(dp, cfg["noise_file"]),
            psf_path=os.path.join(dp, "psf.fits"), pixel_scales=ps)
        info = json.load(open(os.path.join(REAL_ROOT, lens, "info.json")))
        mask = al.Mask2D.circular(shape_native=dataset.shape_native, pixel_scales=ps,
                                  radius=info["mask_radius"])
        dataset = dataset.apply_mask(mask=mask)

        source = tracer.galaxies[-1]
        ny, nx = dataset.shape_native

        def to_masked(arr):
            arr = np.asarray(arr, dtype=np.float64)
            if arr.shape != (ny, nx):  # centre-crop a full stamp to the trimmed shape
                cy, cx = (arr.shape[0] - ny) // 2, (arr.shape[1] - nx) // 2
                arr = arr[cy:cy + ny, cx:cx + nx]
            return al.Array2D.no_mask(values=arr, pixel_scales=ps).apply_mask(
                mask=dataset.mask)

        kwargs = {}
        img_path = os.path.join(tmp, "image", "adapt_images.fits")
        if os.path.exists(img_path):
            img_hdul = afits.open(img_path)
            key = [k for k in img_hdul[1:]
                   if "SOURCE" in str(k.header.get("EXTNAME", "")).upper()]
            if key:
                kwargs["galaxy_image_dict"] = {source: to_masked(img_hdul[key[0]].data)}
        grid_path = os.path.join(tmp, "image", "adapt_image_plane_mesh_grids.fits")
        if os.path.exists(grid_path):
            grid_hdul = afits.open(grid_path)
            kwargs["galaxy_image_plane_mesh_grid_dict"] = {
                source: al.Grid2DIrregular(values=np.asarray(grid_hdul[1].data,
                                                             dtype=np.float64))}
        adapt = al.AdaptImages(**kwargs) if kwargs else None
        fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt)
        inv = fit.inversion
        mapper, values = next(iter(inv.reconstruction_dict.items()))
        lensed = np.asarray(inv.mapped_reconstructed_data_dict[mapper])
        slim_coords = np.asarray(fit.grids.lp.slim)
        return (tracer, lensed, slim_coords,
                np.asarray(mapper.source_plane_mesh_grid), np.asarray(values))


def convert_lens_light(tracer):
    """MGE lens-light bases -> Sersics; returns (kwargs for the new lens Galaxy,
    params dict, fit-quality dict)."""
    coords = fit_coords()
    lens_g = tracer.galaxies[0]
    kwargs, params, quality = {}, {}, {}
    for comp in ("bulge", "point"):
        basis = getattr(lens_g, comp, None)
        if basis is None:
            continue
        target = np.asarray(al.Galaxy(redshift=lens_g.redshift, comp=basis)
                            .image_2d_from(grid=al.Grid2DIrregular(values=coords)))
        prof, q = fit_sersic(target, coords, spherical=(comp == "point"))
        kwargs[comp] = prof
        params[LABELS[("lens", comp)]] = sersic_params_json(prof)
        quality[LABELS[("lens", comp)]] = q
    return kwargs, params, quality


def sub_grid_coords(shape_native, pixel_scale):
    """(y, x) of every sub-pixel centre of the native grid, SUB x SUB per pixel."""
    ny, nx = shape_native
    ys = -(np.arange(ny) - (ny - 1) / 2.0) * pixel_scale
    xs = (np.arange(nx) - (nx - 1) / 2.0) * pixel_scale
    off = (np.arange(SUB) + 0.5) / SUB * pixel_scale - pixel_scale / 2.0
    pts = []
    for dy in off:
        for dx in off:
            yy, xx = np.meshgrid(ys + dy, xs + dx, indexing="ij")
            pts.append(np.column_stack([yy.ravel(), xx.ravel()]))
    # shape (SUB*SUB, ny*nx, 2) -> stack to (N_sub_total, 2) keeping pixel grouping
    return np.stack(pts, axis=1).reshape(-1, SUB * SUB, 2)


def simulate(lens, cfg):
    band, ps = cfg["band"], cfg["pixel_scale"]
    cfg = dict(cfg, _lens=lens)
    real_path = os.path.join(REAL_ROOT, lens, band)
    out_lens = os.path.join(MOCK_ROOT, lens)
    out_band = os.path.join(out_lens, band)
    os.makedirs(out_band, exist_ok=True)

    tracer, lensed_slim, slim_coords, mesh, recon = truth_from_zip(cfg)
    lens_kwargs, light_params, fit_quality = convert_lens_light(tracer)
    lens_g = tracer.galaxies[0]
    source_g = tracer.galaxies[-1]
    mass, shear = lens_g.mass, getattr(lens_g, "shear", None)

    # Parametric source: Sersics fitted to the pixelized reconstruction.
    src_profs, map_info, src_quality = fit_source_parametric(
        mesh, recon, N_SOURCE_COMPONENTS[lens])
    src_map = map_info["src_map"]

    # Point-source AGN. J0959: position and flux from the source_lp point basis,
    # and mass + shear from source_lp too (see LENSES note); the most compact
    # Sersic is still popped so the host fit does not double-count the spike.
    agn = None
    if lens in POINT_AGN:
        i_min = int(np.argmin([float(pr.effective_radius) for pr in src_profs]))
        src_profs.pop(i_min)
    if cfg.get("mass_agn_zip"):
        with tempfile.TemporaryDirectory() as tmp:
            zipfile.ZipFile(cfg["mass_agn_zip"]).extractall(tmp)
            lp_tracer = al.from_json(os.path.join(tmp, "files", "tracer.json"))
        lp_lens, lp_source = lp_tracer.galaxies[0], lp_tracer.galaxies[-1]
        mass, shear = lp_lens.mass, getattr(lp_lens, "shear", None)
        pb = getattr(lp_source, "point", None)
        profs = getattr(pb, "profile_list", None) or getattr(pb, "light_profile_list", None)
        if profs:
            agn = {
                "centre": (float(profs[0].centre[0]), float(profs[0].centre[1])),
                "flux": float(sum(2.0 * np.pi * float(p.intensity) * float(p.sigma) ** 2
                                  for p in profs)),
            }
    agn_flux_fixed = agn is not None  # AGN flux is fitted to the knots below; only the host is rescaled

    # --- forward image ------------------------------------------------------
    real = al.Imaging.from_fits(
        data_path=os.path.join(real_path, "data.fits"),
        noise_map_path=os.path.join(real_path, "noise_map.fits"),
        psf_path=os.path.join(real_path, "psf.fits"), pixel_scales=ps)
    t, b = noise_params_from(real.data.native, real.noise_map.native, ps)

    # Full parametric tracer (Sersic lens light + SIE + shear + Sersic source).
    def make_tracer():
        return al.Tracer(galaxies=[
            al.Galaxy(redshift=lens_g.redshift, mass=mass,
                      **({**lens_kwargs, "shear": shear} if shear is not None
                         else lens_kwargs)),
            al.Galaxy(redshift=source_g.redshift,
                      **{f"comp{i}": pr for i, pr in enumerate(src_profs)}),
        ])

    # Lensed-flux normalisation: a smoothed source redistributes flux across the
    # caustic, so source-plane flux conservation does NOT conserve lensed flux
    # (J0959's compact core gave 6.4x too much). Rescale the source intensities
    # so the lensed flux matches the inversion's lensed image on its bright
    # pixels -- the arcs then have the observed brightness.
    lens_light_image = np.asarray(
        al.Galaxy(redshift=lens_g.redshift, **lens_kwargs)
        .image_2d_from(grid=al.Grid2DIrregular(values=slim_coords)))
    bright = lensed_slim > 0.1 * lensed_slim.max()

    mass_tracer = al.Tracer(galaxies=[
        al.Galaxy(redshift=lens_g.redshift, mass=mass,
                  **({"shear": shear} if shear is not None else {})),
        al.Galaxy(redshift=source_g.redshift),
    ])
    agn_image_list = agn_images_from(mass_tracer, agn["centre"]) if agn else []

    def add_agn(values, coords, pixel_area, flux):
        for p, mu in agn_image_list:
            i = int(np.argmin(np.sum((coords - p) ** 2, axis=1)))
            values[i] += mu * flux / pixel_area
        return values

    lensed_host = np.asarray(make_tracer().image_2d_from(
        grid=al.Grid2DIrregular(values=slim_coords))) - lens_light_image
    lensed_agn = np.zeros_like(lensed_host)
    if agn:
        # AGN flux measured from the REAL data: 3x3 aperture at each model image
        # position, lens light subtracted, per-image F = knot_flux / mu, averaged.
        # (The source_lp point-basis flux would make the knots ~10x brighter than
        # observed -- that fit let the point basis soak up ring flux.)
        lens_sersic = al.Galaxy(redshift=lens_g.redshift, **lens_kwargs)
        sub_tmp = sub_grid_coords(real.shape_native, ps)
        lens_full = np.asarray(lens_sersic.image_2d_from(
            grid=al.Grid2DIrregular(values=sub_tmp.reshape(-1, 2))))
        lens_full = lens_full.reshape(-1, SUB * SUB).mean(axis=1).reshape(real.shape_native)
        ring = real.data.native - lens_full
        nyf, nxf = real.shape_native
        f_est = []
        for p, mu in agn_image_list:
            iy = int(np.rint((nyf - 1) / 2.0 - p[0] / ps))
            ix = int(np.rint(p[1] / ps + (nxf - 1) / 2.0))
            ap = ring[iy - 1:iy + 2, ix - 1:ix + 2]
            knot = max(0.0, float(ap.sum()) * ps ** 2)
            f_est.append(knot / mu)
        agn["flux"] = float(np.mean(f_est)) if f_est else 0.0
        lensed_agn = add_agn(lensed_agn, slim_coords, ps ** 2, agn["flux"])
    if agn_flux_fixed:
        # AGN fixed at its fitted flux; the host is rescaled to the remaining
        # brightness of the inversion's lensed image.
        rescale = float((lensed_slim[bright].sum() - lensed_agn[bright].sum())
                        / lensed_host[bright].sum())
        for pr in src_profs:
            pr.intensity *= rescale
    else:
        rescale = float(lensed_slim[bright].sum()
                        / (lensed_host + lensed_agn)[bright].sum())
        for pr in src_profs:
            pr.intensity *= rescale
        if agn:
            agn["flux"] *= rescale
    src_quality["lensed_flux_rescale"] = rescale

    true_tracer = make_tracer()
    sub = sub_grid_coords(real.shape_native, ps)          # (n_pix, SUB^2, 2)
    grid_irreg = al.Grid2DIrregular(values=sub.reshape(-1, 2))
    img = np.asarray(true_tracer.image_2d_from(grid=grid_irreg))
    if agn:
        img = add_agn(img, sub.reshape(-1, 2), (ps / SUB) ** 2, agn["flux"])
    img = img.reshape(-1, SUB * SUB).mean(axis=1)
    image = al.Array2D.no_mask(values=img.reshape(real.shape_native), pixel_scales=ps)

    # Lensed-image consistency after rescaling.
    lensed_par = np.asarray(true_tracer.image_2d_from(
        grid=al.Grid2DIrregular(values=slim_coords))) - lens_light_image
    if agn:
        lensed_par = add_agn(lensed_par, slim_coords, ps ** 2, agn["flux"])
    src_quality["lensed_frac_rms_bright"] = float(
        np.sqrt(np.mean((lensed_par[bright] - lensed_slim[bright]) ** 2))
        / lensed_slim[bright].mean())
    src_quality["lensed_flux_ratio"] = float(
        lensed_par[bright].sum() / lensed_slim[bright].sum())

    simulator = al.SimulatorImaging(
        exposure_time=t, background_sky_level=b, psf=real.psf,
        add_poisson_noise_to_data=True, noise_seed=cfg["seed"],
    )
    dataset = simulator.via_image_from(image=image)

    al.output_to_fits(values=dataset.data.native,
                      file_path=os.path.join(out_band, "data.fits"), overwrite=True)
    al.output_to_fits(values=dataset.noise_map.native,
                      file_path=os.path.join(out_band, "noise_map.fits"), overwrite=True)
    shutil.copy(os.path.join(real_path, "psf.fits"), os.path.join(out_band, "psf.fits"))
    shutil.copy(os.path.join(real_path, "psf_err.fits"),
                os.path.join(out_band, "psf_err.fits"))
    fits.writeto(os.path.join(out_band, "mask_extra_galaxies.fits"),
                 np.zeros(real.shape_native, dtype=np.int32), overwrite=True)
    psf_err = fits.getdata(os.path.join(real_path, "psf_err.fits")).astype(float)
    var_psf = convolve2d(dataset.data.native ** 2, psf_err ** 2,
                         mode="same", boundary="fill", fillvalue=0.0)
    fits.writeto(os.path.join(out_band, "noise_map_psffloor.fits"),
                 np.sqrt(dataset.noise_map.native ** 2 + var_psf).astype(np.float32),
                 overwrite=True)

    # --- source-plane reference maps ---------------------------------------
    fits.writeto(os.path.join(out_lens, "source_plane_reconstruction.fits"),
                 src_map.astype(np.float32), overwrite=True)
    sy, sx = map_info["sy"], map_info["sx"]
    src_model_map = np.zeros_like(src_map)
    map_grid = al.Grid2DIrregular(
        values=np.column_stack([sy.ravel(), sx.ravel()]))
    for pr in src_profs:
        src_model_map += np.asarray(pr.image_2d_from(grid=map_grid)).reshape(src_map.shape)
    if agn:
        iy = int(np.argmin(np.abs(sy[:, 0] - agn["centre"][0])))
        ix = int(np.argmin(np.abs(sx[0, :] - agn["centre"][1])))
        src_model_map[iy, ix] += agn["flux"] / SRC_GRID_SCALE ** 2
    fits.writeto(os.path.join(out_lens, "source_plane_model.fits"),
                 src_model_map.astype(np.float32), overwrite=True)

    # --- ground-truth tracer (fully parametric) ----------------------------
    al.output_to_json(obj=true_tracer, file_path=os.path.join(out_lens, "tracer.json"))

    q_m, pa_m = ell_to_q_pa(*mass.ell_comps)
    params = {
        "lens": lens, "band": band,
        "note": "Ground truth of this mock -- fully parametric. Lens light is Sersics "
                "fitted to the MGE bases of the source_pix[2] max-likelihood tracer "
                "(flux-conserving); mass and shear are that fit's values; the source "
                "is Sersics fitted to the fit's pixelized reconstruction "
                "(source_plane_reconstruction.fits is the reconstruction, "
                "source_plane_model.fits is the parametric truth).",
        "redshifts": {"lens": float(lens_g.redshift), "source": float(source_g.redshift)},
        "light": light_params,
        "mass": {"profile": f"{type(mass).__name__} (SIE, slope={getattr(mass, 'slope', 2.0)})",
                 "centre": [float(mass.centre[0]), float(mass.centre[1])],
                 "ell_comps": [float(mass.ell_comps[0]), float(mass.ell_comps[1])],
                 "axis_ratio": q_m, "angle_deg": pa_m,
                 "einstein_radius": float(mass.einstein_radius),
                 "from_zip": cfg.get("mass_agn_zip", cfg["source_pix_zip"])},
        "source": {
            "type": "parametric (Sersics fitted to the pixelized reconstruction)",
            "source_pix_zip": cfg["source_pix_zip"],
            "components": {f"comp{i}": sersic_params_json(pr)
                           for i, pr in enumerate(src_profs)},
            "fit_quality": src_quality,
            "reference_map_file": "source_plane_reconstruction.fits",
            "reference_map_pixel_scale": SRC_GRID_SCALE,
            "reference_map_origin_yx": [float(map_info["y0"]), float(map_info["x0"])],
        },
        "fit_quality": fit_quality,
    }
    if agn:
        params["source"]["point_agn"] = {
            "type": "point source (AGN); flux x |magnification| is added to the "
                    "image pixel containing each solved image",
            "centre": [float(agn["centre"][0]), float(agn["centre"][1])],
            "flux": agn["flux"],
            "images": [{"position": [float(p[0]), float(p[1])],
                        "magnification": mu} for p, mu in agn_image_list],
        }
    if shear is not None:
        g1, g2 = float(shear.gamma_1), float(shear.gamma_2)
        params["external_shear"] = {"gamma_1": g1, "gamma_2": g2,
                                    "gamma": float(np.hypot(g1, g2)),
                                    "angle_deg": float(0.5 * np.degrees(np.arctan2(g2, g1)))}
    json.dump(params, open(os.path.join(out_lens, "model_parameters.json"), "w"), indent=4)

    for name in ("positions.json", "primary_band.json"):
        shutil.copy(os.path.join(REAL_ROOT, lens, name), os.path.join(out_lens, name))
    info = json.load(open(os.path.join(REAL_ROOT, lens, "info.json")))
    info["mock"] = True
    json.dump(info, open(os.path.join(out_lens, "info.json"), "w"), indent=4)
    mock_meta = {
        "mock": True,
        "made": "2026-11-14",
        "script": "data_prep/simulate_mock_lenses.py",
        "ground_truth": "model_parameters.json + tracer.json (fully parametric: lens "
                        "Sersics, SIE, shear, source Sersics); "
                        "source_plane_model.fits is the parametric source truth, "
                        "source_plane_reconstruction.fits the reconstruction it fits",
        "source_pix_zip": cfg["source_pix_zip"],
        "real_dataset": os.path.join(real_path, "data.fits"),
        "simulator": {
            "exposure_time_effective": t,
            "background_sky_level_effective": b,
            "noise_model": "sigma^2 = (I + b) / t, Poisson on (I + b) * t counts",
            "noise_seed": cfg["seed"],
            "psf": "real psf.fits kernel, real-space convolution",
            "forward_model": f"full parametric tracer on a {SUB}x{SUB} sub-sampled grid",
            "note": "t and b are effective values fitted to the real noise map; "
                    "the real map's 1/f correction is not reproduced.",
        },
        "caveats": [
            "positions.json and info.json are copied from the real lens.",
            "mask_extra_galaxies.fits is all-clear (no contaminants are simulated).",
            "noise_map_psffloor.fits uses the real psf_err.fits; the mock PSF is exact, "
            "so the floor is for layout parity only.",
        ],
    }
    json.dump(mock_meta, open(os.path.join(out_lens, "mock.json"), "w"), indent=4)

    # --- quick-look ---------------------------------------------------------
    fig, axes = plt.subplots(1, 6, figsize=(32, 5.5))
    mock_img = dataset.data.native
    vmax = np.percentile(mock_img[mock_img > 0], 99.5)
    panels = [(mock_img, "mock data"), (dataset.noise_map.native, "mock noise"),
              (mock_img / dataset.noise_map.native, "mock S/N"),
              (real.data.native, "real data"),
              (src_map, "source reconstruction"),
              (src_model_map, "source truth (parametric)")]
    for ax, (arr, title) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower",
                       vmax=vmax if title in ("mock data", "real data") else None)
        ax.set_title(f"{lens} {band}\n{title}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_band, "preview.png"), dpi=110)
    plt.close(fig)

    print(f"{lens} {band}: t={t:.4g} b={b:.4g} "
          f"noise med={np.median(dataset.noise_map.native):.4g} "
          f"(real {np.median(real.noise_map.native):.4g}) "
          f"peak S/N={np.max(mock_img / dataset.noise_map.native):.0f} -> {out_band}")
    for label, qq in fit_quality.items():
        print(f"   {label}: log_rms={qq['log_rms']:.3f}")
    print(f"   source: {N_SOURCE_COMPONENTS[lens]} Sersics, "
          f"lensed frac rms={src_quality['lensed_frac_rms_bright']:.3f}, "
          f"lensed flux ratio={src_quality['lensed_flux_ratio']:.3f}")
    if agn:
        print(f"   AGN: point source, flux {agn['flux']:.3g}, "
              f"{len(agn_image_list)} images, mu: "
              + " ".join(f"{m:.1f}" for _, m in agn_image_list))


for lens, cfg in LENSES.items():
    simulate(lens, cfg)
