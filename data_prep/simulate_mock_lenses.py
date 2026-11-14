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
    },
}

LABELS = {("lens", "bulge"): "lens_bulge", ("lens", "point"): "lens_nucleus"}

FIT_GRID_SCALE = 0.005  # arcsec/px for the MGE -> Sersic fit
FIT_GRID_HALF = 2.0     # arcsec half-width
SUB = 4                 # sub-sampling of the forward ray-traced image grid
SRC_GRID_SCALE = 0.0025  # arcsec/px of the saved source-plane map


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
    return {"profile": type(prof).__name__,
            "centre": [float(prof.centre[0]), float(prof.centre[1])],
            "ell_comps": [float(e1), float(e2)],
            "axis_ratio": q, "angle_deg": pa,
            "intensity": float(prof.intensity),
            "effective_radius": float(prof.effective_radius),
            "sersic_index": float(prof.sersic_index)}


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

    # --- forward image ------------------------------------------------------
    real = al.Imaging.from_fits(
        data_path=os.path.join(real_path, "data.fits"),
        noise_map_path=os.path.join(real_path, "noise_map.fits"),
        psf_path=os.path.join(real_path, "psf.fits"), pixel_scales=ps)
    t, b = noise_params_from(real.data.native, real.noise_map.native, ps)

    # Lens light: Sersics evaluated on a SUB x SUB sub-sampled full-frame grid.
    sub = sub_grid_coords(real.shape_native, ps)          # (n_pix, SUB^2, 2)
    grid_irreg = al.Grid2DIrregular(values=sub.reshape(-1, 2))
    lens_sersic = al.Galaxy(redshift=lens_g.redshift, **lens_kwargs)
    img_lens = np.asarray(lens_sersic.image_2d_from(grid=grid_irreg))
    img_lens = img_lens.reshape(-1, SUB * SUB).mean(axis=1)

    # Lensed source: the inversion's own unconvolved lensed image (validated to
    # reproduce the fit's convolved model image to machine precision), embedded
    # into the full frame by its (y, x) coordinates. Zero outside the mask --
    # the ring is well inside it, and the PSF convolution below blurs outward.
    nyf, nxf = real.shape_native
    img_src = np.zeros(nyf * nxf)
    iy = np.rint((nyf - 1) / 2.0 - slim_coords[:, 0] / ps).astype(int)
    ix = np.rint(slim_coords[:, 1] / ps + (nxf - 1) / 2.0).astype(int)
    ok = (iy >= 0) & (iy < nyf) & (ix >= 0) & (ix < nxf)
    img_src[iy[ok] * nxf + ix[ok]] = lensed_slim[ok]

    image = al.Array2D.no_mask(values=(img_lens + img_src).reshape(real.shape_native),
                               pixel_scales=ps)

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

    # --- source-plane map on a regular grid --------------------------------
    interp = LinearNDInterpolator(mesh, recon, fill_value=0.0)
    pad = 0.05
    y0, y1 = mesh[:, 0].min() - pad, mesh[:, 0].max() + pad
    x0, x1 = mesh[:, 1].min() - pad, mesh[:, 1].max() + pad
    nys = int((y1 - y0) / SRC_GRID_SCALE) + 1
    nxs = int((x1 - x0) / SRC_GRID_SCALE) + 1
    sy, sx = np.meshgrid(np.linspace(y0, y1, nys), np.linspace(x0, x1, nxs), indexing="ij")
    src_map = interp(sy, sx)
    fits.writeto(os.path.join(out_lens, "source_plane_reconstruction.fits"),
                 src_map.astype(np.float32), overwrite=True)

    # --- ground-truth tracer (source galaxy empty; its truth is the map) ---
    true_lens = al.Galaxy(redshift=lens_g.redshift, mass=mass,
                          **({**lens_kwargs, "shear": shear} if shear is not None
                             else lens_kwargs))
    true_tracer = al.Tracer(galaxies=[true_lens, al.Galaxy(redshift=source_g.redshift)])
    al.output_to_json(obj=true_tracer, file_path=os.path.join(out_lens, "tracer.json"))

    q_m, pa_m = ell_to_q_pa(*mass.ell_comps)
    params = {
        "lens": lens, "band": band,
        "note": "Ground truth of this mock. Lens light is Sersics fitted to the MGE "
                "bases of the source_pix[2] max-likelihood tracer (flux-conserving); "
                "mass and shear are that fit's values; the source is the fit's "
                "pixelized reconstruction (source_plane_reconstruction.fits).",
        "redshifts": {"lens": float(lens_g.redshift), "source": float(source_g.redshift)},
        "light": light_params,
        "mass": {"profile": f"{type(mass).__name__} (SIE, slope={getattr(mass, 'slope', 2.0)})",
                 "centre": [float(mass.centre[0]), float(mass.centre[1])],
                 "ell_comps": [float(mass.ell_comps[0]), float(mass.ell_comps[1])],
                 "axis_ratio": q_m, "angle_deg": pa_m,
                 "einstein_radius": float(mass.einstein_radius)},
        "source": {
            "type": "pixelized reconstruction (linear interpolation of the mesh values)",
            "source_pix_zip": cfg["source_pix_zip"],
            "map_file": "source_plane_reconstruction.fits",
            "map_pixel_scale": SRC_GRID_SCALE,
            "map_origin_yx": [float(y0), float(x0)],
            "n_mesh_pixels": int(len(recon)),
            "total_flux": float(recon.sum()),
            "centroid": [float(np.average(mesh[:, 0], weights=recon)),
                         float(np.average(mesh[:, 1], weights=recon))],
        },
        "fit_quality": fit_quality,
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
        "ground_truth": "model_parameters.json + tracer.json (lens Sersics, SIE, shear) "
                        "+ source_plane_reconstruction.fits (source map)",
        "source_pix_zip": cfg["source_pix_zip"],
        "real_dataset": os.path.join(real_path, "data.fits"),
        "simulator": {
            "exposure_time_effective": t,
            "background_sky_level_effective": b,
            "noise_model": "sigma^2 = (I + b) / t, Poisson on (I + b) * t counts",
            "noise_seed": cfg["seed"],
            "psf": "real psf.fits kernel, real-space convolution",
            "forward_model": f"{SUB}x{SUB} sub-sampled ray-tracing; source map interpolated "
                             "at traced positions",
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
    fig, axes = plt.subplots(1, 5, figsize=(27, 5.5))
    mock_img = dataset.data.native
    vmax = np.percentile(mock_img[mock_img > 0], 99.5)
    panels = [(mock_img, "mock data", None), (dataset.noise_map.native, "mock noise", None),
              (mock_img / dataset.noise_map.native, "mock S/N", None),
              (real.data.native, "real data", None), (src_map, "source map (truth)", None)]
    for ax, (arr, title, _) in zip(axes, panels):
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


for lens, cfg in LENSES.items():
    simulate(lens, cfg)
