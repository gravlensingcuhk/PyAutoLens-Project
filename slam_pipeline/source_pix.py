"""
SOURCE PIX PIPELINE
===================

The two searches that replace the MGE source of SOURCE LP with an adaptive pixelization, and
the adapt-image helpers both runners share.

__The Source Mesh (2026-10)__

Until 2026-10 both searches used a Delaunay mesh on Hilbert image-mesh points with `AdaptSplit`
regularization. On COSJ100018+022138 (J18) `source_pix[1]` ran 13 days / 130k samples without
leaving Nautilus's sampling phase: the posterior had an effective sample size of ~2, because the
likelihood is rugged at the ~100 nat level on 0.1-sigma steps of the mass parameters. Two
things make it so. The Delaunay mesh re-triangulates as the traced points move, and
`AdaptSplit` sets each pixel's regularization from the adapt image divided by the *integer*
number of sub-pixels that land in it, which jumps by whole units.

The setup now follows Sam's HST project (pyauto `z_projects/HST_lensing_mass_light`):

- `RectangularRTUAdaptDensity` (search 1) / `RectangularRTUAdaptImage` (search 2), a fixed
  `RECT_SHAPE` x `RECT_SHAPE` lattice warped by a kernel (erf-sum) CDF of the traced points
  (Enzi et al., arXiv:2606.30620). The topology never changes, so the likelihood is a continuous
  function of the mass. Search 2 also fits the mesh's `weight_power` and `weight_floor`
  (library priors), which the Hilbert scheme hard-coded at 3.5 / 0.01.
- `AdaptPower` regularization (coefficients at lambda^2, single-counted graph Laplacian), with
  `config/general.yaml [inversion] pixel_signal_normalisation: weights`, which normalises the
  pixel signals by the sum of the (continuous) mapping weights instead of the integer count.
  This needs the vendored PyAutoArray `production/hst` branch (see `pyauto/PROVENANCE.md`).
- Nautilus stops shell-building at `SETUP_F_LIVE` (0.05, library 0.01) and every search has a
  likelihood-call cap `SETUP_N_LIKE_MAX` (~2x the most calls a converged HST run needed). A
  capped search still writes its result, so the chain continues, but `report_if_capped` says so
  in the job log and appends to `<output>/capped_searches.txt`.

Scans of J18's `source_pix[1]` maximum-likelihood model (einstein_radius stepped by 0.12 sigma,
2026-10-24): Delaunay+AdaptSplit 120 nats RMS about a quadratic, max step 366; RTU 30x30 +
AdaptPower with weights normalisation 0.01-0.2 RMS, max step 0.5. The rectangular meshes
ignore `image_plane_mesh_grid`, so the Hilbert mesh, the ring of circle edge points and
`zeroed_pixels` are gone; edge zeroing is `use_edge_zeroed_pixels` in general.yaml.

__The Adapt Image__

Both searches cap the source adapt image on a copy at the S/N of its
`HILBERT_ADAPT_SNR_SATURATE`-th (50th) brightest pixel, never below `ADAPT_SNR_CAP` (3). This is
the HST production rule for both mesh families: a fixed cap of 3 flattened the whole arc, so the
run-2 mesh's `weight_power` could only trade arc against background and the v3 RTU mesh did not
zoom (J24 fitted `weight_power` ~0.01). An explicit `adapt_snr_cap` overrides the per-lens value.
The *uncapped* image drives the pixelization over-sampling map (sub-size 4 where the source S/N >
3, else 2), which must see the true S/N. Keep that ordering.

Run 1 adapts to the SOURCE LP source. J24's run 2 uses SOURCE PIX 1's pixelized reconstruction,
the standard two-stage adaption in its better v2 chain. J18 and J0959 deliberately keep run 2 on
SOURCE LP: adapting to a fragmented PIX 1 reconstruction in v2 fed the clumps back into the fit;
`PYAUTOLENS_ADAPT_RUN2=sp1` opts back into PIX 1 adaption (see "__The Adapt Floor__").

__The Hilbert Mesh (re-trial, 2026-10-29)__

`PYAUTOLENS_MESH=hilbert` switches both searches to the thesis scheme, ported from the HST
project's `--mesh hilbert` (its `source_pix.py` "__The Hilbert Mesh__" and "__Edge Points__",
and `docs/current_debugging_and_speedup.md` §0a, 2026-10-29): `al.image_mesh.Hilbert` places
`hilbert_pixels_from_pixel_scale` points (1500 at 0.03"/px, 1250 at 0.06"/px) along a Hilbert
curve weighted by the adapt image (max-normalised, `HILBERT_WEIGHT_POWER` 4, floored at
`HILBERT_WEIGHT_FLOOR` 0.01), `append_with_circle_edge_points` adds `EDGE_PIXELS_TOTAL` points
half a pixel outside the (circular) mask and `al.mesh.Delaunay(zeroed_pixels=...)` regularizes
those to zero, and the regularization is `AdaptSplit` (or `ConstantSplit`, the HST control, via
`PYAUTOLENS_REGULARIZATION=constant_split`). The adapt image is capped per lens at the S/N of
its `HILBERT_ADAPT_SNR_SATURATE`-th (50th) brightest pixel, never below `ADAPT_SNR_CAP`, as for
the RTU mesh.

Why re-try what the RTU switch replaced: the 2026-10 ruggedness was traced to the per-step
recomputation of the pixel signals, and the `weights` normalisation in `config/general.yaml`
that smoothed RTU applies to every adaptive regularization but was never run on the Delaunay
mesh here; and the vendored autoarray's sparse curvature is exact on a Delaunay mapper, so
1250-1500 adaptive points cost about an RTU-30 call on a GPU (HST §0a). The expected gain is
mesh-50 resolution on the arc at mesh-30 cost. Both meshes use the same adapt-image, positions
and over-sampling code; the mesh class changes the search identifier from `source_pix[1]` on,
so a Hilbert chain shares `source_lp[1]` with the RTU chain in the same output directory and
is told apart by `model.info` (mesh class). If that `source_lp[1]` is still being fitted, the
Hilbert runner waits for it instead of dying on the running-search lock (`search_util`
"__Waiting For A Shared Search__", 2026-11-02).

__The Adapt Floor (2026-11-12)__

J18's 2026-11 AdaptSplit chain converged and fit the image plane (noise-like residuals) but
the source plane still fragmented: clumps on the caustics and splattered isolated cells. Two
structural causes, and the two env knobs that address them:

1. The regularization weights come from pixel signals (the adapt image summed per mesh pixel,
   normalised to max 1, `mapper_util.adaptive_pixel_signals_from`), and J18's adapt image is
   the smooth SOURCE LP MGE -- it has ~zero floor away from the ring, so outlying cells get
   ~zero regularization and can hold arbitrary flux. `PYAUTOLENS_ADAPT_FLOOR_FRAC` (0 = off)
   floors the capped adapt image at that fraction of the cap, bounding the regularization
   dynamic range. It does NOT affect the over-sampling map (built from the uncapped image).
2. The Hilbert image mesh starves arc segments that are faint in the adapt image (the knots
   are in the data but NOT in the smooth MGE adapt image): `PYAUTOLENS_HILBERT_WEIGHT_FLOOR`
   (default 0.01) raises the floor of the image-mesh weight map so faint segments keep points;
   `PYAUTOLENS_HILBERT_WEIGHT_POWER` (default 4) and `PYAUTOLENS_HILBERT_PIXELS` (default
   `hilbert_pixels_from_pixel_scale`) are the other two mesh knobs.

None of these enter the model, so they do NOT change the search identifier: tests with them
must run in a fresh `PYAUTOLENS_OUTPUT_DIR` or the runner silently resumes the old result.
`PYAUTOLENS_ADAPT_RUN2=sp1` (runner-side) additionally switches search 2's adapt image from
SOURCE LP to the PIX 1 reconstruction (the standard scheme J24 uses; J18 avoided it while
PIX 1 fragmented), which is the only variant that puts mesh points and regularization onto
the knots themselves.
"""
import os

import autofit as af
import autolens as al

from typing import Optional, Tuple, Union

import numpy as np

from . import search_util

SOURCE_KEY = "('galaxies', 'source')"

# The source mesh, see "__The Source Mesh__". 60 x 60 since 2026-11-05 (50 x 50 in v3, 40 x 40
# before): J18's source is ~0.05" across, one 40 x 40 cell over its 3" mask, and it fragmented;
# the HST project used 30 x 30 on 3.5" masks with more extended sources. A source_pix search's
# GPU memory scales with n_batch x the pixel count (search_util "__Batch Size__").
RECT_SHAPE = (60, 60)

# See "__The Adapt Image__".
ADAPT_SNR_CAP = 3.0  # 10.0 in updated_output_v2/ (2026-10-28) only, see "__The Adapt Image__"
PIX_OVER_SAMPLE_THRESHOLD = 3.0
PIX_OVER_SAMPLE_SIZES = (4, 2)  # (where source S/N > threshold, elsewhere)

# See "__The Hilbert Mesh__". The HST project's production values, env-overridable for the
# 2026-11-12 fragmentation/coverage tests (PYAUTOLENS_HILBERT_*; see "__The Adapt Floor__").
MESHES = ("rtu", "hilbert")
HILBERT_WEIGHT_POWER = float(os.environ.get("PYAUTOLENS_HILBERT_WEIGHT_POWER", 4.0))
HILBERT_WEIGHT_FLOOR = float(os.environ.get("PYAUTOLENS_HILBERT_WEIGHT_FLOOR", 0.01))
HILBERT_PIXELS_OVERRIDE = int(os.environ.get("PYAUTOLENS_HILBERT_PIXELS", 0)) or None
HILBERT_ADAPT_SNR_SATURATE = 50  # cap the adapt image at the S/N of its 50th brightest pixel
EDGE_PIXELS_TOTAL = 30

# See "__The Adapt Floor__". 0.0 = off (the HST behaviour).
ADAPT_FLOOR_FRAC = float(os.environ.get("PYAUTOLENS_ADAPT_FLOOR_FRAC", 0.0))
HILBERT_REGULARIZATIONS = {
    "adapt_split": al.reg.AdaptSplit,
    "constant_split": al.reg.ConstantSplit,
    # lambda^2 coefficient convention, robust to likelihood overflow; see the class docstring
    # and "__The Adapt Floor__" (2026-11-12 test).
    "adapt_split_power": al.reg.AdaptSplitPower,
}


def mesh_from_env() -> str:
    """`PYAUTOLENS_MESH`: `rtu` (default) or `hilbert`; see "__The Hilbert Mesh__"."""
    mesh = os.environ.get("PYAUTOLENS_MESH", "rtu").lower()
    if mesh not in MESHES:
        raise ValueError(f"PYAUTOLENS_MESH must be one of {MESHES}, not {mesh!r}")
    return mesh


def hilbert_regularization_from_env():
    """`PYAUTOLENS_REGULARIZATION` for the Hilbert mesh: `adapt_split` (default) or `constant_split`."""
    name = os.environ.get("PYAUTOLENS_REGULARIZATION", "adapt_split").lower()
    if name not in HILBERT_REGULARIZATIONS:
        raise ValueError(
            f"PYAUTOLENS_REGULARIZATION must be one of {tuple(HILBERT_REGULARIZATIONS)}, not {name!r}"
        )
    return HILBERT_REGULARIZATIONS[name]


def hilbert_snr_cap_for(snr_map, saturate: int = HILBERT_ADAPT_SNR_SATURATE) -> float:
    """The per-lens cap: the S/N of the `saturate`-th brightest pixel, never below `ADAPT_SNR_CAP`."""
    values = np.sort(np.asarray(snr_map, dtype=float).ravel())[::-1]
    nth = values[min(saturate, values.size) - 1]
    return max(ADAPT_SNR_CAP, float(nth))

# Nautilus stopping and caps live in search_util (shared with the other stages); re-exported
# here because the runners and docs refer to them as source_pix's.
SETUP_F_LIVE = search_util.SETUP_F_LIVE
SETUP_N_LIKE_MAX = search_util.SETUP_N_LIKE_MAX
report_if_capped = search_util.report_if_capped


def adapt_images_from(
    result: af.Result,
    adapt_snr_cap: Optional[float] = None,
    mask: Optional[al.Mask2D] = None,
    mesh: Optional[str] = None,
):
    """
    The adapt images for the source mesh, built from `result`'s source image.

    Returns `(adapt_images, source_image_raw)`: the adapt images carry the S/N-capped copy of
    the source image, `source_image_raw` is the uncapped one for the over-sampling map (see
    "__The Adapt Image__"). The default cap is the S/N of the 50th-brightest source pixel,
    never below 3, for both mesh families. An explicit `adapt_snr_cap` takes precedence. For
    the Hilbert mesh (`mesh="hilbert"`, default from `PYAUTOLENS_MESH`) the adapt images also
    carry the image-plane mesh grid with its ring of edge points (see "__The Hilbert Mesh__");
    `mask` is then required.
    """
    mesh = mesh_from_env() if mesh is None else mesh
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result)
    source_image_raw = galaxy_image_name_dict[SOURCE_KEY]

    if adapt_snr_cap is None:
        adapt_snr_cap = hilbert_snr_cap_for(source_image_raw)

    source_adapt_image = source_image_raw.copy()
    source_adapt_image[source_adapt_image > adapt_snr_cap] = adapt_snr_cap
    if ADAPT_FLOOR_FRAC > 0.0:
        # See "__The Adapt Floor__": bound the regularization dynamic range.
        adapt_floor = ADAPT_FLOOR_FRAC * adapt_snr_cap
        source_adapt_image[source_adapt_image < adapt_floor] = adapt_floor
    galaxy_image_name_dict[SOURCE_KEY] = source_adapt_image

    if mesh != "hilbert":
        return al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict), source_image_raw

    if mask is None:
        raise ValueError("adapt_images_from needs the mask for the Hilbert mesh")
    pixels = HILBERT_PIXELS_OVERRIDE or al.model_util.hilbert_pixels_from_pixel_scale(
        mask.pixel_scales[0]
    )
    print(
        f"HILBERT pixels={pixels} (+{EDGE_PIXELS_TOTAL} edge)  weight_power={HILBERT_WEIGHT_POWER}  "
        f"weight_floor={HILBERT_WEIGHT_FLOOR}  regularization={hilbert_regularization_from_env().__name__}  "
        f"adapt_snr_cap={adapt_snr_cap:.2f} (brightest {HILBERT_ADAPT_SNR_SATURATE} px)  "
        f"adapt_floor_frac={ADAPT_FLOOR_FRAC}  "
        f"mask_radius={mask.circular_radius:.2f}\"",
        flush=True,
    )
    image_mesh = al.image_mesh.Hilbert(
        pixels=pixels, weight_power=HILBERT_WEIGHT_POWER, weight_floor=HILBERT_WEIGHT_FLOOR
    )
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=source_adapt_image
    )
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask.circular_radius + mask.pixel_scale / 2.0,
        n_points=EDGE_PIXELS_TOTAL,
    )
    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={SOURCE_KEY: image_plane_mesh_grid},
    )
    return adapt_images, source_image_raw


def apply_pixelization_over_sampling(
    dataset,
    mask,
    source_image_raw,
    over_sample_size_lp=None,
    signal_to_noise_threshold: float = PIX_OVER_SAMPLE_THRESHOLD,
    sizes: Tuple[int, int] = PIX_OVER_SAMPLE_SIZES,
):
    """Sub-size `sizes[0]` where the (uncapped) source image exceeds the S/N threshold, else `sizes[1]`."""
    arc, rest = sizes
    over_sample_size_pixelization = al.Array2D(
        values=np.where(source_image_raw > signal_to_noise_threshold, arc, rest), mask=mask
    )
    kwargs = {} if over_sample_size_lp is None else {"over_sample_size_lp": over_sample_size_lp}
    return dataset.apply_over_sampling(
        over_sample_size_pixelization=over_sample_size_pixelization, **kwargs
    )


def pixelization_from(
    search: int,
    shape: Tuple[int, int] = RECT_SHAPE,
    mesh: Optional[str] = None,
    adapt_images: Optional[al.AdaptImages] = None,
) -> af.Model:
    """
    The `af.Model(al.Pixelization)` of search 1 (density-adapted RTU mesh) or search 2
    (adapt-image RTU mesh with free `weight_power` / `weight_floor`), both with `AdaptPower`;
    or, for the Hilbert mesh, a `Delaunay` of the image-plane mesh grid carried by
    `adapt_images` with its edge points zeroed, and the split regularization from the
    environment (see "__The Hilbert Mesh__").
    """
    mesh = mesh_from_env() if mesh is None else mesh
    if mesh == "hilbert":
        if adapt_images is None:
            raise ValueError("pixelization_from needs the adapt images for the Hilbert mesh")
        grid = adapt_images.galaxy_name_image_plane_mesh_grid_dict[SOURCE_KEY]
        return af.Model(
            al.Pixelization,
            mesh=al.mesh.Delaunay(pixels=grid.shape[0], zeroed_pixels=EDGE_PIXELS_TOTAL),
            regularization=af.Model(hilbert_regularization_from_env()),
        )
    mesh_cls = al.mesh.RectangularRTUAdaptDensity if search == 1 else al.mesh.RectangularRTUAdaptImage
    return af.Model(
        al.Pixelization,
        mesh=af.Model(mesh_cls, shape=shape),
        regularization=af.Model(al.reg.AdaptPower),
    )


def run_1(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_lp_result: af.Result,
    mesh_init: Optional[af.Model] = None,
    regularization_init: Optional[af.Model] = None,
    extra_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
    fixed_mass_model: bool = False,
    n_live: int = 200,
    n_batch: Optional[int] = None,
    f_live: float = SETUP_F_LIVE,
    n_like_max: float = SETUP_N_LIKE_MAX["source_pix_1"],
) -> af.Result:
    """
    The first SLaM SOURCE PIX PIPELINE, which initializes a lens model which uses a pixelized source for the source
    analysis.

    The first SOURCE PIX PIPELINE may require an adapt-image, for example to adapt the regularization scheme to the
    source's unlensed morphology. The adapt image provided by the SOURCE LP PIPELINE may not cover the entire source
    galaxy (e.g. because the MGE only captures part of the source) and produce a suboptimal fit.

    The result of this pipeline is used in the second SOURCE PIX PIPELINE to adapt the source pixelization to the
    source's unlensed morphology via an adapt image, where the adapt image produced in this pipeline will give a robust
    source image because it uses a pixelized source.

    Parameters
    ----------
    settings_search
        The settings used to set up the non-linear search which are general to all SLaM pipelines, for example
        the `path_prefix`.
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_lp_result
        The results of the SLaM SOURCE LP PIPELINE which ran before this pipeline.
    mesh_init
        The mesh of the pixelization; by default `RectangularRTUAdaptDensity` of `RECT_SHAPE`
        (see "__The Source Mesh__").
    regularization_init
        The regularization of the pixelization; by default `AdaptPower`.
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    fixed_mass_model
        Whether the mass model is fixed from the SOURCE LP PIPELINE, which is generally used for multi-band fitting
        where the mass model is fixed to the first band, albeit it may work for standard fitting if the SOURCE LP
        PIPELINE provides a good mass model.
    n_live, n_batch, f_live, n_like_max
        Nautilus settings; the defaults are the HST project's stability choices (n_batch from
        `search_util.n_batch()`: 16 on the CPU jobs, per lens on phygpu).
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using light profiles [parameters fixed to result of SOURCE LP PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [model initialized from the results of the 
     SOURCE LP PIPELINE].
     - The source galaxy's light is a density-adapted rectangular RTU mesh with AdaptPower regularization
     [regularization parameters free to vary] (or, under PYAUTOLENS_MESH=hilbert, a Delaunay mesh on
     Hilbert points with a split regularization; see "__The Hilbert Mesh__").

    This search improves the lens mass model by modeling the source using a pixelization and computes the adapt
    images that are used in search 2.
    """
    if not fixed_mass_model:
        mass = al.util.chaining.mass_from(
            mass=source_lp_result.model.galaxies.lens.mass,
            mass_result=source_lp_result.model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )
        shear = source_lp_result.model.galaxies.lens.shear
    else:
        mass = source_lp_result.instance.galaxies.lens.mass
        shear = source_lp_result.instance.galaxies.lens.shear

    pixelization = pixelization_from(search=1, adapt_images=analysis.adapt_images)
    if mesh_init is not None:
        pixelization.mesh = mesh_init
    if regularization_init is not None:
        pixelization.regularization = regularization_init

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                point=source_lp_result.instance.galaxies.lens.point,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            ),
        ),
        extra_galaxies=extra_galaxies,
        dataset_model=dataset_model,
    )

    if n_batch is None:
        n_batch = search_util.n_batch()

    search = search_util.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        f_live=f_live,
        n_like_max=n_like_max,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    report_if_capped(search, result, "source_pix_1")

    return result


def run_2(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_lp_result: af.Result,
    source_pix_result_1: af.Result,
    mesh: Optional[af.Model] = None,
    regularization: Optional[af.Model] = None,
    dataset_model: Optional[af.Model] = None,
    n_live: int = 300,
    n_batch: Optional[int] = None,
    f_live: float = SETUP_F_LIVE,
    n_like_max: Optional[float] = None,
) -> af.Result:
    """
    The second SLaM SOURCE PIX PIPELINE, which fits a fixed lens model which uses a pixelized source for the source
    analysis.

    The second SOURCE PIX PIPELINE performs a fit using an advanced pixelizaiton which adapt the source's pixelization
    to the source's unlensed morphology.

    This feature requires an adapt-image, which is computed after the first SOURCE PIX PIPELINE.

    Parameters
    ----------
    settings_search
        The settings used to set up the non-linear search which are general to all SLaM pipelines, for example
        the `path_prefix`.
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_lp_result
        The results of the SLaM SOURCE LP PIPELINE which ran before this pipeline.
    source_pix_result_1
        The results of the first SOURCE PIX PIPELINE, whose source image is the adapt image of this search.
    mesh
        The mesh of the pixelization; by default `RectangularRTUAdaptImage` of `RECT_SHAPE` with free
        `weight_power` and `weight_floor` (see "__The Source Mesh__").
    regularization
        The regularization of the pixelization; by default `AdaptPower`.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    n_live, n_batch, f_live, n_like_max
        Nautilus settings; the defaults are the HST project's. `n_like_max` defaults to
        `SETUP_N_LIKE_MAX["source_pix_2"]` scaled by `n_live // 75`.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    Search 2 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using a light profiles [parameters fixed to result of SOURCE LP PIPELINE].
    - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 1].
    - The source galaxy's light is an adapt-image RTU mesh (free weight_power / weight_floor) with AdaptPower
    regularization (or, under PYAUTOLENS_MESH=hilbert, a Delaunay mesh rebuilt from search 1's source).

    This search initializes the pixelization's mesh and regularization.
    """
    if n_like_max is None:
        n_like_max = SETUP_N_LIKE_MAX["source_pix_2"] * max(1, n_live // 75)

    pixelization = pixelization_from(search=2, adapt_images=analysis.adapt_images)
    if mesh is not None:
        pixelization.mesh = mesh
    if regularization is not None:
        pixelization.regularization = regularization

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                point=source_lp_result.instance.galaxies.lens.point,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            ),
        ),
        extra_galaxies=source_pix_result_1.instance.extra_galaxies,
        dataset_model=dataset_model,
    )

    # With the mass fixed, a one-parameter regularization leaves a one-dimensional search, which
    # Nautilus refuses; an unused uniform parameter integrates out of the evidence (HST project).
    if model.prior_count < 2:
        model.nautilus_dummy = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    if n_batch is None:
        n_batch = search_util.n_batch()

    search = search_util.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        f_live=f_live,
        n_like_max=n_like_max,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    report_if_capped(search, result, "source_pix_2")

    return result


def run_1__bypass_lp(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_disk: Optional[af.Model] = af.Model(al.lp.Exponential),
    lens_point: Optional[af.Model] = None,
    mass: af.Model = af.Model(al.mp.Isothermal),
    shear: af.Model(al.mp.ExternalShear) = af.Model(al.mp.ExternalShear),
    mesh_init: af.Model(al.AbstractMesh) = af.Model(al.mesh.Delaunay),
    regularization_init: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.AdaptSplit
    ),
    redshift_lens: float = 0.5,
    redshift_source: float = 1.0,
    extra_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
    n_batch: int = 16,
) -> af.Result:
    """
    The first SLaM SOURCE PIX PIPELINE, which initializes a lens model which uses a pixelized source for the source
    analysis.

    This variant bypasses the source lp pipeline and is typically used for the interferometer SLaM pipeline, albeit
    it can also be used for imaging analysis.

    The first SOURCE PIX PIPELINE may require an adapt-image, for example to adapt the regularization scheme to the
    source's unlensed morphology. The adapt image provided by the SOURCE LP PIPELINE may not cover the entire source
    galaxy (e.g. because the MGE only captures part of the source) and produce a suboptimal fit.

    The result of this pipeline is used in the second SOURCE PIX PIPELINE to adapt the source pixelization to the
    source's unlensed morphology via an adapt image, where the adapt image produced in this pipeline will give a robust
    source image because it uses a pixelized source.

    Parameters
    ----------
    settings_search
        The settings used to set up the non-linear search which are general to all SLaM pipelines, for example
        the `path_prefix`.
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    mesh_init
        The mesh, which defines how the source is reconstruction in the source-plane, used by the pixelization
        in the first search which initializes the source.
    regularization_init
        The regularization, which places a smoothness prior on the source reconstruction, used by the pixelization
        which fits the source light in the initialization search (`search[1]`).
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    fixed_mass_model
        Whether the mass model is fixed from the SOURCE LP PIPELINE, which is generally used for multi-band fitting
        where the mass model is fixed to the first band, albeit it may work for standard fitting if the SOURCE LP
        PIPELINE provides a good mass model.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using light profiles [parameters fixed to result of SOURCE LP PIPELINE].

     - The lens galaxy mass is modeled using a total mass distribution [model initialized from the results of the 
     SOURCE LP PIPELINE].

     - The source galaxy's light is the input initialization image mesh, mesh and regularization scheme [parameters of 
     regularization free to vary].

    This search improves the lens mass model by modeling the source using a pixelization and computes the adapt
    images that are used in search 2.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=lens_disk,
                point=lens_point,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
        extra_galaxies=extra_galaxies,
        dataset_model=dataset_model,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result


def run_1__multi(
    settings_search: af.SettingsSearch,
    analysis_list: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_lp_result: af.Result,
    mesh_init: af.Model(al.AbstractMesh) = af.Model(al.mesh.RectangularRTUAdaptDensity),
    regularization_init: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.Constant
    ),
    extra_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
    fixed_mass_model: bool = False,
    n_batch: int = 16,
) -> af.Result:
    """
    The first SLaM SOURCE PIX PIPELINE, which initializes a lens model which uses a pixelized source for the source
    analysis.

    The first SOURCE PIX PIPELINE may require an adapt-image, for example to adapt the regularization scheme to the
    source's unlensed morphology. The adapt image provided by the SOURCE LP PIPELINE may not cover the entire source
    galaxy (e.g. because the MGE only captures part of the source) and produce a suboptimal fit.

    The result of this pipeline is used in the second SOURCE PIX PIPELINE to adapt the source pixelization to the
    source's unlensed morphology via an adapt image, where the adapt image produced in this pipeline will give a robust
    source image because it uses a pixelized source.

    This is a variant of the above SLaM pipeline which fits multiple datasets simultaneously using a list of
    analysis objects and the factor graph functionality. Its purpose and model are identical to the single
    dataset version, except for changes required to fit multiple datasets.

    Parameters
    ----------
    settings_search
        The settings used to set up the non-linear search which are general to all SLaM pipelines, for example
        the `path_prefix`.
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_lp_result
        The results of the SLaM SOURCE LP PIPELINE which ran before this pipeline.
    mesh_init
        The mesh, which defines how the source is reconstruction in the source-plane, used by the pixelization
        in the first search which initializes the source.
    regularization_init
        The regularization, which places a smoothness prior on the source reconstruction, used by the pixelization
        which fits the source light in the initialization search (`search[1]`).
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    fixed_mass_model
        Whether the mass model is fixed from the SOURCE LP PIPELINE, which is generally used for multi-band fitting
        where the mass model is fixed to the first band, albeit it may work for standard fitting if the SOURCE LP
        PIPELINE provides a good mass model.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using light profiles [parameters fixed to result of SOURCE LP PIPELINE].

     - The lens galaxy mass is modeled using a total mass distribution [model initialized from the results of the 
     SOURCE LP PIPELINE].

     - The source galaxy's light is the input initialization image mesh, mesh and regularization scheme [parameters of 
     regularization free to vary].

    This search improves the lens mass model by modeling the source using a pixelization and computes the adapt
    images that are used in search 2.
    """

    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):

        if not fixed_mass_model:

            mass = al.util.chaining.mass_from(
                mass=source_lp_result[i].model.galaxies.lens.mass,
                mass_result=source_lp_result[i].model.galaxies.lens.mass,
                unfix_mass_centre=True,
            )

            if i > 0:

                mass.centre = model.galaxies.lens.mass.centre

            shear = source_lp_result[i].model.galaxies.lens.shear

        else:

            mass = source_lp_result[i].instance.galaxies.lens.mass
            shear = source_lp_result[i].instance.galaxies.lens.shear

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result[i].instance.galaxies.lens.redshift,
                    bulge=source_lp_result[i].instance.galaxies.lens.bulge,
                    disk=source_lp_result[i].instance.galaxies.lens.disk,
                    point=source_lp_result[i].instance.galaxies.lens.point,
                    mass=mass,
                    shear=shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result[i].instance.galaxies.source.redshift,
                    pixelization=af.Model(
                        al.Pixelization,
                        mesh=mesh_init,
                        regularization=regularization_init,
                    ),
                ),
            ),
            extra_galaxies=extra_galaxies,
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

    return result


def run_2__multi(
    settings_search: af.SettingsSearch,
    analysis_list: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_lp_result: af.Result,
    source_pix_result_1: af.Result,
    mesh: af.Model(al.AbstractMesh) = af.Model(al.mesh.RectangularRTUAdaptDensity),
    regularization: af.Model(al.AbstractRegularization) = af.Model(al.reg.AdaptSplit),
    dataset_model: Optional[af.Model] = None,
    n_batch: int = 16,
) -> af.Result:
    """
    The second SLaM SOURCE PIX PIPELINE, which fits a fixed lens model which uses a pixelized source for the source
    analysis.

    The second SOURCE PIX PIPELINE performs a fit using an advanced pixelizaiton which adapt the source's pixelization
    to the source's unlensed morphology.

    This feature requires an adapt-image, which is computed after the first SOURCE PIX PIPELINE.

    This is a variant of the above SLaM pipeline which fits multiple datasets simultaneously using a list of
    analysis objects and the factor graph functionality. Its purpose and model are identical to the single
    dataset version, except for changes required to fit multiple datasets.

    Parameters
    ----------
    settings_search
        The settings used to set up the non-linear search which are general to all SLaM pipelines, for example
        the `path_prefix`.
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_lp_result
        The results of the SLaM SOURCE LP PIPELINE which ran before this pipeline.
    mesh
        The mesh, which defines how the source is reconstruction in the source-plane, used by the pixelization
        in the final search which improves the source adaption.
    regularization
        The regularization, which places a smoothness prior on the source reconstruction, used by the pixelization
        in the final search which improves the source adaption.
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    Search 2 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using a light profiles [parameters fixed to result of SOURCE LP PIPELINE].
    - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 2].
    - The source galaxy's light is the input final mesh and regularization.

    This search initializes the pixelization's mesh and regularization.
    """
    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result[i].instance.galaxies.lens.redshift,
                    bulge=source_lp_result[i].instance.galaxies.lens.bulge,
                    disk=source_lp_result[i].instance.galaxies.lens.disk,
                    point=source_lp_result[i].instance.galaxies.lens.point,
                    mass=source_pix_result_1[i].instance.galaxies.lens.mass,
                    shear=source_pix_result_1[i].instance.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result[i].instance.galaxies.source.redshift,
                    pixelization=af.Model(
                        al.Pixelization,
                        mesh=mesh,
                        regularization=regularization,
                    ),
                ),
            ),
            extra_galaxies=source_pix_result_1[i].instance.extra_galaxies,
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    """
    __Search (Search 2)__

    This search uses the nested sampling algorithm Dynesty, in contrast to nearly every other search throughout the
    autolens workspace which use `DynestyStatic`.

    The reason is quite technical, but in a nutshell it is because the likelihood function sampled in `source_pix[2]`
    is often not smooth. This leads to behaviour where the `DynestyStatic` search gets stuck sampling small regions of
    parameter space indefinitely, and does not converge and terminate.

    Dynesty has proven more robust to these issues, because it uses a random walk nested sampling algorithm which
    is less susceptible to a noisy likelihood function.

    The reason this likelihood function is noisy is because it has parameters which change the distribution of source
    pixels. For example, the parameters may mean more or less source pixels cluster over the brightest regions of the
    image. In all other searches, the source pixelization parameters are fixed, ensuring that the likelihood function
    is smooth.
    """
    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

    return result
