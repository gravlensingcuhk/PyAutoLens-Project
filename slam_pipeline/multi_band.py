"""
MULTI-BAND PIPELINE (2026-11-12)
================================

Joint multi-band SLaM stages with ONE shared mass + shear model across bands.

__Why Not The Template `run__multi`__

The vendored `source_pix.run_1__multi` / `mass_total.run__multi` give every band its OWN mass
model and only tie the centre (`mass.centre = model.galaxies.lens.mass.centre`). That cannot
break the mass/source degeneracy we see in J18 (the mass model moves 10-20% between source
reconstruction variants, 2026-11-12). Here the mass and shear `af.Model` objects are built
ONCE -- with priors chained from the PRIMARY band's result -- and the same objects are placed
in every band's `af.Collection`, so the factor graph samples a single shared mass model. Lens
light and the source pixelization stay per-band (different bands, different light/morphology).

__Layout__

- Per-band `source_lp` (single-band `source_lp.run`) with unique_tag `<lens>_<band>` -- the
  dataset is not in the search identifier, so same-lens bands would collide without the tag.
- `run_1__multi_shared` / `run_2__multi_shared`: shared free (run 1) / fixed (run 2) mass+shear,
  per-band pixelization from `source_pix.pixelization_from` (PYAUTOLENS_MESH honoured per band).
- Per-band `light_lp.run` on the indexed multi results (`result[i]` per band).
- `mass__multi_shared`: EPL or EPL+multipoles, priors chained from band 0, SHEAR_MAX honoured,
  per-band light/source fixed. `mass_light_dark` has no shared multi variant yet (deferred).
- Non-primary bands get a free `DatasetModel.grid_offset` (the reduction's band-to-band
  astrometry is unverified); the primary band carries the positions likelihood.
"""
import os
from typing import List, Optional

import autofit as af
import autolens as al

from . import search_util
from .source_pix import pixelization_from, SETUP_F_LIVE, SETUP_N_LIKE_MAX, report_if_capped
from .mass_total import link_multipole, apply_ell_comps_prior_from_light


def _shared_mass_shear_from(source_lp_result_0):
    """One mass + one shear model, priors from the PRIMARY band's source_lp (shared objects)."""
    mass = al.util.chaining.mass_from(
        mass=source_lp_result_0.model.galaxies.lens.mass,
        mass_result=source_lp_result_0.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    shear = source_lp_result_0.model.galaxies.lens.shear
    return mass, shear


def _dataset_model_for(index: int):
    """Band-to-band astrometry freedom: free grid offset on non-primary bands."""
    if index == 0:
        return None
    return af.Model(al.DatasetModel)


def run_1__multi_shared(
    settings_search: af.SettingsSearch,
    analysis_list: List[al.AnalysisImaging],
    source_lp_result_list: List[af.Result],
    extra_galaxies: Optional[af.Collection] = None,
    n_live: int = 200,
    n_batch: Optional[int] = None,
    f_live: float = SETUP_F_LIVE,
    n_like_max: float = SETUP_N_LIKE_MAX["source_pix_1"],
) -> af.Result:
    """SOURCE PIX 1 multi-band: shared free mass+shear, per-band light fixed to source_lp."""
    mass, shear = _shared_mass_shear_from(source_lp_result_list[0])

    analysis_factor_list = []
    for i, analysis in enumerate(analysis_list):
        lp = source_lp_result_list[i]
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=lp.instance.galaxies.lens.redshift,
                    bulge=lp.instance.galaxies.lens.bulge,
                    disk=lp.instance.galaxies.lens.disk,
                    point=lp.instance.galaxies.lens.point,
                    mass=mass,
                    shear=shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=lp.instance.galaxies.source.redshift,
                    pixelization=pixelization_from(search=1, adapt_images=analysis.adapt_images),
                ),
            ),
            extra_galaxies=extra_galaxies,
            dataset_model=_dataset_model_for(i),
        )
        analysis_factor_list.append(af.AnalysisFactor(prior_model=model, analysis=analysis))

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

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

    result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)
    report_if_capped(search, result, "source_pix_1")
    return result


def run_2__multi_shared(
    settings_search: af.SettingsSearch,
    analysis_list: List[al.AnalysisImaging],
    source_lp_result_list: List[af.Result],
    source_pix_result_1: af.Result,
    n_live: int = 300,
    n_batch: Optional[int] = None,
    f_live: float = SETUP_F_LIVE,
    n_like_max: Optional[float] = None,
) -> af.Result:
    """SOURCE PIX 2 multi-band: mass+shear fixed to run 1 (hence shared), per-band run-2 meshes."""
    if n_like_max is None:
        n_like_max = SETUP_N_LIKE_MAX["source_pix_2"] * max(1, n_live // 75)

    analysis_factor_list = []
    for i, analysis in enumerate(analysis_list):
        lp = source_lp_result_list[i]
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=lp.instance.galaxies.lens.redshift,
                    bulge=lp.instance.galaxies.lens.bulge,
                    disk=lp.instance.galaxies.lens.disk,
                    point=lp.instance.galaxies.lens.point,
                    mass=source_pix_result_1[0].instance.galaxies.lens.mass,
                    shear=source_pix_result_1[0].instance.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=lp.instance.galaxies.source.redshift,
                    pixelization=pixelization_from(search=2, adapt_images=analysis.adapt_images),
                ),
            ),
            extra_galaxies=source_pix_result_1[i].instance.extra_galaxies,
            dataset_model=_dataset_model_for(i),
        )
        analysis_factor_list.append(af.AnalysisFactor(prior_model=model, analysis=analysis))

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)
    model = factor_graph.global_prior_model

    # As in the single-band run_2: Nautilus refuses a one-parameter search.
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

    result = search.fit(model=model, analysis=factor_graph)
    report_if_capped(search, result, "source_pix_2")
    return result


def mass__multi_shared(
    settings_search: af.SettingsSearch,
    analysis_list: List[al.AnalysisImaging],
    source_pix_result_1: af.Result,
    source_pix_result_2: af.Result,
    light_result_list: List[af.Result],
    multipole_3: Optional[af.Model] = None,
    multipole_4: Optional[af.Model] = None,
    extra_galaxies: Optional[af.Collection] = None,
    n_batch: Optional[int] = None,
    name: str = "mass_EPL",
) -> af.Result:
    """
    MASS multi-band: ONE shared EPL(+multipoles)+shear; per-band light and pixelized source fixed
    from the per-band light_lp / source_pix[2] results. Priors chain from band 0 (the primary
    band): mass from source_pix[1], shear reset as in the single-band mass_total (SHEAR_MAX
    honoured), ell_comps prior centred on the primary band's lens light.
    """
    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=source_pix_result_1[0].model_centred.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    apply_ell_comps_prior_from_light(mass=mass, light_result=light_result_list[0])

    if multipole_3 is not None:
        link_multipole(multipole_3, m=3, mass=mass)
    if multipole_4 is not None:
        link_multipole(multipole_4, m=4, mass=mass)

    # Same as mass_total.run(reset_shear_prior=True): fresh shear, optionally widened.
    shear = af.Model(al.mp.ExternalShear)
    shear_max = os.environ.get("PYAUTOLENS_SHEAR_MAX")
    if shear_max is not None:
        smax = float(shear_max)
        shear.gamma_1 = af.TruncatedGaussianPrior(mean=0.0, sigma=0.05, lower_limit=-smax, upper_limit=smax)
        shear.gamma_2 = af.TruncatedGaussianPrior(mean=0.0, sigma=0.05, lower_limit=-smax, upper_limit=smax)

    analysis_factor_list = []
    for i, analysis in enumerate(analysis_list):
        light = light_result_list[i]
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=light.instance.galaxies.lens.redshift,
                    bulge=light.instance.galaxies.lens.bulge,
                    disk=light.instance.galaxies.lens.disk,
                    point=light.instance.galaxies.lens.point,
                    mass=mass,
                    multipole_3=multipole_3,
                    multipole_4=multipole_4,
                    shear=shear,
                ),
                source=al.util.chaining.source_from(result=source_pix_result_2[i]),
            ),
            extra_galaxies=extra_galaxies,
            dataset_model=_dataset_model_for(i),
        )
        analysis_factor_list.append(af.AnalysisFactor(prior_model=model, analysis=analysis))

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    if n_batch is None:
        n_batch = search_util.n_batch()

    nlive = 400 if (multipole_3 is not None or multipole_4 is not None) else 300
    search = search_util.Nautilus(
        name=name,
        **settings_search.search_dict,
        n_live=nlive,
        n_batch=n_batch,
    )

    return search.fit(model=factor_graph.global_prior_model, analysis=factor_graph, **settings_search.fit_dict)
