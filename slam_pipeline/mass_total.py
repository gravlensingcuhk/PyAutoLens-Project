import logging
import os

import autofit as af
import autolens as al
import numpy as np

from . import search_util

from typing import Union, Optional, Tuple

logger = logging.getLogger(__name__)

# See "__The Mass Ellipticity Prior__": sigma of the Gaussian prior on each of the power law's
# `ell_comps`, centred on the lens light's. Widened 0.2 -> 0.4 on 2026-11-06: at 0.2 every v4
# mass fit pinned the shear at its +/-0.1 boundary while the mass ellipticity sat ~2 sigma from
# the light -- the prior capped the mass quadrupole and the overflow went into shear. 0.4 still
# seeds the light-aligned mode but lets the data move the ellipticity.
ELL_COMPS_FROM_LIGHT_SIGMA = 0.4


def light_ell_comps_from(light_result: af.Result) -> Tuple[float, float]:
    """
    The `ell_comps` of the lens light's brightest Gaussian group in `light_result`.

    __The Mass Ellipticity Prior__

    The mass stages chain the power law's `ell_comps` from `source_pix[1]`'s isothermal posterior,
    and the shear from a fresh prior (`reset_shear_prior`). With one arc and a counter-image the
    mass position angle and the shear are degenerate: on J24 (`output_sam/`, 2026-11-03) the RTU
    chain settled with the mass perpendicular to the lens light and the shear rotated by 50
    degrees to compensate, the Hilbert chains with the mass aligned to the light and a higher
    evidence. Centring a Gaussian prior on the light's `ell_comps` (sigma
    `ELL_COMPS_FROM_LIGHT_SIGMA`) starts the power law in the physical mode: the lens light and
    total mass of an early-type lens are aligned and similarly flattened (the HST project's
    mass-versus-light premise), while the prior is wide enough for the data to move it.

    For a multi-group lens light, the group carrying more flux (the sum over its Gaussians of
    intensity x sigma^2 x axis ratio, with the intensities the maximum-likelihood fit solved)
    supplies the shape. The groups are ordered only to remove label degeneracy, so the physically
    dominant group is identified from the fit rather than from its index.
    """
    fit = light_result.max_log_likelihood_fit
    tracer = fit.tracer_linear_light_profiles_to_light_profiles
    lens = next(g for g in tracer.galaxies if getattr(g, "bulge", None) is not None)
    flux_by_ell_comps = {}
    for gaussian in lens.bulge.profile_list:
        ell_comps = tuple(float(c) for c in gaussian.ell_comps)
        ell = float(np.hypot(*ell_comps))
        axis_ratio = (1.0 - ell) / (1.0 + ell)
        flux = float(gaussian.intensity) * float(gaussian.sigma) ** 2 * axis_ratio
        flux_by_ell_comps[ell_comps] = flux_by_ell_comps.get(ell_comps, 0.0) + flux
    ell_comps = max(flux_by_ell_comps, key=flux_by_ell_comps.get)
    logger.info(
        f"Lens light ell_comps by Gaussian group (flux-weighted): {flux_by_ell_comps}; using {ell_comps}"
    )
    return ell_comps


def apply_ell_comps_prior_from_light(
    mass: af.Model, light_result: af.Result, sigma: float = ELL_COMPS_FROM_LIGHT_SIGMA
) -> af.Model:
    """Centre `mass.ell_comps` on the lens light's (`light_ell_comps_from`), Gaussian priors of width `sigma`."""
    ell_comps = light_ell_comps_from(light_result=light_result)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=ell_comps[0], sigma=sigma)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=ell_comps[1], sigma=sigma)
    return mass



def link_multipole(multipole: af.Model, m: int, mass: af.Model) -> af.Model:
    """
    Tie a multipole of order `m` to `mass`'s geometry by prior links (not copies).

    `al.mp.PowerLawEllipticalMultipole` (the project default since 2026-10-26) perturbs the
    power law's *elliptical* isodensity contours, so it must share the power law's `centre`,
    `ell_comps`, `einstein_radius` and `slope`; the legacy circular `al.mp.PowerLawMultipole`
    has no `ell_comps` and only the other three are linked.
    """
    multipole.m = m
    multipole.centre = mass.centre
    if hasattr(multipole, "ell_comps"):
        multipole.ell_comps = mass.ell_comps
    multipole.einstein_radius = mass.einstein_radius
    multipole.slope = mass.slope
    return multipole

def run(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: Optional[af.Result],
    mass: af.Model = af.Model(al.mp.Isothermal),
    name: str = "mass_total[1]",
    multipole_1: Optional[af.Model] = None,
    multipole_3: Optional[af.Model] = None,
    multipole_4: Optional[af.Model] = None,
    smbh: Optional[af.Model] = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    extra_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
    reset_shear_prior: bool = False,
    n_batch: Optional[int] = None,
    use_prior_chaining=False,
    ell_comps_from_light: bool = False,
) -> af.Result:
    """
    The SLaM MASS TOTAL PIPELINE, which fits a lens model with a total mass distribution (e.g. a power-law).

    `ell_comps_from_light` replaces the chained `ell_comps` prior of `mass` with Gaussian priors
    centred on the lens light's shape in `light_result` (see `light_ell_comps_from`).

    Parameters
    ----------
    settings_search
        The settings used to set up the non-linear search which are general to all SLaM pipelines, for example
        the `path_prefix`.
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_result_for_lens
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline,
        used for initializing model components associated with the lens galaxy.
    source_result_for_source
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline,
        used for initializing model components associated with the source galaxy.
    light_result
        The result of the SLaM LIGHT LP PIPELINE which ran before this pipeline.
    mass
        The `MassProfile` used to fit the lens galaxy mass in this pipeline.
    light_linear_to_standard
        If `True`, convert all linear light profiles in the model to standard light profiles, whose `intensity` values
        use the max likelihood result of the LIGHT PIPELINE.
    multipole_1
        Optionally include a first order multipole mass profile component in the mass model.
    multipole_3
        Optionally include a third order multipole mass profile component in the mass model.
    multipole_4
        Optionally include a fourth order multipole mass profile component in the mass model.
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the MASS TOTAL PIPELINE fits a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [Priors initialized from SOURCE PIPELINE].
     - The source galaxy's light is parametric or a pixelization depending on the previous pipeline [Model and priors 
     initialized from SOURCE PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the SOURCE PIPELINE
    """

    if use_prior_chaining:
        mass = al.util.chaining.mass_from(
            mass=mass,
            mass_result=source_result_for_lens.model_centred.galaxies.lens.mass,
            unfix_mass_centre=True,
        )
    else:
        mass = al.util.chaining.mass_from(
            mass=mass,
            mass_result=source_result_for_lens.model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )
    print(mass)

    if mass_centre is not None:
        mass.centre = mass_centre

    if smbh is not None:
        smbh.centre = mass.centre

    if light_result is None:
        bulge = None
        disk = None
        point = None

    else:
        bulge = light_result.instance.galaxies.lens.bulge
        disk = light_result.instance.galaxies.lens.disk
        point = light_result.instance.galaxies.lens.point

    if ell_comps_from_light:
        if light_result is None:
            raise ValueError("ell_comps_from_light needs a light_result")
        # Before the multipoles are linked, so they share the new priors.
        apply_ell_comps_prior_from_light(mass=mass, light_result=light_result)

    if multipole_1 is not None:
        link_multipole(multipole_1, m=1, mass=mass)

    if multipole_3 is not None:
        link_multipole(multipole_3, m=3, mass=mass)

    if multipole_4 is not None:
        link_multipole(multipole_4, m=4, mass=mass)

    source = al.util.chaining.source_from(
        result=source_result_for_source,
    )

    if not reset_shear_prior:
        shear = source_result_for_lens.model.galaxies.lens.shear
    else:
        shear = af.Model(al.mp.ExternalShear)
        # 2026-11-07 diagnostic (PYAUTOLENS_SHEAR_MAX=0.2): every v4 mass fit pinned the shear
        # at the config's +/-0.1 bounds (both lenses, both meshes, both data reductions,
        # sigma-0.2 and 0.4 mass ellipticity priors). Widening here -- mass stages only, so
        # the source/light searches upstream keep their identifiers and are reused -- shows
        # where the shear actually wants to land; if it rails at 0.2 the EPL family cannot
        # express these lenses. Unset keeps the config prior, so the PSF-floor v5 chain runs
        # with the v4 baseline shear.
        shear_max = os.environ.get("PYAUTOLENS_SHEAR_MAX")
        if shear_max is not None:
            smax = float(shear_max)
            shear.gamma_1 = af.TruncatedGaussianPrior(mean=0.0, sigma=0.05, lower_limit=-smax, upper_limit=smax)
            shear.gamma_2 = af.TruncatedGaussianPrior(mean=0.0, sigma=0.05, lower_limit=-smax, upper_limit=smax)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=bulge,
                disk=disk,
                point=point,
                mass=mass,
                multipole_1=multipole_1,
                multipole_3=multipole_3,
                multipole_4=multipole_4,
                shear=shear,
                smbh=smbh,
            ),
            source=source,
        ),
        extra_galaxies=extra_galaxies,
        dataset_model=dataset_model,
    )

    if (multipole_1 is not None) or (multipole_3 is not None) or (multipole_4 is not None):
        nlive = 400
    else:
        nlive = 300

    if n_batch is None:
        n_batch = search_util.n_batch()

    search = search_util.Nautilus(
        name=name,
        **settings_search.search_dict,
        n_live=nlive,
        n_batch=n_batch,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result


def run__multi(
    settings_search: af.SettingsSearch,
    analysis_list: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: Optional[af.Result],
    mass: af.Model = af.Model(al.mp.Isothermal),
    multipole_1: Optional[af.Model] = None,
    multipole_3: Optional[af.Model] = None,
    multipole_4: Optional[af.Model] = None,
    smbh: Optional[af.Model] = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    extra_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
    n_batch: Optional[int] = None,
) -> af.Result:
    """
    The SLaM MASS TOTAL PIPELINE, which fits a lens model with a total mass distribution (e.g. a power-law).

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
    source_result_for_lens
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline,
        used for initializing model components associated with the lens galaxy.
    source_result_for_source
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline,
        used for initializing model components associated with the source galaxy.
    light_result
        The result of the SLaM LIGHT LP PIPELINE which ran before this pipeline.
    mass
        The `MassProfile` used to fit the lens galaxy mass in this pipeline.
    light_linear_to_standard
        If `True`, convert all linear light profiles in the model to standard light profiles, whose `intensity` values
        use the max likelihood result of the LIGHT PIPELINE.
    multipole_1
        Optionally include a first order multipole mass profile component in the mass model.
    multipole_3
        Optionally include a third order multipole mass profile component in the mass model.
    multipole_4
        Optionally include a fourth order multipole mass profile component in the mass model.
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the MASS TOTAL PIPELINE fits a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [Priors initialized from SOURCE PIPELINE].
     - The source galaxy's light is parametric or a pixelization depending on the previous pipeline [Model and priors 
     initialized from SOURCE PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the SOURCE PIPELINE
    """
    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):

        mass = al.util.chaining.mass_from(
            mass=mass,
            mass_result=source_result_for_lens[i].model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )

        if mass_centre is not None:
            mass.centre = mass_centre

        if smbh is not None:
            smbh.centre = mass.centre

        if light_result is None:
            bulge = None
            disk = None
            point = None

        else:
            bulge = light_result[i].instance.galaxies.lens.bulge
            disk = light_result[i].instance.galaxies.lens.disk
            point = light_result[i].instance.galaxies.lens.point

        if multipole_1 is not None:
            link_multipole(multipole_1, m=1, mass=mass)

        if multipole_3 is not None:
            link_multipole(multipole_3, m=3, mass=mass)

        if multipole_4 is not None:
            link_multipole(multipole_4, m=4, mass=mass)

        source = al.util.chaining.source_from(
            result=source_result_for_source[i],
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_result_for_lens[i].instance.galaxies.lens.redshift,
                    bulge=bulge,
                    disk=disk,
                    point=point,
                    mass=mass,
                    multipole_1=multipole_1,
                    multipole_3=multipole_3,
                    multipole_4=multipole_4,
                    shear=source_result_for_lens[i].model.galaxies.lens.shear,
                    smbh=smbh,
                ),
                source=source,
            ),
            extra_galaxies=extra_galaxies,
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    if n_batch is None:
        n_batch = search_util.n_batch()

    search = search_util.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=300,
        n_batch=n_batch,
    )

    result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

    return result
