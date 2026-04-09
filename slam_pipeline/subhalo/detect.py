"""
Subhalo Detection: Start Here
=============================

Strong gravitational lenses can be used to detect the presence of small-scale dark matter (DM) subhalos. This occurs
when the DM subhalo overlaps the lensed source emission, and therefore gravitationally perturbs the observed image of
the lensed source galaxy.

When a DM subhalo is not included in the lens model, residuals will be present in the fit to the data in the lensed
source regions near the subhalo. By adding a DM subhalo to the lens model, these residuals can be reduced. Bayesian
model comparison can then be used to quantify whether or not the improvement to the fit is significant enough to
claim the detection of a DM subhalo.


__SLaM Pipelines__

The Source, (lens) Light and Mass (SLaM) pipelines are advanced lens modeling pipelines which automate the fitting
of complex lens models. The SLaM pipelines are used for all DM subhalo detection analyses. Therefore
you should be familiar with the SLaM pipelines before performing DM subhalo detection yourself. If you are unfamiliar
with the SLaM pipelines, checkout the
example `autolens_workspace/notebooks/guides/modeling/slam_start_here`.

Dark matter subhalo detection runs the standard SLaM pipelines, and then extends them with a SUBHALO PIPELINE which
performs the following three chained non-linear searches:

 1) Fits the lens model fitted in the MASS PIPELINE again, without a DM subhalo, to estimate the Bayesian evidence
    of the model without a DM subhalo.

 2) Performs a grid-search of non-linear searches, where each grid cell includes a DM subhalo whose (y,x) centre is
    confined to a small 2D section of the image plane via uniform priors (we explain this in more detail below).

 3) Fit the lens model again, including a DM subhalo whose (y,x) centre is initialized from the highest log evidence
    grid cell of the grid-search. The Bayesian evidence estimated in this model-fit is compared to the model-fit
    which did not include a DM subhalo, to determine whether or not a DM subhalo was detected.

__Grid Search__

The second stage of the SUBHALO PIPELINE uses a grid-search of non-linear searches to determine the highest log
evidence model with a DM subhalo. This grid search confines each DM subhalo in the lens model to a small 2D section
of the image plane via priors on its (y,x) centre. The reasons for this are as follows:

 - Lens models including a DM subhalo often have a multi-model parameter space. This means there are multiple lens
   models with high likelihood solutions, each of which place the DM subhalo in different (y,x) image-plane location.
   Multi-modal parameter spaces are synonomously difficult for non-linear searches to fit, and often produce
   incorrect or inefficient fitting. The grid search breaks the multi-modal parameter space into many single-peaked
   parameter spaces, making the model-fitting faster and more reliable.

 - By inferring how placing a DM subhalo at different locations in the image-plane changes the Bayesian evidence, we
   map out spatial information on where a DM subhalo is detected. This can help our interpretation of the DM subhalo
   detection.

__Pixelized Source__

Detecting a DM subhalo requires the lens model to be sufficiently accurate that the residuals of the source's light
are at a level where the subhalo's perturbing lensing effects can be detected.

This requires the source reconstruction to be performed using a pixelized source, as this provides a more detailed
reconstruction of the source's light than fits using light profiles.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__SUBHALO PIPELINE (no subhalo)__

The first search of the SUBHALO PIPELINE refits the lens model from the MASS TOTAL PIPELINE without a DM subhalo.
This establishes a Bayesian evidence baseline for model comparison with the fits that include a subhalo.
"""


def subhalo_no_subhalo(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    n_batch: int = 20,
    use_chained_model: bool = False,
    reset_multipoles: bool = True,
    reset_shear: bool = True

) -> af.Result:

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            mass_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
    )

    source = al.util.chaining.source_from(result=mass_result)

    if use_chained_model:
        lens=mass_result.model_centred.galaxies.lens
        if reset_multipoles:
            if hasattr(lens, 'multipole_1'):
                lens.multipole_1 = af.Model(al.mp.PowerLawMultipole)
                lens.multipole.m = 1
            if hasattr(lens, 'multipole_3'):
                lens.multipole_3 = af.Model(al.mp.PowerLawMultipole)
                lens.multipole.m = 3
            if hasattr(lens, 'multipole_4'):
                lens.multipole_4 = af.Model(al.mp.PowerLawMultipole)
                lens.multipole.m = 4
        if reset_shear:
            lens.shear = af.Model(al.mp.ExternalShear)
    else:
        lens = mass_result.model.galaxies.lens

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source=source),
    )

    search = af.Nautilus(
        name="subhalo_base",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    # Return a dict object containing the settings used so that they can be copied in the next stages without errors
    settings_dict = {'chaining': use_chained_model, 'reset_multipoles': reset_multipoles, 'reset_shear': reset_shear}

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict), settings_dict


"""
__SUBHALO PIPELINE (grid search)__

The second search of the SUBHALO PIPELINE performs a [number_of_steps x number_of_steps] grid search of
non-linear searches. Each grid cell includes a DM subhalo whose (y,x) centre is confined to a small 2D section
of the image plane via uniform priors.

This grid search maps out where in the image plane including a DM subhalo provides a better fit to the data.
"""


def subhalo_grid_search(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    subhalo_no_subhalo_result: af.Result,
    subhalo_no_subhalo_settings_dict: dict,
    subhalo_mass: af.Model,
    grid_dimension_arcsec: float = 3.0,
    subhalo_mass_limits: list = [1e6, 1e11],
    number_of_steps: int = 2,
    n_batch: int = 20,
) -> af.Result:

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            mass_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
    )

    subhalo = af.Model(al.Galaxy, mass=subhalo_mass)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=subhalo_mass_limits[0],
                                                  upper_limit=subhalo_mass_limits[1])
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )

    subhalo.redshift = subhalo_no_subhalo_result.instance.galaxies.lens.redshift
    subhalo.mass.redshift_object = (
        subhalo_no_subhalo_result.instance.galaxies.lens.redshift
    )
    subhalo.mass.redshift_source = (
        subhalo_no_subhalo_result.instance.galaxies.source.redshift
    )

    if subhalo_no_subhalo_settings_dict['chaining']:
        lens=mass_result.model_centred.galaxies.lens
        if subhalo_no_subhalo_settings_dict['reset_multipoles']:
            if hasattr(lens, 'multipole_1'):
                lens.multipole_1 = af.Model(al.mp.PowerLawMultipole)
                lens.multipole.m = 1
            if hasattr(lens, 'multipole_3'):
                lens.multipole_3 = af.Model(al.mp.PowerLawMultipole)
                lens.multipole.m = 3
            if hasattr(lens, 'multipole_4'):
                lens.multipole_4 = af.Model(al.mp.PowerLawMultipole)
                lens.multipole.m = 4
        if subhalo_no_subhalo_settings_dict['reset_shear']:
            lens.shear = af.Model(al.mp.ExternalShear)
    else:
        lens = mass_result.model.galaxies.lens

    source = al.util.chaining.source_from(result=mass_result)

    model = af.Collection(
        galaxies=af.Collection(lens=lens, subhalo=subhalo, source=source),
    )

    search = af.Nautilus(
        name="subhalo_grid",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    subhalo_grid_search = af.SearchGridSearch(
        search=search,
        number_of_steps=number_of_steps,
    )

    return subhalo_grid_search.fit(
        model=model,
        analysis=analysis,
        grid_priors=[
            model.galaxies.subhalo.mass.centre_1,
            model.galaxies.subhalo.mass.centre_0,
        ],
        info=settings_search.info,
    )


"""
__SUBHALO PIPELINE (refine)__

The third search of the SUBHALO PIPELINE refits the lens model including a DM subhalo, initializing the
subhalo centre from the highest log evidence grid cell of the grid search.

The Bayesian evidence from this fit is compared to the no-subhalo fit to determine whether a DM subhalo
was detected.
"""


def subhalo_refine(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    subhalo_no_subhalo_result: af.Result,
    subhalo_grid_search_result: af.Result,
    subhalo_mass: af.Model,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            mass_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
    )

    subhalo = af.Model(
        al.Galaxy,
        redshift=subhalo_no_subhalo_result.instance.galaxies.lens.redshift,
        mass=subhalo_mass,
    )

    subhalo.redshift = subhalo_no_subhalo_result.instance.galaxies.lens.redshift
    subhalo.mass.redshift_object = (
        subhalo_no_subhalo_result.instance.galaxies.lens.redshift
    )
    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre = subhalo_grid_search_result.model_centred_absolute(
        a=1.0
    ).galaxies.subhalo.mass.centre

    subhalo.redshift = subhalo_grid_search_result.model.galaxies.subhalo.redshift
    subhalo.mass.redshift_object = subhalo.redshift

    model = af.Collection(
        galaxies=af.Collection(
            lens=subhalo_grid_search_result.model.galaxies.lens,
            subhalo=subhalo,
            source=subhalo_grid_search_result.model.galaxies.source,
        ),
    )

    search = af.Nautilus(
        name="subhalo_refine",
        **settings_search.search_dict,
        n_live=600,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

