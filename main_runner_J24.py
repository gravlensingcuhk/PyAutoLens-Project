import json
from autonerves import jax_wrapper  # Sets JAX environment before other imports 
import sys
from os import path
import autofit as af
import autolens as al
from autonerves import conf
import numpy as np
import slam_pipeline

import os

# The cluster workspace root; set PYAUTOLENS_PROJECT_ROOT to run from another checkout (e.g. locally).
cosma_path = os.environ.get('PYAUTOLENS_PROJECT_ROOT', path.join(path.sep, 'users', 'wing-yan.chan', 'PyAutoLens-Project'))
workspace_path = cosma_path
output_path = path.join(cosma_path, 'output')
config_path = path.join(cosma_path, 'config')
conf.instance.push(new_path=config_path, output_path=output_path)
data_path = path.join(workspace_path, 'data', 'cowls')


use_jax = True

"""
__Dataset__ 
Load and mask the data.
"""
dataset_name = str(sys.argv[1])
filt = "F444W"
ps = 0.06
dataset_path = path.join(data_path, dataset_name, filt)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf95pc.fits"),
    pixel_scales=ps,
    check_noise_map=False
)

with open(path.join(data_path, dataset_name, 'info.json'), 'r') as file:
    info_file = json.load(file)

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(data_path, dataset_name, "positions.json"))
)
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=path.join(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=dataset.pixel_scales,
    invert=True,
)
dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
mask_radius = info_file['mask_radius']
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, 
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

print(f'RUNNING Dataset: {dataset_name}, Mask: {mask_radius}')

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.2, 0.4],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
# No `apply_sparse_operator_cpu()`: its Numba inversion path is not JAX-traceable, and autofit >= 2026.10
# jit-traces the likelihood before Nautilus starts (the HST runners use the default JAX path too).


"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix='model_setup_free_source',
    unique_tag=f'{dataset_name}',
    info=None,
    session=None,
    number_of_cores = 16,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = info_file['lens_spec_z']
redshift_source = info_file['source_spec_z']


"""
__SOURCE LP PIPELINE__
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=use_jax,
                              positions_likelihood_list=[al.PositionsLH(threshold=0.2, positions=positions)],)

source_bulge = slam_pipeline.mge_model_from(
    total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-3, np.log10(1), 30),
    centre_prior_is_uniform=False,  # GaussianPrior(mean=centre, sigma=0.3) -> unrails the source centre
)

lens_bulge = slam_pipeline.mge_model_from(
    total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-2, np.log10(mask_radius-0.1), 30)
)

lens_disk = slam_pipeline.mge_model_from(
    total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-2, np.log10(mask_radius-0.1), 30)
)

lens_point = slam_pipeline.mge_model_from(
    total_gaussians=10, gaussian_per_basis=1, log10_sigma_list=np.linspace(-4, np.log10(0.025), 10),
    centre=(0.0, 0.0)
)

mass = af.Model(al.mp.Isothermal)
mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
    lens_point=lens_point,
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__SOURCE PIX PIPELINE__

Rectangular RTU mesh (25 x 25) + AdaptPower regularization, the HST project's stability setup;
see slam_pipeline/source_pix.py "__The Source Mesh__" for why the Hilbert/Delaunay/AdaptSplit
scheme was dropped. The adapt image is the SOURCE LP source capped at S/N 3 on a copy, while the
uncapped image drives the pixelization over-sampling map (4 on the arcs, 2 elsewhere).
"""

settings_search = af.SettingsSearch(
    path_prefix='model_setup_free_source',
    unique_tag=f'{dataset_name}',
    info=None,
    session=None,
    number_of_cores = 16,
)

adapt_images, source_image_raw = slam_pipeline.source_pix.adapt_images_from(result=source_lp_result)

dataset = slam_pipeline.source_pix.apply_pixelization_over_sampling(
    dataset=dataset, mask=mask, source_image_raw=source_image_raw, over_sample_size_lp=over_sample_size,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.075)
    ],
    use_jax=use_jax,
)

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
)

"""
__SOURCE PIX PIPELINE 2__

The adapt image is now the pixelized source of search 1; the mesh adapts to it and its
weight_power / weight_floor are fitted.
"""

adapt_images, source_image_raw = slam_pipeline.source_pix.adapt_images_from(result=source_pix_result_1)

dataset = slam_pipeline.source_pix.apply_pixelization_over_sampling(
    dataset=dataset, mask=mask, source_image_raw=source_image_raw, over_sample_size_lp=over_sample_size,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    use_jax=use_jax,
)

source_pix_result_2 = slam_pipeline.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
)

"""
__LIGHT LP PIPELINE__
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
)

light_result = slam_pipeline.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
    lens_point=lens_point
)


"""
__MASS TOTAL PIPELINE__
"""
settings_search = af.SettingsSearch(
    path_prefix='mass_models',
    unique_tag=f'{dataset_name}',
    info=None,
    session=None,
    number_of_cores=16
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.075)
    ],
)

mass_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    reset_shear_prior=True,
    name='mass_EPL'
)


"""
Multipole

Elliptical multipoles (al.mp.PowerLawEllipticalMultipole, PyAutoGalaxy feature/elliptical-multipole):
the m=3 and m=4 terms deform the power law's elliptical isodensity contours, with strength k_m and
shape angle s_m measured from the major axis, rather than adding a circular multipole in the sky
frame (the old al.mp.PowerLawMultipole). They share the power law's centre, ell_comps,
einstein_radius and slope by prior links (slam_pipeline.mass_total.link_multipole).
"""

multipole_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    multipole_1=None,
    multipole_3=af.Model(al.mp.PowerLawEllipticalMultipole),
    multipole_4=af.Model(al.mp.PowerLawEllipticalMultipole),
    reset_shear_prior=True,
    name='mass_multipole'
)

