import json
from autoconf import jax_wrapper  # Sets JAX environment before other imports 
import sys
from os import path
import autofit as af
import autolens as al
from autoconf import conf
import numpy as np
import slam_pipeline

cosma_path = path.join(path.sep, 'home', 'vian', 'PyAutoLens-Project')
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
filt = "F277W"
ps = 0.0315
dataset_path = path.join(data_path, dataset_name, filt)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf_71x71.fits"),
    pixel_scales=ps,
    check_noise_map=False
)

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(data_path, dataset_name, "positions.json"))
)
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=path.join(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=dataset.pixel_scales,
    invert=True,
)
dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
mask_radius = mask_extra_galaxies.circular_radius
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


"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix='model_setup',
    unique_tag=f'{dataset_name}',
    info=None,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 1.53
redshift_source = 3.42


"""
__SOURCE LP PIPELINE__
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=use_jax,
                              positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],)

source_bulge = slam_pipeline.mge_model_from(
    total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-3, np.log10(1), 30)
)

lens_bulge = slam_pipeline.mge_model_from(
    total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-2, np.log10(3), 30)
)

lens_disk = slam_pipeline.mge_model_from(
    total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-2, np.log10(3), 30)
)

lens_point = slam_pipeline.mge_model_from(
    total_gaussians=10, gaussian_per_basis=1, log10_sigma_list=np.linspace(-4, -1, 10),
    centre=(0.0, 0.0)
)

miso = af.Model(al.mp.Isothermal)
#if dataset_name == 'slacs1143-0144':
#    miso.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(mean = -0.144, sigma = 0.01, lower_limit=-0.18, upper_limit=-0.12)
#    miso.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(mean = -0.054, sigma = 0.01, lower_limit=-0.08, upper_limit=-0.02)

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    mass=miso,
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
"""
hilbert_pixels = al.model_util.hilbert_pixels_from_pixel_scale(ps)

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_lp_result
)

image_mesh = al.image_mesh.Hilbert(pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01)


image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
)

edge_pixels_total = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

signal_to_noise_threshold = 3.0
over_sample_size_pixelization = np.where(
    galaxy_image_name_dict["('galaxies', 'source')"] > signal_to_noise_threshold,
    4,
    2,
)
over_sample_size_pixelization = al.Array2D(
    values=over_sample_size_pixelization, mask=mask
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=over_sample_size_pixelization,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Delaunay(pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total),
    regularization_init=af.Model(al.reg.AdaptSplit),
)

"""
__SOURCE PIX PIPELINE 2__
"""

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_lp_result
)

image_mesh = al.image_mesh.Hilbert(pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01)


image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
)

edge_pixels_total = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

over_sample_size_pixelization = np.where(
    galaxy_image_name_dict["('galaxies', 'source')"] > signal_to_noise_threshold,
    4,
    2,
)

over_sample_size_pixelization = al.Array2D(
    values=over_sample_size_pixelization, mask=mask
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=over_sample_size_pixelization,
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
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
    ),
    regularization=af.Model(al.reg.AdaptSplit)
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
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
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
"""

multipole_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    multipole_1=None,
    multipole_3=af.Model(al.mp.PowerLawMultipole),
    multipole_4=af.Model(al.mp.PowerLawMultipole),
    reset_shear_prior=True,
    name='mass_multipole'
)


sys.exit()

## The below is to be used in case there are issues with the priors etc
"""
Chained results
"""
settings_search = af.SettingsSearch(
    path_prefix='chained_mass_models_',
    unique_tag=f'{dataset_name}',
    info=None,
    session=None,
)

chained_mass_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=mass_result,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    reset_shear_prior=True,
    name='mass_EPL_chained',
    use_new_mass=True
)

chained_multipole_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=mass_result,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    multipole_1=None,
    multipole_3=af.Model(al.mp.PowerLawMultipole),
    multipole_4=af.Model(al.mp.PowerLawMultipole),
    reset_shear_prior=True,
    name='mass_multipole_chained',
    use_new_mass=True
)
