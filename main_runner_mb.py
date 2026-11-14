import json
from autonerves import jax_wrapper  # Sets JAX environment before other imports
import sys
import os
from os import path
import autofit as af
import autolens as al
from autonerves import conf
import numpy as np
import slam_pipeline
import slam_pipeline.multi_band

PIXEL_SCALES = {"F115W": 0.03, "F150W": 0.03, "F277W": 0.06, "F444W": 0.06}


def bands_for(data_path, dataset_name):
    """
    PYAUTOLENS_BANDS="F150W,F277W,..." overrides. Default: the primary band first, then the
    rest by measured arc S/N (primary_band.json), i.e. strongest constraint first.
    """
    env = os.environ.get("PYAUTOLENS_BANDS")
    if env:
        return [b.strip() for b in env.split(",")]
    with open(path.join(data_path, dataset_name, "primary_band.json")) as f:
        pb = json.load(f)
    scores = pb["scores"]
    rest = sorted(scores, key=lambda b: -scores[b]["arc_snr"])
    bands = [pb["primary_band"]] + [b for b in rest if b != pb["primary_band"]]
    return bands


def dataset_for_band(data_path, dataset_name, band, noise_map_file):
    """The single-band runners' dataset block, per band (trim_to_mask applies inside)."""
    dataset_path = path.join(data_path, dataset_name, band)
    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, noise_map_file),
        psf_path=path.join(dataset_path, "psf.fits"),
        pixel_scales=PIXEL_SCALES[band],
        check_noise_map=False,
    )
    mask_extra_galaxies = al.Mask2D.from_fits(
        file_path=path.join(dataset_path, "mask_extra_galaxies.fits"),
        pixel_scales=dataset.pixel_scales,
        invert=True,
    )
    dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
    with open(path.join(data_path, dataset_name, "info.json")) as file:
        mask_radius = json.load(file)["mask_radius"]
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    dataset = dataset.apply_mask(mask=mask)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.2, 0.4],
        centre_list=[(0.0, 0.0)],
    )
    return dataset.apply_over_sampling(over_sample_size_lp=over_sample_size), over_sample_size


def main():
    cosma_path = os.environ.get('PYAUTOLENS_PROJECT_ROOT', path.join(path.sep, 'users', 'wing-yan.chan', 'PyAutoLens-Project'))
    output_path = path.join(cosma_path, os.environ.get('PYAUTOLENS_OUTPUT_DIR', 'output_sam_mb'))
    config_path = path.join(cosma_path, 'config')
    conf.instance.push(new_path=config_path, output_path=output_path)
    data_path = path.join(cosma_path, 'data', 'cowls_sam')

    use_jax = True
    analysis_settings = slam_pipeline.search_util.analysis_settings()

    dataset_name = str(sys.argv[1])
    bands = bands_for(data_path, dataset_name)
    noise_map_file = f'noise_map_{os.environ.get("PYAUTOLENS_NOISE")}.fits' if os.environ.get("PYAUTOLENS_NOISE") else "noise_map.fits"
    positions_threshold_max = 0.10

    with open(path.join(data_path, dataset_name, 'info.json')) as file:
        info_file = json.load(file)
    redshift_lens = info_file['lens_spec_z']
    redshift_source = info_file['source_spec_z']

    positions = slam_pipeline.positions_util.positions_sets_from(
        file_path=path.join(data_path, dataset_name, "positions.json")
    )

    print(f'RUNNING MULTI-BAND Dataset: {dataset_name}, bands: {bands}, noise: {noise_map_file}')
    print(f'MESH: {slam_pipeline.source_pix.mesh_from_env()}', flush=True)

    datasets, over_sample_sizes = [], []
    for b in bands:
        d, oss = dataset_for_band(data_path, dataset_name, b, noise_map_file)
        datasets.append(d)
        over_sample_sizes.append(oss)

    # Positions are marked in the primary band; with a shared mass model one band's
    # positions likelihood constrains all (multi_band.py module docstring).
    def analysis_for(i, **kwargs):
        if i == 0 and "positions_likelihood_list" not in kwargs:
            kwargs["positions_likelihood_list"] = [al.PositionsLH(threshold=0.2, positions=p) for p in positions]
        return al.AnalysisImaging(
            dataset=datasets[i],
            settings=analysis_settings,
            use_jax=use_jax,
            **kwargs,
        )

    """
    __SOURCE LP (per band)__
    """
    source_lp_results = []
    for i, band in enumerate(bands):
        settings_search = af.SettingsSearch(
            path_prefix='model_setup_mb',
            unique_tag=f'{dataset_name}_{band}',  # the dataset is not in the identifier
            info=None,
            session=None,
            number_of_cores=16,
        )
        source_bulge = slam_pipeline.mge_model_from(
            total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-3, np.log10(1), 30),
            centre_prior_is_uniform=False,
        )
        source_disk = slam_pipeline.mge_model_from(
            total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-3, np.log10(1), 30),
            centre_prior_is_uniform=False,
        )
        mask_radius = info_file['mask_radius']
        lens_bulge = slam_pipeline.mge_model_from(
            total_gaussians=30, gaussian_per_basis=1,
            log10_sigma_list=np.linspace(-2, np.log10(mask_radius - 0.1), 30),
        )
        lens_point = slam_pipeline.mge_model_from(
            total_gaussians=10, gaussian_per_basis=1, log10_sigma_list=np.linspace(-4, np.log10(0.025), 10),
            use_spherical=True,
            centre_priors=(lens_bulge.profile_list[0].centre.centre_0, lens_bulge.profile_list[0].centre.centre_1),
        )
        mass = af.Model(al.mp.Isothermal)
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        source_lp_results.append(slam_pipeline.source_lp.run(
            settings_search=settings_search,
            analysis=analysis_for(i),
            mass=mass,
            shear=af.Model(al.mp.ExternalShear),
            lens_bulge=lens_bulge,
            lens_disk=None,
            lens_point=lens_point,
            source_bulge=source_bulge,
            source_disk=source_disk,
            redshift_lens=redshift_lens,
            redshift_source=redshift_source,
        ))

    settings_search = af.SettingsSearch(
        path_prefix='model_setup_mb',
        unique_tag=f'{dataset_name}_mb',
        info=None,
        session=None,
        number_of_cores=16,
    )

    """
    __SOURCE PIX 1 (multi-band, shared mass)__
    """
    analysis_list = []
    for i, band in enumerate(bands):
        adapt_images, source_image_raw = slam_pipeline.source_pix.adapt_images_from(
            result=source_lp_results[i], mask=datasets[i].mask)
        datasets[i] = slam_pipeline.source_pix.apply_pixelization_over_sampling(
            dataset=datasets[i], mask=datasets[i].mask, source_image_raw=source_image_raw,
            over_sample_size_lp=over_sample_sizes[i],
        )
        analysis_list.append(analysis_for(
            i,
            adapt_images=adapt_images,
            positions_likelihood_list=slam_pipeline.positions_util.positions_likelihood_list_from(
                result=source_lp_results[i], positions=positions, maximum_threshold=positions_threshold_max
            ) if i == 0 else None,
        ))

    source_pix_result_1 = slam_pipeline.multi_band.run_1__multi_shared(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_lp_result_list=source_lp_results,
    )

    """
    __SOURCE PIX 2 (multi-band, mass fixed)__
    """
    analysis_list = []
    for i, band in enumerate(bands):
        # As for the single-band J18/J0959: run 2 adapts to SOURCE LP (the smooth MGE), not to
        # run 1's reconstruction, unless PYAUTOLENS_ADAPT_RUN2=sp1.
        adapt_result = source_pix_result_1[i] if os.environ.get("PYAUTOLENS_ADAPT_RUN2") == "sp1" else source_lp_results[i]
        adapt_images, source_image_raw = slam_pipeline.source_pix.adapt_images_from(
            result=adapt_result, mask=datasets[i].mask)
        analysis_list.append(analysis_for(i, adapt_images=adapt_images, positions_likelihood_list=None))

    source_pix_result_2 = slam_pipeline.multi_band.run_2__multi_shared(
        settings_search=settings_search,
        analysis_list=analysis_list,
        source_lp_result_list=source_lp_results,
        source_pix_result_1=source_pix_result_1,
    )

    """
    __LIGHT LP (per band)__
    """
    light_results = []
    for i, band in enumerate(bands):
        settings_light = af.SettingsSearch(
            path_prefix='model_setup_mb',
            unique_tag=f'{dataset_name}_{band}',
            info=None,
            session=None,
            number_of_cores=16,
        )
        mask_radius = info_file['mask_radius']
        lens_bulge = slam_pipeline.mge_model_from(
            total_gaussians=30, gaussian_per_basis=1,
            log10_sigma_list=np.linspace(-2, np.log10(mask_radius - 0.1), 30),
        )
        lens_point = slam_pipeline.mge_model_from(
            total_gaussians=10, gaussian_per_basis=1, log10_sigma_list=np.linspace(-4, np.log10(0.025), 10),
            use_spherical=True,
            centre_priors=(lens_bulge.profile_list[0].centre.centre_0, lens_bulge.profile_list[0].centre.centre_1),
        )
        light_results.append(slam_pipeline.light_lp.run(
            settings_search=settings_light,
            analysis=analysis_for(i, adapt_images=analysis_list[i].adapt_images, positions_likelihood_list=None),
            source_result_for_lens=source_pix_result_1[i],
            source_result_for_source=source_pix_result_2[i],
            lens_bulge=lens_bulge,
            lens_disk=None,
            lens_point=lens_point,
        ))

    """
    __MASS (multi-band, shared EPL / EPL+multipoles)__
    """
    settings_mass = af.SettingsSearch(
        path_prefix='mass_models_mb',
        unique_tag=f'{dataset_name}_mb',
        info=None,
        session=None,
        number_of_cores=16,
    )
    analysis_mass = []
    for i, band in enumerate(bands):
        analysis_mass.append(analysis_for(
            i,
            adapt_images=analysis_list[i].adapt_images,
            positions_likelihood_list=slam_pipeline.positions_util.positions_likelihood_list_from(
                result=source_pix_result_2[0], positions=positions, maximum_threshold=positions_threshold_max
            ) if i == 0 else None,
        ))

    mass_result = slam_pipeline.multi_band.mass__multi_shared(
        settings_search=settings_mass,
        analysis_list=analysis_mass,
        source_pix_result_1=source_pix_result_1,
        source_pix_result_2=source_pix_result_2,
        light_result_list=light_results,
        name='mass_EPL',
    )

    multipole_result = slam_pipeline.multi_band.mass__multi_shared(
        settings_search=settings_mass,
        analysis_list=analysis_mass,
        source_pix_result_1=source_pix_result_1,
        source_pix_result_2=source_pix_result_2,
        light_result_list=light_results,
        multipole_3=af.Model(al.mp.PowerLawEllipticalMultipole),
        multipole_4=af.Model(al.mp.PowerLawEllipticalMultipole),
        name='mass_multipole',
    )


if __name__ == "__main__":
    # Restarts with a smaller n_batch on a GPU out-of-memory failure (search_util "__Out-Of-Memory Retry__").
    slam_pipeline.search_util.run_with_oom_retry(main)
