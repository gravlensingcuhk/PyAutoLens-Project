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


def main():
    """
    The whole chain lives in a function so that multiprocessing's spawned workers, which import
    this script as __mp_main__ (slam_pipeline.search_util's Nautilus sampler pool), do not run it.
    """
    import os

    # The cluster workspace root; set PYAUTOLENS_PROJECT_ROOT to run from another checkout (e.g. locally).
    cosma_path = os.environ.get('PYAUTOLENS_PROJECT_ROOT', path.join(path.sep, 'users', 'wing-yan.chan', 'PyAutoLens-Project'))
    workspace_path = cosma_path
    output_path = path.join(cosma_path, 'output_sam_v3')  # fits from 2026-11-04 (single-group lens light, 50x50 RTU, run-2 adapt image from source_lp); output_sam_v2/ = the 2026-11-03/04 two-group-light fits, output_sam/ = the 2026-11-02/03 separate-disk fits (both gitignored except .zip archives); updated_output*/ = the data/cowls/ fits of 2026-10-26 to 11-02 (gitignored archives), output_v3/ = the pre-2026-10-26 Hilbert/Delaunay fits
    config_path = path.join(cosma_path, 'config')
    conf.instance.push(new_path=config_path, output_path=output_path)
    data_path = path.join(workspace_path, 'data', 'cowls_sam')


    use_jax = True

    # Mixed precision / sparse curvature from the environment (GPU runs); plain al.Settings() on the
    # CPU cluster. See slam_pipeline/search_util.py "__Analysis Settings__".
    analysis_settings = slam_pipeline.search_util.analysis_settings()

    """
    __Dataset__ 
    Load and mask the data.
    """
    dataset_name = str(sys.argv[1])
    filt = "F444W"
    ps = 0.06
    # Cap on the chained positions threshold (arcsec, 2.5 px at 0.06"/px); see slam_pipeline.positions_util "__The Threshold__".
    positions_threshold_max = 0.15
    dataset_path = path.join(data_path, dataset_name, filt)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),  # the reduction's STPSF kernel, 37 x 37 (SW) / 41 x 41 (LW), sums to 1 (data/cowls_sam/README.md)
        pixel_scales=ps,
        check_noise_map=False
    )

    with open(path.join(data_path, dataset_name, 'info.json'), 'r') as file:
        info_file = json.load(file)

    # One grid per source feature (slam_pipeline.positions_util "__Position Sets__"); every stage
    # gets one PositionsLH per set. J18's E+W and N+S pairs are two sets (2026-11-03).
    positions = slam_pipeline.positions_util.positions_sets_from(
        file_path=path.join(data_path, dataset_name, "positions.json")
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
    print(f'MESH: {slam_pipeline.source_pix.mesh_from_env()}  (PYAUTOLENS_MESH={os.environ.get("PYAUTOLENS_MESH", "<unset>")}, PYAUTOLENS_REGULARIZATION={os.environ.get("PYAUTOLENS_REGULARIZATION", "<unset>")})', flush=True)

    dataset = dataset.apply_mask(mask=mask)  # trims the frame to the mask + PSF (config [dataset] trim_to_mask); from here on use dataset.mask, not mask

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
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=use_jax, settings=analysis_settings,
                                  positions_likelihood_list=[al.PositionsLH(threshold=0.2, positions=p) for p in positions],)

    source_bulge = slam_pipeline.mge_model_from(
        total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-3, np.log10(1), 30),
        centre_prior_is_uniform=False,  # GaussianPrior(mean=centre, sigma=0.3) -> unrails the source centre
    )

    # The lens light (2026-11-04): one basis of 30 Gaussians with a single free ellipticity and
    # centre (sigmas 0.01" to the mask edge) plus a spherical point component on the same centre.
    # The second Gaussian group (2026-11-03, gaussian_per_basis=2) is gone: on J18 it fit q ~ 0.2
    # at every sigma, a razor-thin sheet under the arc whose subtraction residuals the pixelized
    # source then fit as fake fragmented flux (output_sam_v2/, 2026-11-04); on J24 it was milder
    # (q ~ 0.6) and added nothing. The separate disk basis before 2026-11-03 (own centre and
    # ellipticity) likewise only absorbed arc light; see slam_pipeline.model_util.mge_model_from.
    lens_bulge = slam_pipeline.mge_model_from(
        total_gaussians=30, gaussian_per_basis=1, log10_sigma_list=np.linspace(-2, np.log10(mask_radius-0.1), 30)
    )

    lens_disk = None

    lens_point = slam_pipeline.mge_model_from(
        total_gaussians=10, gaussian_per_basis=1, log10_sigma_list=np.linspace(-4, np.log10(0.025), 10),
        use_spherical=True,
        centre_priors=(lens_bulge.profile_list[0].centre.centre_0, lens_bulge.profile_list[0].centre.centre_1),
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

    Rectangular RTU mesh (40 x 40) + AdaptPower regularization, the HST project's stability setup;
    see slam_pipeline/source_pix.py "__The Source Mesh__" for why the Hilbert/Delaunay/AdaptSplit
    scheme was dropped, and "__The Hilbert Mesh__" for its re-trial under PYAUTOLENS_MESH=hilbert. The adapt image is the SOURCE LP source capped at S/N 3 on a copy, while the
    uncapped image drives the pixelization over-sampling map (4 on the arcs, 2 elsewhere).
    """

    settings_search = af.SettingsSearch(
        path_prefix='model_setup_free_source',
        unique_tag=f'{dataset_name}',
        info=None,
        session=None,
        number_of_cores = 16,
    )

    adapt_images, source_image_raw = slam_pipeline.source_pix.adapt_images_from(result=source_lp_result, mask=dataset.mask)

    dataset = slam_pipeline.source_pix.apply_pixelization_over_sampling(
        dataset=dataset, mask=dataset.mask, source_image_raw=source_image_raw, over_sample_size_lp=over_sample_size,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        settings=analysis_settings,
        adapt_images=adapt_images,
        # The marked positions, threshold from the previous result (slam_pipeline.positions_util).
        positions_likelihood_list=slam_pipeline.positions_util.positions_likelihood_list_from(
            result=source_lp_result, positions=positions, maximum_threshold=positions_threshold_max
        ),
        use_jax=use_jax,
    )

    source_pix_result_1 = slam_pipeline.source_pix.run_1(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
    )

    """
    __SOURCE PIX PIPELINE 2__

    The adapt image is the same SOURCE LP one search 1 used, deliberately NOT source_pix[1]'s
    reconstruction: when the pixelized source fragments (J18), adapting this mesh to it feeds the
    clumps back into the fit (output_sam_v2/, 2026-11-04). The mesh's weight_power / weight_floor
    are still fitted. The uncapped image drives the pixelization over-sampling map.
    """

    adapt_images, source_image_raw = slam_pipeline.source_pix.adapt_images_from(result=source_lp_result, mask=dataset.mask)

    dataset = slam_pipeline.source_pix.apply_pixelization_over_sampling(
        dataset=dataset, mask=dataset.mask, source_image_raw=source_image_raw, over_sample_size_lp=over_sample_size,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        settings=analysis_settings,
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
        settings=analysis_settings,
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
        settings=analysis_settings,
        adapt_images=adapt_images,
        positions_likelihood_list=slam_pipeline.positions_util.positions_likelihood_list_from(
            result=source_pix_result_2, positions=positions, maximum_threshold=positions_threshold_max
        ),
    )

    mass_result = slam_pipeline.mass_total.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        mass=af.Model(al.mp.PowerLaw),
        reset_shear_prior=True,
        ell_comps_from_light=True,  # ell_comps prior centred on the lens light (mass_total "__The Mass Ellipticity Prior__")
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
        ell_comps_from_light=True,
        name='mass_multipole'
    )



if __name__ == "__main__":
    # Restarts with a smaller n_batch on a GPU out-of-memory failure (search_util "__Out-Of-Memory Retry__").
    slam_pipeline.search_util.run_with_oom_retry(main)
