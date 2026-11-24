"""
Mass-stage likelihood roughness scan (J0959, 2026-11-24).

Hypothesis: the Delaunay mesh re-triangulates every likelihood call as the trial mass model
moves, so the mass-stage likelihood is discontinuous in the mass parameters (the nested
sampling stalls), while the RTU mesh keeps a fixed rectangular topology and stays smooth.

Fixes every model parameter at the v9 RTU mass_EPL best-fit (positions-consistent mode) and
scans einstein_radius in fine steps, once with the Delaunay chain's source pixelization
(pix2 64ae909a + light 2100855e) and once with the RTU chain's (96bd6885 + 3b22723b).

The zip archives carry no result.pickle, so the source pixelization and lens light are read
from each stage's files/tracer.json. Adapt images are omitted: they shift the absolute logL
(regularization weights) but not the per-call triangulation roughness under test.

Run: PYTHONPATH=. uv run python data_prep/scan_mass_likelihood_roughness.py
"""
import json
import zipfile
from os import path

import numpy as np

import autofit as af
import autolens as al
from autonerves import conf
from autonerves.dictable import from_dict

import slam_pipeline

LENS = "COSJ095921+020638"
BAND = "F150W"

conf.instance.push(new_path="config", output_path="/tmp/scan_roughness_output")

with open(path.join("data/cowls_sam", LENS, "info.json")) as f:
    info = json.load(f)

dataset_path = path.join("data/cowls_sam", LENS, BAND)
dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map_psfcoh.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.03,
    check_noise_map=False,
)
mask_extra = al.Mask2D.from_fits(
    file_path=path.join(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=dataset.pixel_scales,
    invert=True,
)
dataset = dataset.apply_noise_scaling(mask=mask_extra)
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=info["mask_radius"],
)
dataset = dataset.apply_mask(mask=mask)
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid, sub_size_list=[4, 2, 1], radial_list=[0.2, 0.4], centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)


def tracer_from(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return from_dict(json.loads(z.read("files/tracer.json")))


def adapt_images_from_zip(pix2_zip):
    """The pix2 stage's adapt images + Delaunay image-plane mesh grid, from the zip's FITS."""
    import io
    import autoarray as aa
    from astropy.io import fits as astropy_fits
    from autonerves.fitsable import ndarray_via_hdu_from

    with zipfile.ZipFile(pix2_zip) as z:
        images_raw = z.read("image/adapt_images.fits")
        grids_raw = (
            z.read("image/adapt_image_plane_mesh_grids.fits")
            if "image/adapt_image_plane_mesh_grids.fits" in z.namelist()
            else None
        )

    with astropy_fits.open(io.BytesIO(images_raw)) as hdul:
        header = hdul[0].header
        pixel_scales = (header["PIXSCAY"], header["PIXSCAX"])
        origin = (header["ORIGINY"], header["ORIGINX"])
        mask_cached = aa.Mask2D(
            mask=ndarray_via_hdu_from(hdul[0]), pixel_scales=pixel_scales, origin=origin
        )
        galaxy_name_image_dict = {}
        for hdu in hdul[1:]:
            image = aa.Array2D.no_mask(
                values=ndarray_via_hdu_from(hdu), pixel_scales=pixel_scales, origin=origin
            )
            galaxy_name_image_dict[hdu.header["EXTNAME"].lower()] = image.apply_mask(mask_cached)

    if grids_raw is None:
        return al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

    with astropy_fits.open(io.BytesIO(grids_raw)) as hdul:
        galaxy_name_mesh_grid_dict = {
            hdu.header["EXTNAME"].lower(): aa.Grid2DIrregular(values=ndarray_via_hdu_from(hdu))
            for hdu in hdul[1:]
        }

    return al.AdaptImages(
        galaxy_name_image_dict=galaxy_name_image_dict,
        galaxy_name_image_plane_mesh_grid_dict=galaxy_name_mesh_grid_dict,
    )


SETUP = "output_sam_v9/model_setup_free_source/COSJ095921+020638"
# v9 J0959 RTU mass_EPL max-likelihood mass+shear (zip 9ac0b146)
tr_mass = tracer_from("output_sam_v9/mass_models/COSJ095921+020638/mass_EPL/9ac0b146fbf7ba54cab4a6facfc2e8dc.zip")
mass_ml = tr_mass.galaxies[0].mass
shear_ml = tr_mass.galaxies[0].shear
print("RTU best-fit mass:", mass_ml)

for tag, pix2_zip, light_zip in [
    ("delaunay", f"{SETUP}/source_pix[2]/64ae909a325de70039cc38558f965deb.zip",
     f"{SETUP}/light[1]/2100855eedb0d4a44357892d12ad81c6.zip"),
    ("delaunay_nn", f"{SETUP}/source_pix[2]/64ae909a325de70039cc38558f965deb.zip",
     f"{SETUP}/light[1]/2100855eedb0d4a44357892d12ad81c6.zip"),
]:
    pix2_tr = tracer_from(pix2_zip)
    light_tr = tracer_from(light_zip)
    pixelization = pix2_tr.galaxies[1].pixelization
    if tag == "delaunay_nn":
        # Same vertices/connectivity/regularization, Sibson natural-neighbour weights:
        # continuous through the triangle flips that make the plain Delaunay surface rough.
        from autoarray.inversion.mesh.mesh.delaunay_nn import DelaunayNN

        mesh = pixelization.mesh
        pixelization = al.Pixelization(
            mesh=DelaunayNN(
                pixels=mesh.pixels,
                zeroed_pixels=mesh.zeroed_pixels,
                areas_factor=mesh.areas_factor,
            ),
            regularization=pixelization.regularization,
        )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        settings=slam_pipeline.search_util.analysis_settings(),
        adapt_images=adapt_images_from_zip(pix2_zip),
        use_jax=False,
    )

    lens = al.Galaxy(
        redshift=info["lens_spec_z"],
        bulge=light_tr.galaxies[0].bulge,
        point=light_tr.galaxies[0].point,
        mass=al.mp.PowerLaw(
            centre=mass_ml.centre,
            ell_comps=mass_ml.ell_comps,
            einstein_radius=mass_ml.einstein_radius,
            slope=mass_ml.slope,
        ),
        shear=al.mp.ExternalShear(gamma_1=shear_ml.gamma_1, gamma_2=shear_ml.gamma_2),
    )
    source_gal = al.Galaxy(redshift=info["source_spec_z"], pixelization=pixelization)

    instance = af.ModelInstance()
    instance.galaxies = af.ModelInstance()
    instance.galaxies.lens = lens
    instance.galaxies.source = source_gal

    print(f"\n=== {tag} ===")
    centre_te = mass_ml.einstein_radius
    tes = np.concatenate([
        np.arange(centre_te - 0.02, centre_te + 0.0201, 0.004),
        np.arange(centre_te - 0.0015, centre_te + 0.00151, 0.00025),
    ])
    tes = np.sort(np.unique(tes))
    lls = []
    for te in tes:
        instance.galaxies.lens.mass.einstein_radius = float(te)
        ll = analysis.log_likelihood_function(instance=instance)
        lls.append(ll)
        print(f"{te:.5f} {ll:.4f}", flush=True)
    lls = np.array(lls)
    coarse = lls[tes <= centre_te - 0.004]
    fine = lls[np.abs(tes - centre_te) <= 0.0015]
    print(f"{tag}: coarse diff std (dtheta=0.004): {np.std(np.diff(coarse)):.3f}")
    print(f"{tag}: fine diffs (dtheta=0.00025): {np.round(np.diff(fine), 3)}")
    print(f"{tag}: fine diff std: {np.std(np.diff(fine)):.4f}")
