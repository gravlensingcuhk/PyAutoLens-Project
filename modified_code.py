import os
from pathlib import Path
import autolens as al
import autolens.plot as aplt
import autofit as af

workspace_path = Path("/users/wing-yan.chan/autolens_workspace")
os.chdir(workspace_path)
print(f"Working Directory successfully set to: {os.getcwd()}")

dataset_name = "slacs1250+0523"
dataset_path = Path("dataset") / "imaging" / dataset_name

# load mask
mask = al.Mask2D.from_fits(
    file_path=dataset_path / "mask.fits",
    pixel_scales=0.05,
    hdu=0
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits", 
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.05,
    check_noise_map=False
)
dataset = dataset.apply_mask(mask=mask)
plot_path = workspace_path / "output" / dataset_name / "plots"
plot_path.mkdir(parents=True, exist_ok=True)

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=plot_path, filename="dataset_plot", format="png")
)
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot_2d)
dataset_plotter.subplot_dataset()

# load positions
positions = al.from_json(
    file_path=dataset_path / "positions.json"
)

mass = af.Model(al.mp.IsothermalSph)
mass_bulge = af.Model(al.lp.Sersic)
lens = af.Model(al.Galaxy, redshift=0.2318, mass=mass, bulge=mass_bulge)
bulge = af.Model(al.lp_linear.ExponentialCoreSph)
source = af.Model(al.Galaxy, redshift=0.7953, bulge=bulge)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

# tie centres 
model.galaxies.lens.bulge.centre = model.galaxies.lens.mass.centre

# priors
model.galaxies.lens.mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.05)
model.galaxies.lens.mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
model.galaxies.lens.mass.einstein_radius = af.GaussianPrior(mean=1.0, sigma=0.3) 
model.galaxies.lens.bulge.intensity = af.GaussianPrior(mean=1.0, sigma=0.5)
model.galaxies.lens.bulge.effective_radius = af.GaussianPrior(mean=1.0, sigma=0.5)
model.galaxies.lens.bulge.sersic_index = af.GaussianPrior(mean=4.0, sigma=1.0)
model.galaxies.lens.bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
model.galaxies.lens.bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
model.galaxies.source.bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
model.galaxies.source.bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
model.galaxies.source.bulge.effective_radius = af.GaussianPrior(mean=0.5, sigma=0.2)

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=al.analysis.positions.PositionsLH(
        positions=positions,
        threshold=1.0
    )
)

if __name__ == "__main__":
    search = af.LBFGS(
        path_prefix=os.path.join("output", dataset_name),
        name="basic_fit_lbfgs",
        iterations_per_quick_update=10
    )
    result = search.fit(model=model, analysis=analysis)
    
    # saving
    mat_plot_2d_fit = aplt.MatPlot2D(
        output=aplt.Output(path=plot_path, filename="fit_plot_lbfgs", format="png")
    )
    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit, mat_plot_2d=mat_plot_2d_fit)
    fit_plotter.subplot_fit()

    # tracer
    mat_plot_2d_tracer = aplt.MatPlot2D(
        output=aplt.Output(path=plot_path, filename="tracer_plot_lbfgs", format="png")
    )
    tracer_plotter = aplt.TracerPlotter(
        tracer=result.max_log_likelihood_tracer,
        grid=dataset.grid,
        mat_plot_2d=mat_plot_2d_tracer
    )
    tracer_plotter.subplot_tracer()
