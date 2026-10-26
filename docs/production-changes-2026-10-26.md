# COWLS SLaM: production changes on branch `sam`

As of 2026-10-26.

## Summary

Branch `sam` of PyAutoLens-Project replaces the Hilbert/Delaunay + AdaptSplit source
reconstruction with the HST project's rectangular RTU mesh + AdaptPower setup, on an updated
PyAuto stack (Python 3.13, autolens/autonerves 2026.10.23.1, vendored dev branches of autoarray,
autofit and autogalaxy). The change was made because J18's `source_pix[1]` ran 13 days without
converging; a likelihood scan showed the old scheme is discontinuous in the mass parameters and
the new one is smooth. The same commit adds Nautilus stopping rules and likelihood-call caps, and
follow-ups make autofit's running-search lock work under HTCondor, SLURM or a plain local run and
switch the mass multipoles to the elliptical formalism. Both lenses must be refitted from
`source_pix[1]`; `source_lp` results reload unchanged. The branch is committed locally and not
yet pushed.

## Why: the J18 source_pix[1] stall

COSJ100018+022138 `source_pix[1]` was not hung; Nautilus could not finish its sampling phase.
After 130,000 samples over 13.6 days the posterior had an effective sample size of 2.2 against
the required 500, because a handful of samples carried almost all the weight. J24 showed the
same tendency in milder form (ESS 122 in pix 1, 13.7 in light 1).

The mechanism is a likelihood that is discontinuous in the mass parameters, from two sources in
the old scheme:

- The Delaunay mesh is rebuilt from ray-traced image points on every call, so triangles flip
  and the neighbour graph changes as the mass moves.
- AdaptSplit sets each pixel's regularization from the adapt image divided by the integer
  number of sub-pixels landing in it, which jumps by whole units; with the fitted inner and
  outer coefficients six decades apart (inner at effectively zero, outer about 76), and the
  coefficients entering at the fourth power, those jumps dominate the evidence.

More live points do not help: Nautilus's stop condition is the effective sample size, which a
rugged likelihood keeps near one regardless of `n_live`. A secondary slowdown (9 s per sample on
the cluster, 2.4x the J24 rate) comes from the larger J18 data vector under JAX vmap and is not
addressed here.

## Source reconstruction: old and new

Both lenses now fit a 25 x 25 rectangular RTU mesh with AdaptPower regularization, the HST
project's production setup, sized down from its 30 x 30 for the 1.5" COWLS masks. Everything
else in the chain (MGE lens light, Isothermal + shear, S/N 3 adapt-image cap, over-sampling map,
positive-only solver, border relocator) is unchanged.

| | Old (to 2026-10-25) | New (branch `sam`) |
| --- | --- | --- |
| Source pixels | Hilbert image mesh, 1500 points, `weight_power` 3.5, `weight_floor` 0.01, + 30 circle edge points; Delaunay triangulation in the source plane, rebuilt every call | Fixed 25 x 25 lattice warped by a kernel-CDF (erf-sum) of the traced points: `RectangularRTUAdaptDensity` in pix 1, `RectangularRTUAdaptImage` in pix 2 |
| Mapping | Barycentric over 3 vertices; edge ring zeroed via `zeroed_pixels` | Bilinear over 4 cell centres; edge cells zeroed by `use_edge_zeroed_pixels` |
| Regularization | `AdaptSplit`: signals / integer sub-pixel count, cross-of-four split, coefficients at lambda^4 | `AdaptPower`: signals / sum of mapping weights (`pixel_signal_normalisation: weights`), graph Laplacian, coefficients at lambda^2 |
| Fitted in pix 2 | inner, outer, signal_scale | inner, outer, signal_scale, `weight_power`, `weight_floor` |
| Adapt image for pix 2 | source_lp source | pix 1 pixelized source (standard SLaM) |
| Nautilus | n_live 150 / 75, library f_live, no cap | n_live 200 / 300, f_live 0.05, caps 100k / 10k x (n_live // 75), capped runs logged to `output/capped_searches.txt` |
| n_batch | 16 | 16 (CPU, matches `CPUS_PER_TASK`) |

The scan behind the decision stepped `einstein_radius` through J18's pix 1 maximum-likelihood
model in 41 steps of 1e-4" (about 0.12 sigma) and measured the roughness of the log-likelihood
curve. Values are nats.

| Scheme | Library | RMS about a quadratic | Max adjacent step |
| --- | --- | --- | --- |
| Delaunay + AdaptSplit (old) | pinned 2026.2.26.4 | 120 | 366 |
| Delaunay + ConstantSplit | pinned | 0.7 | 2.0 |
| RTU AdaptImage + AdaptPower, count normalisation | dev branches | 2.3 | 4.6 |
| RTU AdaptDensity + AdaptPower, count normalisation | dev | 1.0 | 2.1 |
| RTU AdaptImage + AdaptPower, weights normalisation | dev | 0.17 | 0.5 |
| RTU AdaptDensity + AdaptPower, weights normalisation | dev | 0.01 | 0.02 |
| Bilinear AdaptImage + AdaptPower, weights | dev | 2.8 | 5.2 |

The ConstantSplit row shows the roughness came from the adaptive weighting, not the Delaunay mesh
alone; the weights rows show the normalisation change removes what the RTU mesh leaves. Caveats:
one parameter scanned at one point; coefficients and mesh shape untuned; absolute likelihoods are
not comparable across schemes. RTU cost 2.0 to 3.2 s per evaluation on a Mac against 3.7 s for
Delaunay; the smoke test on the full runner sampled at about 0.3 s per call on 8 threads.

## Environment and config

The project moved from the pinned PyAuto release 2026.2.26.4 on Python 3.11 to the October 2026
stack on Python 3.13. `uv sync` rebuilds the environment; `uv.lock` is regenerated and
cross-platform.

| Package | Source | Version |
| --- | --- | --- |
| autolens | PyPI (on `main` in the codespace) | 2026.10.23.1 |
| autonerves (renamed autoconf) | PyPI | 2026.10.23.1 |
| autoarray | vendored `pyauto/PyAutoArray`, branch `production/hst` @ `40ae61b4` | 2026.10.23.1.post3 |
| autofit | vendored `pyauto/PyAutoFit`, branch `production/hst` head (lock patch included) | 2026.10.23.1.post1 |
| autogalaxy | vendored `pyauto/PyAutoGalaxy`, branch `feature/elliptical-multipole` @ `b1f4cabf` | 2026.10.16.1.post5 |

The vendored copies hold only the package, `pyproject.toml`, `setup.py`, README, LICENSE and
MANIFEST (no tests, docs or git). `setup.py` stamps the static version so the intra-family `>=`
floors resolve. `pyauto/PROVENANCE.md` records commits and the re-export command; refresh by
re-running it and updating that table.

Code and config changes that follow from the new library:

- `from autoconf import ...` is `from autonerves import ...` in both runners,
  `slam_pipeline/subhalo/` and all of `proposed_subhalo/`.
- `config/general.yaml [inversion]` gains `pixel_signal_normalisation: weights`,
  `rtu_forward: tabulated`, `rtu_forward_knots: 1024` and `use_edge_zeroed_pixels: true`; a
  `version:` block disables the workspace version check. A pushed config overrides library
  defaults key by key, so only changed keys are needed.
- `config/non_linear/` is removed: this autofit has no non-linear config, Nautilus settings are
  `af.Nautilus` kwargs. The old `nest.yaml` values (n_eff 500, discard_exploration true) were
  dead; defaults now follow the library and the HST project (discard_exploration false).
- `apply_sparse_operator_cpu()` is dropped from the runners. Its Numba inversion path is not
  JAX-traceable, and autofit now jit-traces the likelihood before Nautilus starts.
- Prior files for the RTU meshes and AdaptPower are not in `config/priors/`; they fall back to
  autogalaxy's own, so pix 2's `weight_power` is Uniform 0 to 10 and `weight_floor` LogUniform
  1e-5 to 1.
- `PYAUTOLENS_PROJECT_ROOT` overrides the hard-coded cluster root for local runs.

Not adopted from the HST project: mixed precision (GPU-oriented), the sampler process pool,
light and mass stage caps.

## Mass multipoles: elliptical formalism

The `mass_multipole` stage and the subhalo stages' multipole reset now use
`al.mp.PowerLawEllipticalMultipole` (PyAutoGalaxy `feature/elliptical-multipole`) in place of the
circular `al.mp.PowerLawMultipole`. The m=3 and m=4 terms deform the power law's elliptical
isodensity contours, with strength k_m (a fractional contour deformation in units of the major
axis) and shape angle s_m measured from the major axis, so they are directly comparable with
isophote multipoles. The old class added a circular multipole in the sky frame and its
deflection carries a 1 / (m^2 - (3 - slope)^2) factor that is singular for m=1.

- `slam_pipeline.mass_total.link_multipole` ties each multipole's `centre`, `ell_comps`,
  `einstein_radius` and `slope` to the power law by prior links (the circular class had no
  `ell_comps` link). Each order adds exactly two free parameters: 12 in total for PowerLaw +
  shear + m=3 + m=4.
- `proposed_subhalo/slam_pipeline/subhalo/detect.py` rebuilds the multipoles the same way in
  both places it resets them; before, the reset left them unlinked.
- `config/priors/mass/total/power_law_elliptical_multipole.yaml` carries the project prior on
  `shape_multipole_comps`: TruncatedGaussian(0, 0.01) on [-0.1, 0.1], the same as this project
  used for the circular `multipole_comps`. The HST project samples broader (sigma 0.05, bounds
  0.15 for m=3) and reweights afterwards.

## Running-search lock (autofit)

autofit's `production/hst` branch guards an unfinished search's output folder with a lock
directory `<output>.running`, so a resubmitted job cannot write into a folder another process is
still fitting. As committed it only queried SLURM, and its fallback heartbeat was touched on
search updates, which this project disables (`iterations_per_quick_update: 1e99`), so under
HTCondor a lock left by a killed job on another node would have blocked a resubmission for six
hours.

The lock is now scheduler-agnostic. The holder file records host, pid, the pid's start time and
a job id from `PYAUTO_JOB_ID`, SLURM, PBS, LSF, SGE or the HTCondor job ad. Liveness is decided
in order:

1. Same process, or same job id: its own lock, taken over.
2. Same host: the pid is probed, with the start time (from `ps etime`, locale-free, 10 s
   tolerance) guarding against a reused pid.
3. A scheduler whose client is on PATH: its queue is asked (`squeue`, `condor_q`, `qstat`,
   `bjobs`).
4. Otherwise the heartbeat, which a daemon thread refreshes every 60 s while the lock is held;
   stale after 30 min. Both are class attributes on `AbstractPaths`.

The change lives in `autofit/non_linear/paths/abstract.py` on PyAutoFit `production/hst` and
`feature/paths-running-lock-replay` (pushed to `samlange04/PyAutoFit`) and in the vendored copy
here. Side effect: the heartbeat thread makes the process multi-threaded, so a search that forks
a multiprocessing pool sees Python's fork warning; the JAX path used here has no pool.

## GPU runs on phygpu, CPU replay on HTCondor

The chain through `mass_multipole` runs on phygpu (SLURM, one lens per GPU); the archives are then
copied to the HTCondor cluster's `output/` for the CPU-side subhalo stages. `phygpu/README.md` has
the bring-up, submission and sync commands. Everything GPU-specific is an environment switch with
a CPU default, read by `slam_pipeline/search_util.py` (ported from the HST project):

| Variable | phygpu | CPU default | Effect |
| --- | --- | --- | --- |
| `PYAUTOLENS_N_BATCH` | 20 | 16 | JAX vmap batch of live points; not in the search identifier |
| `PYAUTOLENS_MIXED_PRECISION` | 1 | off | PyAutoArray mixed precision (fp32 mapping cube and PSF FFT, fp64 solve); not in the identifier |
| `PYAUTOLENS_SAMPLER_PROCESSES` | 4 | 1 (no pool) | CPU process pool for Nautilus's network training, so the GPU is not idle-killed |
| `PYAUTOLENS_SPARSE_CURVATURE` | unset | off | autoarray's new sparse curvature matrix (`production/hst` 82cc4396, re-vendored); unbenchmarked on COWLS data |

Infrastructure: `pyproject.toml` gains a `gpu` extra (`uv sync --extra gpu` installs the CUDA 12
jaxlib on Linux); `phygpu/activate.sh` (environment, GPU-use log), `phygpu/multi_run2` (exclusive
2-GPU node, both lenses), `phygpu/run_lens` (one GPU, tests), `phygpu/sync_to_condor.sh` (rsync of
`*.zip` only, same `output/<prefix>/<lens>/<search>/<id>.zip` layout, no checkpoints, locks or
replay folders). Both runners are wrapped in `main()`: the pool's spawned workers import the
script as `__mp_main__`, and without the guard they re-ran the whole chain (caught in the local
smoke test).

Why CPU replay works: archives hold only JSON, CSV and images; Nautilus's checkpoint is dropped on
completion; the saved analysis settings (`use_mixed_precision`) are valid on CPU JAX; the subhalo
stages build their own analyses with CPU defaults; none of the switches enters a search identifier
(checked: `source_pix[1]` has the same identifier at n_batch 16 and 20). The one requirement is
that the HTCondor side runs the same commit of this repository (vendored `pyauto/`, pinned
autolens/autonerves) with a plain `uv sync`.

## Operational notes

- **Refit from source_pix[1].** The model changed, so the old `source_pix`, `light` and `mass`
  results have new identifiers and are not reused. `source_lp[1]` results reload under their old
  identifier (verified for J18; its sampler state is rebuilt from `samples.csv`, with a benign
  warning).
- **Cluster.** `uv sync` on the cluster needs Python 3.13 (uv downloads it). `run_job_*.sh`, the
  `.sub` files and `request_cpus = 16` are unchanged; `n_batch` stays 16. The three vendored
  packages travel with the repo, so nothing outside it has to be checked out.
- **Verification done.** Smoke run of the J18 runner against a scratch copy of `output/`:
  `source_lp[1]` loaded, `source_pix[1]` built with the new model, passed autofit's JAX trace
  preflight and sampled 944 likelihood calls in 5 min on 8 threads; stopped there. The lock
  change passes its scenario test and autofit's paths and Nautilus tests on both branches. No
  full chain has been run under the new stack yet.
- **Watch for.** `output/capped_searches.txt`: any line there is a setup search that hit its call
  cap and should be inspected before its result is trusted downstream. The 25 x 25 shape and the
  AdaptPower coefficients are untuned for COWLS; the scan fixed them at the old
  maximum-likelihood values.
- **Subhalo stages.** They import `autonerves` now and rebuild elliptical multipoles. They were
  not re-run; `prepare_subhalo_baseline.py` and `main_subhalo.py` load `source_pix[1]` and
  `mass_multipole` results by name, so they will pick up the new results once those exist.
- **Not pushed.** Branch `sam` of PyAutoLens-Project is local. The two PyAutoFit branches are on
  the fork.

## Commit map

| Repository / branch | Commit | Change |
| --- | --- | --- |
| PyAutoLens-Project `sam` | `155b071` | RTU 25x25 + AdaptPower source_pix, pyauto dev stack, config, caps |
| PyAutoLens-Project `sam` | `f4f79fa` | re-vendor autofit from `production/hst` `c9264d112` |
| PyAutoLens-Project `sam` | `ef307be` | autofit lock: scheduler-agnostic liveness, heartbeat thread |
| PyAutoLens-Project `sam` | `4efaf22` | autofit lock: locale-free pid start time (vendored follow-up) |
| PyAutoLens-Project `sam` | `7c05a2f` | elliptical multipoles in mass_total and subhalo reset, project prior |
| PyAutoLens-Project `sam` | `e606db7` | GPU runs on phygpu: search_util, env settings, SLURM scripts, autoarray re-vendor |
| PyAutoFit `production/hst` (fork) | `e6fcb6bd8`, `8a9ae493f` | lock change and follow-up |
| PyAutoFit `feature/paths-running-lock-replay` (fork) | `068a66dae`, `f7422754d` | same two, on top of `db6ad6b45` |
