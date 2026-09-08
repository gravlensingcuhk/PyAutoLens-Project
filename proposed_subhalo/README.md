# Tiled subhalo (substructure) analysis

Split the image plane into an `n x n` grid of tiles, farm each tile out to an
**independent job** (one HTCondor job = one Nautilus fit), assemble the per-tile
delta-log-evidence into a map, then **refine** the strongest tile with a
whole-frame fit whose subhalo position is a Gaussian centred on the detection.

This is the cluster-friendly equivalent of the reference `detect.py` grid-search
pipeline (`subhalo_grid_search` + `subhalo_refine`), which runs every grid cell
inside a single process.

This directory is **self-contained**: it ships its own complete
`slam_pipeline/subhalo/` package (`detect.py`, `database.py`, `tiling.py`,
`loaders.py`) and its `.sub` files point `executable`/`initialdir` at
`proposed_subhalo/`, so you run it straight from here with **no assembly step**.

## Files

| File | Purpose |
|------|---------|
| `slam_pipeline/subhalo/tiling.py` | `make_tiles()`, `subhalo_tile()` (one tile fit), `subhalo_refine()` (final whole-frame fit), and the shared `analysis_from()` / lens-model builder. |
| `slam_pipeline/subhalo/loaders.py` | Load completed results from `output/` via the directory `Aggregator` **without re-running** them. |
| `slam_pipeline/subhalo/detect.py` | Reference in-process pipeline (`subhalo_no_subhalo`, `subhalo_grid_search`, `subhalo_refine`). The baseline stage calls `detect.subhalo_no_subhalo`. |
| `slam_pipeline/subhalo/database.py` | Standalone example that scrapes results into a `.sqlite`. Run it directly; it is **not** imported by the pipeline. |
| `prepare_subhalo_baseline.py` | Run **once**: fits the no-subhalo baseline under a tile-independent `unique_tag` so all 16 tiles *and* the refine share one reference evidence. |
| `main_subhalo.py` | The per-tile job. Loads `source_pix[1]`, `mass_multipole` and the baseline, then runs only its one tile. |
| `combine_tiles.py` | Reads the 16 `tile_###.json` summaries and prints/saves the delta-log-evidence + best-fit mass/centre maps. |
| `subhalo_refine.py` | Final job (run once, after the tiles): loads the highest-evidence tile and refits the whole frame with the subhalo centre a Gaussian at that position; writes `refine.json`. |
| `run_job_subhalo*.sh` | Job wrappers. They let XLA use all allocated cores (`CPUS_PER_TASK=16`), matching the JAX-aware `run_job_J18.sh` / `run_job_J24.sh`. |
| `submission_subhalo_baseline.sub`, `submission_subhalo.sub`, `submission_subhalo_refine.sub` | HTCondor submit files (1 baseline + 16 tiles + 1 refine). |

## Why this is not the "rerun everything per tile" version

The reference `main_subhalo.py` you were given re-runs the **entire**
source → light → mass chain inside every one of the 16 tile jobs. At ~days per
chain that is ~16× the chain cost, and each job also fits its own slightly
different no-subhalo baseline (Nautilus is stochastic), which contaminates a
clean ΔlogE comparison.

This version instead:

1. **Loads** `source_pix[1]` and `mass_multipole` from your already-finished
   `main_runner_*.py` output via the directory `Aggregator` (chain not re-run).
2. Fits the **no-subhalo baseline once** (`prepare_subhalo_baseline.py`) under a
   tile-independent `unique_tag = <dataset>_subhalo_baseline`; every tile and the
   refine load that exact result.
3. Each of the 16 jobs runs **only its single subhalo-tile Nautilus fit**.
4. One **refine** job refits the whole frame at the strongest tile.

If a tile starts before the baseline job has finished, it polls and waits
(default up to 6 h); if the baseline never appears it falls back to fitting it
itself (the shared output dir + resume means concurrent tiles don't duplicate
it).

## How to run

Self-contained — run the condor jobs straight from `proposed_subhalo/`, no
assembly. After the SLaM runs producing `source_pix[1]` and `mass_multipole`
have finished:

```bash
# 1. Baseline (one job) -- submit first, or together with the tiles (they wait).
condor_submit submission_subhalo_baseline.sub

# 2. The 16 tile jobs.
condor_submit submission_subhalo.sub

# 3. After all 16 finish, assemble the delta-log-evidence map:
uv run python combine_tiles.py COSJ100024+021749 F444W

# 4. Final refine (one job, after the tiles): refit the whole frame with the
#    subhalo centre a Gaussian at the strongest tile's position. Writes
#    refine.json = refined log-evidence vs the baseline (final significance).
condor_submit submission_subhalo_refine.sub
```

Edit the `arguments =` line in each `.sub` for dataset / filter / prefixes. To
run a stage by hand instead of via condor, e.g.
`uv run python subhalo_refine.py COSJ100024+021749 F444W`.

### path_prefix / name defaults

Both active runners (`main_runner_J24.py`, `main_runner_J18.py`) now write under
the **same** output root trees, so the defaults cover both:

* source prefix `model_setup_free_source`, mass prefix `mass_models`.
* mass **name** `mass_multipole` — the final smooth model (PowerLaw + m=3,4
  multipoles), fit after `mass_EPL` in the same `mass_models` prefix.

The two datasets are distinguished by `unique_tag` (the dataset name), not by
prefix. Override via argv:

* `main_subhalo.py` — argv[4] source prefix, argv[5] mass prefix, argv[6] mass name.
* `prepare_subhalo_baseline.py` / `subhalo_refine.py` — argv[3] source prefix,
  argv[4] mass prefix, argv[5] mass name.

If you re-run `source_pix` into a different prefix (e.g. after a fit-quality
change), update the `arguments =` line in **all three** `.sub` files to match.
(The archived `main_runner_J24_regularization.py` used `model_setup_free_source_regularized_0`.)

## Output layout

```
output/
  subhalo/
    <dataset>_subhalo_baseline/subhalo_base/...        # one shared baseline
    <dataset>_subhalo_tile_000/subhalo_tile_000/...    # per-tile fits
    ...
    <dataset>_subhalo_tile_015/subhalo_tile_015/...
    <dataset>_subhalo_refine/subhalo_refine/...        # final whole-frame fit
  subhalo_tiles/<dataset>/<filter>/
    tile_000.json ... tile_015.json                    # per-tile summaries
    delta_log_evidence.csv                             # written by combine_tiles.py
    best_fit_mass_at_200.csv
    best_fit_concentration.csv
    best_fit_kappa_s.csv
    best_fit_scale_radius.csv
    refine.json                                        # final detection (subhalo_refine.py)
```

## Tile numbering

Row-major, y outer / x inner, matching PyAutoLens' `centre_0 = y`, `centre_1 = x`:

```
for iy in range(n):       # y, top -> bottom
    for ix in range(n):   # x, left -> right
        index = iy*n + ix
```

`combine_tiles.py` prints maps with rows = y, cols = x.

## Bugs in the reference code that are fixed here

1. **`lens.multipole.m = ...`** typo in `detect.py` (sets `m` on a non-existent
   `lens.multipole` for the `multipole_1/_3/_4` cases) — fixed in the bundled
   `detect.py` and in `tiling._lens_model_from`, which set `m` on each multipole
   model itself.
2. **16× chain re-run + 16 stochastic baselines** (see above) — replaced with
   Aggregator loading + one shared baseline.
3. The reference `main_subhalo.py` hard-coded the cosma path but mixed in a
   `summary_dir` relative to cwd; paths are now all rooted at the workspace
   `output/` consistently.
4. `combine_tiles.py` now reports **missing tiles** (NaN cells) instead of
   silently producing a partial map, and writes a mass CSV too.

## Notes / things to double-check on your side

* **Self-contained.** This directory bundles a complete `slam_pipeline/subhalo/`
  package and the three `.sub` files point at `proposed_subhalo/`, so no
  copy/merge step is needed. `database.py` is a standalone example script (it
  runs at import time), so `__init__.py` deliberately does **not** import it.

* **Refine stage.** `subhalo_refine.py` reads the per-tile summaries, picks the
  max-`delta_log_evidence` tile, loads its Result, and refits the whole frame
  with the subhalo centre =
  `winning_tile_result.model_centred_absolute(a=CENTRE_SIGMA_SCALE).galaxies.subhalo.mass.centre`
  (a GaussianPrior at the detected position; mass and multipoles freed from the
  winning tile). `refine.json` holds `delta_log_evidence_refined` = refined logE
  − baseline logE, the final detection significance. Tune `CENTRE_SIGMA_SCALE`
  (Gaussian width scaling) and `N_LIVE` at the top of `subhalo_refine.py`.

* **Aggregator API.** `loaders.py` uses
  `autofit.aggregator.aggregator.Aggregator.from_directory(...)` — **not**
  `af.Aggregator`, which is the *database* aggregator and has no `from_directory`
  (calling it raises `AttributeError`). It still auto-extracts zipped results on
  first encounter.

* **Mass model = `mass_multipole`.** Detection builds on the final smooth model
  (PowerLaw + m=3,4 multipoles), fit *after* `mass_EPL` under `mass_models`. The
  baseline, tiles and refine all default to it, so they agree. The whole pipe
  runs with `chaining=False`, so the lens is taken from
  `mass_result.model.galaxies.lens` (multipoles free, initialised from the
  result) and the `reset_multipoles` branch is not entered — and the bundled
  `detect.py` has the multipole typo fixed anyway if you do turn chaining on.

* **Threading / cores (JAX).** With `use_jax=True` the Nautilus search takes the
  JAX path (`autofit .../nautilus/search.py:130`): parallelisation is *disabled*
  (no process pool) and `number_of_cores` is **ignored**. All parallelism is a
  JAX `vmap` over the `n_batch` live points, executed by XLA across CPU threads.
  The wrappers set `CPUS_PER_TASK=16` (was 1) so XLA uses all allocated cores —
  no pool, so no oversubscription. `request_cpus = 16` in the `.sub` files matches.

* **`N_BATCH` matches the core count.** Under the JAX path `n_batch` is the
  width of the vmapped batch (how many of the 16 cores are used per wave). All
  three stages (`main_subhalo.py`, `prepare_subhalo_baseline.py`,
  `subhalo_refine.py`) set `N_BATCH=16` = one full wave over the 16 cores. If you
  change `request_cpus`, change these to match (or a multiple).

* **Other subhalo knobs** — `N_LIVE`, `GRID_DIMENSION_ARCSEC`, `NUMBER_OF_TILES`,
  `SUBHALO_KAPPA_S_LIMITS`, `SUBHALO_SCALE_RADIUS_KPC_LIMITS` at the top of
  `main_subhalo.py` (4×4 / 3.0" / 200 live points). If you change
  `NUMBER_OF_TILES`, change `queue N**2` in `submission_subhalo.sub` (currently
  `queue 16`) to match.

* **Subhalo mass profile** — the subhalo is an `al.mp.NFWSph` sampled **directly
  in `kappa_s` and `scale_radius` (r_s)**, matching Amvrosiadis et al.:
  independent log-uniform priors `-4 < log₁₀(κ_s) < 0` and `-3 < log₁₀(r_s/kpc) <
  1`. Concentration is **free** — derived post-hoc from κ_s/r_s, *not* tied to a
  mass–concentration relation. The `SUBHALO_SCALE_RADIUS_KPC_LIMITS` knob is in
  **kpc** (as the paper quotes) and is converted to arcsec at the lens redshift
  inside `tiling.py` (`scale_radius_arcsec_limits_from_kpc`). The per-tile JSON
  records `kappa_s`, `scale_radius` (arcsec), `scale_radius_kpc`, and — derived
  under Planck15 from the fitted κ_s/r_s and the lens/source redshifts —
  `concentration` and `mass_at_200`. Plot κ_s vs r_s (Figure 5) directly from the
  chains; overlay constant-M₂₀₀ / Ludlow-offset tracks post-hoc if desired.

* **Cosmetic:** a few `.sub` comments still say "mass_EPL result" when describing
  the mass *prefix* — harmless; the loaded search *name* is `mass_multipole`.
