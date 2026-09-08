# Tiled subhalo (substructure) analysis

Split the image plane into an `n x n` grid of tiles and farm each tile out to an
**independent job** (one HTCondor job = one Nautilus fit), then assemble the
per-tile delta-log-evidence into a map.

This is the cluster-friendly alternative to `detect.subhalo_grid_search`
(which runs every grid cell inside one process).

## Files

| File | Purpose |
|------|---------|
| `slam_pipeline/subhalo/tiling.py` | `make_tiles()` + `subhalo_tile()` (one tile fit) and the shared `analysis_from()` / lens-model builder. Mirrors `detect.subhalo_grid_search` exactly. |
| `slam_pipeline/subhalo/loaders.py` | Load completed results from `output/` via the Aggregator **without re-running** them. |
| `prepare_subhalo_baseline.py` | Run **once**: fits the no-subhalo baseline under a tile-independent `unique_tag` so all 25 tiles share one reference evidence. |
| `main_subhalo.py` | The per-tile job. Loads `source_pix[1]`, `mass_multipole`, and the baseline from `output/`, then runs only its one tile. |
| `combine_tiles.py` | Reads the 25 `tile_###.json` summaries and prints/saves the delta-log-evidence + best-fit mass/centre maps. |
| `run_job_subhalo.sh`, `run_job_subhalo_baseline.sh` | Job wrappers. They let XLA use all allocated cores (`CPUS_PER_TASK=16`), matching the JAX-aware `run_job_J18.sh` / `run_job_J24.sh`. |
| `submission_subhalo_baseline.sub`, `submission_subhalo.sub` | HTCondor submit files (1 baseline job + 25 tile jobs). |

## Why this is not the "rerun everything per tile" version

The reference `main_subhalo.py` you were given re-runs the **entire**
source → light → mass chain inside every one of the 25 tile jobs. At ~days per
chain that is ~25× the chain cost, and each job also fits its own slightly
different no-subhalo baseline (Nautilus is stochastic), which contaminates a
clean ΔlogE comparison.

This version instead:

1. **Loads** `source_pix[1]` and `mass_multipole` from your already-finished
   `main_runner_*.py` output via the directory `Aggregator` (chain not re-run).
2. Fits the **no-subhalo baseline once** (`prepare_subhalo_baseline.py`) under a
   tile-independent `unique_tag = <dataset>_subhalo_baseline`; every tile loads
   that exact result.
3. Each of the 25 jobs runs **only its single subhalo-tile Nautilus fit**.

If a tile starts before the baseline job has finished, it polls and waits
(default up to 6 h); if the baseline never appears it falls back to fitting it
itself (the shared output dir + resume means concurrent tiles don't duplicate
it).

## How to run

After your SLaM pipeline submissions (the `.sub` files producing
`source_pix[1]` and `mass_multipole`) have finished:

```bash
# 0. ONE-TIME ASSEMBLY. This directory is NOT self-contained: its
#    slam_pipeline/subhalo/ ships only tiling.py + loaders.py, while detect.py
#    and database.py live in the repo-root slam_pipeline/subhalo/. Neither tree
#    is complete alone, and the .sub files point executable/initialdir at the
#    repo root, so assemble everything there:
#
#      cd <repo root>
#      cp proposed_subhalo/slam_pipeline/subhalo/{tiling.py,loaders.py,__init__.py} slam_pipeline/subhalo/
#      cp proposed_subhalo/{main_subhalo.py,prepare_subhalo_baseline.py,combine_tiles.py} .
#      cp proposed_subhalo/{run_job_subhalo.sh,run_job_subhalo_baseline.sh} .
#      cp proposed_subhalo/{submission_subhalo.sub,submission_subhalo_baseline.sub} .
#      chmod +x run_job_subhalo.sh run_job_subhalo_baseline.sh
#
#    (Alternatively keep everything here: copy detect.py + database.py INTO
#    proposed_subhalo/slam_pipeline/subhalo/ and repoint each .sub's
#    executable/initialdir at proposed_subhalo/.)

# 1. Baseline (one job) -- can be submitted first and allowed to finish, or
#    submitted together with the tiles (tiles wait for it).
condor_submit submission_subhalo_baseline.sub

# 2. The 25 tile jobs.
condor_submit submission_subhalo.sub

# 3. After all 25 finish, assemble the map:
uv run python combine_tiles.py COSJ100024+021749 F444W
```

Edit the `arguments =` line in each `.sub` to set dataset / filter /
path_prefixes.

### path_prefix / name defaults

Both active runners (`main_runner_J24.py`, `main_runner_J18.py`) now write under
the **same** output root trees, so the defaults cover both:

* source prefix `model_setup_free_source`, mass prefix `mass_models`.
* mass **name** `mass_multipole` — the final smooth model (PowerLaw + m=3,4
  multipoles), fit after `mass_EPL` in the same `mass_models` prefix.

The two datasets are distinguished by `unique_tag` (the dataset name), not by
prefix. Pass overrides as argv to `main_subhalo.py` (argv[4] source prefix,
argv[5] mass prefix, argv[6] mass name) / `prepare_subhalo_baseline.py`
(argv[3], argv[4], argv[5]), or edit the defaults at the top of those scripts.
If you re-run `source_pix` into a different prefix (e.g. after a fit-quality
change), update the `arguments =` line in **both** `.sub` files to match.
(The archived `main_runner_J24_regularization.py` used `model_setup_free_source_regularized_0`.)

## Output layout

```
output/
  subhalo/
    <dataset>_subhalo_baseline/subhalo_base/...      # one shared baseline
    <dataset>_subhalo_tile_000/subhalo_tile_000/... # per-tile fits
    ...
    <dataset>_subhalo_tile_024/subhalo_tile_024/...
  subhalo_tiles/<dataset>/<filter>/
    tile_000.json ... tile_024.json                  # per-tile summaries
    delta_log_evidence.csv                           # written by combine_tiles.py
    best_fit_mass.csv
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

1. **`lens.multipole.m = ...`** in `detect.py` / the reference tile function
   sets `m` on a non-existent `lens.multipole` for the `multipole_1/_3/_4`
   cases. `tiling._lens_model_from` sets `m` on each multipole model itself.
2. **25× chain re-run + 25 stochastic baselines** (see above) — replaced with
   Aggregator loading + one shared baseline.
3. The reference `main_subhalo.py` hard-coded the cosma path but mixed in a
   `summary_dir` relative to cwd; paths are now all rooted at the workspace
   `output/` consistently.
4. `combine_tiles.py` now reports **missing tiles** (NaN cells) instead of
   silently producing a partial map, and writes a mass CSV too.

## Notes / things to double-check on your side

* **Assembly is required first.** See step 0 in "How to run" — the tiled modules
  and the reference `detect.py`/`database.py` live in different trees and must be
  combined into one `slam_pipeline/subhalo/` package, with the runnable scripts at
  the repo root (where the `.sub` files' `executable`/`initialdir` point).

* **Aggregator API.** `loaders.py` uses
  `autofit.aggregator.aggregator.Aggregator.from_directory(...)` — **not**
  `af.Aggregator`, which is the *database* aggregator and has no `from_directory`
  (calling it raises `AttributeError`). It still auto-extracts zipped results on
  first encounter.

* **Mass model = `mass_multipole`.** Detection builds on the final smooth model
  (PowerLaw + m=3,4 multipoles), which the runner fits *after* `mass_EPL` under
  `mass_models`. Both `main_subhalo.py` and `prepare_subhalo_baseline.py` default
  to it, so the baseline and the tiles agree. This is safe only because the whole
  pipe runs with `use_chained_model=False` / `chaining=False`: the lens is taken
  from `mass_result.model.galaxies.lens` (multipoles free, initialised from the
  result) and the `reset_multipoles` branch is never entered — which matters
  because the reference `detect.py` still has the `lens.multipole.m = ...` typo
  (only `tiling._lens_model_from` fixes it). **If you ever set chaining on, fix
  that typo in `detect.py` first.**

* **Threading / cores (JAX).** With `use_jax=True` the Nautilus search takes the
  JAX path (`autofit .../nautilus/search.py:130`): parallelisation is *disabled*
  (no process pool) and `number_of_cores` is **ignored**. All parallelism is JAX
  `vmap` over the `n_batch` live points, executed by XLA across CPU threads. The
  wrappers therefore set `CPUS_PER_TASK=16` (was 1) so XLA can use all allocated
  cores — there is no pool, so no oversubscription. `request_cpus = 16` in the
  `.sub` files matches.

* **`N_BATCH` should match the core count.** Under the JAX path, `n_batch` is the
  width of the vmapped batch, i.e. how many of the 16 cores are used per wave. The
  SLaM stages were retuned to `n_batch=16` this session, but `main_subhalo.py`
  still sets `N_BATCH=20` (top of file) → two uneven waves (16+4, ~62% packed).
  Set it to **16** (one full wave) or 32 to fully use the 16 cores.

* **Other subhalo knobs** — `N_LIVE`, `GRID_DIMENSION_ARCSEC`, `NUMBER_OF_TILES`,
  `SUBHALO_MASS_LIMITS` at the top of `main_subhalo.py` (5×5 / 3.0" / 200 live
  points). If you change `NUMBER_OF_TILES`, change `queue N**2` in
  `submission_subhalo.sub` (currently `queue 25`) to match.

* **Cosmetic:** a few `.sub` comments still say "mass_EPL result" when describing
  the mass *prefix* — harmless; the loaded search *name* is `mass_multipole`.
