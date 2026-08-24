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
| `main_subhalo.py` | The per-tile job. Loads `source_pix[1]`, `mass_EPL`, and the baseline from `output/`, then runs only its one tile. |
| `combine_tiles.py` | Reads the 25 `tile_###.json` summaries and prints/saves the delta-log-evidence + best-fit mass/centre maps. |
| `run_job_subhalo.sh`, `run_job_subhalo_baseline.sh` | Thread-pinned wrappers (match the existing `run_job_*.sh`). |
| `submission_subhalo_baseline.sub`, `submission_subhalo.sub` | HTCondor submit files (1 baseline job + 25 tile jobs). |

## Why this is not the "rerun everything per tile" version

The reference `main_subhalo.py` you were given re-runs the **entire**
source → light → mass chain inside every one of the 25 tile jobs. At ~days per
chain that is ~25× the chain cost, and each job also fits its own slightly
different no-subhalo baseline (Nautilus is stochastic), which contaminates a
clean ΔlogE comparison.

This version instead:

1. **Loads** `source_pix[1]` and `mass_EPL` from your already-finished
   `main_runner_*.py` output via `af.Aggregator` (chain not re-run).
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
`source_pix[1]` and `mass_EPL`) have finished:

```bash
# 0. Copy these files into the repo root (or run from this dir with PYTHONPATH
#    set so `import slam_pipeline.subhalo.tiling` resolves).

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

### path_prefix defaults

* `main_runner_J24.py` / `main_runner_J24_regularization.py`:
  source prefix `model_setup_free_source` (or `model_setup_free_source_regularized_0`),
  mass prefix `mass_models`.
* `main_runner_J18*.py`: source prefix `model_setup_free_source_weight_3` (or `_4`),
  mass prefix `mass_models`.

Pass them as argv[4]/argv[5] to `main_subhalo.py` / `prepare_subhalo_baseline.py`,
or edit the defaults at the top of those scripts.

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

* `N_LIVE`, `N_BATCH`, `GRID_DIMENSION_ARCSEC`, `NUMBER_OF_TILES`, and
  `SUBHALO_MASS_LIMITS` are at the top of `main_subhalo.py`. The 5×5 / 3.0" /
  200-live-point defaults match the reference.
* `request_cpus = 16` in the `.sub` files matches your existing submissions;
  `number_of_cores=16` is set in each `SettingsSearch`. The shell wrappers pin
  BLAS/OMP threads to 1 (same as your other wrappers).
* The loader uses the standard `af.Aggregator.from_directory(...)` API (the same
  one in `slam_pipeline/subhalo/database.py`), so it auto-extracts zipped
  results on first encounter.
