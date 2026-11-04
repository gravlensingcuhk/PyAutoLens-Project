# Running the SLaM chain on phygpu (GPU), replaying on HTCondor (CPU)

The two runners (`main_runner_J18.py`, `main_runner_J24.py`) fit `source_lp` through
`mass_multipole` on phygpu's GPUs, one lens per GPU. Their archives are committed to the branch
from the phygpu clone and merged onto `master`; the original user then pulls on the HTCondor
cluster and runs `proposed_subhalo/` on CPUs. Everything
GPU-specific is an environment switch with a CPU default, so the same code and config serve both
sides (`slam_pipeline/search_util.py` has the reasoning for each setting).

## Bring-up on phygpu (once)

```bash
ssh phygpu
git clone <this repo> /home/user/lange/phygpu/PyAutoLens-Project
cd /home/user/lange/phygpu/PyAutoLens-Project && git checkout sam   # or master, same commit
uv sync --extra gpu          # CUDA 12 jaxlib; the vendored pyauto/ packages install from the clone
source phygpu/activate.sh && python -c "import jax; print(jax.devices())"   # expect cuda devices
```

The data (`data/cowls_sam/`) is tracked in the repository, so the clone has it. Update the cluster by
`git pull` and `uv sync --extra gpu` again. Nothing on phygpu is edited in place.

## Submitting

```bash
sbatch -J cowls phygpu/multi_run2 COSJ100018+022138 COSJ100024+021749   # both lenses, one node
sbatch -J cowls phygpu/multi_run2 COSJ100018+022138:4 COSJ100024+021749 # J18 at n_batch 4
sbatch -J j18   phygpu/run_lens   COSJ100018+022138                     # one lens, shared node: tests only

# Hilbert chains (Delaunay-on-Hilbert source mesh from source_pix[1] on), one per regularization;
# the tag suffix says which: _a = AdaptSplit (the default), _c = the constant_split control chain
sbatch -J hilb_a --export=ALL,PYAUTOLENS_MESH=hilbert phygpu/multi_run2 COSJ100018+022138 COSJ100024+021749
sbatch -J hilb_c --export=ALL,PYAUTOLENS_MESH=hilbert,PYAUTOLENS_REGULARIZATION=constant_split phygpu/multi_run2 COSJ100018+022138 COSJ100024+021749
```

Each lens runs at its own Nautilus batch size: `cowls_n_batch_for` in `activate.sh` (J18 6, J24 12
with the 50 x 50 mesh), an exported `PYAUTOLENS_N_BATCH` for every lens of a job, or `<lens>:<n>`
for one lens. A runner that runs out of GPU memory restarts itself at `floor(0.8 n_batch)`. Its
log shows `OUT OF MEMORY at n_batch=...: restarting with n_batch=...`, and completed searches
are not refitted (`slam_pipeline/search_util.py` "__Out-Of-Memory Retry__"). Put a lens's
working value back into `cowls_n_batch_for` so the next job does not pay for the restarts.

Logs: `phygpu/outputs/m2_<tag>.<job>.out` (SLURM), `phygpu/outputs/<lens>.<job>.log` (each
runner), `phygpu/outputs/gpu_<tag>.<job>.out` (GPU use every 30 s; summarise with the HST
project's `hpc/gpu_usage.py`). Results: `output_sam_v3/<prefix>/<lens>/<search>/<id>.zip`, the layout
the HTCondor side reads, tracked in git. A job that hits the 48 h wall is resubmitted with the same command; every
search resumes from its folder. `output_sam_v3/capped_searches.txt` lists setup searches that stopped at
their likelihood-call cap; look at those before trusting what they seeded.

Settings `phygpu/activate.sh` exports (unset on the CPU cluster):

| Variable | phygpu | Effect |
| --- | --- | --- |
| `PYAUTOLENS_N_BATCH` | per lens | Nautilus / JAX vmap batch (16 on CPU), set for each runner by the batch scripts (see Submitting). Not in the search identifier. Lowered automatically on an out-of-memory failure. |
| `PYAUTOLENS_MIXED_PRECISION` | 1 | PyAutoArray mixed precision (mapping cube and PSF FFT in fp32, solve in fp64). Not in the identifier. |
| `PYAUTOLENS_SAMPLER_PROCESSES` | 4 | CPU pool for Nautilus's network training, so the GPU is not idle-killed. |
| `PYAUTOLENS_SPARSE_CURVATURE` | 1 | Sparse curvature matrix (autoarray 2026-10-26), on since 2026-10-30 (`PYAUTOLENS_SPARSE_CURVATURE=0` turns it off). The HST project measured 1.8x per call on an A6000 at mesh 50 and an exact result on Delaunay mappers; it is not in the identifier. |
| `PYAUTOLENS_MESH` | unset (`rtu`) | `hilbert` runs the Delaunay-on-Hilbert-points source mesh from `source_pix[1]` on (`slam_pipeline/source_pix.py` "__The Hilbert Mesh__"). Changes the identifier from `source_pix[1]`; shares `source_lp[1]` with the RTU chain: `sbatch -J hilb --export=ALL,PYAUTOLENS_MESH=hilbert phygpu/multi_run2 ...` (an env-prefix `PYAUTOLENS_MESH=hilbert sbatch ...` is not forwarded reliably on phygpu: such a job ran the RTU chain on 2026-11-02; the SLURM log's `mesh ...` line and the runner log's `MESH:` line say what arrived); `PYAUTOLENS_REGULARIZATION=constant_split` gives the smooth control chain, which changes the identifier again. Since 2026-11-02 a runner that reaches a `source_lp[1]` another process is still fitting waits for it (`search_util` "__Waiting For A Shared Search__"), so the Hilbert jobs can be submitted at any time. The per-lens batch sizes (6 / 12) cover the Hilbert meshes too (`cowls_n_batch_for`). |
| `PYAUTOLENS_REGULARIZATION` | unset (`adapt_split`) | Hilbert mesh only: `constant_split` for the HST control. |
| `PYAUTOLENS_PROJECT_ROOT` | the clone | Where the runners find `config/`, `data/` and write `output_sam_v3/`. |

## Getting results to the HTCondor cluster

Results travel through git. On phygpu, once a stage (or the whole chain) has finished:

```bash
cd /home/user/lange/phygpu/PyAutoLens-Project
git add output_sam_v3/model_setup_free_source output_sam_v3/mass_models
git commit -m "J18/J24 outputs from phygpu"
git push origin sam            # then merge sam onto master
```

On the HTCondor cluster: `git pull`, `uv sync`, run `proposed_subhalo/` as before. `.gitignore`
keeps Nautilus checkpoints (`*.hdf5`), `.running` lock directories and `*.replay-*` folders out
of the commit, so only finished archives and their folders are tracked. Commit only finished
stages: a partially written search folder is of no use on the other side.

## What the CPU side needs for a correct replay

- **The same commit of this repository.** The vendored `pyauto/` packages and the pinned
  `autolens`/`autonerves` versions define how an archive's model, settings and samples are read.
  On the HTCondor cluster: `git pull`, `uv sync` (no extra), and run `proposed_subhalo/` as before.
- **Nothing else.** Archives hold JSON and CSV (`files/model.json`, `settings.json`,
  `samples.csv`, ...) and images; Nautilus's checkpoint is dropped on completion. The analysis
  settings saved with a GPU search (`use_mixed_precision`) are also valid on CPU JAX, and the
  subhalo stages build their own analyses with CPU defaults, exactly as before. Finished archives
  replay read-only, so several CPU jobs may read one chain at once.
- **Commit finished stages only.** Checkpoints are not tracked, so an unfinished GPU search cannot
  be resumed on the CPU side anyway; let phygpu finish each stage (resubmit on the wall clock),
  then commit. A search's identifier does not depend on `n_batch`,
  mixed precision or the pool, so the CPU side sees exactly the searches phygpu ran.
