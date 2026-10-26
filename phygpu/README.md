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

The data (`data/cowls/`) is tracked in the repository, so the clone has it. Update the cluster by
`git pull` and `uv sync --extra gpu` again. Nothing on phygpu is edited in place.

## Submitting

```bash
sbatch -J cowls phygpu/multi_run2 COSJ100018+022138 COSJ100024+021749   # both lenses, one node
sbatch -J j18   phygpu/run_lens   COSJ100018+022138                     # one lens, shared node: tests only
```

Logs: `phygpu/outputs/m2_<tag>.<job>.out` (SLURM), `phygpu/outputs/<lens>.<job>.log` (each
runner), `phygpu/outputs/gpu_<tag>.<job>.out` (GPU use every 30 s; summarise with the HST
project's `hpc/gpu_usage.py`). Results: `updated_output/<prefix>/<lens>/<search>/<id>.zip`, the layout
the HTCondor side reads, tracked in git. A job that hits the 48 h wall is resubmitted with the same command; every
search resumes from its folder. `updated_output/capped_searches.txt` lists setup searches that stopped at
their likelihood-call cap; look at those before trusting what they seeded.

Settings `phygpu/activate.sh` exports (unset on the CPU cluster):

| Variable | phygpu | Effect |
| --- | --- | --- |
| `PYAUTOLENS_N_BATCH` | 20 | Nautilus / JAX vmap batch (16 on CPU). Not in the search identifier. Lower it if a job dies out of GPU memory. |
| `PYAUTOLENS_MIXED_PRECISION` | 1 | PyAutoArray mixed precision (mapping cube and PSF FFT in fp32, solve in fp64). Not in the identifier. |
| `PYAUTOLENS_SAMPLER_PROCESSES` | 4 | CPU pool for Nautilus's network training, so the GPU is not idle-killed. |
| `PYAUTOLENS_SPARSE_CURVATURE` | unset | Sparse curvature matrix (autoarray 2026-10-26). Off until benchmarked on COWLS data. |
| `PYAUTOLENS_PROJECT_ROOT` | the clone | Where the runners find `config/`, `data/` and write `updated_output/`. |

## Getting results to the HTCondor cluster

Results travel through git. On phygpu, once a stage (or the whole chain) has finished:

```bash
cd /home/user/lange/phygpu/PyAutoLens-Project
git add updated_output/model_setup_free_source updated_output/mass_models
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
