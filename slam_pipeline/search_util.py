"""
Search and analysis settings shared by every stage
==================================================

One home for what every SLaM stage passes to Nautilus and to `al.AnalysisImaging`, and for the
environment switches that let the same runners fit on a GPU node (phygpu, SLURM) and on the
HTCondor CPU cluster. Ported from Sam's HST project (`z_projects/HST_lensing_mass_light/hpc/
slam_pipeline/search_util.py` and `dataset_util.py`, 2026-10-26).

__Contents__

- **Setup Search Stopping:** The `f_live` the setup searches stop at.
- **Likelihood-Call Caps:** Why every setup search has a hard ceiling, and how a capped search is reported.
- **The Sampler Pool:** Training Nautilus's networks on several CPU cores so a GPU is not idle-killed.
- **Batch Size:** `n_batch` from the environment: 16 on the CPU cluster, 20 on a GPU.
- **Analysis Settings:** Mixed precision and sparse curvature, from the environment, GPU only.
- **CPU Replay:** What keeps a GPU-written archive loadable by the CPU-side subhalo stages.

__Setup Search Stopping__

Nautilus stops building shells once the live set holds less than `f_live` of the evidence. The
setup searches use `SETUP_F_LIVE = 0.05`, not the library's 0.01: searches such as the HST
project's J0029-0055 SOURCE PIX 1 stalled for hours after converging while the last fraction of
the evidence trickled out of the live set. These stages only prepare the source, lens light and
mass start for the mass stages, so the small extra evidence error costs nothing downstream.
`f_live` is not part of a search's output identifier.

__Likelihood-Call Caps__

`f_live` does not catch every stall: a pixelized-source likelihood can be rugged enough that
Nautilus chases it indefinitely (J18's old Delaunay/AdaptSplit `source_pix[1]`: 130k calls, 13
days, ESS 2). Every setup search therefore has a hard ceiling, `SETUP_N_LIKE_MAX`, at about twice
the most calls a converged HST run of that stage needed. A search that reaches it stops and
writes its result as usual (good enough to seed the next stage), but `report_if_capped` says so
in the job log and appends a line to `<output>/capped_searches.txt`, so a capped lens is looked
at rather than trusted silently. `n_like_max` is not part of the output identifier either.

__The Sampler Pool__

On the vectorised (JAX) path Nautilus's own calculations between likelihood batches -- training
the bounding networks, drawing from the bounds -- run on the host, single-threaded, with the GPU
idle. phygpu auto-kills a job whose GPU sits below 5% utilisation for long. `Nautilus` here is
`af.Nautilus` with a CPU process pool for those calculations, sized by
`PYAUTOLENS_SAMPLER_PROCESSES` (default 1 = no pool; the phygpu activation script sets 4). The
pool's workers are spawned with JAX forced to the CPU and CUDA hidden, so none of them ever
touches the GPU, and each gets an equal share of `CPUS_PER_TASK` BLAS threads.

__Batch Size__

`n_batch()` is the number of live points whose likelihoods JAX evaluates in one vmap call:
`PYAUTOLENS_N_BATCH`, default 16 (the CPU jobs' `CPUS_PER_TASK`; CLAUDE.md "JAX and cores"). The
phygpu activation sets 20, the HST project's GPU value. A pixelized fit's peak GPU memory scales
with it; lower it if a job dies out of memory. It is not part of the output identifier (checked
2026-10-26: `source_pix[1]` has the same identifier at 16 and 20), so a chain may change it on
resume, e.g. when moving between the GPU and CPU clusters.

__Analysis Settings__

`analysis_settings()` builds the `al.Settings` every `al.AnalysisImaging` receives.
`PYAUTOLENS_MIXED_PRECISION=1` (set on phygpu) is PyAutoArray's `use_mixed_precision`: the
mapping cube and forward PSF FFT in single precision, upcast at the kernel multiply, so the
curvature matrix, positive-only solve and log-determinants stay double. The HST project measured
0.167 against 0.183 s per call and at most 0.005 nats shift. `PYAUTOLENS_SPARSE_CURVATURE=1`
turns on `use_sparse_curvature` (PyAutoArray `production/hst` 2026-10-26: the curvature matrix's
mapper rows from a transposed PSF blur and a sparse contraction instead of the dense B^T B
product; exact in exact arithmetic, JAX backend only). It is off by default and not yet
benchmarked on COWLS-sized data. Neither is part of the output identifier.

__CPU Replay__

The archives written on phygpu are read on the HTCondor CPU cluster by `proposed_subhalo/`,
which loads `source_pix[1]` and `mass_multipole` by directory. Nothing in an archive is
device-specific: `files/` holds JSON (model, search, settings, samples summary) and CSV
(samples), the images are PNG/FITS, and Nautilus's checkpoint is deleted on completion
(`output.yaml search_internal: false`). What has to match is the code: the CPU side must run
the same commit of this repository (the vendored `pyauto/` packages travel with it) with a plain
`uv sync`; the GPU side adds `--extra gpu`. The settings above are environment-driven with CPU
defaults, so the CPU runs build exactly the analyses they did before. The archives travel through
git: committed from the phygpu clone, merged onto `master`, pulled on the HTCondor cluster
(`phygpu/README.md`).
"""
import atexit
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import autofit as af
import autolens as al
from autonerves import conf

# See "__Setup Search Stopping__".
SETUP_F_LIVE = 0.05

# See "__Likelihood-Call Caps__": ~2x the most calls a converged HST run of each stage needed.
SETUP_N_LIKE_MAX = {
    "source_pix_1": 100_000,
    "source_pix_2": 10_000,  # at 75 live points; source_pix.run_2 scales it by n_live // 75
}


def n_batch() -> int:
    """See "__Batch Size__"."""
    return int(os.environ.get("PYAUTOLENS_N_BATCH", "16"))


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def analysis_settings() -> al.Settings:
    """See "__Analysis Settings__". Says on stdout what it turned on, so a job log shows it."""
    kwargs = {}
    if _flag("PYAUTOLENS_MIXED_PRECISION"):
        kwargs["use_mixed_precision"] = True
    if _flag("PYAUTOLENS_SPARSE_CURVATURE"):
        kwargs["use_sparse_curvature"] = True
    if kwargs:
        print(f"ANALYSIS SETTINGS: {kwargs}", flush=True)
    return al.Settings(**kwargs)


def report_if_capped(search: af.Nautilus, result, stage: str) -> None:
    """
    Warn, and record in `<output>/capped_searches.txt`, when a search stopped at its
    `n_like_max` rather than converging. See "__Likelihood-Call Caps__".
    """
    n_like_max = search.n_like_max
    try:
        total = result.samples.total_samples
    except (AttributeError, KeyError, TypeError):
        return
    if n_like_max is None or n_like_max == float("inf") or total < n_like_max:
        return

    path = str(search.paths.output_path)
    print(
        f"\nWARNING: {stage} STOPPED AT ITS LIKELIHOOD-CALL CAP ({total} >= {n_like_max}) "
        f"without converging: {path}\nCheck this fit by eye before trusting what it seeds.\n"
    )

    log = Path(conf.instance.output_path) / "capped_searches.txt"
    seen = log.read_text() if log.is_file() else ""
    if path not in seen:
        with open(log, "a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M}  {stage}  {total}/{n_like_max}  {path}\n")


# ---- The sampler pool (see "__The Sampler Pool__") -------------------------------------------

_POOL = None


def _noop(_):
    return None


class _ExecutorPool:
    """
    The executor as Nautilus's `NautilusPool` sees it: `map` and a size. Nautilus reads the size
    from the first of `_processes` / `_max_workers` it finds, and an executor's own `_processes`
    is a dict of its worker processes, not a count.
    """

    def __init__(self, executor, processes: int):
        self._executor = executor
        self._max_workers = processes

    def map(self, func, iterable):
        return self._executor.map(func, iterable)


def sampler_processes() -> int:
    """Size of the sampler pool (`PYAUTOLENS_SAMPLER_PROCESSES`, default 1); <= 1 means no pool."""
    return int(os.environ.get("PYAUTOLENS_SAMPLER_PROCESSES", "1"))


def _sampler_pool(processes: int):
    """The process-wide CPU-only spawn pool for Nautilus's sampler calculations."""
    global _POOL
    if _POOL is None:
        threads = str(max(1, int(os.environ.get("CPUS_PER_TASK", "1")) // processes))
        cpu_only = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            cpu_only[key] = threads
        saved = {key: os.environ.get(key) for key in cpu_only}
        os.environ.update(cpu_only)
        try:
            executor = ProcessPoolExecutor(
                max_workers=processes, mp_context=multiprocessing.get_context("spawn")
            )
            # The executor starts workers on demand; start them all now, while the CPU-only
            # environment is in place, so none of them ever sees a GPU.
            list(executor.map(_noop, range(2 * processes)))
            _POOL = _ExecutorPool(executor, processes)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        atexit.register(executor.shutdown, wait=False, cancel_futures=True)
        print(f"NAUTILUS sampler pool: {processes} CPU processes x {threads} threads (spawn executor, JAX on CPU)")
    return _POOL


class Nautilus(af.Nautilus):
    """
    `af.Nautilus` with a CPU pool for the sampler's own calculations on the vectorised (JAX)
    path. See "__The Sampler Pool__". With `PYAUTOLENS_SAMPLER_PROCESSES` unset it is plain
    `af.Nautilus`.
    """

    @property
    def sampler_cls(self):
        from nautilus import Sampler

        processes = sampler_processes()
        if processes <= 1:
            return Sampler

        class PooledSampler(Sampler):
            def __init__(self, *args, pool=None, **kwargs):
                if pool is None and kwargs.get("vectorized"):
                    pool = (None, _sampler_pool(processes))
                super().__init__(*args, pool=pool, **kwargs)

        return PooledSampler
