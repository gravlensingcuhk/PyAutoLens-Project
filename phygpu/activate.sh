#!/bin/bash -l
#
# phygpu (SLURM, GPU) environment for the COWLS SLaM runners. Sourced by every batch script here.
# Adapted from the HST project's hpc/activate.sh (z_projects/HST_lensing_mass_light), which
# carries the full reasoning behind each setting.
#
# Layout on phygpu: this repository cloned at $PROJECT_ROOT (branch sam/master), with its venv
# built by `uv sync --extra gpu` (the extra installs the CUDA 12 jaxlib; the vendored pyauto/
# packages come with the clone). Bring-up:
#
#   ssh phygpu
#   git clone <repo> /home/user/lange/phygpu/PyAutoLens-Project && cd $_
#   uv sync --extra gpu
#   sbatch -J cowls phygpu/multi_run2 COSJ100018+022138 COSJ100024+021749
#
# Results land in $PROJECT_ROOT/updated_output with the same layout the HTCondor subhalo stages read
# (updated_output/<prefix>/<lens>/<search>/<id>.zip) and travel through git (README.md here).

export PROJECT_ROOT=${PROJECT_ROOT:-/home/user/lange/phygpu/PyAutoLens-Project}
export PYAUTOLENS_PROJECT_ROOT=$PROJECT_ROOT   # read by main_runner_J18.py / J24.py

source "$PROJECT_ROOT/.venv/bin/activate"

# The cluster profile puts /usr/lib/x86_64-linux-gnu on LD_LIBRARY_PATH, which holds an older
# system CUDA. It outranks the RUNPATH the pip CUDA wheels use to find each other, the wheel's
# cuSPARSE binds the system nvJitLink, fails, and jax silently falls back to CPU.
unset LD_LIBRARY_PATH

# One thread per allocated CPU: the parallelism comes from the GPU (JAX) and the sampler pool.
# multi_run2 owns a whole node and sets COWLS_THREADS to each runner's share of its CPUs.
export CPUS_PER_TASK=${COWLS_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK
export OMP_NUM_THREADS=$CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$CPUS_PER_TASK

# GPU-side settings read by slam_pipeline/search_util.py (all off / CPU values when unset, which
# is what the HTCondor CPU jobs get). None of them enters the search identifier. The batch size
# (PYAUTOLENS_N_BATCH) is per lens: the batch scripts set it for each runner from
# cowls_n_batch_for below.
export PYAUTOLENS_MIXED_PRECISION=${PYAUTOLENS_MIXED_PRECISION:-1}
export PYAUTOLENS_SAMPLER_PROCESSES=${PYAUTOLENS_SAMPLER_PROCESSES:-4}   # Nautilus CPU pool, keeps the GPU from idling
export PYAUTOLENS_SPARSE_CURVATURE=${PYAUTOLENS_SPARSE_CURVATURE:-1}  # exact; HST measured 1.8x per call at large meshes (on since 2026-10-30)

export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1

# Nautilus rewrites checkpoint.hdf5 after every shell under an h5py file lock; a concurrent
# reader's lock would kill the search. Ignore HDF5 locks (HST project, 2026-10-08).
export HDF5_USE_FILE_LOCKING=FALSE

# Headroom on the GPU: XLA's default 0.9 preallocation leaves too little for transient buffers.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# Per-job caches, so concurrent tasks on one node cannot collide.
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-$$}
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-$$}
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

# GPU use every 30 s for the whole job (phygpu auto-kills jobs whose GPU sits below 5% for long);
# written beside the SLURM log as gpu_<job name>.<job id>.out.
cowls_gpu_log_start() {   # $1: the directory to write the log to
    command -v nvidia-smi >/dev/null || return 0
    local select=()
    [ -n "$CUDA_VISIBLE_DEVICES" ] && select=(-i "$CUDA_VISIBLE_DEVICES")
    COWLS_GPU_LOG="$1/gpu_${SLURM_JOB_NAME:-job}.${SLURM_JOB_ID:-$$}.out"
    nvidia-smi "${select[@]}" --query-gpu=timestamp,index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits -l 30 > "$COWLS_GPU_LOG" 2>&1 &
    COWLS_GPU_LOG_PID=$!
    trap cowls_gpu_log_stop EXIT
    echo "GPU use logged every 30 s to $COWLS_GPU_LOG"
}

cowls_gpu_log_stop() {
    [ -n "$COWLS_GPU_LOG_PID" ] && kill "$COWLS_GPU_LOG_PID" 2>/dev/null
    COWLS_GPU_LOG_PID=
}

# The Nautilus / JAX vmap batch for a lens argument `<lens>[:<n>]`: the explicit `:<n>`, else an
# exported PYAUTOLENS_N_BATCH, else the lens's default here. A source_pix search's GPU memory goes
# as n_batch x source pixels x the FFT frame (search_util "__Batch Size__"): J18 (419 px frame) ran
# out of the A6000's 48 GB at 20 x 625 = 12,500 FFTs (job 57684). With the 40 x 40 mesh (1600 pixels)
# J18 starts at 6 (9,600 FFTs); J24's 209 px frame makes each FFT ~4x smaller, so 12. A guess
# that is too high costs one restart: the runners lower it on an out-of-memory failure
# (search_util "__Out-Of-Memory Retry__").
cowls_n_batch_for() {
    case "$1" in *:*) echo "${1#*:}"; return ;; esac
    if [ -n "$PYAUTOLENS_N_BATCH" ]; then echo "$PYAUTOLENS_N_BATCH"; return; fi
    case "$1" in
        COSJ100018+022138) echo 6 ;;
        COSJ100024+021749) echo 12 ;;
        *) echo 20 ;;   # the HST project's GPU batch size
    esac
}

# The runner for a lens: the two runners differ only in filter and pixel scale.
cowls_runner_for() {
    case "$1" in
        COSJ100018+022138) echo "$PROJECT_ROOT/main_runner_J18.py" ;;
        COSJ100024+021749) echo "$PROJECT_ROOT/main_runner_J24.py" ;;
        *) echo "cowls_runner_for: no runner for lens '$1'" >&2; return 1 ;;
    esac
}
