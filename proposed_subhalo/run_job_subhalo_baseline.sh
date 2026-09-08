#!/bin/bash
# HTCondor job wrapper for the single no-subhalo baseline fit.
# Run this FIRST; it must complete before (or alongside) the tile jobs.

# use_jax=True -> Nautilus takes the JAX path (autofit search.py:130): parallelization
# is DISABLED (no multiprocessing pool, number_of_cores ignored). All parallelism is JAX
# vmap over the n_batch live points, executed by XLA on CPU. Pinning threads to 1 forces
# XLA single-threaded (observed speed-up ~1x). Let XLA use all allocated cores instead;
# there is no process pool, so no oversubscription risk.
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"   # match request_cpus in the .sub file
export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$CPUS_PER_TASK
export OMP_NUM_THREADS=$CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$CPUS_PER_TASK

DATASET="$1"
FILT="${2:-F444W}"
SRC_PREFIX="${3:-model_setup_free_source}"
MASS_PREFIX="${4:-mass_models}"

exec uv run python prepare_subhalo_baseline.py "$DATASET" "$FILT" "$SRC_PREFIX" "$MASS_PREFIX"
