#!/bin/bash
# HTCondor job wrapper for one subhalo tile.
#
# CPUS_PER_TASK is 1 here (matching the other run_job_*.sh wrappers): the tile
# search is a single Nautilus run whose parallelism is controlled by
# number_of_cores / n_batch; we pin BLAS/MKL/OMP threads to 1 so they do not
# oversubscribe the cores Nautilus uses.

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

# $1 = dataset name, $2 = tile index, $3 = filter (optional),
# $4 = source path_prefix (optional), $5 = mass path_prefix (optional)
DATASET="$1"
TILE_INDEX="$2"
FILT="${3:-F444W}"
SRC_PREFIX="${4:-model_setup_free_source}"
MASS_PREFIX="${5:-mass_models}"

exec uv run python main_subhalo.py "$DATASET" "$TILE_INDEX" "$FILT" "$SRC_PREFIX" "$MASS_PREFIX"
