#!/bin/bash
# HTCondor job wrapper for the final subhalo REFINE fit. Run ONCE, after all
# tile jobs finish: it loads the highest-evidence tile and refits the whole
# frame with the subhalo centre a Gaussian centred at that position.

# use_jax=True -> Nautilus takes the JAX path (no multiprocessing pool;
# number_of_cores ignored). Parallelism is a JAX vmap over n_batch on the XLA
# CPU threads, so let XLA use all allocated cores (no pool -> no oversubscription).
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"   # match request_cpus in the .sub file
export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$CPUS_PER_TASK
export OMP_NUM_THREADS=$CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$CPUS_PER_TASK

# $1 dataset, $2 filter, $3 source prefix, $4 mass prefix, $5 mass name
DATASET="$1"
FILT="${2:-F444W}"
SRC_PREFIX="${3:-model_setup_free_source}"
MASS_PREFIX="${4:-mass_models}"
MASS_NAME="${5:-mass_multipole}"

exec uv run python subhalo_refine.py "$DATASET" "$FILT" "$SRC_PREFIX" "$MASS_PREFIX" "$MASS_NAME"
